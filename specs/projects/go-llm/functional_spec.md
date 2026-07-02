---
status: draft
---

# Functional Spec: go-llm

## 1. Overview

`go-llm` is a low-level Go client library providing a single, clean, unified
interface to LLM chat providers. It normalizes the request/response model,
streaming, tool use, structured output, reasoning, multimodal input, errors,
and usage accounting across providers, while always leaving an escape hatch to
provider-specific features and raw clients.

It is a **provider client library**, not an agent framework. A future,
separate `go-agent` package will build agent primitives (loops, planning,
memory persistence) on top of go-llm.

Target Go version: **1.26.x** (latest). Streaming uses standard iterators
(`iter`).

## 2. Providers

| Provider | Implementation | Notes |
|---|---|---|
| Anthropic | Wraps official `anthropic-sdk-go` | Messages API |
| OpenAI | Wraps official `openai-go` | Chat Completions API |
| OpenRouter | Shared OpenAI-compatible adapter (via `openai-go`), preset base URL + auth + typed extensions | `https://openrouter.ai/api/v1` |
| ZAI | Shared OpenAI-compatible adapter, preset base URL + auth + typed extensions | `https://api.z.ai/api/paas/v4/` (also coding-plan base URL constant) |
| Custom | Public `Provider` interface — anyone can implement and register their own | |

Decisions:

- **Wrap official SDKs** (`anthropic-sdk-go`, `openai-go`). They are
  well-maintained, updated same-day with API changes, and provide
  battle-tested SSE parsing, retries, and auth. The SDKs are implementation
  details of each provider package; go-llm's public API is its own.
- The core `llm` package (interface + types) has **zero third-party
  dependencies**. SDK dependencies live in provider subpackages
  (`providers/anthropic`, `providers/openai`, `providers/openrouter`,
  `providers/zai`).
- OpenRouter and ZAI are **presets over one shared OpenAI-compatible
  adapter**, each adding typed provider-specific options and response
  mappings (researched from their current docs; see §6).

## 3. In Scope / Out of Scope

**In scope (v1):**

- Chat/messages (single-shot and multi-turn)
- Streaming
- Tool use / function calling (including parallel tool calls and streaming
  tool-call arguments)
- Structured output (JSON schema + JSON mode)
- Multimodal input: images; PDFs/files where supported
- Reasoning / thinking (unified config + normalized output)
- Prompt caching (unified cache hints + normalized cache usage)
- Retries with backoff; rate-limit awareness (Retry-After)
- Conversation history helper (in-memory message management only)
- Unified errors (normalized top layer + provider cause underneath)
- Normalized usage (tokens) and cost (USD)
- Capabilities discovery
- `Models()` listing of available models per configured provider

**Out of scope (v1):**

- Agent loops, planning, tool execution orchestration (→ `go-agent`)
- Embeddings, image generation, audio (TTS/STT)
- Batch APIs (deferred; candidate for v1.x — the `Provider` interface must
  not preclude adding a batch surface later)
- RAG helpers (chunking, vector stores, retrieval glue)
- Prompt templating (Go's `text/template` already covers it; app-level concern)
- Conversation persistence, summarization, cross-session memory (→ `go-agent`)
- MCP client support (revisit later)
- Provider pricing/catalog maintenance beyond a best-effort, overridable
  price table (see §11)

## 4. Core Model

### 4.1 Provider interface

Every provider implements a small public interface (exact Go signatures in
architecture.md):

- `Name() string` — provider identifier (`"anthropic"`, `"openai"`, …)
- `Capabilities() []Capability` — see §12
- `Models(ctx) ([]ModelInfo, error)` — see §13
- `Chat(ctx, Request) (*Response, error)` — blocking completion
- `ChatStream(ctx, Request) Stream` — streaming completion (iterator-based)

Escape hatch: each provider package exposes its concrete type; a type
assertion gives access to the raw underlying SDK client at runtime
(e.g. `p.(*anthropic.Provider).Client()`).

### 4.2 Request

Unified request fields (all optional unless noted):

- `Model string` — **required**; provider-native model ID passed verbatim
  (no aliasing layer)
- `Messages []Message` — **required**
- `System` — system prompt (text; multiple parts allowed)
- `MaxTokens int` — optional; for providers that require it (Anthropic),
  the adapter applies a documented, configurable default (16384) when unset
- `Temperature`, `TopP` — optional pointers (unset ≠ zero); passed through;
  provider-specific range constraints are the provider's problem (errors
  surface normally)
- `StopSequences []string`
- `Tools []Tool`, `ToolChoice` — §7
- `ResponseFormat` — §8
- `Effort` — reasoning/thinking effort level (§9)
- `SessionID string` — session/routing affinity hint (§9A)
- `ProviderOptions` — typed per-provider extension struct (§14)

### 4.3 Messages and content parts

`Message` = role + ordered content parts.

Roles: `user`, `assistant`, `tool` (tool results), `system` (only where a
provider models it as a message; normally use the request-level `System`).

Content part types:

- `Text`
- `Image` — URL or raw bytes + media type
- `File` — document input (PDF etc.) where supported
- `ToolCall` — assistant-issued tool invocation (id, name, JSON arguments)
- `ToolResult` — result for a tool call (tool-call id, content, `IsError`)
- `Reasoning` — reasoning/thinking output; carries normalized text **plus
  opaque raw payload** (e.g. Anthropic thinking-block signatures, OpenRouter
  `reasoning_details`) so multi-turn round-tripping works when messages are
  replayed
- Provider-specific parts (e.g. ZAI `video_url`, `file_url`) via extension

### 4.4 Response

- `Content []Part` — ordered parts (text, tool calls, thinking)
- Convenience accessors: `Text()` (concatenated text), `ToolCalls()`
- `StopReason` — normalized (§5)
- `Usage` — normalized (§11)
- `Model string` — the model that actually served the request (matters for
  OpenRouter fallbacks)
- `Raw` — access to the raw provider response; provider packages expose
  typed extras (e.g. OpenRouter's `provider`, `native_finish_reason`,
  annotations)

## 5. Stop Reasons (normalized)

`StopReason` values: `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`,
`content_filter`, `refusal`, `context_overflow`, `paused`, `error`, `other`.
The raw provider value is always preserved alongside.

| Provider raw | Normalized |
|---|---|
| Anthropic `end_turn` / OpenAI `stop` / ZAI `stop` | `end_turn` |
| Anthropic `max_tokens` / OpenAI+ZAI `length` | `max_tokens` |
| Anthropic `stop_sequence` | `stop_sequence` |
| Anthropic `tool_use` / OpenAI+ZAI `tool_calls` | `tool_use` |
| OpenAI `content_filter` / ZAI `sensitive` | `content_filter` |
| Anthropic `refusal` | `refusal` |
| Anthropic + ZAI `model_context_window_exceeded` | `context_overflow` |
| Anthropic `pause_turn` | `paused` |
| OpenRouter `error`, ZAI `network_error` | `error` |
| Anything unrecognized | `other` (raw preserved) |

## 6. Streaming

- `ChatStream` returns an iterator of unified events; errors are yielded
  in-stream (`iter.Seq2[Event, error]` or equivalent stream object — final
  shape in architecture.md).
- Event types: `MessageStart` (id, model), `TextDelta`, `ReasoningDelta`,
  `ToolCallStart` (index, id, name), `ToolCallDelta` (argument JSON
  fragment), `ToolCallEnd`, `MessageEnd` (stop reason + usage).
- A helper (`llm.Collect`) accumulates any stream into a complete
  `*Response`.
- **Per-provider streaming nuances** — normalizing these is a core adapter
  responsibility:

| Provider | Wire format | Nuances the adapter handles |
|---|---|---|
| Anthropic | SSE with typed events (`message_start`, `content_block_start/delta/stop`, `message_delta`, `message_stop`) | Block-indexed deltas (text, `input_json_delta` for tool args, `thinking_delta`); usage on `message_delta`; thinking blocks precede text |
| OpenAI | SSE `chat.completion.chunk` + `data: [DONE]` | Usage only in final chunk and **only if** `stream_options: {include_usage: true}` — the adapter sets this automatically; tool-call deltas indexed via `delta.tool_calls[].index` |
| OpenRouter | OpenAI-style SSE | SSE **comment keep-alives** (`: OPENROUTER PROCESSING`) must be skipped; **mid-stream errors** arrive as an HTTP-200 chunk with `finish_reason: "error"` + `error{}` object → normalized in-stream error; usage+cost auto-included in final chunk; `reasoning`/`reasoning_details` on deltas |
| ZAI | OpenAI-style SSE + `data: [DONE]` | Reasoning streams as `delta.reasoning_content` (arrives **before** `delta.content`) → `ReasoningDelta`; tool-argument streaming requires `tool_stream: true` — set automatically when streaming with tools; usage included in final chunk by default |

- Usage always surfaces on the `MessageEnd` event regardless of where the
  provider reports it.
- Cancellation: context cancellation closes the stream and releases the
  connection. (Note: on OpenRouter, upstream billing stops on disconnect
  only for providers that support cancellation — informational, not
  something go-llm can control.)

## 7. Tool Use

- `Tool` = name, description, JSON Schema input (any JSON-marshalable
  value or raw `json.RawMessage`), optional `Strict` flag (mapped to
  Anthropic/OpenAI strict mode where supported).
- `ToolChoice`: `auto` | `none` | `required` | `tool(name)`. Capability-gated:
  ZAI supports only `auto`; anything else returns `ErrUnsupported` before any
  network call.
- Parallel tool calls are supported; the conversation helper (§10) appends
  all tool results as a **single** message (an Anthropic API requirement,
  and best practice everywhere).
- Provider-hosted tools (Anthropic web search, ZAI `web_search`/`retrieval`
  tool types, OpenRouter plugins) are **not unified** in v1 — they are
  reachable via provider extensions (§14). Anthropic's `pause_turn` from
  server tools is normalized to `StopReason: paused`.

## 8. Structured Output

Two levels, capability-flagged separately:

- `json-schema` — full schema-constrained output: Anthropic
  `output_config.format`, OpenAI/OpenRouter `response_format: json_schema`.
- `json-mode` — valid-JSON-but-unschema'd: OpenAI/ZAI `json_object`.

`ResponseFormat` carries name + schema + strict flag (schema variant) or
JSON-mode marker. Requesting a level a provider lacks → `ErrUnsupported`.
OpenRouter note: schema support varies by upstream provider; the OpenRouter
extension offers `provider.require_parameters` to route only to providers
honoring it.

## 9. Effort (Reasoning / Thinking Control)

Naming decision: the unified knob is called **`Effort`** — it is the common
term across providers (Anthropic `output_config.effort`, OpenAI
`reasoning_effort`, OpenRouter `reasoning.effort`, ZAI `reasoning_effort`).
The *output* is called **`Reasoning`** (majority naming; Anthropic calls it
"thinking").

`Effort` is a single typed enum on the request:

- `""` (zero value) — provider default; nothing is sent
- `none` — reasoning explicitly off
- `minimal` | `low` | `medium` | `high` | `xhigh` | `max`

Mapping:

| Provider | Mapping |
|---|---|
| Anthropic | adaptive thinking + `output_config.effort`; `display` summarized so reasoning content is returned |
| OpenAI | `reasoning_effort` |
| OpenRouter | `reasoning: {effort}` (its `exclude`/`max_tokens` variants live in `openrouter.Options`) |
| ZAI | `thinking: {type: enabled/disabled}` + `reasoning_effort` (GLM-5.2) |

**Per-provider effort level support** (adapters map the unified level to the
nearest supported native level; the table is documented in code):

| Unified level | Anthropic (`output_config.effort`) | OpenAI (`reasoning_effort`) | OpenRouter (`reasoning.effort`) | ZAI (`reasoning_effort`, GLM-5.2 only) |
|---|---|---|---|---|
| `minimal` | `low` (nearest) | `minimal` | `minimal` | `minimal` |
| `low` | `low` | `low` | `low` | `low` |
| `medium` | `medium` | `medium` | `medium` | `medium` |
| `high` | `high` | `high` | `high` | `high` |
| `xhigh` | `xhigh` (Opus 4.7+/Sonnet 5/Fable; else `high`) | `high` (nearest) | `xhigh` | `xhigh` |
| `max` | `max` (4.6+; else `high`) | `high` (nearest) | `max` | `max` |
| `none` | thinking disabled/omitted | `none`/omit | `enabled: false` | `thinking: disabled` |

Rules: nearest-level mapping is deterministic and documented; go-llm does
not maintain per-model capability tables — if a *model* rejects a level the
provider error surfaces normally. Reasoning output is normalized to
`Reasoning` parts / `ReasoningDelta` events, with raw payloads preserved for
round-tripping (§4.3).

## 9A. Session Affinity (sticky routing)

A unified request-level `SessionID` provides best-effort routing/cache
affinity across providers — set it once per logical conversation/agent run:

| Provider | Mapping | Effect |
|---|---|---|
| OpenRouter | `session_id` body field | Sticky provider routing across turns |
| OpenAI | `prompt_cache_key` | Cache-hit routing affinity |
| ZAI | `user_id` | Request attribution/affinity (best effort) |
| Anthropic | no-op | Prefix-based caching is automatic; no session key exists |

Capability flag: `session-affinity` (present when the mapping has a real
effect). The field is always safe to set — providers without a mapping
ignore it.

## 10. Conversation History Helper

A minimal in-memory helper (no persistence, no summarization):

- Append user messages
- Append an assistant `*Response` (correctly carrying tool calls and
  reasoning parts, preserving raw payloads for replay)
- Append tool results (grouped into one message)
- Produce `[]Message` for the next request

That is the entire feature. Persistence and memory strategies belong to
`go-agent`.

## 11. Usage & Cost

Every response (and `MessageEnd` stream event) carries a normalized `Usage`:

- `InputTokens`, `OutputTokens`, `TotalTokens`
- `CacheReadTokens`, `CacheWriteTokens` (zero when unreported)
- `ReasoningTokens` (zero when unreported)
- `CostUSD *float64` — `nil` when unknown
- Raw provider usage accessible underneath

Cost sourcing:

1. **Provider-reported** (OpenRouter reports `usage.cost` in USD on every
   response) — used verbatim.
2. **Estimated** — an optional, user-overridable price table
   (per provider/model: USD per MTok input/output/cache-read/cache-write).
   go-llm ships a best-effort snapshot with a clearly documented snapshot
   date; estimates are marked as estimates. No table entry → `CostUSD` nil.

## 12. Capabilities

- `Capabilities() []Capability` where `Capability` is a typed string.
- Standard constants in core: `streaming`, `tools`, `tool-choice-required`,
  `tool-streaming`, `parallel-tools`, `strict-tools`, `json-schema`,
  `json-mode`, `reasoning`, `image-input`, `pdf-input`, `prompt-caching`,
  `session-affinity`, `cost-reporting`, `models-listing`.
- Provider-specific capabilities are namespaced strings, e.g.
  `openrouter/routing`, `openrouter/plugins`, `zai/web-search-tool`,
  `zai/video-input`.
- Helper `llm.CustomCapabilities(p)` filters to non-standard entries.
- Requests using a capability a provider lacks fail fast with
  `ErrUnsupported` (checked before the network call where determinable).

## 13. Models Listing

`Models(ctx)` returns `[]ModelInfo` — `ID` (always), plus best-effort
`DisplayName`, `ContextWindow`, `MaxOutputTokens`, `Pricing` where the
provider reports them:

- Anthropic: `GET /v1/models` (rich capability data)
- OpenAI: `GET /models` (IDs, minimal metadata)
- OpenRouter: `GET /models` (rich: pricing, context length, modalities)
- ZAI: no documented listing endpoint — returns a curated static list
  shipped with the provider package (documented as such)

## 14. Provider-Specific Extensions

Requests accept a typed, per-provider options struct (carried opaquely by
core; defined in each provider package). Unified fields always win on
conflict. Initial extension surface (from docs research, mid-2026):

**OpenRouter** (`openrouter.Options`):
- `Models []string` (fallback list), `Provider` routing prefs (`order`,
  `only`/`ignore`, `allow_fallbacks`, `require_parameters`, `sort`,
  `max_price`, `quantizations`, `zdr`, …), `Plugins` (web search,
  context-compression), `Prediction`, reasoning variants beyond the unified
  `Effort` (`reasoning.exclude`, `reasoning.max_tokens`), extra sampling
  params (`top_k`, `min_p`, `top_a`, `repetition_penalty`), attribution
  headers (`HTTP-Referer`, `X-Title`).
- Response extras: `provider`, `native_finish_reason`, `cost_details`,
  `annotations` (web citations), `reasoning_details`.
- Deliberately **not** exposed (deprecated upstream): `transforms`,
  `usage: {include}`.

**ZAI** (`zai.Options`):
- `Thinking` (`clear_thinking`), `DoSample`, `ToolStream`, `RequestID`,
  `UserID`; non-standard tool types (`web_search`, `retrieval`); extra
  content parts (`video_url`, `file_url`); base URL selection (general vs
  coding-plan endpoint; Anthropic-compatible endpoint constant provided).
- Response extras: `request_id`, `web_search[]` results.

**Anthropic** (`anthropic.Options`): beta headers, provider-native fields not
in the unified surface (e.g. fine-grained cache TTLs beyond the unified
hint, service tiers), raw SDK access via `Client()`.

**OpenAI** (`openai.Options`): equivalent pass-through for OpenAI-specific
fields; raw SDK access via `Client()`.

## 15. Prompt Caching

- Unified: an optional `CacheHint` on content parts / system prompt
  (with optional TTL). Providers with explicit breakpoints honor it
  (Anthropic `cache_control`; OpenRouter passes `cache_control` through to
  Anthropic-family models). Providers with automatic caching (OpenAI, ZAI)
  ignore the hint.
- Cache effectiveness is always observable via `Usage.CacheReadTokens` /
  `CacheWriteTokens`.

## 16. Errors

Two layers, always:

1. **Normalized sentinel errors** (match with `errors.Is`): `ErrAuth`,
   `ErrPermission`, `ErrNotFound` (bad model/endpoint), `ErrBadRequest`,
   `ErrRateLimited`, `ErrInsufficientCredits` (OpenRouter 402),
   `ErrOverloaded`, `ErrServer`, `ErrTimeout`, `ErrContentFiltered`,
   `ErrContextTooLong`, `ErrUnsupported` (capability mismatch),
   plus standard `context.Canceled`/`DeadlineExceeded` pass-through.
2. **Provider cause** (match with `errors.As`): `*llm.ProviderError` with
   `Provider`, `HTTPStatus`, `Code` (string — handles OpenAI error types,
   OpenRouter numeric codes, ZAI numeric-string business codes),
   `Message`, `RetryAfter`, `Metadata`, and the raw body.

Mapping notes from research: ZAI uses numeric-string business codes
(1000/1001/1003 auth → `ErrAuth`; 1261 → `ErrContextTooLong`; 1301 →
`ErrContentFiltered`; 1302 → `ErrRateLimited`; 1308+ → quota); OpenRouter
403 moderation carries structured metadata (surfaced via `ProviderError`);
mid-stream errors normalize identically to pre-stream errors.

## 17. Configuration & Operations

- Per-provider constructors with functional options: `WithAPIKey` (env-var
  fallback: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`,
  `ZAI_API_KEY`), `WithBaseURL`, `WithHTTPClient`, `WithMaxRetries`,
  `WithTimeout`.
- Retries: delegated to the wrapped SDKs where available; the shared
  adapter retries 408/429/5xx and connection errors with exponential
  backoff, honoring `Retry-After`. Default max retries: 2 (SDK convention).
- All calls take `context.Context`; no global state.

## 18. Edge Cases & Behaviors

- **Feature/provider mismatch** → `ErrUnsupported` before the network call
  when statically determinable (e.g. `ToolChoice: required` on ZAI).
- **Refusals** (Anthropic) → `StopReason: refusal`; content may be empty or
  partial; callers must check stop reason before reading content — the
  `Text()` accessor returns `""` safely.
- **OpenRouter warm-up empty choices** (HTTP 200, no content) → normalized
  to `ErrServer` (retryable).
- **Reasoning round-trip**: replaying history containing `Reasoning` parts
  re-emits the preserved raw payloads for the same provider; other providers
  receive only what their API accepts (raw payloads dropped).
- **Tool-call arguments** are always parsed/emitted as JSON — no raw string
  matching guarantees (providers vary in escaping).
- **Sampling range differences** (e.g. ZAI temperature ∈ [0,1]) are passed
  through; provider validation errors surface via the normalized error
  layer. go-llm does not silently clamp.
- **Verbatim model IDs**: no validation of model names client-side;
  unknown model → provider's `ErrNotFound`/`ErrBadRequest`.
