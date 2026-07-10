---
status: complete
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

The minimum Go version for users of the module is **1.26**. Releases are
verified with Go 1.26.5 or newer. Streaming uses standard iterators (`iter`).

## 2. Providers

| Provider | Implementation | Notes |
|---|---|---|
| Anthropic | Wraps official `anthropic-sdk-go` | Messages API |
| OpenAI | Wraps official `openai-go` (direct) | **Responses API** — OpenAI's recommended surface for new projects; enables reasoning continuity + content (see provider_capabilities.md) |
| OpenAI Codex | `providers/openaicodex` — shares the Responses mapping with `providers/openai` | **ChatGPT Plus/Pro subscription** via OAuth — `chatgpt.com/backend-api/codex` (§17C) |
| OpenRouter | Shared OpenAI-compatible adapter (via `openai-go`), preset base URL + auth + typed extensions | `https://openrouter.ai/api/v1` |
| vLLM | `providers/vllm` — preset over the public `chatcompletions` engine; **host-first construction** (self-hosted: `http://host:8000/v1`, key optional via `--api-key` bearer), era-aware (modern v0.12+ default, `WithLegacyEra()`) | Live-tested. Includes fuzzy model resolution and concrete-provider `/tokenize`, `/detokenize`, and `/tokenizer_info` extensions. Full upstream research: `vllm_research.md` |
| Ollama | `providers/ollama` — data-only preset over the public `chatcompletions` engine (`http://localhost:11434/v1` convention) | Offline conformance-tested; not in the credentialed live matrix |
| Custom | Public `Provider` interface — anyone can implement their own; or `chatcompletions.New(baseURL, opts...)` for any OpenAI-compatible server, with quirks declared via the `Compat` struct | |

Decisions:

- **Wrap official SDKs** (`anthropic-sdk-go`, `openai-go`) for their native
  request surfaces. The SDKs are implementation details: ordinary provider
  options use go-llm or standard-library types. Raw clients and the advanced
  `chatcompletions.Dialect`/`Config`/`BuildParams` surface are explicitly
  vendor-coupled and stability-exempt before v1.
- The core `llm` package (interface + types) has **zero third-party
  dependencies**. SDK dependencies live in provider subpackages
  (`providers/anthropic`, `providers/openai`, and packages built on
  `providers/chatcompletions`).
- The OpenAI provider targets the **Responses API** (verified July 2026:
  "Responses is recommended for all new projects"; Chat Completions never
  returns reasoning content and discards reasoning between tool-call
  turns). Stateless operation: the adapter sets `store: false` and
  round-trips encrypted reasoning items. Chat Completions remains fully
  supported upstream; legacy-only knobs are reachable via the raw client.
- OpenRouter, vLLM, and Ollama are presets over one public
  OpenAI-compatible, chat-completions-shaped engine. OpenRouter and vLLM add
  typed provider options and response mappings; Ollama is a data-only preset.
- **Subscription auth is in scope** (§17C): Anthropic gains an OAuth mode
  (Claude Pro/Max) and a separate `openai-codex` provider serves ChatGPT
  Plus/Pro — both *consume and refresh* credentials minted by existing
  CLIs; interactive login flows are deferred.

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
- Conversation history helper (in-memory message management only) + a
  minimal `Session` convenience wrapper — §10, §10D
- Offline model lookup + context-window accounting (`ContextUsage`) — §13
- Unified errors (normalized top layer + provider cause underneath)
- Normalized usage (tokens) and cost (USD)
- Capabilities discovery
- `Models()` listing of available models per configured provider
- Canonical JSON serialization of messages/responses (persistence-safe;
  reasoning raw payloads survive round-trips) — §10A
- Generic structured-output helper `Parse[T]` with schema-from-struct — §8
- Tool schema generation from Go types + tool-argument validation — §7
- Client-level middleware (chat/stream interceptor seam) — §10B
- Minimal `PromptTemplate` helper (`text/template` wrapper) — §10C
- Observability: `slog` logging, usage-telemetry aggregation, wire-level
  debug capture — §17B
- `llmtest` fake provider for downstream testing — §17A
- `cmd/llm-cli` — a curl-like command-line frontend to the library — §19
- Self-hosted / any OpenAI-compatible server: public
  `chatcompletions.New(baseURL)` engine (key-optional, declarative
  `Compat`) + `providers/vllm` preset + data-only `providers/ollama` — §2

**Out of scope (v1):**

- Agent loops, planning, tool execution orchestration (→ `go-agent`)
- Embeddings, image generation, audio (TTS/STT)
- Batch APIs (deferred; candidate for v1.x — the `Provider` interface must
  not preclude adding a batch surface later)
- RAG helpers (chunking, vector stores, retrieval glue)
- Prompt-template *machinery* beyond the minimal helper in §10C: template
  registries, versioning, alternate syntaxes (Jinja/f-string), few-shot
  machinery, message splicing (→ `go-agent` / app layer)
- Skills (prompt bundles with progressive disclosure → `go-agent`, which
  needs an agent loop to load them; provider *server-side* skills —
  Anthropic container skills, OpenAI hosted `skills` tool — are reachable
  via provider extensions, shapes too divergent to unify in v1)
- Conversation persistence, compaction/summarization (no
  `Session.Compact()` — it's an LLM call + policy; go-llm ships the
  primitives: `ContextUsage` trigger, `Chat`/`Parse` for the summary,
  `Messages()` + rebuild for the splice; Anthropic *server-side*
  compaction reachable via `anthropic.Options`), cross-session memory
  (→ `go-agent`)
- MCP client support (revisit later)
- **ZAI provider**: deferred future work. Historical capability research is
  retained in `provider_capabilities.md`, but it is non-normative and no ZAI
  package, model rows, fixture entry, CLI provider, or live scenario ships.
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
- `ChatStream(ctx, Request) iter.Seq2[Event, error]` — streaming completion

Escape hatch: each provider package exposes its concrete type; a type
assertion gives access to the raw underlying SDK client at runtime
(e.g. `p.(*anthropic.Provider).Client()`).

Interface contract (binding for custom providers too):

- **Goroutine-safe**: a `Provider` must be safe for concurrent use — one
  instance, many goroutines (the wrapped SDK clients already are).
- **Streams are single-use**: the iterator returned by `ChatStream` may
  be ranged once; a second range yields a single `ErrBadRequest`-wrapped
  error, never silent emptiness.
- **No panics**: the library never panics on user input or provider
  behavior; the only panicking functions are the documented `Must*`
  variants.

### 4.2 Request

Unified request fields (all optional unless noted):

- `Model string` — **required**; provider-native model ID passed verbatim
  (no aliasing layer)
- `Messages []Message` — **required**
- `System string` — system prompt text; `SystemCache *CacheHint` optionally
  marks it for provider-side prompt caching (§15)
- `MaxTokens int` — optional; for providers that require it (Anthropic),
  the adapter applies a documented, configurable default (16384) when unset
- `Temperature`, `TopP` — optional pointers (unset ≠ zero); passed through;
  provider-specific range constraints are the provider's problem (errors
  surface normally)
- `StopSequences []string` — capability-gated (`stop-sequences`): the
  OpenAI Responses API has no stop parameter → setting it on OpenAI
  returns `ErrUnsupported`
- `Tools []Tool`, `ToolChoice` — §7
- `ResponseFormat` — §8
- `Effort` — reasoning/thinking effort level (§9)
- `SessionID string` — session/routing affinity hint (§9A)
- `ProviderOptions` — typed per-provider extension struct (§14). The field
  is deliberately **singular** — a request targets one provider; routing/
  failover layers that re-dispatch a request across providers must swap or
  strip `ProviderOptions` per target provider (adapters reject options
  whose `ForProvider()` does not match)

### 4.3 Messages and content parts

`Message` = role + ordered content parts + optional provenance.

Roles: `user`, `assistant`, `tool` (tool results), `system` (only where a
provider models it as a message; normally use the request-level `System`).

**Provenance** (pi-validated): assistant messages carry optional
`Provider`/`Model` metadata — stamped by `History.AddResponse`, preserved
by serialization (§10A) — so persisted history is self-describing and
cross-provider replay decisions don't depend on external state.

Content part types:

- `Text`
- `Image` — URL or raw bytes + media type
- `File` — document input (PDF etc.) where supported
- `ToolCall` — assistant-issued tool invocation (id, name, JSON arguments)
- `ToolResult` — result for a tool call (tool-call id, tool name, content,
  `IsError`)
- `Reasoning` — reasoning/thinking output; carries normalized text **plus
  opaque raw payload** (e.g. Anthropic thinking-block signatures, OpenRouter
  `reasoning_details`) so multi-turn round-tripping works when messages are
  replayed
- Provider-specific parts may be registered through the extension mechanism

### 4.4 Response

- `Parts []Part` — ordered parts (text, tool calls, reasoning)
- Convenience accessors: `Text()` (concatenated text), `ToolCalls()`
- `StopReason` — normalized (§5)
- `Usage` — normalized (§11)
- `Provider string` / `Model string` — the provider and model that actually
  served the request (`Model` matters for OpenRouter fallbacks); the source
  for `History.AddResponse` provenance stamping (§4.3, §10)
- `Raw` — access to the raw provider response; provider packages expose
  typed extras (e.g. OpenRouter's `provider`, `native_finish_reason`,
  annotations)

## 5. Stop Reasons (normalized)

`StopReason` values: `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`,
`content_filter`, `refusal`, `context_overflow`, `paused`, `error`, `other`.
The raw provider value is always preserved alongside.

| Provider raw | Normalized |
|---|---|
| Anthropic `end_turn` / CC-shaped `stop` | `end_turn` |
| Anthropic `max_tokens` / CC-shaped `length` | `max_tokens` |
| Anthropic `stop_sequence` | `stop_sequence` |
| Anthropic `tool_use` / CC-shaped `tool_calls` | `tool_use` |
| CC-shaped `content_filter` | `content_filter` |
| Anthropic `refusal` | `refusal` |
| Anthropic `model_context_window_exceeded` | `context_overflow` |
| Anthropic `pause_turn` | `paused` |
| OpenRouter `error` | `error` |
| Anything unrecognized | `other` (raw preserved) |

OpenAI (Responses) has no `finish_reason` — the adapter maps
`status`/`incomplete_details.reason`: `completed` → `end_turn` (or
`tool_use` when `function_call` items are present), `incomplete` +
`max_output_tokens` → `max_tokens`, `incomplete` + `content_filter` →
`content_filter`, `failed` → `error`. The `OpenAI` entries in the table
above are chat-completions vocabulary, which applies to the CC-shaped
adapter and its OpenRouter, vLLM, and Ollama presets.

## 6. Streaming

- `ChatStream` returns an iterator of unified events; errors are yielded
  in-stream (`iter.Seq2[Event, error]`; design rationale in architecture.md).
- Event types: `MessageStart` (id, provider, model — so `Collect` can
  populate `Response.Provider`/`Model`), `TextDelta` (index, text),
  `ReasoningDelta` (index, text), `ToolCallStart` (index, id, name),
  `ToolCallDelta` (index, argument JSON fragment), `ToolCallIDChanged`
  (index, old id, new id), `ToolCallEnd` (index),
  `ToolCallDropped` (index, reason — §7 malformed-call contract),
  `MessageEnd` (stop reason + usage). All content events carry a block
  index. Indexes are stable provider/wire output positions; malformed or
  filtered items may leave gaps and content blocks are not guaranteed
  contiguous.
- **Partial content is never lost**: an in-stream error terminates the
  sequence, and `llm.Collect` returns the partial `*Response` accumulated
  so far *alongside* the error — callers (and `go-agent`) can persist
  aborted/failed turns.
- `ToolCallIDChanged` is an identity correction for an already-started,
  still-provisional tool call, not a drop/restart signal. Direct stream
  consumers replace `OldID` with `NewID` for the active call at `Index`;
  `OldID` must match the identity established by `ToolCallStart` or the
  preceding change. `Collect` applies the same replacement only to that
  matching active call and treats a missing call, ended call, index/type
  collision, or old-id mismatch as a canonical malformed-stream error.
  Identity corrections never create `Response.DroppedToolCalls` metadata.
  `ToolCallDropped` remains reserved for calls the adapter actually abandons.
- A helper (`llm.Collect`) accumulates any stream into a complete
  `*Response`. A text-only consumer adapter, `llm.StreamText`, filters a
  stream to plain text deltas (`iter.Seq2[string, error]`); its
  `WithDebounce(window)` option batches deltas on a time window to
  rate-limit UI re-renders (adopted from fugue-labs/gollem, collapsed to
  one function + options).
- **Per-provider streaming nuances** — normalizing these is a core adapter
  responsibility:

| Provider | Wire format | Nuances the adapter handles |
|---|---|---|
| Anthropic | SSE with typed events (`message_start`, `content_block_start/delta/stop`, `message_delta`, `message_stop`) | Block-indexed deltas (text, `input_json_delta` for tool args, `thinking_delta`); usage on `message_delta`; thinking blocks precede text |
| OpenAI (Responses) | SSE **semantic typed events** (`response.output_text.delta`, `response.function_call_arguments.delta`, `response.reasoning_summary_text.delta`, `response.output_item.added/done`, `response.completed`, …) | Deltas keyed by output-item/content index (no `choices[].delta`); reasoning summary deltas → `ReasoningDelta`; usage arrives on `response.completed`; `response.failed`/`error` events → normalized in-stream error |
| CC-shaped adapter (OpenRouter, vLLM, Ollama, custom) | SSE `chat.completion.chunk` + `data: [DONE]` | Choice index 0 only; tool-call deltas indexed via `delta.tool_calls[].index`; `stream_options.include_usage` set where a dialect requires it; usage-only tails and comment keep-alives tolerated |
| OpenRouter (dialect specifics) | as CC-shaped row | SSE **comment keep-alives** (`: OPENROUTER PROCESSING`) must be skipped; **mid-stream errors** arrive as an HTTP-200 chunk with `finish_reason: "error"` + `error{}` object → normalized in-stream error; usage+cost auto-included in final chunk; `reasoning`/`reasoning_details` on deltas |
| vLLM (dialect specifics) | as CC-shaped row | Reads modern `reasoning` and legacy `reasoning_content`; choice-less error events normalize in-stream; forced tool calls ending with wire `stop` normalize to `tool_use` |

- Usage always surfaces on the `MessageEnd` event regardless of where the
  provider reports it.
- **Every fully consumed successful stream terminates with exactly one
  `MessageEnd`** (enforced by a shared stream-contract wrapper across all
  engines). Failed or canceled streams terminate with their error instead. A
  provider's own completion signal is authoritative: Anthropic's
  `message_stop` yields a `MessageEnd` even without a preceding
  `message_delta` (empty `StopReason` in that case — the server signaled
  completion, so this is not an error).
  A stream that drains to clean EOF *without* any terminal signal is a
  truncation → `ProviderError` wrapping `ErrServer`, never a silent success;
  a 2xx stream with zero content events is likewise `ErrServer`. Consumer
  early-break (abandoning the iterator) is preserved and is not an error.
- Cancellation: context cancellation closes the stream and releases the
  connection. (Note: on OpenRouter, upstream billing stops on disconnect
  only for providers that support cancellation — informational, not
  something go-llm can control.)

## 7. Tool Use

- `Tool` = name, description, JSON Schema input (any JSON-marshalable
  value or raw `json.RawMessage`), optional `Strict` flag (mapped to
  Anthropic/OpenAI strict mode where supported).
- **Schema from Go types**: `InputSchema` may be generated from a Go struct
  via the `schema` subpackage — reflection over `json` + ecosystem-standard
  `jsonschema:"description=…,enum=a|b|c,required|optional"` tags (the
  invopop/eino/fugue-convergent convention), with a per-field modifier
  option and a `JSONSchemaer` self-describe hook — emitting the
  strict-mode-compliant subset providers accept. Hand-written JSON Schema
  remains first-class.
- **Argument validation**: an opt-in helper validates model-emitted tool
  arguments against the tool's schema before the caller dispatches them —
  uniform protection against malformed tool calls across providers.
  Before checking arguments, validation checks the structural requirements
  used by the focused subset: every visited node must have one supported
  string `type`; arrays require object-valued `items`; and `properties`,
  `required`, `enum`, and `additionalProperties` must have their expected
  shapes. Malformed schemas fail with `ErrBadRequest` rather than silently
  passing every argument. This is not a full JSON Schema dialect: unsupported
  constraint keywords such as `oneOf`, `$ref`, and `pattern` are not enforced
  and must not be relied upon. Reflection-derived `schema.For` output conforms
  to the subset; schemas supplied by `JSONSchemaer` or changed through
  `WithModifier` remain caller-controlled and may be rejected.
- **Annotations**: optional, MCP-aligned behavioral hints on `Tool`
  (`ReadOnly`, `Destructive`, `Idempotent`, `OpenWorld`). Informational
  only — never sent to providers; consumed by callers (e.g. `go-agent`
  approval policies, MCP interop).
- `ToolChoice`: `auto` | `none` | `required` | `tool(name)`. Unsupported
  choices return `ErrUnsupported` before any network call when the provider's
  fixed capabilities make that determinable.
- Parallel tool calls are supported; the conversation helper (§10) appends
  all tool results as a **single** message (an Anthropic API requirement,
  and best practice everywhere).
- **Tool-result content modality**: `ToolResultPart.Content` may carry
  images (and PDF files) alongside text — screenshot-returning agent tools.
  Anthropic maps them to native tool_result content blocks; the OpenAI
  Responses adapters (openai, openai-codex) map them to the
  `function_call_output` content-array form (`input_text` / `input_image` /
  `input_file`; text-only results keep the plain string form). The
  chat-completions wire (OpenRouter) genuinely accepts only string content
  for tool messages, so non-text tool-result parts return `ErrUnsupported`
  there.
- **Malformed tool calls never fail the turn and are never silently
  swallowed** (adopted from zero's production design — models and
  providers occasionally emit unusable calls: missing id/name,
  unparseable/truncated arguments, duplicate ids). Adapters must:
  (1) **rescue** what's rescuable — synthesize ids for missing/duplicate
  ones, buffer argument deltas until id+name arrive; (2) when a call is
  truly unusable, **drop it visibly** — a `ToolCallDropped` stream event
  (§6) and a `Response.DroppedToolCalls` entry (index + reason) — so
  callers (`go-agent`) can nudge the model to retry instead of the
  conversation silently derailing.
- **Bounded auto-retry (opt-in)**: `llm.RetryDroppedToolCalls(n)` — a
  shipped middleware (§10B seam) that, when a `Chat` response contains
  dropped tool calls, appends the assistant turn plus a **fixed,
  mechanical** correction and re-calls, up to `n` times (mirroring
  `Parse[T]`'s retry precedent: same failure class — output violating a
  machine-checkable contract; retry turns are likewise ephemeral). The
  correction is **tool-result-shaped when tool calls survived** in the
  assistant turn — each surviving call answered with
  `ToolResultPart{ToolCallID, IsError: true}` carrying the correction
  text (providers reject unanswered tool calls); plain user text only
  when none survived. A failed retry attempt returns the **prior
  successful response alongside the retry error** — both non-nil, mirroring
  `Collect`'s partial contract — so the salvageable turn is persistable and
  the failure (including context cancellation, which propagates) is
  observable. Chat-path only — streaming consumers receive
  `ToolCallDropped` and decide themselves. Ordering note: place it
  *outside* `UsageTracker` so every attempt is metered. Distinct from
  transport retries (`WithMaxRetries` resends a failed request; this
  constructs a new turn after a successful-but-defective response) —
  never one knob. Open-ended nudge *policy* (custom wording, escalation,
  model switching) remains `go-agent`.
- Provider-hosted tools (Anthropic server tools, OpenAI Responses hosted
  tools, OpenRouter plugins) are
  **not unified** in v1 — they are reachable via provider extensions (§14);
  all four have hosted web search in incompatible shapes (v2 unification
  candidate, see provider_capabilities.md). Anthropic's `pause_turn` from
  server tools is normalized to `StopReason: paused`.

## 8. Structured Output

Two levels, capability-flagged separately:

- `json-schema` — full schema-constrained output: Anthropic
  `output_config.format`, OpenAI `text: {format: json_schema}` (Responses),
  OpenRouter `response_format: json_schema`.
- `json-mode` — valid JSON without a supplied schema, where advertised by a
  provider.

`ResponseFormat` carries name + schema + strict flag (schema variant) or
JSON-mode marker. Requesting a level a provider lacks → `ErrUnsupported`.
Adapters **adapt schemas to provider dialects fail-open**: strict-mode
sanitization (e.g. OpenAI strict rejects `format`/`pattern`) degrades to
`strict: false` rather than failing the request (oh-my-pi's
production-proven behavior); the same applies to `Tool.InputSchema`.
OpenRouter note: schema support varies by upstream provider; the OpenRouter
extension offers `provider.require_parameters` to route only to providers
honoring it.

**Generic helper — `llm.Parse[T]`**: derives the JSON schema from `T` (same
generator as tool schemas), sets `ResponseFormat`, calls the provider, and
unmarshals the response into `T`.

- Mode resolution, best-first (override via `WithParseMode`):
  1. **native** — `json-schema` capability: schema enforced server-side;
  2. **forced-tool extraction** — provider has tools + forced tool choice:
     the schema is bound as a single synthetic tool, tool choice forced,
     and the arguments parsed as the result. The convergent "most
     reliable" pattern across eino and fugue-labs/gollem; useful e.g. on
     OpenRouter when the upstream model lacks schema support;
  3. **json-mode + guidance** — JSON mode plus schema guidance appended to
     the system prompt with client-side validation.
- Optional bounded retry (`WithParseRetries(n)`, default 0): on
  invalid/unparseable output, the failed turn plus a correction is
  appended and the request retried, up to `n` times. The correction is
  **mode-aware**: in forced-tool mode it must be a tool *result*
  (`ToolResultPart{ToolCallID, IsError: true}` carrying the error text) —
  providers reject an assistant tool-call turn not followed by matching
  tool results; in the other modes it's a plain user-text correction.
- Optional semantic validation (`WithParseValidator(func(T) error)`): a
  validator failure feeds the same bounded retry with the error text.
- Parse options are **non-generic** (`ParseOption`): `WithParseMode` and
  `WithParseRetries` take no type parameter. Only `WithParseValidator[T]`
  is generic; `Parse[T]` type-asserts the stored validator and a mismatched
  validator type fails the call with `ErrBadRequest` before any provider
  call (exact signatures in architecture §2.9).

## 9. Effort (Reasoning / Thinking Control)

Naming decision: the unified knob is called **`Effort`** — it is the common
term across providers (Anthropic `output_config.effort`, OpenAI
`reasoning.effort`, OpenRouter `reasoning.effort`, vLLM
`reasoning_effort`).
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
| OpenAI | `reasoning: {effort, summary: "auto"}` (Responses) — summaries → `ReasoningPart.Text`, full reasoning items (incl. `encrypted_content`) → `ReasoningPart.Raw` |
| OpenRouter | `reasoning: {effort}` (its `exclude`/`max_tokens` variants live in `openrouter.Options`) |
| vLLM | `reasoning_effort` (top-level; vLLM accepts its own native `max`); thinking toggles via `chat_template_kwargs` live in `vllm.Options` |

**Per-provider effort level support** (adapters map the unified level to the
nearest supported native level; the table is documented in code):

| Unified level | Anthropic (`output_config.effort`) | OpenAI (`reasoning.effort`, Responses) | OpenRouter (`reasoning.effort`) | vLLM (`reasoning_effort`) |
|---|---|---|---|---|
| `minimal` | `low` (nearest) | `minimal` | `minimal` | `minimal` |
| `low` | `low` | `low` | `low` | `low` |
| `medium` | `medium` | `medium` | `medium` | `medium` |
| `high` | `high` | `high` | `high` | `high` |
| `xhigh` | `xhigh` where accepted, otherwise `high` | `xhigh` | `xhigh` | `high` (nearest) |
| `max` | `max` where accepted, otherwise `high` | `xhigh` (nearest) | `max` | `max` (native) |
| `none` | thinking disabled/omitted | `none` | `enabled: false` | `none` |

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
| Anthropic | no-op | Prefix-based caching is automatic; no session key exists |

Capability flag: `session-affinity` (present when the mapping has a real
effect). The field is always safe to set — providers without a mapping
ignore it.

## 10. Conversation History Helper

A minimal in-memory helper (no persistence, no summarization):

- Append user messages
- Append an assistant `*Response` (correctly carrying tool calls and
  reasoning parts, preserving raw payloads for replay, stamping
  `Provider`/`Model` provenance — §4.3)
- Append tool results (grouped into one message)
- Produce `[]Message` for the next request

That is the entire feature. Persistence and memory strategies belong to
`go-agent`. Cross-provider replay drops foreign raw reasoning payloads,
per §18.

## 10A. Canonical Serialization

`Message`, `Part`, `Response`, and `Usage` have a **canonical, versioned
JSON encoding** so applications (and `go-agent`) can persist and reload
conversations without inventing their own format. Raw-byte-preserving
persistence uses the `MarshalMessage` / `MarshalMessages` /
`MarshalResponse` helper APIs; direct `encoding/json` marshaling is only
ordinary JSON interop because the standard library compacts and HTML-escapes
`MarshalJSON` output.

- Parts serialize with a `"type"` discriminator (`text`, `image`, `file`,
  `tool_call`, `tool_result`, `reasoning`, …).
- The helper round trip is **lossless for replay**: `ReasoningPart.Raw` and
  `.Provider` survive helper marshal → unmarshal → helper marshal
  byte-identically, so same-provider reasoning replay (§18) works on
  reloaded history.
- Provider extension parts serialize under a namespaced type
  (`"provider/kind"`); provider packages register their part types for
  decoding.
- An envelope helper encodes `[]Message` with a format `version` field for
  forward migration.
- `Response.Raw` and `Usage.Raw` (in-memory SDK values) are **not**
  serialized — documented; everything normalized is, including
  `Response.Provider`/`Model` provenance (mirroring
  `Message.Provider`/`Model`), so a reloaded `Response` can still stamp
  history via `AddResponse`.

## 10B. Middleware

A single, minimal interceptor seam (validated by convergent designs in
eino and gollem):

- `Middleware` provides optional wrappers for `Chat` and `ChatStream`
  (wrapping a stream iterator is a plain function decoration — no teeing
  machinery).
- Applied by **composition**, not provider support:
  `llm.Wrap(provider, mw...)` returns a decorated `Provider`. Adapters know
  nothing about middleware.
- Use cases: tracing, redaction, request mutation, custom policies. Shipped
  built-ins on this seam: the observability trio (§17B) and
  `RetryDroppedToolCalls` (§7); third-party integrations (OpenTelemetry,
  metrics backends) build on the same seam.

## 10C. Prompt Templates (minimal)

A deliberately thin, stdlib-only helper over `text/template` for
parameterizing prompts inside the library's own vocabulary:

- `NewPromptTemplate(name, text)` / `MustPromptTemplate` — parses a
  standard `text/template`.
- `Format(vars)` — renders to a string (for `System` or `UserText`);
  **strict**: missing variables error (fail loud, no `<no value>`).
- `Partial(vars)` — returns a new template with some variables pre-bound
  (merged at `Format` time; call-time vars win). Templates are immutable.

Hard scope line: no registry, no versioning, no alternate syntaxes, no
message splicing (history splicing is `History`'s job), no few-shot
machinery — those are `go-agent`/app concerns. This is ergonomics over
`text/template`, nothing more.

## 10D. Session (convenience wrapper)

A thin, optional wrapper tying together what multi-turn callers otherwise
wire up by hand — sugar, not a runtime:

- `NewSession(provider, model, opts...)` — options for `System`, `Effort`,
  default `MaxTokens`, `Tools` (`WithSessionTools`) and `ToolChoice`
  (`WithSessionToolChoice`); a `SessionID` is auto-generated (session
  affinity §9A for free) unless overridden.
- `Chat` / `ChatStream` — build the `Request` from session defaults +
  `History` (including session tools), append the user turn and the
  assistant response automatically (streams append on completion via
  `Collect`).
- **Tool loop**: when a response stops at `tool_use`, execute the calls
  yourself, append results with `AddToolResults`, then call
  `Continue(ctx)` — a request from the current history WITHOUT appending
  a user turn. On error, `Chat`/`Continue` leave history unchanged
  (rollback contract).
- `History()`, `Messages()` (serialize via §10A yourself), cumulative
  `Usage()` across the session, and `ContextUsage()` (§13) for
  window-remaining checks.

Hard lines (documented): no tool *execution*, no persistence methods, no
`Compact()`, no memory, not goroutine-safe — those live in `go-agent`,
built on this.

## 11. Usage & Cost

Every response (and `MessageEnd` stream event) carries a normalized `Usage`:

- `InputTokens`, `OutputTokens`, `TotalTokens`
- `CacheReadTokens`, `CacheWriteTokens` (zero when unreported)
- `ReasoningTokens` (zero when unreported)
- `CostUSD *float64` — `nil` when unknown
- `CostSource string` — cost provenance: `"native"` (provider-reported,
  billing-grade) or `"estimated"` (price-table estimate); empty when
  `CostUSD` is nil. Serialized as `cost_source` (omitempty; additive key,
  no envelope version bump). `UsageTracker` sums keep `native` only when
  every component is native; any estimated component marks the sum
  estimated.
- Raw provider usage accessible underneath

**Normalization invariant** (binding on every adapter — providers disagree
natively, so this is where cross-adapter drift is prevented; inspired by
zero's `NormalizeUsage` contract):

- `InputTokens` **excludes** cache tokens; `CacheReadTokens` +
  `CacheWriteTokens` are disjoint additions (total prompt occupancy =
  `InputTokens + CacheReadTokens + CacheWriteTokens`). Anthropic reports
  this shape natively; **OpenAI reports `cached_tokens` as a subset of
  its prompt total, so its adapter subtracts**. The OpenAI Responses and
  Codex adapters apply that subtraction before exposing normalized usage.
- `ReasoningTokens` is an **informational subset of `OutputTokens`**,
  never additive — `ContextUsage` and cost math count output exactly once.
  Exception: when a provider's *native* accounting violates the subset
  property (observed live: OpenRouter reports upstream
  `reasoning_tokens > completion_tokens` for some models), adapters pass
  the native values through unaltered (§18: no silent clamping) — so the
  subset property is a normalization goal, not a cross-provider guarantee.
- `TotalTokens` = prompt occupancy + `OutputTokens`.

Cost sourcing:

1. **Provider-reported** (OpenRouter reports `usage.cost` in USD on every
   response) — used verbatim, `CostSource: "native"`.
2. **Estimated** (`CostSource: "estimated"`) — an optional, user-overridable price table
   (per provider/model: USD per MTok input/output/cache-read/cache-write).
   The shipped table is a **trimmed JSON snapshot of the
   community-maintained models.dev database** plus a hand-maintained
   overrides file (pi's production experience: upstream metadata needs a
   patch table), refreshed by a dev-time script, embedded via `go:embed`,
   parsed lazily on first use, and stamped with a generation date. The
   library never fetches model data at runtime. Estimates are marked as
   estimates via `Usage.CostSource`. No table entry → `CostUSD` nil (and
   `CostSource` empty).

## 12. Capabilities

- `Capabilities() []Capability` where `Capability` is a typed string.
- Standard constants in core: `streaming`, `tools`, `tool-choice-required`,
  `tool-streaming`, `parallel-tools`, `strict-tools`, `json-schema`,
  `json-mode`, `reasoning`, `image-input`, `pdf-input`, `stop-sequences`,
  `prompt-caching`, `session-affinity`, `cost-reporting`, `models-listing`.
- `cost-reporting` means the provider NATIVELY reports the request's cost in
  its responses (e.g. OpenRouter's `usage.cost`); price-table cost
  *estimation* is universal, works for every provider, and is not
  capability-gated.
- Provider-specific capabilities are namespaced strings, e.g.
  `openrouter/routing` and `openrouter/plugins`.
- Helper `llm.CustomCapabilities(p)` filters to non-standard entries.
- Requests using a capability a provider lacks fail fast with
  `ErrUnsupported` (checked before the network call where determinable).

## 13. Models Listing

`Models(ctx)` returns `[]ModelInfo` — `ID` (always), plus best-effort
`DisplayName`, `ContextWindow`, `MaxOutputTokens`, `Pricing`, and
`CanonicalID` (the upstream provider-qualified model identity behind an
aggregator entry, e.g. OpenRouter's `anthropic/claude-x` →
`anthropic/claude-x`; empty when unknown or identical to the row's own
`provider/id`; populated from the generated catalog where known — enables
pricing/capability lookup and same-model handoff across providers) where the
provider reports them:

- Anthropic: `GET /v1/models` (rich capability data)
- OpenAI: `GET /models` (IDs, minimal metadata)
- OpenAI Codex: curated static list (the subscription backend has no public
  model-list endpoint)
- OpenRouter: `GET /models` (rich: pricing, context length, modalities)
- vLLM: `GET /v1/models`, including `max_model_len` when the server reports
  it. `ResolveModel(ctx, preference)` chooses an exact, substring/normalized,
  or token-overlap match, then prefers a Qwen model and finally the first row.
- Ollama and generic Chat Completions: the compatible `/models` endpoint

**Offline lookup & context accounting** (no network; from the embedded
models snapshot §11):

- `llm.LookupModelInfo(provider, modelID) (ModelInfo, bool)` — context
  window, max output, pricing, canonical ID from the embedded table.
- `Usage.ContextUsage(window) ContextUsage` — `{UsedTokens, Window,
  Remaining, UsedPercent}`. Correctly sums prompt occupancy as
  `InputTokens + CacheReadTokens + CacheWriteTokens (+ OutputTokens)` —
  cached tokens are *not* included in `InputTokens`, the classic
  miscalculation this helper exists to prevent. The practical pattern:
  the last response's `ContextUsage` ≈ current conversation size; use it
  as the compaction/pruning trigger (in `go-agent` or your app).
- `Session.ContextUsage()` composes the two (last-turn usage + embedded
  window for the session's model; `ok=false` when the model is unknown
  to the table).

## 14. Provider-Specific Extensions

Requests accept a typed, per-provider options struct (carried opaquely by
core; defined in each provider package). Unified fields always win on
conflict. Current extension surface:

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

**vLLM** (`vllm.Options`):
- Extra sampling (`TopK`, `MinP`, `RepetitionPenalty`, `StopTokenIDs`),
  `ChatTemplateKwargs` (incl. the `EnableThinking` sugar for Qwen-style
  toggles), `XArgs` (`vllm_xargs` engine passthrough). Era selection via
  `vllm.WithLegacyEra()` (pre-v0.12 servers: `reasoning_content` replay).
- `StructuredOutputs` (native `structured_outputs` constraint modes,
  v0.12+): exactly one of `Regex`, `Choice`, `Grammar`, `StructuralTag`
  (raw JSON), plus `WhitespacePattern`; JSON-schema output stays on the
  unified `ResponseFormat` (era-portable spelling). Conflicts fail loud at
  build: set together with `ResponseFormat` → `ErrBadRequest` (one
  constraint system per request; unified owns the slot), set under
  `WithLegacyEra()` → `ErrUnsupported` (modern servers silently ignore the
  removed `guided_*` spelling, so no fallback is emitted), zero/multiple
  modes → `ErrBadRequest`.
- Tokenizer extension methods (escape hatch on the concrete provider, not
  `llm.Provider`): `Tokenize(ctx, *llm.Request)` → `TokenizeResult{Count,
  MaxModelLen, Tokens}` (exact chat-template-aware prompt accounting via
  `POST /tokenize` at the SERVER ROOT — live-probed: `/v1/tokenize` 404 —
  reusing the engine's message/tool conversion and mirroring the
  server-side `reasoning_effort`→`enable_thinking` injection; live parity:
  count == a real chat's `prompt_tokens`, diff 0), `Detokenize`, and
  `TokenizerInfo` (raw JSON; endpoint flag-gated server-side →
  `ErrNotFound` when absent). `TokenizeResult.ContextUsage()` bridges to
  `llm.ContextUsage` — exact occupancy vs the §13 estimate path.
- Known server-coupled behaviors are documented on the package: named
  forced tool calls finishing `stop` (normalized), the constrained-decoding
  × thinking conflict observed on some builds (structured_outputs modes
  corrupt output while thinking is active — doubled choice text observed
  live; disable thinking when constraining), `cached_tokens` requiring
  `--enable-prompt-tokens-details`.

**Anthropic** (`anthropic.Options`): provider-native fields not in the
unified surface (e.g. fine-grained cache TTLs beyond the unified hint and
service tiers), raw SDK access via `Client()`.

**OpenAI** (`openai.Options`): Responses-specific `Store`,
`PreviousResponseID`, `ConversationID`/`Conversation`, `Include`,
`Background`, `HostedTools`, `Verbosity`, `Metadata`, `ServiceTier`,
`SafetyIdentifier`, and `PromptCacheRetention`. These use library-owned or
standard-library types; `HostedTools` is a slice of validated JSON objects so
future hosted-tool shapes do not expose SDK unions. `Client()` remains an
advanced vendor-typed escape hatch.

## 15. Prompt Caching

- Unified: an optional `CacheHint` on content parts / system prompt
  (with optional TTL). Providers with explicit breakpoints honor it
  (Anthropic `cache_control`; OpenRouter passes `cache_control` through to
  Anthropic-family models). Providers with automatic caching such as OpenAI
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
   `Provider`, `HTTPStatus`, `Code` (string — handles OpenAI error types and
   OpenRouter numeric codes),
   `Message`, `RetryAfter`, `Metadata`, and the raw body.

**Canonical HTTP-status mapping** (binding on every provider engine — one
shared classifier in `providerutil` enforces it, so identical statuses
yield identical sentinels across providers):

| Status | Sentinel |
|---|---|
| 401 | `ErrAuth` |
| 402 | `ErrInsufficientCredits` |
| 403 | `ErrPermission` |
| 404 | `ErrNotFound` |
| 408 | `ErrTimeout` |
| 429 | `ErrRateLimited` |
| 503 and 529 | `ErrOverloaded` |
| any other 5xx | `ErrServer` |
| any other 4xx | `ErrBadRequest` |

Provider-native error codes/messages classify first (scoped heuristics —
e.g. `context_length…` codes and "context window/length/limit" messages →
`ErrContextTooLong`; a bare "context" substring deliberately does NOT
match), then the status table, then weak code fallbacks for status-less
in-stream errors (`invalid_request` → `ErrBadRequest`, `server_error` →
`ErrServer`).

OpenRouter 403 moderation carries structured metadata (surfaced via `ProviderError`)
and its status-less mid-stream numeric code `"402"` →
`ErrInsufficientCredits`; mid-stream errors otherwise normalize identically
to pre-stream errors. An empty-but-2xx SSE stream (zero events before EOF)
normalizes to an in-stream `ErrServer` ("provider returned an empty
stream") — never a silent empty success.

## 17. Configuration & Operations

- Per-provider constructors with functional options: `WithAPIKey` (env-var
  fallback: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`),
  `WithAPIKeyFunc(func(ctx) (string, error))` for
  rotating/expiring credentials (OAuth-era tokens, gateways),
  `WithBaseURL`, `WithHTTPClient`, `WithMaxRetries`, `WithTimeout`.
- **`llm.DefaultHTTPClient()`** — go-llm ships its own default HTTP
  client, tuned for LLM workloads, used by every provider constructor
  when `WithHTTPClient` is not passed (adopted from zero's production
  transport hardening):
  - `Client.Timeout: 0` — **never** a whole-response timeout (it would
    kill SSE streams mid-body; the footgun naive custom clients hit);
    time-to-first-byte is bounded by `ResponseHeaderTimeout` (120s), and
    per-request deadlines come from `WithTimeout`/`context`.
  - `IdleConnTimeout: 30s` — evict stale pooled connections quickly.
  - `MaxIdleConnsPerHost` raised (16; Go's default of 2 cripples
    concurrent fan-out against a single API host).
  - **darwin**: `DisableKeepAlives: true` — works around the documented
    macOS stale-connection hang after sleep/wake/network changes (cost:
    per-request HTTP/1.1 + handshake; worth it per zero's production
    data). Overridable like everything else.
  - Every call returns a fresh `*http.Client`; clients share only a private,
    lazily initialized transport and its connection pool. Mutating client
    fields therefore cannot change another caller's client.
- Retries: delegated to the wrapped SDKs where available; the shared
  adapter retries 408/429/5xx and connection errors with exponential
  backoff, honoring `Retry-After`. Default max retries: 2 (SDK convention).
- All calls take `context.Context`; no global mutable request state. The
  write-once part registry, lazily parsed model table, and shared private
  default transport are the only process-wide infrastructure.
- go-llm **never reads configuration files implicitly** — credentials
  come from explicit options or environment variables. For applications
  that keep provider credentials in a file, one explicit opt-in loader is
  provided: **`llm.LoadAuthFile(path)`**, parsing the **pi-compatible
  credential format** — a map of provider-id → credential union
  discriminated by `type`: `{"type": "api_key", "key": "..."}` or
  `{"type": "oauth", "access": "...", "refresh": "...", "expires": <ms>,
  "accountId": "..."}` — accepted either bare (pi's `~/.pi/agent/auth.json`
  parses verbatim) or nested under a `"providers"` wrapper (as the e2e
  harness's `gollm-test.json` does; architecture §9). The caller passes
  loaded credentials to constructors (`WithAPIKey` / `WithAPIKeyFunc` /
  `WithOAuth`); `oauth` credentials serve the subscription auth paths
  (§17C), where providers handle refresh and hand renewed credentials
  back for persistence.

## 17A. Testing Support (`llmtest`)

A small `llmtest` package ships a scriptable fake `Provider` so downstream
code (and `go-agent`) can be tested offline:

- Enqueue canned responses, canned event streams, or errors — consumed in
  order by `Chat`/`ChatStream`.
- Records every received `*Request` for assertions.
- Configurable `Name`/`Capabilities` to exercise capability-gated paths.
- Implements the full `Provider` interface — it is also a permanent
  dogfooding check that the interface stays implementable outside the
  built-in providers.
- Exports `RunConformance`, so the Provider contract is machine-checked:
  every provider package (and `llmtest` itself) runs the suite offline
  against fixture servers — single-use streams, mid-stream cancellation
  without goroutine leaks, concurrent use, panic-freedom on odd requests,
  and `Collect`'s partial-on-error shape.

## 17B. Observability: Logging, Telemetry, Debugging

Three layers — all stdlib-only, all **silent by default**, no globals:

**Logging (`log/slog`).** Every provider constructor accepts
`WithLogger(*slog.Logger)`. With a logger set, adapters log: call
summaries at `Debug` (provider, model, duration, stop reason, usage,
request id where available), retries and rate-limit waits at `Warn`,
request failures at `Error`. No logger (the default) = fully silent.

**Telemetry (usage aggregation).** `llm.NewUsageTracker()` returns a
goroutine-safe aggregator whose `Middleware()` plugs into `llm.Wrap`:
per provider/model — call counts, success/error counts, token sums
(input/output/cache read+write/reasoning), cost sums, durations. Streams
are covered (usage observed on `MessageEnd`). `Stats()` returns a
snapshot suitable for feeding any metrics system. OpenTelemetry and
metrics-backend *integrations* are deliberately not in core — the
tracker, middleware seam, and wire capture are their raw material.
Budget **enforcement** (halting/refusing calls once a token or cost cap
is hit) is workload policy → `go-agent`; it composes as a few-line
middleware over `UsageTracker.Stats()`, and Anthropic's server-side task
budgets remain reachable via `anthropic.Options`. go-llm's job is that
the numbers are always accurate and available; spending decisions belong
to the caller.

**Debugging (wire capture / trace mode).** Every provider constructor
accepts `WithWireCapture(fn func(WireCapture))` — a transport-level tap
beneath the SDK that hands the callback the exact HTTP exchange per
attempt: method, URL, redacted request headers, request body, status,
response headers, buffered response body (SSE streams included), timing,
and transport error. Sensitive headers (`Authorization`, `x-api-key`,
cookies) are **always redacted** — not optional. Each SDK retry attempt
is captured separately. `llm.WireCaptureToLogger(l)` adapts a logger into
a capture fn (full payloads at `Debug`) for instant trace mode. Bodies may
contain user data — storage/retention is the application's
responsibility. (Naming: the whole capture surface uses the "wire"
vocabulary — `WireCapture`, `NewWireTap`, `WithWireCapture`,
`WireCaptureToLogger`.)

## 17C. Subscription Auth (OAuth)

In scope (promoted from the deferred list): **consuming** subscription
credentials minted by existing CLIs (pi, claude, codex). go-llm does not
implement interactive login flows in v1 (PKCE/device-code minting stays
deferred; natural future home: `llm-cli auth login`).

- **Credential shape**: the `oauth` variant of the pi-compatible format
  (§17): `{access, refresh?, expires?, accountId?}`.
- **Provider option**: subscription-capable providers accept
  `WithOAuth(cred, persist)` where `persist` is
  `llm.OAuthPersistenceFunc` (`func(context.Context, AuthCredential) error`)
  — bearer auth, automatic refresh on expiry (plus one forced-refresh retry
  on 401), with renewed credentials handed to the caller for persistence
  (go-llm never writes files). The callback MUST honor its generation context
  and return only after durable persistence. Cancellation, deadline, or any
  returned error prevents publication of the renewed credential and is
  propagated to all waiters with `errors.Is` compatibility. A credential with
  a non-empty refresh token is rejected at provider construction when persist
  is nil. An access-only credential may omit persist because it cannot renew.
  Callers may deliberately pass an explicit context-aware no-op for
  in-memory-only rotation, but doing so risks restart with a stale stored
  refresh token.
- **Anthropic (Claude Pro/Max)** — an auth *mode* on the existing
  provider, not a new one: same Messages API, `Authorization: Bearer` +
  refresh via Anthropic's OAuth token endpoint. **Identity requirements**:
  subscription tokens are served only to Claude-Code-identified traffic —
  per the reference implementations the request path must send
  `anthropic-beta: claude-code-20250219,oauth-2025-04-20`, the
  `claude-cli` user-agent + `x-app: cli` headers, and the Claude Code
  identity line as the first system block (verified against pi's Anthropic
  OAuth implementation). Must be **live-verified with a real
  subscription credential** before this mode is considered done.
- **`openai-codex` (ChatGPT Plus/Pro)** — a separate provider (id
  `openai-codex`, pi-compatible naming; zero calls it "chatgpt"): the
  Responses wire shape against `chatgpt.com/backend-api/codex`, extra
  headers (`chatgpt-account-id` resolved per request from the stored
  credential, `originator`), refresh via OpenAI's OAuth token endpoint.
  Capabilities mirror `providers/openai`.
- Reference implementations: pi `packages/ai/src/utils/oauth/*`, zero's
  codex client — exact endpoints and public client ids are verified from
  those sources at implementation time, not hardcoded here.

## 18. Edge Cases & Behaviors

- **Feature/provider mismatch** → `ErrUnsupported` before the network call
  when statically determinable from fixed provider capabilities.
- **Refusals** (Anthropic) → `StopReason: refusal`; content may be empty or
  partial; callers must check stop reason before reading content — the
  `Text()` accessor returns `""` safely.
- **OpenRouter warm-up empty choices** (HTTP 200, no content) → normalized
  to `ErrServer` (retryable).
- **Reasoning round-trip**: replaying history containing `Reasoning` parts
  re-emits the preserved raw payloads for the same provider; other providers
  receive only what their API accepts (raw payloads dropped).
- **OpenAI statelessness**: the adapter always sends `store: false` +
  `include: ["reasoning.encrypted_content"]` — no server-side conversation
  state unless explicitly opted into via `openai.Options`. Encrypted
  reasoning items round-trip through `ReasoningPart.Raw`, giving OpenAI
  turn-to-turn reasoning continuity in tool loops with zero stored state.
- **Tool-call arguments** are always parsed/emitted as JSON — no raw string
  matching guarantees (providers vary in escaping).
- **Sampling range differences** are passed through; provider validation
  errors surface via the normalized error layer. go-llm does not silently
  clamp model-specific ranges.
- **Verbatim model IDs**: no validation of model names client-side;
  unknown model → provider's `ErrNotFound`/`ErrBadRequest`.
- **Single-choice contract**: go-llm always requests and consumes exactly
  one completion choice (`n` is not part of the unified surface; adapters
  read choice 0).

## 19. `llm-cli` — Command-Line Frontend

A small CLI at `cmd/llm-cli` — "the curl of go-llm": try the library
directly, demo it, and permanently dogfood it (the CLI is the library's
first real consumer and exercises the public API only — no internal
imports).

Usage shape (single-shot, stateless, streams to stdout by default):

```
llm-cli -p anthropic -m claude-opus-4-8 "explain me this error: ..."
echo "long doc" | llm-cli -p openai -m gpt-5.5 -s "summarize stdin"
llm-cli -p openrouter -m z-ai/glm-4.7 --effort high --json "..."
llm-cli -p openai-codex --auth-file ~/.pi/agent/auth.json -m gpt-5.4 "..."
llm-cli models -p openrouter
```

- **Prompt sources**: positional arg, stdin (piped), or both (stdin
  appended as a text part) — curl-style composability.
- **Flags**: `-p/--provider` (anthropic|openai|openai-codex|openrouter),
  `-m/--model`, `-s/--system`, `--effort`, `--max-tokens`, `--temp`
  (alias `--temperature`), `--image <path|url>` / `--file <path|url>`
  (repeatable), `--schema <file.json>` (structured output; prints
  validated JSON; forces the non-streaming path), `--no-stream` (buffer
  and print complete response), `--json` (emit the full canonical
  `Response` JSON incl. usage/cost), `--usage` (usage + cost summary to
  stderr), `--reasoning` (print reasoning deltas to stderr as they
  stream), `--cache-system` (set `SystemCache` — pair with `--usage` to
  observe cache hits), `--session-id`, `--debug` (wire capture to stderr
  via `WireCaptureToLogger`), `--api-key` (for `openai-codex` this carries the
  OAuth **access token**, not an API key — a deliberate semantic overload
  stated in the flag's help text), `--auth-file` (OpenAI Codex only),
  `--base-url`, `--timeout`, `--version`.
- **Conversation files** — multi-turn the curl way (state in files, not
  the process): `--load <file>` prepends a saved conversation (canonical
  envelope, §10A), `--save <file>` writes the updated conversation after
  the response. Loading with a *different* `-p` exercises cross-provider
  history replay directly from the shell.
- **Commands**: default = chat; `models` = list models for a provider
  (table; `--json` for machine output).
- **Keys** from provider env vars (library convention) or `--api-key`.
  OpenAI Codex auth precedence is explicit `--auth-file`, then
  `OPENAI_CODEX_ACCESS_TOKEN`, then compatibility `--api-key`; files are
  never loaded unless the flag is supplied. The compatibility flag exposes
  its value through process argv and often shell history, so the file or env
  forms are preferred. Refreshes from an auth file are persisted atomically.
- **Tool calls are printed, not executed**: with `--tool <file.json>`
  (repeatable tool declarations), a `tool_use` stop prints the tool calls
  as JSON and exits — execution loops are `go-agent` territory; the CLI
  stays a client.
- **Constraints**: stdlib `flag` only (no CLI-framework dependency in the
  module), errors to stderr with the normalized error text, exit code 0
  on success / 1 on error, streaming output unbuffered.
