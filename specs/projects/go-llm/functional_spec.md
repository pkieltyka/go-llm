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
verified with Go 1.26.6 or newer. Streaming uses standard iterators (`iter`).

## 2. Providers

| Provider | Implementation | Notes |
|---|---|---|
| Anthropic | Wraps official `anthropic-sdk-go` | Messages API |
| OpenAI | Wraps official `openai-go` (direct) | **Responses API** — OpenAI's recommended surface for new projects; enables reasoning continuity + content (see provider_capabilities.md). Authenticates with an API key or **Sign in with ChatGPT** OAuth (a ChatGPT subscription using the API, §17C) |
| OpenRouter | Shared OpenAI-compatible adapter (via `openai-go`), preset base URL + auth + typed extensions | `https://openrouter.ai/api/v1` |
| vLLM | `providers/vllm` — current-stable v0.26.0 preset over the public `chatcompletions` engine; **host-first construction** (self-hosted: `http://host:8000/v1`, key optional via `--api-key` bearer) | Live-tested. Includes fuzzy model resolution and concrete-provider `/tokenize`, `/detokenize`, and `/tokenizer_info` extensions. Full upstream research: `vllm_research.md` |
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
  supported upstream; Chat Completions-only knobs are reachable via the raw
  client.
- OpenRouter, vLLM, and Ollama are presets over one public
  OpenAI-compatible, chat-completions-shaped engine. OpenRouter and vLLM add
  typed provider options and response mappings; Ollama is a data-only preset.
- **Subscription auth is in scope for OpenAI** (§17C): the `openai`
  provider accepts Sign in with ChatGPT OAuth credentials from a
  `CredentialStore`, refreshes them, and mints them through a browser PKCE
  login (`openai.NewLoginFlow`). The former `openai-codex` provider was
  removed (2026-10-22), mirroring pi. Anthropic supports API keys only.

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
The raw provider value is always preserved alongside. `context_overflow`,
`content_filter`, and `refusal` describe a generation the provider stopped;
a rejected request surfaces as the matching error sentinel instead, never
both (the one-channel rule in §16).

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
- **Incremental collection**: `llm.Collector` (zero value ready) is the
  event-at-a-time form of `Collect`, which is defined as a loop over
  `Collector.Add` followed by `Collector.Finish`, so the two share one
  precedence. `Add` normalizes and folds one event and returns the canonical
  malformed-stream error, which is sticky (later `Add` calls return it, and
  it wins over a terminal stream error in `Finish`). `Snapshot` returns an
  owned deep copy of what `Collect` would return if the stream failed now
  (nil before `MessageStart` and any content), so a consumer can forward
  deltas while periodically persisting the partial response. As with
  `Response.Clone`, the in-memory SDK payloads (`Response.Raw`,
  `Usage.Raw`) are shared by reference. `Finish(err)`
  is idempotent, and `Add` after it is `ErrBadRequest`. `Session` streaming
  collects through the same type. `llm.NormalizeEvent` exports the
  pointer-to-value event normalization these paths share.
- A helper (`llm.Collect`) accumulates any stream into a complete
  `*Response`. A text-only consumer adapter, `llm.StreamText`, filters a
  stream to plain text deltas (`iter.Seq2[string, error]`); its
  `WithDebounce(window)` option batches deltas on a time window to
  rate-limit UI re-renders.
- **Per-provider streaming nuances** — normalizing these is a core adapter
  responsibility:

| Provider | Wire format | Nuances the adapter handles |
|---|---|---|
| Anthropic | SSE with typed events (`message_start`, `content_block_start/delta/stop`, `message_delta`, `message_stop`) | Block-indexed deltas (text, `input_json_delta` for tool args, `thinking_delta`); usage on `message_delta`; thinking blocks precede text |
| OpenAI (Responses) | SSE **semantic typed events** (`response.output_text.delta`, `response.function_call_arguments.delta`, `response.reasoning_summary_text.delta`, `response.output_item.added/done`, `response.completed`, …) | Deltas keyed by output-item/content index (no `choices[].delta`); reasoning summary deltas → `ReasoningDelta`; usage arrives on `response.completed`; `response.failed`/`error` events → normalized in-stream error |
| CC-shaped adapter (OpenRouter, vLLM, Ollama, custom) | SSE `chat.completion.chunk` + `data: [DONE]` | Choice index 0 only; tool-call deltas indexed via `delta.tool_calls[].index`; `stream_options.include_usage` set where a dialect requires it; usage-only tails and comment keep-alives tolerated |
| OpenRouter (dialect specifics) | as CC-shaped row | SSE **comment keep-alives** (`: OPENROUTER PROCESSING`) must be skipped; **mid-stream errors** arrive as an HTTP-200 chunk with `finish_reason: "error"` + `error{}` object → normalized in-stream error; usage+cost auto-included in final chunk; `reasoning`/`reasoning_details` on deltas |
| vLLM (dialect specifics) | as CC-shaped row | Reads `reasoning`; choice-less nested error events normalize in-stream; forced tool calls ending with wire `stop` normalize to `tool_use` |

- Usage always surfaces on the `MessageEnd` event regardless of where the
  provider reports it.
- **Every fully consumed successful stream terminates with exactly one
  `MessageEnd`** (enforced by a shared stream-contract wrapper across all
  engines), and its normalized `StopReason` is non-empty. Failed or canceled
  streams terminate with their error instead. Anthropic requires a
  stop-bearing `message_delta` before `message_stop`. Chat Completions
  requires a choice-0 `finish_reason`; a narrowly configured
  `InferMissingFinishReason` compatibility flag may infer `tool_use` or
  `end_turn` only at clean `[DONE]`/EOF while leaving the raw reason empty.
  Missing terminal state, an empty 2xx stream, or an empty `MessageEnd`
  normalizes to a partial-stream `ProviderError` wrapping `ErrServer`.
  Consumer early-break is preserved and is not an error.
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
- The shared Chat Completions adapter requires an object tool-schema root
  (an absent root `type` or `type: "object"` is accepted). Missing or null
  `properties` is normalized on a decoded copy to `{}` so no-argument tools
  work across compatible endpoints. A non-object root/type or non-object
  `properties` value fails with `ErrBadRequest` before network I/O; caller
  schemas are never mutated.
- `ToolChoice`: `auto` | `none` | `required` | `tool(name)`. Unsupported
  choices return `ErrUnsupported` before any network call when the provider's
  fixed capabilities make that determinable.
- Parallel tool calls are supported; the conversation helper (§10) appends
  all tool results as a **single** message (an Anthropic API requirement,
  and best practice everywhere).
- **Tool-result content modality**: `ToolResultPart.Content` may carry
  images (and PDF files) alongside text — screenshot-returning agent tools.
  Anthropic maps them to native tool_result content blocks; the OpenAI
  Responses adapter maps them to the
  `function_call_output` content-array form (`input_text` / `input_image` /
  `input_file`; text-only results keep the plain string form). Chat
  Completions uses string content by default. Its explicit
  `ToolMessageContentBlocks` compatibility mode preserves text boundaries
  and `CacheHint` values as `cache_control` blocks; the OpenRouter preset
  enables this live-verified form. Image/file tool-result parts remain
  `ErrUnsupported` on the Chat Completions surface.
- **Malformed tool calls never fail the turn and are never silently
  swallowed**. Models and providers occasionally emit unusable calls:
  missing id/name, unparseable/truncated arguments, or duplicate ids.
  Adapters must:
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
`strict: false` rather than failing the request; the same applies to
`Tool.InputSchema`.
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
     and the arguments parsed as the result; useful e.g. on OpenRouter when
     the upstream model lacks schema support;
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
| OpenAI | `reasoning: {effort, summary: "auto"}` (Responses) by default; `openai.Options.ReasoningSummary` can select the closed `auto`/`concise`/`detailed` vocabulary without changing `Effort` — summaries → `ReasoningPart.Text`, full reasoning items (incl. `encrypted_content`) → `ReasoningPart.Raw` |
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
| `xhigh` | `xhigh` where accepted, otherwise `high` | `xhigh` | `xhigh` | `xhigh` |
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
ignore it. OpenAI preserves IDs up to 64 runes exactly; longer IDs
become a 47-rune prefix plus `-` and a 16-character SHA-256 suffix only at
the Responses wire boundary. Stored history and OpenRouter `session_id`
remain unchanged.

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

A single, minimal interceptor seam:

- `Middleware` provides optional wrappers for `Chat` and `ChatStream`
  (wrapping a stream iterator is a plain function decoration — no teeing
  machinery).
- Applied by **composition**, not provider support:
  `llm.Wrap(provider, mw...)` returns a decorated `Provider`. Adapters know
  nothing about middleware.
- Use cases: tracing, redaction, request mutation, custom policies. Shipped
  built-ins on this seam: the observability trio (§17B),
  `RetryDroppedToolCalls` (§7), and `ContextGuard` (§13A); third-party
  integrations (OpenTelemetry, metrics backends) build on the same seam.
- A wrapped provider exposes `Unwrap() Provider`, so optional extensions
  such as `TokenCounter` stay discoverable through `Wrap` (`llm.CountTokens`
  follows the chain).

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
- `CacheWrite1hTokens`: the informational subset of `CacheWriteTokens`
  written with a one-hour lifetime (Anthropic `cache_creation.
  ephemeral_1h_input_tokens`, produced by `CacheHint.TTL > 5m`), never
  additive. Serialized as `cache_write_1h_tokens` (omitempty; additive key,
  no envelope version bump).
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
natively, so this is where cross-adapter drift is prevented):

- `InputTokens` **excludes** cache tokens; `CacheReadTokens` +
  `CacheWriteTokens` are disjoint additions (total prompt occupancy =
  `InputTokens + CacheReadTokens + CacheWriteTokens`). Anthropic reports
  this shape natively; **OpenAI reports `cached_tokens` as a subset of
  its prompt total, so its adapter subtracts**. The OpenAI Responses
  adapter applies that subtraction before exposing normalized usage.
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
   Pricing may include ascending input-occupancy tiers. The highest threshold
   strictly exceeded by `InputTokens + CacheReadTokens + CacheWriteTokens`
   supplies the rates for the entire request; exact equality stays on the
   lower tier. The snapshot fills a rate missing from a tier with the base
   rate; a rate absent from both stays unknown rather than reading as free.
   At estimation time a selected tier never falls back to base rates.
   Invalid caller-supplied tiers are ignored. Native cost always wins.
   **One-hour cache writes** have no published rate in any upstream table,
   so a nonzero `CacheWrite1hTokens` leaves an estimated `CostUSD` nil
   (unknown) rather than pricing those writes at the default write rate,
   which would understate cost (Anthropic bills one-hour writes at 2× input
   versus 1.25× for five-minute writes). This follows the
   no-curated-pricing rule: a multiplier constant is never hand-maintained.
   When upstream publishes a one-hour rate, it joins `ModelPricing`.
   The shipped table is a **trimmed JSON snapshot of the
   community-maintained models.dev database** and OpenRouter's catalog,
   refreshed by a dev-time script, embedded via `go:embed`,
   parsed lazily on first use, and stamped with a generation date. Upstream
   data is not hand-curated: a gap stays unknown and a wrong value is fixed
   upstream (models.dev or OpenRouter). The overrides file is kept empty as an emergency escape
   hatch. The
   embedded-table API never fetches model data at runtime. Explicit provider
   `Models(ctx)` calls may use that provider's remote model endpoint.
   Estimates are marked as estimates via `Usage.CostSource`. No table entry →
   `CostUSD` nil (and `CostSource` empty).

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
pricing/capability lookup and same-model handoff across providers),
`SupportedEfforts`, advisory `DefaultEffort`, advisory `ReasoningRequired`,
and per-model `Capabilities` where the provider reports them.
For `SupportedEfforts`, nil means the ladder was omitted or is unknown; a
non-nil empty slice means the provider supplied a ladder containing no known
unified effort values. The ladder lists the unified values the upstream source
advertises for the model; it is not the set go-llm can map, since §9
nearest-level mapping accepts every `Effort` (e.g. Anthropic rows list native
levels such as `low..max` while `EffortNone` still disables thinking).
`llm.SupportedEffortsForModel` reads the embedded table only; a model without
upstream effort metadata returns nil, and no ladder is inferred from its name.
`ModelPricing.Availability`, when non-nil, is authoritative per component and
distinguishes an explicit zero/free input, output, cache-read, or cache-write
rate from an unavailable component. A nil availability value preserves the
legacy `ModelPricing` interpretation for callers constructing values directly.
Cost estimation remains unknown when usage consumes an unavailable component.
`ModelPricingTier.Availability` follows the same contract for a selected
request-wide tier (nil means all four tier rates are known), so a tier is
estimable whenever the usage avoids its unavailable components.
`ReasoningRequired == false` means false or unknown. These per-model fields
never reject or rewrite a request; provider-wide `Capabilities()` remains the
sole request-preflight authority, and `Request.Effort` is forwarded unchanged.
Model discovery is caller-triggered network I/O only when `Models(ctx)` is
explicitly invoked; construction, chat, and streaming do not fetch catalogs.

- Anthropic: `GET /v1/models` (rich capability data)
- OpenAI: `GET /models` (IDs, minimal metadata)
- OpenRouter: one `GET /models` per explicit call (rich: pricing, context
  length, input modalities, supported parameters, and reasoning policy). No
  models.dev or other fallback request is performed. Explicit catalog fields
  populate advisory tool, parallel-tool, reasoning, structured-output,
  image-input, and stop capabilities plus normalized supported/default effort
  and mandatory-reasoning metadata. Unknown or malformed advisory fields stay
  in `Raw` without failing the row. Every input, output, cache-read, and
  cache-write component is decoded independently: explicit zero is known/free;
  missing, JSON `null`, non-numeric, malformed, negative/dynamic, overflowing,
  and non-finite values are unknown rather than negative or free. Valid sibling
  rates and the complete copied live row remain available.
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

## 13A. Context Fitting

Explicit, opt-in helpers that answer "how big is this request, and does it
fit?". None runs implicitly, and none changes a request unless the caller
asks (ARCH §10: no silent clamping).

- **`llm.EstimateRequestTokens(req)`** is a deterministic,
  provider-neutral estimate covering system text, message parts, tool
  definitions, and response format. It is calibrated (2026-10-22) against
  vLLM's exact `/tokenize` on Qwen3.8: prose, Go code, indented JSON, tool
  loops, and a 20-message conversation land at 0.95–1.17× of the exact
  count. Two known cases run higher; both are described below. The rules:
  - Text by character class: ASCII letters and digits at 4 per token,
    punctuation at 0.6 tokens each, each newline and each whitespace
    character after the first in a run at 0.5, and each non-ASCII rune at
    1 token. CJK is about 1.5× high on CJK-efficient tokenizers such as
    Qwen's, and lower on others.
  - Fixed overheads: 8 per request, 4 per message, a one-time 300-token
    tool-use preamble when tools are offered, plus 8 per tool and its
    definition JSON. The preamble dominates tiny prompts, which therefore
    estimate higher (1.36× for a one-tool, two-message request).
  - Tool-call IDs (in calls and results) count as text.
  - Images count 1600 tokens each. That is Anthropic's per-image cap, so on
    providers that tile large images (for example vLLM-served vision models)
    it is a floor, not a bound.
  - Inline PDFs count 2500 per page object found (minimum 2000). Other
    non-text files count 2000. Text-typed file data counts as text.
  - Reasoning counts its visible text only, and extension parts count
    nothing.
  - It never claims exactness. Constants change only with recorded
    calibration evidence.
- **`llm.EstimateContinuationTokens(last, trailing)`** is the last
  response's full occupancy (input, cache reads and writes, and output) plus
  the estimate of the messages added since. `trailing` excludes the
  assistant message built from that response. Hidden reasoning counted in
  `OutputTokens` but not replayed makes the result run high, the safe
  direction.
- **`llm.TokenCounter`** is an optional exact-count extension,
  `CountTokens(ctx, req) (TokenCount, error)`, found by type assertion.
  vLLM implements it over `/tokenize`. Anthropic implements it over
  `/v1/messages/count_tokens`, with the Chat conversion reduced to the
  fields that endpoint accepts.
- **`llm.CountTokens(ctx, p, req)`** uses the first counter on `p` or the
  providers it wraps, and otherwise returns the estimate with
  `Exact: false`. A counter error is returned unchanged.
- **`llm.ResolveModelInfo(ctx, p, model)`** merges the live `Models()` row
  (exact ID, then `SameModel`; among several inexact matches the lexically
  greatest ID, normally the latest dated snapshot, wins) with the embedded
  table, filling only zero or nil fields and never inventing one. A `Models()` failure degrades to
  the table, and the error surfaces only when neither source knows the
  model.
- **`llm.FitMaxTokens(prompt, window, requested, margin)`** is a pure
  function: `min(requested, window-prompt-margin)`, with requested ≤ 0
  meaning "as much as fits". It returns `ErrContextTooLong` when no output
  fits.
- **`llm.EstimateCostBound(prompt, maxOutput, pricing)`** prices the prompt
  as uncached input, as cache reads, and as cache writes, and returns the
  largest; it reports unknown when a rate is missing.
- **`llm.ContextGuard(GuardConfig{MaxOccupancy, Margin, ClampMaxTokens,
  OnNearFull, Window})`** is `Wrap` middleware. Per request:
  - It measures with `CountTokens`, falling back to the estimate if the
    counter fails.
  - It resolves the window per model, caching successful lookups and
    unknown models, and passes the request through when the window is
    unknown. A transient lookup failure is retried on the next request
    rather than disabling the guard.
  - It fails before the network call with `ErrContextTooLong` only for an
    exact count with prompt + Margin ≥ window. An estimate never fails a
    request.
  - It runs `OnNearFull` at or above `MaxOccupancy` (a fraction; default
    0.9, and values above 1 count as 1). The hook's error aborts the call.
    The hook must not mutate the request, because the guard does not
    re-measure. This is the go-agent compaction trigger: the primitive
    lives here and the policy in go-agent.
  - With `ClampMaxTokens`, it lowers an explicit `MaxTokens` on a copy of
    the request, to fit the window and to a known `MaxOutputTokens`. It
    never sets an unset `MaxTokens`: raising a provider's default can change
    behavior (Anthropic's SDK requires streaming for large non-streaming
    budgets).
  - For `ChatStream`, the check runs when the stream is iterated.
  - Place the guard last among the `Wrap` arguments (innermost), so that
    request-rewriting middleware runs first and the measured request is the
    one sent. `CountTokens` reaches the provider directly.
  - The counting call is a visible extra wire request; it is never folded
    into `Chat`/`ChatStream`.

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
  toggles), `ThinkingTokenBudget`, and `XArgs` (`vllm_xargs` engine
  passthrough). The budget emits `thinking_token_budget` only when explicitly
  set, must be positive and compatible with effective thinking enablement,
  and reserves 1,024 visible-answer tokens when `MaxTokens` is set.
- `StructuredOutputs` (native `structured_outputs` constraint modes,
  current vLLM): exactly one of `Regex`, `Choice`, `Grammar`, `StructuralTag`
  (raw JSON), plus `WhitespacePattern`; JSON-schema output stays on the
  unified `ResponseFormat` (era-portable spelling). Conflicts fail loud at
  build: set together with `ResponseFormat` → `ErrBadRequest` (one
  constraint system per request; unified owns the slot), zero/multiple modes
  → `ErrBadRequest`.
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
`SafetyIdentifier`, `PromptCacheOptions`, and `ReasoningSummary`.
`ReasoningSummary` is empty by default, preserving effort-driven
`summary: "auto"`; its currently closed values are `auto`, `concise`, and
`detailed`, while OpenAI owns per-model acceptance. Invalid values fail before
network I/O. This option is not part of `llm.Request`. These extensions use library-owned or standard-library types;
`HostedTools` is a slice of validated JSON objects so future hosted-tool shapes
do not expose SDK unions. `Client()` remains an advanced vendor-typed escape
hatch.

## 15. Prompt Caching

- Unified: an optional `CacheHint` on content parts / system prompt
  (with optional TTL). Providers with explicit breakpoints honor it
  (Anthropic `cache_control`; OpenRouter passes `cache_control` through to
  Anthropic-family models). Providers with automatic caching such as OpenAI
  ignore the hint.
- Anthropic preserves hints on text nested inside tool results. OpenRouter
  emits cache-aware role-tool text blocks only when a nested hint exists;
  no-hint tool messages keep the traditional concatenated string wire form.
- **`Request.CacheRetention`** is a request-wide policy, applied by every
  engine before conversion without mutating the caller's request:
  - `""`: hints and provider defaults are unchanged.
  - `none`: removes every `CacheHint` (system, message parts, nested tool
    results), and the OpenAI Responses `prompt_cache_key` is
    omitted. Use it for one-off calls such as summarization; providers with
    automatic caching may still cache internally. Session affinity is
    unaffected: OpenRouter's `session_id` is routing (§9A), not a cache key.
  - `short`: resets every hint to the provider's default lifetime
    (five minutes on Anthropic).
  - `long`: sets every hint to the long tier (Anthropic and
    Anthropic-family OpenRouter: one hour).
  - It does not toggle OpenAI's `prompt_cache_options`, which stay explicit
    on `openai.Options` because model acceptance is server-defined.
  - It needs no capability; unknown values fail validation with
    `ErrBadRequest`.
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

`Code`, `Message`, `Metadata`, and `RawBody` are provider-controlled and may
echo request content. They remain available for programmatic diagnosis, while
operational logs use `ProviderError.SafeSummary()` / `llm.SafeError(err)`,
which include only a valid bounded ASCII adapter identity, HTTP status, and
normalized sentinel. The label is omitted when it is not `[A-Za-z0-9._-]` or
exceeds 64 bytes. `SafeError` deliberately returns non-provider errors
verbatim; it is not a general-purpose secret scrubber.

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
| 413 | `ErrContextTooLong` (request-size or context overflow) |
| 429 | `ErrRateLimited` |
| 503 and 529 | `ErrOverloaded` |
| any other 5xx | `ErrServer` |
| any other 4xx | `ErrBadRequest` |

Provider-native error codes/messages classify first (scoped heuristics),
then the status table, then weak code fallbacks for status-less in-stream
errors (`invalid_request` → `ErrBadRequest`, `server_error` → `ErrServer`).

**Context overflow** uses one scoped matcher shared by every engine,
including Anthropic and in-stream errors:

- **Codes:** `context_length…`, `request_too_large`, and
  `model_context_window_exceeded`.
- **Messages:** an attributed corpus of real provider phrasings, covering
  "context window/length/limit/size", "maximum context",
  "prompt (is) too long", "maximum prompt length", "maximum model length",
  "reduce the length of the messages", "maximum allowed input length",
  "exceeded model token limit", "range of input length should be", and
  "input token count".
- **Vetoes:** throttling texts ("rate limit", "too many requests",
  "throttl…", "per minute", "tokens per min") never classify as overflow.
  A bare "context" or "tokens" substring deliberately does not match.
- **413 is request size.** It can come from oversized attachments (images,
  PDFs) rather than the conversation, so compaction alone may not fix it.
- **Not overflow:** `max_tokens` above the model window (vLLM `max_tokens …
  cannot be greater than max_model_len`) stays `ErrBadRequest`. The fix is
  a smaller `MaxTokens`, not a shorter prompt.

**One channel per event.** Overflow and content filtering each reach the
caller through exactly one signal, never both:

- A request the provider rejects, and any provider error event, including
  a mid-stream error that names filtering, surfaces as the error sentinel
  (`ErrContextTooLong`, `ErrContentFiltered`).
- A generation that ends normally with a provider stop reason surfaces as
  `StopReasonContextOverflow`, `StopReasonContentFilter`, or
  `StopReasonRefusal` on a successful `MessageEnd`.
- Consumers that handle a condition must therefore check both forms: the
  error sentinel for rejections and the stop reason for stopped
  generations.

OpenRouter 403 moderation carries structured metadata (surfaced via `ProviderError`)
and its status-less mid-stream numeric code `"402"` →
`ErrInsufficientCredits`; mid-stream errors otherwise normalize identically
to pre-stream errors. An empty-but-2xx SSE stream (zero events before EOF)
normalizes to an in-stream `ErrServer` ("provider returned an empty
stream") — never a silent empty success.

**Classification helpers** (consumer-facing, typed-chain only — never
message matching):

- `llm.ErrorClass(err)` returns the normalized class in err's chain: one of
  the package sentinels (including the `ErrLogin*` login sentinels) or
  `context.Canceled` / `context.DeadlineExceeded`; nil when unclassified.
  Precedence when a chain matches several: cancellation, then deadline, then
  specific conditions (`ErrContextTooLong`, `ErrContentFiltered`,
  `ErrUnsupported`, `ErrInsufficientCredits`, `ErrRateLimited`,
  `ErrOverloaded`, `ErrTimeout`, `ErrAuth`, `ErrPermission`, `ErrNotFound`,
  login sentinels), then the generic `ErrBadRequest` and `ErrServer`.
- `llm.SafeErrorInfo(err)` returns a serializable `llm.ErrorInfo{Class,
  Provider, HTTPStatus, RetryAfter}` following the `SafeSummary` policy: it
  never contains `Code`, `Message`, `Metadata`, or `RawBody`. `Class` uses
  the stable vocabulary `canceled`, `deadline_exceeded`,
  `context_too_long`, `content_filtered`, `unsupported`,
  `insufficient_credits`, `rate_limited`, `overloaded`, `timeout`, `auth`,
  `permission`, `not_found`, `login_expired`, `login_state_mismatch`,
  `login_closed`, `bad_request`, `server`.
- `llm.Transient(err)` is true exactly for `ErrRateLimited`,
  `ErrOverloaded`, `ErrServer`, and `ErrTimeout` (and false when the chain
  also carries cancellation). It is classification only: attempt budgets,
  backoff, and jitter are caller policy, applied after go-llm's own
  transport retries. An expired provider `WithTimeout` surfaces as
  `context.DeadlineExceeded` and is not transient.

## 17. Configuration & Operations

- Per-provider constructors with functional options: `WithAPIKey` (env-var
  fallback: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`),
  `WithAPIKeyFunc(func(ctx) (string, error))` for
  rotating/expiring credentials (OAuth-era tokens, gateways),
  `WithBaseURL`, `WithHTTPClient`, `WithMaxRetries`, `WithTimeout`.
- **`llm.DefaultHTTPClient()`** — go-llm ships its own default HTTP
  client, tuned for LLM workloads, used by every provider constructor
  when `WithHTTPClient` is not passed:
  - `Client.Timeout: 0` — **never** a whole-response timeout (it would
    kill SSE streams mid-body; the footgun naive custom clients hit);
    time-to-first-byte is bounded by `ResponseHeaderTimeout` (120s), and
    per-request deadlines come from `WithTimeout`/`context`.
  - `IdleConnTimeout: 30s` — evict stale pooled connections quickly.
  - `MaxIdleConnsPerHost` raised (16; Go's default of 2 cripples
    concurrent fan-out against a single API host).
  - **darwin**: `DisableKeepAlives: true` — works around the documented
    macOS stale-connection hang after sleep/wake/network changes (cost:
    per-request HTTP/1.1 + handshake). Overridable like everything else.
  - Every call returns a fresh `*http.Client`; clients share only a private,
    lazily initialized transport and its connection pool. Mutating client
    fields therefore cannot change another caller's client.
- Retries: one shared transport policy covers blocking and streaming calls;
  wrapped SDK retry loops are disabled so policies never stack. By default,
  only replayable requests with typed proof that no HTTP bytes were sent
  (temporary DNS, dial, or proxy-connect failures) are retried. Ambiguous
  EOF/reset/broken-pipe/read/write-timeout and TLS errors are not replayed.
  `WithResponseRetries(true)` separately opts into at-least-once replay of
  429/503/529 responses; it is not billing-safe. Redirects and
  `X-Should-Retry: false` veto replay. Backoff is context-aware; a provider
  delay over 30 seconds returns the original response untouched for normal
  error mapping. Default maximum: 2 additional attempts per auth epoch.
- All calls take `context.Context`; no global mutable request state. The
  write-once part registry, lazily parsed model table, and shared private
  default transport are the only process-wide infrastructure.
- **`llm.CredentialEnvNames()`** lists every credential-bearing environment
  variable read in-process, so hosts that spawn subprocesses can scrub them.
  go-llm code reads `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  `OPENROUTER_API_KEY`, and `OPENAI_CUSTOM_HEADERS`. openai-go's environment defaults also read
  `OPENAI_ADMIN_KEY` and `OPENAI_WEBHOOK_SECRET`, which the adapters override
  rather than use; the Anthropic SDK's environment defaults are disabled.
  Self-hosted presets read no environment variables. A repository test scans
  provider and CLI sources, resolving constant arguments, to keep the
  go-llm-read names exact.
- go-llm **never reads configuration files implicitly** — credentials
  come from explicit options or environment variables. For applications
  that keep provider credentials in a file, one explicit opt-in loader is
  provided: **`llm.LoadAuthFile(path)`**, parsing the documented credential
  format — a map of provider-id → credential union
  discriminated by `type`: `{"type": "api_key", "key": "..."}` or
  `{"type": "oauth", "access": "...", "refresh": "...", "expires": <ms>,
  "accountId": "...", "clientId": "..."}` — accepted either bare or
  nested under a `"providers"` wrapper (as the e2e harness's
  `gollm-test.json` does; architecture §9). Callers either pass a static
  key (`WithAPIKey` / `WithAPIKeyFunc`) or hand the provider a
  `llm.CredentialStore` (`WithCredentials`, typically
  `authstore.NewFile`); `oauth` credentials serve the subscription auth
  path (§17C), where the provider refreshes inside the store's
  serialized `Modify`.

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
- Exports additive `RunCapabilityConformance` for a reviewed set of
  activation-sensitive standard capabilities. Profiles preserve real request
  models, use `CapabilityInvocation` only as out-of-band fixture control data,
  assert provider-native request fields in provider packages, and assert the
  normalized result in the common runner. Every advertised reviewed
  capability needs one or more cases or one explicit non-empty exemption;
  exemptions identify offline evidence gaps rather than capability denials.
- `RunConformance` proves common lifecycle/normalization behavior;
  `RunCapabilityConformance` proves deterministic native activation plus a
  normalized result. Neither is evidence of live model/account availability,
  quota, cache admission, or service reliability; §9's credentialed suite is
  authoritative for those properties.

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

**OpenTelemetry integration (`contrib/otelllm`).** This is a separate
nested module, `github.com/pkieltyka/go-llm/contrib/otelllm`, with its own
`go.mod`, so the root module never requires OpenTelemetry.
`otelllm.Middleware(opts...)` is `llm.Wrap` middleware that uses the global
providers by default, or `WithTracerProvider` and `WithMeterProvider`. Place
it outermost.

- **Spans.** One CLIENT span per `Chat`/`ChatStream`, named
  `chat {model}`. The parent comes from the request context, and the span's
  context is passed to the wrapped provider.
- **Naming.** Attributes and metrics follow the OpenTelemetry GenAI
  semantic conventions as published in semconv v1.41.0 (development
  status), written as string constants so no semconv package is imported:
  - request: `gen_ai.operation.name`, `gen_ai.provider.name`,
    `gen_ai.request.model`,
    `gen_ai.request.max_tokens`, `gen_ai.request.temperature`,
    `gen_ai.request.top_p`, and `gen_ai.request.stream`;
  - response: `gen_ai.response.id`, `gen_ai.response.model`,
    `gen_ai.response.finish_reasons`, and
    `gen_ai.response.time_to_first_chunk`;
  - usage: `gen_ai.usage.input_tokens` (including cache reads and writes,
    per the conventions), `gen_ai.usage.output_tokens`,
    `gen_ai.usage.cache_read.input_tokens`,
    `gen_ai.usage.cache_creation.input_tokens`, and
    `gen_ai.usage.reasoning.output_tokens`;
  - errors: `error.type`, carrying the `SafeErrorInfo` class or `_OTHER`.
- **go-llm names.** Concepts the conventions lack use `go_llm.*`:
  - the raw provider, effort, and tool count;
  - cost in USD and its source;
  - dropped tool calls;
  - `go_llm.canceled` and `go_llm.stream.abandoned`;
  - a `go_llm.first_chunk` event, and one `go_llm.block` event per content
    block carrying only type, index, character count, and tool name.
- **Metrics:** `gen_ai.client.operation.duration`,
  `gen_ai.client.token.usage` (by `gen_ai.token.type`),
  `gen_ai.client.operation.time_to_first_chunk` (streams only), and the
  `go_llm.client.cost` counter. They use the convention-recommended bucket
  boundaries.
- **Content-free by construction.** No prompt, system, completion,
  reasoning, tool-argument, or tool-result bytes are recorded, and error
  status descriptions are `llm.SafeError` truncated to 500 bytes for
  provider errors, and only the error class for any other error, so text
  from caller middleware is never recorded. A canary test enforces this.
- **Error status.** Caller cancellation and a consumer abandoning a stream
  are not errors.
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
  cookies) are **always redacted** — not optional. Each transport attempt
is captured separately. `llm.WireCaptureToLogger(l)` adapts a logger into
a capture fn (full payloads at `Debug`) for instant trace mode. Bodies may
contain user data — storage/retention is the application's
responsibility. (Naming: the whole capture surface uses the "wire"
vocabulary — `WireCapture`, `NewWireTap`, `WithWireCapture`,
`WireCaptureToLogger`.)

## 17C. Authentication: credential stores and OAuth

go-llm mirrors pi's authentication model (pi v1.4, `packages/ai/src/auth`).

- **Credentials.** An `llm.AuthCredential` is pi's auth.json credential
  union: `{"type":"api_key","key"}` or `{"type":"oauth","access",
  "refresh","expires","clientId"?,"accountId"?}`, plus go-llm's optional
  `model`/`base_url` configuration fields. `llm.LoadAuthFile` and
  `ParseAuthFile` read bare and `{"providers":...}`-wrapped files; unknown
  fields such as pi's `scopes` are ignored. `ClientID` is the OAuth client
  issued per login, which refresh requires.
- **Credential stores.** `llm.CredentialStore` is application-owned storage
  keyed by provider ID with `Read`, `Modify`, and `Delete`:
  - `Modify` is the only write path, serialized per provider ID (across
    processes where supported). It passes the current credential to `fn`
    and stores `fn`'s result.
  - `ctx` bounds only the wait for exclusivity: once `fn` runs, its write
    completes, because a started OAuth refresh may already have rotated the
    refresh token.
  - Implementations live in `authstore`. `NewMemory` is in-process.
    `NewFile(path)` is a pi-compatible auth.json:
    - It re-reads the file on every `Read`, without the lock; a file that
      reads as empty or unparsable is re-read under the lock, because pi
      rewrites auth.json in place.
    - `Modify` and `Delete` hold an in-process lock plus pi's
      cross-process lock (proper-lockfile's protocol): `<path>.lock` is
      created as a directory, the holder refreshes its mtime every 5 s
      (pi's synchronous reload treats 10 s as stale), a lock untouched for
      30 s is stale and removed, and release removes it. pi and go-llm
      therefore exclude each other on a shared auth.json. A holder whose
      lock was removed or replaced meanwhile writes nothing and leaves the
      new lock alone. Waits are bounded at 45 s (`ErrTimeout`).
    - Writes go through a symlinked path to its target.
    - Writes are atomic (temporary file, fsync, rename, directory fsync)
      with mode 0600.
    - Other entries, the `providers` wrapper, and unknown entry fields are
      preserved. A credential type change drops the old type's fields but
      keeps `model`/`base_url`.
    - Errors never quote file content.
- **Provider resolution** (`WithCredentials(store)` on `openai`,
  `anthropic`, and `openrouter`) runs on every request:
  - A stored credential owns the provider: a stored credential of a type
    the provider cannot use fails with `ErrAuth` and never falls back to
    the environment.
  - With nothing stored, or a stored `api_key` entry without a key (as in
    pi), the provider's environment variable is used.
  - `WithCredentials` is mutually exclusive with `WithAPIKey`/
    `WithAPIKeyFunc`.
  - Anthropic and OpenRouter accept stored `api_key` credentials only.
    Anthropic explains a stored `oauth` entry (pi's Claude Pro/Max login):
    subscription credentials are limited to Anthropic's own apps, so the
    entry must be replaced with an API key or removed.
- **OAuth refresh** (`provideroauth.Resolver`) mirrors pi's
  `resolveStoredOAuth`:
  - An optimistic `Read`; a token with less than five minutes left, or
    without an expiry, is refreshed inside `Modify` after re-checking under
    the lock, so concurrent requests and processes refresh exactly once.
  - A minted or refreshed credential stores `expires` as now +
    `expires_in` − 3 minutes, as pi does, so a shared file means the same
    to both.
  - The renewed credential is stored before it is used.
  - A started refresh ignores caller cancellation and is bounded by a
    15 s timeout (pi's).
  - Identity fields (`clientId`, `accountId`) survive a refresh response
    that omits them.
  - After an HTTP 401, the request is retried once with a forced refresh,
    skipped if the stored token has already changed.
  - Refresh endpoints never follow redirects.
- **Sign in with ChatGPT (`openai`).** An `oauth` credential lets a
  ChatGPT subscription use the ordinary Responses API: its access token is
  sent to `api.openai.com` as a bearer token.
  - Login (`openai.NewLoginFlow`) is pi's public-client flow:
    - It authorizes at `auth.openai.com/api/accounts/authorize` with
      `client_id=dynamic_agent_client`, `agent_name_hint` (default
      `go-llm`), `ext_agent_host_id=urn:uuid:<device ID>`, `resource=
      https://api.openai.com/v1`, scope `openid profile email
      offline_access resource.invoke chatgpt.tokens.use.direct`, S256
      PKCE, state, and nonce.
    - The callback must carry the issued `client_id`.
    - The token exchange requires an ID token, a refresh token, a positive
      lifetime, and the `chatgpt.tokens.use.direct` scope.
  - Refresh sends `grant_type=refresh_token`, the stored `clientId`, and
    `resource`, with the same response requirements.
  - `WithLoginDeviceID` is required: a stable per-installation UUID that
    the host creates once and persists.
  - The flow binds only `127.0.0.1:1455` with redirect `/auth/callback`,
    and fails with `ErrServer` while another login holds the port.
  - Subscription usage limits (`subscription_sharing_*usage_limit_exceeded`)
    classify as `ErrInsufficientCredits`, which is not transient.
  - The token reaches only the subscription's model catalog, and the
    endpoint accepts only streaming requests. Mirroring pi, on that path
    the provider:
    - serves `Chat` by draining a stream;
    - omits `temperature`, `top_p`, `max_output_tokens`,
      `prompt_cache_retention`, and `prompt_cache_options`, which the
      endpoint rejects (so `MaxTokens`, `Temperature`, and `TopP` are
      ignored);
    - maps `Models` from the `{"models": [...]}` catalog the endpoint
      returns in place of `{"data": [...]}` (listed rows only, with
      context window, reasoning ladder and default, parallel tools, and
      image input).
  - Rejections arrive as `{"detail": ...}` bodies; their message maps
    into `ProviderError.Message`.
  - No prompt-cache reads were observed on this path (2026-10-22 probes);
    `CacheReadTokens` reports what the endpoint returns.
- **Login-flow contract.** `llm.LoginFlow` is a single-use, context-aware
  `Begin`/`Complete`/`Submit`/`Cancel` lifecycle.
  - `Begin` returns a redacting `LoginAuthorization` with a short-lived
    loopback launch URL.
  - `Complete` waits for the automatic callback or `Submit` (a copied
    callback URL).
  - The shared loopback engine (`provideroauth.LoopbackFlow`) provides
    IPv4-only listening, constant-time state checks, bounded callback
    fields, flow and exchange timeouts, single use, and closing of idle
    keep-alive connections (so a later login's callback cannot hit a stale
    server).
  - The provider owns endpoints, parameters, and exchange.
  - The returned credential is never persisted by go-llm: the host writes
    it with `store.Modify`.
- **Anthropic** — API-key authentication only. go-llm does not consume or
  mint Anthropic subscription credentials and does not emulate Claude Code
  identity (pi does; go-llm deliberately does not, because Anthropic
  restricts subscription credentials to its own clients). An Agent SDK
  layer, if desired by a host application, belongs outside this transport
  library.
- **Removed:** the `openai-codex` provider (ChatGPT backend at
  `chatgpt.com/backend-api/codex`), `WithOAuth(cred, persist)`, and
  `llm.OAuthPersistenceFunc`. pi superseded its Codex provider with Sign in
  with ChatGPT on `openai`, and go-llm followed without a deprecation
  period.

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
- **Model-aware replay**: a `ReasoningPart` replays only when its assistant
  message was produced by the request's model.
  - Every engine applies one shared pre-pass. When `Message.Model` is set
    and `!llm.SameModel(Message.Model, Request.Model)`, the message's
    reasoning parts are dropped, never converted to text.
  - An assistant message left empty is dropped, as with provider filtering.
  - `Message.Model` is the server-reported model, which often differs from
    the requested alias (Anthropic answers `claude-opus-4-0` as
    `claude-opus-4-20250514` and `…-latest` with a dated ID; OpenAI appends
    a date). `llm.SameModel` therefore compares IDs case-insensitively
    after trimming and removing, in order, one `:variant` routing suffix,
    one trailing date snapshot (`-YYYYMMDD`, `_YYYYMMDD`, `@YYYYMMDD`,
    `-YYYY-MM-DD`), one `-latest`, and one trailing `-0`.
  - The normalization deliberately leans toward matching. A false match
    reproduces the earlier provider-gated replay, while a false mismatch
    would strip the signed thinking a same-model Anthropic tool loop
    requires. Different versions or sizes (`gpt-5` vs `gpt-5-mini`) still
    differ.
  - History with an empty `Message.Model` (legacy) keeps provider-gated
    replay. Persisting assistant provenance (as `History.AddResponse` does)
    is what enables the rule.
  - The rule matters most on aggregators such as OpenRouter, where one
    provider fronts many model families. Live probes (2026-10-22) found
    OpenRouter accepts cross-family `reasoning_details` without rejecting
    them, and never requires them to continue a tool loop. Dropping them is
    therefore safe there and keeps another model's signed reasoning off the
    wire.
  - Unverified on direct Anthropic: switching models mid tool loop with
    thinking enabled leaves the final assistant turn without a thinking
    block, which Anthropic may reject. Before this rule the previous model's
    signed block was sent instead, and whether Anthropic accepts that across
    models is equally unverified. Callers
    switching models mid tool loop on direct Anthropic should disable
    thinking for the switch turn until this is live-verified.
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
llm-cli -p openai --auth-file ~/.pi/agent/auth.json -m gpt-5.5 "..."
llm-cli models -p openrouter
```

- **Prompt sources**: positional arg, stdin (piped), or both (stdin
  appended as a text part) — curl-style composability.
- **Flags**: `-p/--provider` (anthropic|openai|openrouter),
  `-m/--model`, `-s/--system`, `--effort`, `--max-tokens`, `--temp`
  (alias `--temperature`), `--image <path|url>` / `--file <path|url>`
  (repeatable), `--schema <file.json>` (structured output; prints
  validated JSON; forces the non-streaming path), `--no-stream` (buffer
  and print complete response), `--json` (emit the full canonical
  `Response` JSON incl. usage/cost), `--usage` (usage + cost summary to
  stderr), `--reasoning` (print reasoning deltas to stderr as they
  stream), `--cache-system` (set `SystemCache` — pair with `--usage` to
  observe cache hits), `--session-id`, `--debug` (wire capture to stderr
  via `WireCaptureToLogger`), `--api-key`, `--auth-file` (a pi-compatible
  credential store for every provider; exclusive with `--api-key`),
  `--base-url`, `--timeout`, `--version`.
- **Conversation files** — multi-turn the curl way (state in files, not
  the process): `--load <file>` prepends a saved conversation (canonical
  envelope, §10A), `--save <file>` writes the updated conversation after
  the response. Loading with a *different* `-p` exercises cross-provider
  history replay directly from the shell.
- **Commands**: default = chat; `models` = explicitly fetch and list models
  for a provider (table; `--json` for machine output). The table exposes
  supported efforts, default effort, and positive reasoning-required claims;
  JSON uses optional `supported_efforts`, `default_effort`, and
  `reasoning_required` fields.
- **Credentials**: `--auth-file` opens an `authstore.File` (§17C) that every
  provider reads per request: API keys for any provider, and Sign in with
  ChatGPT OAuth for `openai`, refreshed and written back under the file's
  cross-process lock. Without it, providers read their environment variable
  (library convention), or `--api-key`, whose value is exposed through
  process argv and shell history. Files are never read unless the flag is
  supplied.
- **Tool calls are printed, not executed**: with `--tool <file.json>`
  (repeatable tool declarations), a `tool_use` stop prints the tool calls
  as JSON and exits — execution loops are `go-agent` territory; the CLI
  stays a client.
- **Constraints**: stdlib `flag` only (no CLI-framework dependency in the
  module), errors to stderr with the normalized error text, exit code 0
  on success / 1 on error, streaming output unbuffered.
- `make build` compiles all packages and writes an executable
  `bin/llm-cli`; CI runs credential-free `--help` and `--version` smoke tests.
