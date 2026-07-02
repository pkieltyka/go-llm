---
status: complete
---

# Architecture: go-llm

Companion to `functional_spec.md` — that doc defines *what*; this doc defines
*how*. Mapping tables (stop reasons, effort levels, session affinity,
streaming nuances, error codes) are specified there and not restated.

## 1. Module & Package Layout

Single Go module: `github.com/pkieltyka/go-llm`. Root package name: `llm`.

```
go-llm/
├── llm.go, request.go, message.go, response.go   # core types (package llm)
├── stream.go                                     # Event types, Collect()
├── errors.go                                     # sentinels + ProviderError
├── capability.go                                 # Capability + helpers
├── history.go                                    # History helper
├── pricing.go, pricing_table.go                  # Usage cost estimation (+ snapshot)
├── validate.go                                   # capability/request validation
├── serialize.go                                  # canonical JSON for Message/Part/Response
├── middleware.go                                 # Middleware + Wrap decorator
├── parse.go                                      # Parse[T] structured-output helper
├── schema/                                       # schema-from-struct + arg validation (stdlib-only)
├── llmtest/                                      # scriptable fake Provider
├── internal/e2e/                                 # live e2e scenario harness (build tag: live)
├── providers/
│   ├── anthropic/                                # wraps anthropic-sdk-go (Messages API, direct)
│   ├── openai/                                   # wraps openai-go Responses API (direct)
│   ├── openrouter/                               # openaicompat + OpenRouter dialect
│   ├── zai/                                      # openaicompat + ZAI dialect
│   └── internal/openaicompat/                    # shared chat-completions-shaped adapter
└── go.mod
```

Dependency rules:

- Root `llm` package: **stdlib only**. It defines the entire public vocabulary.
- Provider packages import `llm` + their SDK. Nothing in `llm` imports providers.
- `providers/internal/openaicompat` is `internal/` — its API can change freely.

Dependencies (pinned at implementation time):

| Dependency | Where | Purpose |
|---|---|---|
| `github.com/anthropics/anthropic-sdk-go` (v1.55.x verified 2026-07) | `providers/anthropic` | Official Anthropic SDK — Messages API confirmed as the current recommended client surface |
| `github.com/openai/openai-go/v3` (v3.41.x verified 2026-07) | `providers/openai` (Responses surface, direct) + `providers/{openrouter,zai}` via openaicompat (Chat Completions surface) | Official OpenAI SDK — one dependency serves both surfaces |
| `github.com/google/go-cmp` | tests only | Diffs in table tests |

Go version: 1.26.

## 2. Core Types (package `llm`)

### 2.1 Messages and parts

```go
type Role string // RoleUser, RoleAssistant, RoleTool, RoleSystem

type Message struct {
    Role  Role
    Parts []Part
}

// Part is a sealed interface — only types in this package implement it.
type Part interface{ part() }

type TextPart      struct { Text string; Cache *CacheHint }
type ImagePart     struct { URL string; Data []byte; MediaType string; Cache *CacheHint }
type FilePart      struct { URL string; Data []byte; MediaType string; Name string; Cache *CacheHint }
type ToolCallPart  struct { ID, Name string; Args json.RawMessage }
type ToolResultPart struct { ToolCallID string; Parts []Part; IsError bool }
type ReasoningPart struct {
    Text string          // normalized readable reasoning (may be "")
    Raw  json.RawMessage // opaque provider payload for round-trip (thinking block
                         // w/ signature, reasoning_details, etc.)
    Provider string      // which provider produced Raw
}

type CacheHint struct { TTL time.Duration } // zero = provider default
```

Ergonomic constructors: `llm.UserText(s)`, `llm.AssistantText(s)`,
`llm.UserParts(parts...)`, `llm.Text(s)`, `llm.ImageURL(u)`,
`llm.ImageData(b, mediaType)`, `llm.ToolResult(id, content)`.

Provider-specific parts (ZAI video/file URL) are defined in the provider
package as types implementing `llm.Part` — the sealed interface has an
exported escape: `type ExtensionPart interface { Part; ExtensionProvider() string }`.
Adapters skip extension parts from other providers with `ErrUnsupported`.

### 2.2 Request

```go
type Request struct {
    Model         string        // required, verbatim provider model ID
    Messages      []Message     // required
    System        string
    SystemCache   *CacheHint    // cache hint for the system prompt
    MaxTokens     int           // 0 = unset (adapter default if required)
    Temperature   *float64
    TopP          *float64
    StopSequences []string
    Tools         []Tool
    ToolChoice    ToolChoice    // zero value = auto/omitted
    ResponseFormat *ResponseFormat
    Effort        Effort        // "" = provider default
    SessionID     string
    ProviderOptions ProviderOptions // nil ok; typed per-provider
}

type Tool struct {
    Name        string
    Description string
    InputSchema any  // JSON-marshalable schema, json.RawMessage, or schema.For[T]()
    Strict      bool
    Annotations ToolAnnotations // informational; never sent to providers
}

type ToolAnnotations struct { // MCP-aligned behavioral hints
    ReadOnly, Destructive, Idempotent, OpenWorld bool
}

type ToolChoice struct { Mode ToolChoiceMode; Name string }
type ToolChoiceMode string // "", auto, none, required, tool

type ResponseFormat struct {
    Type   ResponseFormatType // FormatJSONSchema | FormatJSONMode
    Name   string
    Schema any
    Strict bool
}

type Effort string // "", EffortNone, EffortMinimal, EffortLow, EffortMedium,
                   // EffortHigh, EffortXHigh, EffortMax

// ProviderOptions is implemented by each provider's Options struct.
type ProviderOptions interface{ ForProvider() string }
```

`ProviderOptions` contract: adapters check `ForProvider() == p.Name()`; a
mismatch returns `ErrBadRequest` (fail loud, never silently ignore).

### 2.3 Response

```go
type Response struct {
    ID            string
    Model         string      // model that actually served (OpenRouter fallbacks)
    Parts         []Part
    StopReason    StopReason
    StopReasonRaw string
    Usage         Usage
    Raw           any         // raw SDK response value
}

func (r *Response) Text() string               // concatenated TextParts ("" safe on refusal)
func (r *Response) Reasoning() string          // concatenated ReasoningParts
func (r *Response) ToolCalls() []ToolCallPart

type StopReason string // per functional spec §5

type Usage struct {
    InputTokens, OutputTokens, TotalTokens int64
    CacheReadTokens, CacheWriteTokens      int64
    ReasoningTokens                        int64
    CostUSD *float64 // nil = unknown
    Raw     any
}
```

### 2.4 Provider interface

```go
type Provider interface {
    Name() string
    Capabilities() []Capability
    Models(ctx context.Context) ([]ModelInfo, error)
    Chat(ctx context.Context, req *Request) (*Response, error)
    ChatStream(ctx context.Context, req *Request) iter.Seq2[Event, error]
}

type ModelInfo struct {
    ID              string
    DisplayName     string
    ContextWindow   int
    MaxOutputTokens int
    Pricing         *ModelPricing // when reported (OpenRouter) or from table
    Raw             any
}
```

Design notes:

- **Streaming returns a bare `iter.Seq2[Event, error]`** — one method, no
  stream object, no `Close()`. Connection/setup errors are yielded as the
  first pair. Early `break` triggers iterator cleanup (range-over-func
  defer semantics) which closes the underlying SSE connection. Context
  cancellation likewise terminates the sequence.
- Batch APIs later attach as a separate optional interface
  (`interface{ Batch() BatchClient }` shape) without touching `Provider`.
- Raw-client escape hatch is per-package via type assertion:
  `p.(*anthropic.Provider).Client()` returns the SDK client.

### 2.5 Stream events

```go
type Event interface{ event() } // sealed

type MessageStart  struct { ID, Model string }
type TextDelta     struct { Text string }
type ReasoningDelta struct { Text string }
type ToolCallStart struct { Index int; ID, Name string }
type ToolCallDelta struct { Index int; ArgsFragment string }
type ToolCallEnd   struct { Index int }
type MessageEnd    struct { StopReason StopReason; StopReasonRaw string; Usage Usage }

// Collect drains a stream into a complete Response.
func Collect(events iter.Seq2[Event, error]) (*Response, error)
```

`Collect` is also the internal building block for testing adapters: every
recorded stream fixture must `Collect` into the same `Response` the
non-streaming path produces for the equivalent payload.

### 2.6 Errors

```go
// Sentinels (match with errors.Is) — full list in functional spec §16.
var ErrAuth, ErrPermission, ErrNotFound, ErrBadRequest, ErrRateLimited,
    ErrInsufficientCredits, ErrOverloaded, ErrServer, ErrTimeout,
    ErrContentFiltered, ErrContextTooLong, ErrUnsupported error

type ProviderError struct {
    Provider   string
    HTTPStatus int
    Code       string            // stringly: OpenAI type, OpenRouter number, ZAI business code
    Message    string
    RetryAfter time.Duration     // 0 = not provided
    Metadata   map[string]any    // e.g. OpenRouter moderation metadata
    RawBody    []byte
    Kind       error             // one of the sentinels above
}

func (e *ProviderError) Error() string { /* "llm/openrouter: 429 rate_limited: ..." */ }
func (e *ProviderError) Unwrap() error { return e.Kind } // errors.Is(err, ErrRateLimited)
```

Both layers in one type: `errors.Is` matches the sentinel via `Unwrap`;
`errors.As` extracts `*ProviderError` for provider detail. Context errors
(`context.Canceled`, `DeadlineExceeded`) pass through unwrapped.

`ErrUnsupported` is produced by pre-flight validation (§6) and always wrapped
with the capability name: `fmt.Errorf("%w: tool-choice-required (zai)", ErrUnsupported)`.

### 2.7 Canonical serialization (`serialize.go`)

- Every `Part` type implements `MarshalJSON` emitting a `"type"`
  discriminator; `Message.UnmarshalJSON` dispatches on it. Discriminators:
  `text`, `image`, `file`, `tool_call`, `tool_result`, `reasoning`.
- `ReasoningPart` round-trips `raw` (as `json.RawMessage`, byte-preserved)
  and `provider` — the invariant that keeps same-provider replay working on
  reloaded history (tested as a property, §9).
- Extension parts serialize as `"<provider>/<kind>"` (e.g. `zai/video_url`).
  Provider packages register decoders in `init()` via
  `llm.RegisterPartType(name string, decode func(json.RawMessage) (Part, error))`.
  Unknown types decode to an `UnknownPart{Type string; Data json.RawMessage}`
  (preserved on re-marshal, skipped by adapters) — forward-compatible, never
  lossy.
- Envelope helpers: `llm.MarshalMessages([]Message) ([]byte, error)` /
  `UnmarshalMessages` wrap `{"version": 1, "messages": [...]}`.
- `Response` marshals everything normalized; `Response.Raw` and `Usage.Raw`
  are excluded (documented).

### 2.8 Middleware (`middleware.go`)

Middleware is a **Provider decorator** — adapters have zero middleware
awareness:

```go
type ChatFunc   func(ctx context.Context, req *Request) (*Response, error)
type StreamFunc func(ctx context.Context, req *Request) iter.Seq2[Event, error]

type Middleware struct {
    Chat   func(next ChatFunc) ChatFunc     // optional
    Stream func(next StreamFunc) StreamFunc // optional
}

// Wrap composes middleware around a Provider (first mw = outermost).
// Name/Capabilities/Models delegate untouched.
func Wrap(p Provider, mw ...Middleware) Provider
```

Wrapping a stream is plain function decoration over the returned iterator
(observe/transform events as they pass) — no teeing machinery needed.

### 2.9 Parse[T] (`parse.go`)

```go
func Parse[T any](ctx context.Context, p Provider, req *Request,
    opts ...ParseOption) (T, *Response, error)

func WithParseRetries(n int) ParseOption // default 0
```

Flow: derive schema via `schema.For[T]()` (unless `req.ResponseFormat` is
already set) → capability switch: `json-schema` → server-enforced;
`json-mode` only → JSON mode + schema guidance appended to `System` +
client-side validation; neither → `ErrUnsupported` → call `p.Chat` →
`json.Unmarshal` into `T`. On unmarshal/validation failure with retries
remaining: append the assistant turn + a correction user message, retry.

## 3. Provider Implementation Architecture

### 3.1 Anthropic (direct wrap)

`providers/anthropic` wraps `anthropic-sdk-go` directly (import-aliased
`sdk`). Responsibilities:

- Request build: `llm.Request` → `sdk.MessageNewParams`. System → `system`
  block (`SystemCache` → `cache_control` on it); part-level `CacheHint` →
  `cache_control` (TTL ≤5m → 5m, else 1h); Effort →
  adaptive thinking + `output_config.effort` with `display: summarized`;
  MaxTokens default **16384** when unset (constructor-overridable via
  `WithDefaultMaxTokens`).
- Response map: content blocks → parts (`thinking` blocks → `ReasoningPart`
  with the full block JSON in `Raw`); stop reasons per spec table.
- Stream map: typed SDK events → unified events (block-index bookkeeping for
  tool `input_json_delta`).
- Replay: `ReasoningPart.Raw` with `Provider == "anthropic"` re-emits the
  original thinking block verbatim (API requirement); other providers'
  reasoning parts are dropped from outgoing Anthropic requests.
- Errors: `*sdk.Error` → `ProviderError` (status → sentinel; `retry-after`
  header honored).
- `Models()`: `GET /v1/models` via SDK, mapped to `ModelInfo` (context
  window, max output, capabilities available on that endpoint).

### 3.2 OpenAI (direct wrap, Responses API)

`providers/openai` wraps `openai-go`'s **Responses** surface directly
(`client.Responses.New` / `NewStreaming`) — OpenAI's recommended API for
new projects, and the only one returning reasoning content and preserving
reasoning across tool-call turns. Decision record:
`provider_capabilities.md` → Decision note.

- **Request build**: `Messages` → `input` items (`ToolResultPart` →
  `function_call_output` items keyed by `call_id`); `System` →
  `instructions`; `MaxTokens` → `max_output_tokens`; `Effort` →
  `reasoning: {effort, summary: "auto"}` (`none` supported natively);
  `ResponseFormat` → `text: {format}`; tools → flattened function shape
  (`strict` default-on per Responses convention, disabled when
  `Tool.Strict` is false); `SessionID` → `prompt_cache_key`.
- **Statelessness**: always `store: false` +
  `include: ["reasoning.encrypted_content"]` unless `openai.Options`
  explicitly opts into server-side state (`Store`, `PreviousResponseID`,
  `Conversation`).
- **Response map**: `output` items → parts in order — `reasoning` item →
  `ReasoningPart{Text: joined summary, Raw: full item JSON incl.
  encrypted_content}`; `message`/`output_text` → `TextPart` (annotations
  preserved in extras); `function_call` → `ToolCallPart{ID: call_id}`.
  Stop reason from `status` + `incomplete_details` (FS §5 note). Usage
  from `input_tokens`/`output_tokens` (+ `cached_tokens`,
  `reasoning_tokens` details).
- **Replay**: `ReasoningPart.Raw` with `Provider == "openai"` re-emits the
  reasoning item verbatim in `input` — reasoning continuity in tool loops
  with zero stored state; foreign reasoning parts are dropped.
- **Stream map**: semantic events → unified events —
  `response.output_text.delta` → `TextDelta`,
  `response.reasoning_summary_text.delta` → `ReasoningDelta`,
  `response.output_item.added(function_call)` → `ToolCallStart`,
  `response.function_call_arguments.delta` → `ToolCallDelta`,
  `response.output_item.done` → `ToolCallEnd`, `response.completed` →
  `MessageEnd` (usage), `response.failed`/`error` → normalized in-stream
  error. Accumulation keys off `(output_index, content_index)`.
- **Errors**: `*openai.Error` → `ProviderError` (type/code/param preserved).
- `Models()`: `GET /models`; `Client()` returns the SDK client (raw access
  incl. Chat Completions for legacy knobs).

### 3.3 OpenAI-compatible providers — OpenRouter & ZAI (shared adapter + Dialect)

`providers/internal/openaicompat` implements one adapter over `openai-go`'s
Chat Completions surface; OpenRouter and ZAI each supply a `Dialect`
(chat completions is OpenRouter's canonical surface — its `/responses`
endpoint is beta — and ZAI's only OpenAI-style surface):

```go
package openaicompat

type Dialect interface {
    Name() string
    DefaultBaseURL() string
    APIKeyEnv() string
    Capabilities() []llm.Capability

    // ApplyRequest injects dialect-specific fields (effort mapping, session
    // affinity, ProviderOptions extras) into the outgoing params, using
    // openai-go's extra-fields mechanism for non-standard fields.
    ApplyRequest(req *llm.Request, params *openai.ChatCompletionNewParams) error

    // ExtractParts maps a choice message to parts, letting dialects surface
    // extras (ZAI reasoning_content, OpenRouter reasoning_details) read from
    // the SDK's preserved unknown fields (JSON.ExtraFields / RawJSON()).
    ExtractParts(msg *openai.ChatCompletionMessage) ([]llm.Part, error)

    // ExtractDeltaEvents does the same per stream chunk (reasoning deltas,
    // mid-stream error chunks). Returning an error aborts the stream with a
    // normalized error.
    ExtractDeltaEvents(chunk *openai.ChatCompletionChunk) ([]llm.Event, error)

    MapStopReason(raw string) llm.StopReason
    MapError(err error) *llm.ProviderError
    MapUsage(u *openai.CompletionUsage, raw []byte) llm.Usage // dialect cost/cache extras
}
```

The adapter owns everything common: message/part conversion, tools, response
format, streaming loop (`stream_options.include_usage` set where a dialect
requires it; OpenAI's `obfuscation` chunk field tolerated), tool-call-delta
index tracking, `Collect`-equivalence, retries config, and the `Provider`
interface plumbing. Dialects stay small and declarative.

Dialect specifics (surface per functional spec §14):

- **openrouter**: attribution headers at construction; `session_id`,
  `models`, `provider`, `plugins`, `reasoning{...}` via extra fields;
  detects mid-stream error chunks (`finish_reason == "error"` or `error`
  extra field) → normalized in-stream error; maps `usage.cost` → `CostUSD`;
  extracts `provider`, `native_finish_reason`, annotations,
  `reasoning_details` into typed response extras (accessor
  `openrouter.Extras(resp *llm.Response) (*ResponseExtras, bool)`).
- **zai**: `thinking`/`reasoning_effort`/`do_sample`/`tool_stream`/
  `request_id`/`user_id` extras; auto-sets `tool_stream: true` when
  streaming with tools; `delta.reasoning_content` → `ReasoningDelta`;
  numeric-string business-code error table; base URL selector
  (`zai.General`, `zai.CodingPlan`); curated static model list.

**SSE comment lines** (OpenRouter keep-alives): the SSE spec says comment
lines are ignored, and Stainless SDK decoders follow it. **Verification item
(resolved in the OpenRouter implementation phase)**: a recorded-fixture test proving `openai-go` tolerates
`: OPENROUTER PROCESSING` lines; if it doesn't, mitigation is an
`http.RoundTripper` wrapper in openaicompat that strips comment lines —
isolated, no public API impact.

### 3.4 Construction & configuration

Every provider package:

```go
func New(opts ...Option) (*Provider, error)

// Common options (mirrored per package, mapped onto SDK options):
WithAPIKey(string)        // default: provider env var
WithBaseURL(string)
WithHTTPClient(*http.Client)
WithMaxRetries(int)       // default 2, delegated to the SDK's retry layer
WithTimeout(time.Duration)
WithPriceTable(llm.PriceTable)   // cost estimation override
WithDefaultMaxTokens(int)        // anthropic only
```

No global state; constructors return errors (e.g. missing API key) rather
than panicking. Retries live **only** in the SDK layer — go-llm never adds a
second retry loop.

## 4. Request Pipeline (all providers)

```
Chat/ChatStream(ctx, req)
  1. validate(req, provider)     — required fields; capability pre-flight (§6);
                                   ProviderOptions type/name check
  2. build                       — llm.Request → SDK params (+ dialect extras)
  3. call                        — SDK request (SDK handles retries/backoff)
  4. map                         — SDK response/stream → llm.Response / events
                                   (errors normalized at this boundary only)
```

Error normalization happens in exactly one place per provider (step 3/4
boundary) so the taxonomy can't drift between code paths.

## 5. Usage & Cost

- Adapters populate token fields from native usage (mapping in spec §6/§11).
- `CostUSD` resolution order: provider-reported (OpenRouter `usage.cost`) →
  `PriceTable` estimate → nil.
- `PriceTable` is `map[string]ModelPricing` keyed `"provider/model-id"`,
  with prefix fallback (`"anthropic/claude-opus-4"` matches dated variants).
  `DefaultPriceTable` ships in `pricing_table.go` with a `PriceTableDate`
  constant documenting the snapshot; users override per-provider via
  `WithPriceTable` or globally by mutating a copy.

## 6. Capabilities & Pre-flight Validation

- `capability.go` defines standard constants; provider `Capabilities()`
  returns a fixed slice per provider (not per model).
- `validate.go` implements `validateRequest(caps []Capability, req *Request) error`
  shared by all adapters. Checks (each → wrapped `ErrUnsupported`):
  tools, tool-choice mode, response-format level, reasoning, image/file
  parts, stop sequences, streaming-specific requirements. Model-level
  rejections are NOT predicted client-side — provider errors surface
  normally.

## 7. History Helper

```go
type History struct{ msgs []Message }

func (h *History) Add(msgs ...Message)
func (h *History) AddUserText(text string)
func (h *History) AddResponse(resp *Response)         // assistant turn incl. tool calls + reasoning
func (h *History) AddToolResults(results ...ToolResultPart) // one message, all results
func (h *History) Messages() []Message                // defensive copy
```

Not goroutine-safe (documented); it's a builder, not a store.

## 7A. `schema` Subpackage (stdlib-only)

Hand-rolled, minimal reflection-based generator — deliberately **not** a
full JSON Schema implementation, only the strict-mode subset providers
accept (see functional spec §8):

```go
package schema

// For generates a JSON Schema for T: objects from structs (json tags;
// fields required unless pointer or `,omitempty`), strings/numbers/bools,
// slices, maps[string]X, nested structs. Tags: `description:"..."` and
// `enum:"a,b,c"`. Always emits additionalProperties: false.
func For[T any]() (json.RawMessage, error)
func MustFor[T any]() json.RawMessage

// ValidateArgs checks model-emitted tool arguments against a tool's schema
// (types, required fields, enums — the supported subset, not full JSON
// Schema validation).
func ValidateArgs(t llm.Tool, args json.RawMessage) error
```

Rejected alternative: `invopop/jsonschema` dependency — the supported schema
subset is small enough that ~300 lines of `reflect` keeps `go.mod` clean.
Unsupported Go types (channels, funcs, recursive structs) return errors, not
panics.

## 7B. `llmtest` Package

```go
package llmtest

type Provider struct { /* configurable Name, Capabilities */ }

func New(opts ...Option) *Provider           // implements llm.Provider

func (p *Provider) EnqueueResponse(r *llm.Response)
func (p *Provider) EnqueueStream(events ...llm.Event)
func (p *Provider) EnqueueError(err error)
func (p *Provider) Requests() []*llm.Request // copies of every request received
```

Steps are consumed FIFO by `Chat`/`ChatStream`; an exhausted queue fails the
test loudly. `ChatStream` yields enqueued events as a real
`iter.Seq2[Event, error]` (honoring context cancellation) so consumer
streaming code is exercised realistically. Goroutine-safe. Also serves as
the reference third-party `Provider` implementation.

## 8. Technical Challenges (solved here)

1. **Stream normalization state machine.** Anthropic indexes content blocks;
   OpenAI-compat indexes tool-call deltas within choices. Each adapter keeps
   a per-stream `map[int]*toolCallState` (id, name, args buffer) and emits
   `ToolCallStart` exactly once per index, `ToolCallEnd` on block stop /
   finish. Invariant tested via fixture streams: event sequence is always
   `MessageStart, (TextDelta|ReasoningDelta|ToolCall*)*, MessageEnd`.
2. **Reasoning round-trip.** `ReasoningPart.Raw` + `Provider` tag; adapters
   replay raw payloads only when `Provider` matches, else drop the part
   (documented, matches provider behavior of ignoring/rejecting foreign
   reasoning). OpenRouter: `reasoning_details` must be echoed back verbatim —
   handled the same way.
3. **Two error layers in one chain.** `ProviderError.Unwrap() → sentinel`
   gives `errors.Is` + `errors.As` with a single wrap. No multi-error trees.
4. **Typed ProviderOptions without generics contortions.** Marker interface
   + name check. Compile-time safety inside each provider package; runtime
   name check guards cross-provider mistakes.
5. **openai-go extra fields.** Non-standard request fields via
   `SetExtraFields`; non-standard response fields via preserved raw JSON
   (`.JSON.ExtraFields` / `RawJSON()`). This is the load-bearing mechanism
   for OpenRouter/ZAI dialects and gets dedicated fixture tests.
6. **Auto-set flags.** `store: false` + `include:
   ["reasoning.encrypted_content"]` (OpenAI Responses), `tool_stream`
   (ZAI, when streaming+tools), `stream_options.include_usage` (CC dialects
   where required) — set in the adapter, invisible to callers, covered by
   request-build golden tests.
7. **Serialization forward-compatibility.** Unknown part types decode to
   `UnknownPart` (raw JSON preserved, re-marshaled verbatim, skipped by
   adapters) instead of erroring — histories written by a newer go-llm
   remain loadable by an older one. Extension-part decoding requires the
   provider package to be imported (its `init()` registers the decoder);
   otherwise the part degrades to `UnknownPart`, which is safe.
8. **Parse[T] on json-mode providers.** Schema guidance is appended to
   `System` (never injected into user messages) and validation happens
   client-side; the retry turn includes the validation error verbatim so
   the model can self-correct. Bounded by `WithParseRetries` — no infinite
   loops.

## 9. Testing Strategy

- **Golden request-build tests** (per provider): `llm.Request` → serialized
  SDK params JSON compared against golden files. Covers effort mapping,
  cache hints, tools, extras, defaults (MaxTokens, include_usage,
  tool_stream, OpenAI `store: false` + encrypted-reasoning `include`).
- **Response/stream fixture tests**: recorded provider JSON/SSE payloads
  replayed via `httptest.Server` → assert parts, stop reasons, usage, and
  unified event sequences. Fixtures include: OpenRouter comment keep-alives,
  OpenRouter mid-stream error chunk, ZAI `reasoning_content`, Anthropic
  thinking + tool-use blocks, refusal stop, parallel tool calls.
- **Collect-equivalence property**: for every stream fixture with a matching
  non-stream fixture, `Collect(stream)` equals the `Chat` response.
- **Error-mapping table tests**: status/code → sentinel for all providers
  (incl. ZAI business codes, OpenRouter 402/403-with-metadata).
- **Validation tests**: capability mismatches → `ErrUnsupported`.
- **Serialization round-trip property**: for every fixture response and
  hand-built message set, `marshal → unmarshal → marshal` is byte-identical,
  and `ReasoningPart.Raw` survives untouched; unknown-part-type fixtures
  decode to `UnknownPart` and re-marshal verbatim.
- **Schema generator golden tests**: Go structs → golden schema JSON
  (required/optional, enums, nesting, error cases for unsupported types);
  `ValidateArgs` table tests (valid, missing required, wrong type, bad enum).
- **Parse[T] tests**: run against `llmtest.Provider` — json-schema path,
  json-mode fallback path, retry-on-invalid path, retries-exhausted error.
- **Middleware tests**: `Wrap` ordering, chat and stream wrappers, pass-through
  of `Name/Capabilities/Models`.
- **`llmtest` self-tests** double as the `Provider` interface conformance
  suite.
- **Live e2e suite** (`internal/e2e`, `//go:build live`): a
  capability-driven scenario harness run against the real provider APIs.
  One scenario per standard capability, written once and parameterized by
  `llm.Provider` — the runner iterates `provider.Capabilities()` and
  executes exactly the declared scenarios, so capability declarations are
  themselves verified against reality. Scenarios (real prompts, loose
  assertions — contains-word / non-nil / >0, never exact-match):
  - chat ("reply with exactly one word: pong"), streaming (≥2 deltas,
    usage on `MessageEnd`, coherent `Collect`)
  - tools: forced call → parseable args → tool-result round-trip second
    turn; parallel-tools variant
  - structured output + `Parse[T]` (contact extraction → schema-valid)
  - effort/reasoning: `high` → `Reasoning` parts present; `none` → absent
  - reasoning replay: two-turn same-provider round-trip of raw payloads
  - multimodal: bundled red-square PNG → answer contains "red"
  - prompt caching (Anthropic): >4K-token cached system prompt, second
    call shows `CacheReadTokens > 0`
  - usage/cost: tokens > 0; `CostUSD` non-nil on OpenRouter
  - live error mapping: bogus model → `ErrNotFound`/`ErrBadRequest`;
    invalid key → `ErrAuth`
  - models listing non-empty and containing the model under test
  - cross-provider handoff: tool-using conversation started on provider A
    continues on provider B without error
  Cost discipline: cheapest suitable model per provider pinned in one
  constants file, tight `MaxTokens`. Never runs in CI by default.
  **Credentials/config**: the harness loads `gollm-test.json` from the
  repo root — gitignored, with `gollm-test.json.sample` committed:
  ```json
  {
    "providers": {
      "anthropic":  {"api_key": "sk-ant-..."},
      "openai":     {"api_key": "sk-..."},
      "openrouter": {"api_key": "sk-or-...", "model": "optional-override"},
      "zai":        {"api_key": "...", "base_url": "optional"}
    }
  }
  ```
  Per-provider fields: `api_key` (required to run), optional `model`
  (overrides the pinned cheap default) and `base_url`. Providers missing
  from the file fall back to their env var (`ANTHROPIC_API_KEY`, …);
  still missing → that provider's scenarios **skip with a visible SKIP**,
  never fail. The library itself never reads this file (FS §17).
- **Fixture recording**: the e2e harness accepts a `-record` flag that
  captures live wire payloads into the offline fixture corpus — fixtures
  stay honest as providers evolve instead of being hand-edited.
- Tooling: stdlib `testing` + `go-cmp`; `go vet` + `golangci-lint`; race
  detector on. Coverage target: ≥85% on mapping/adapters, no vanity target
  on plumbing.

## 10. Non-Goals Reaffirmed

No built-in logging or metrics — the middleware seam (`llm.Wrap`) and
`WithHTTPClient` are the instrumentation points; go-llm ships the seam, not
integrations. No global registries (the sole exception: the part-type
decoder registry for serialization, write-once at `init()`). No model
capability database, no silent parameter clamping.
