# go-llm

One clean Go interface to chat LLMs — Anthropic, OpenAI, ChatGPT
subscription plans, OpenRouter, and self-hosted servers (vLLM, Ollama, any
OpenAI-compatible endpoint) — with normalized streaming, tool calling,
structured output, reasoning, usage, cost, and errors.

`go-llm` is a low-level **provider client library**: it does one layer well
and stops there. No agent loops, no prompt frameworks, no magic. The core
package has **zero third-party dependencies**; official vendor SDKs are used
where they exist and pulled in only by the provider package you import.

## Highlights

- **One `llm.Provider` interface** — blocking `Chat` and streaming
  `ChatStream` (`iter.Seq2` iterators), the same request/response model
  everywhere, per-provider escape hatches down to the raw SDK client.
- **Providers**: Anthropic (API key), OpenAI
  (Responses API; API key or Sign in with ChatGPT), OpenRouter, and
  self-hosted vLLM. First-party presets run the shared offline
  conformance suite, and credentialed presets have capability-driven live
  scenarios. `chatcompletions.New(baseURL)` covers any other
  OpenAI Chat Completions-compatible server (Ollama, llama.cpp, Groq,
  Together, ...).
- **Credentials, the pi way**: a `llm.CredentialStore` (in memory, or a
  pi-compatible `auth.json` via `authstore.NewFile`) owns each provider's
  credential. Sign in with ChatGPT tokens refresh inside the store's locked
  `Modify`, so concurrent requests and processes refresh exactly once.
- **Tools**: parallel calls, streamed arguments, and a defined contract for
  malformed tool calls — rescue what's rescuable, drop the rest *visibly*
  (`ToolCallDropped`), with an opt-in `llm.RetryDroppedToolCalls` middleware.
  No-argument tool schemas are sent with an explicit empty `properties: {}`
  (OpenAI-compatible `parameters`, Anthropic `input_schema`); non-object
  `properties` values fail locally.
- **Cache-stable prompt changes**: a `system` message with
  `SystemDeltaPart`/`ToolsDeltaPart` changes the instructions or tools
  mid-conversation without rewriting the cached prefix. Models that take it
  in place (per-model `behavior` in `models.json`, as pi tracks it) get
  Anthropic `tool_addition`/`tool_removal` or OpenAI `additional_tools`;
  everything else folds deterministically into the system prompt.
- **Structured output**: `llm.Parse[T]` decodes model output straight into
  your struct (native JSON-schema mode where supported, forced-tool or
  JSON-mode fallback elsewhere), and `schema.For[T]` generates JSON Schema
  from Go types for tool inputs, with `schema.ValidateArgs` for checking
  model-emitted arguments.
- **Reasoning**: one `Effort` dial across providers; reasoning output is
  normalized, and raw provider payloads (signed thinking blocks, encrypted
  reasoning items) round-trip so same-provider replay just works.
- **Persistence & portability**: canonical, versioned JSON for messages and
  responses. A tool-using conversation started on one provider can be
  serialized and *continued on another* — cross-provider handoff is part of
  the live test matrix, not a hope.
- **Usage & cost**: normalized tokens (including cache reads/writes and
  reasoning tokens), `CostUSD` from native provider reporting (OpenRouter) or
  estimated from an embedded, refreshable [models.dev](https://models.dev)
  price table, including request-wide long-context pricing tiers — with
  provenance in `CostSource` (`native` vs `estimated`) — plus `ContextUsage`
  for context-window accounting. Live model catalogs preserve independent
  input/output/cache-read/cache-write availability, so unknown rates are not
  displayed as free while explicit zero rates remain visible. Anthropic
  one-hour cache writes are tracked separately (`CacheWrite1hTokens`); no
  price table publishes their rate, so an estimated cost containing them is
  reported as unknown rather than understated.
- **Capabilities**: provider-wide `Capabilities()` discovery with pre-flight
  request validation, plus advisory per-model capabilities from rich live
  catalogs such as OpenRouter. Unsupported provider features fail
  fast with `ErrUnsupported`; incomplete model metadata never blocks a call.
- **Conveniences**: `Session` (auto-managed history, session tools with
  `AddToolResults` + `Continue`, cumulative usage), `History`,
  `PromptTemplate`, `llm.Ptr` for optional scalar fields, middleware via
  `llm.Wrap`, and observability built in but silent by default (`slog`
  logging, `UsageTracker`, wire capture via `WithWireCapture`), plus an
  optional, content-free OpenTelemetry middleware in the separate
  `contrib/otelllm` module (GenAI semantic conventions). Sensitive
  headers are always redacted; captured URLs and bodies remain
  application-sensitive data. Built-in failure logs use safe provider-error
  summaries and never emit provider-controlled bodies or metadata.
- **Testing**: `llmtest` — like `net/http/httptest`, but for code that
  consumes go-llm.
- **CLI**: `llm-cli`, a curl-like frontend built entirely on the public API.

## Install

```sh
go get github.com/pkieltyka/go-llm
```

Go 1.26 is the minimum version for users of the module. Development and releases
use Go 1.27.3 or newer, while CI retains a Go 1.26 compatibility lane. Provider SDK
dependencies are pulled only when importing provider packages.

## Quick start

```go
package main

import (
	"context"
	"fmt"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/openai"
)

func main() {
	ctx := context.Background()

	p, err := openai.New() // reads OPENAI_API_KEY
	if err != nil {
		panic(err)
	}

	resp, err := p.Chat(ctx, &llm.Request{
		Model:    "gpt-5.5",
		Messages: []llm.Message{llm.UserText("Explain Go iterators in one sentence.")},
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(resp.Text())
}
```

Streaming:

```go
for text, err := range llm.StreamText(p.ChatStream(ctx, req)) {
	if err != nil {
		return err
	}
	fmt.Print(text)
}
```

Structured output:

```go
type Summary struct {
	Title string   `json:"title"`
	Tags  []string `json:"tags"`
}

summary, resp, err := llm.Parse[Summary](ctx, p, &llm.Request{
	Model:    "gpt-5.5",
	Messages: []llm.Message{llm.UserText("Summarize this release note.")},
})
_, _ = summary, resp
```

## Providers

| Package | Auth | Notes |
|---|---|---|
| `providers/anthropic` | `ANTHROPIC_API_KEY` or `WithAPIKey` | Messages API |
| `providers/openai` | `OPENAI_API_KEY`, `WithAPIKey`, or `WithCredentials` (API key or Sign in with ChatGPT) | Responses API — reasoning survives across tool-call turns |
| `providers/openrouter` | `OPENROUTER_API_KEY` or `WithAPIKey` | Chat Completions; routing/plugins, explicit rich model discovery, and native per-request cost reporting |
| `providers/vllm` | optional `WithAPIKey` (vLLM `--api-key`) | Self-hosted vLLM preset: host-first, era-aware, live-tested |
| `providers/chatcompletions` | optional `WithAPIKey` | Common entry point for OpenAI Chat Completions-compatible servers, including Ollama: `New(baseURL, ...)` + declarative `Compat` quirks |

```go
anthropic.New()  // ANTHROPIC_API_KEY
openai.New()     // OPENAI_API_KEY
openrouter.New() // OPENROUTER_API_KEY
```

OpenAI Responses callers can select the user-visible reasoning-summary detail
without adding an OpenAI-specific field to `llm.Request`:

```go
resp, err := p.Chat(ctx, &llm.Request{
	Model:    "gpt-5.5",
	Effort:   llm.EffortHigh,
	Messages: []llm.Message{llm.UserText("Compare the two proposals.")},
	ProviderOptions: openai.Options{
		ReasoningSummary: openai.ReasoningSummaryConcise,
	},
})
```

The supported selectors are `ReasoningSummaryAuto`,
`ReasoningSummaryConcise`, and `ReasoningSummaryDetailed`. Leaving the field
empty preserves the existing behavior (`summary: "auto"` when `Effort` is
set); unknown values fail before a request is sent. OpenAI remains
authoritative about which models accept each selector.

### Credentials

Authentication mirrors [pi](https://github.com/earendil-works/pi). A
`llm.CredentialStore` holds one credential per provider id, and
`WithCredentials(store)` makes a provider read it on every request. A stored
credential owns its provider (no environment fallback); with nothing stored,
or a keyless `api_key` entry as pi allows, the provider's usual environment
variable is used. `WithCredentials` is
mutually exclusive with `WithAPIKey`/`WithAPIKeyFunc`.

```go
store := authstore.NewFile(filepath.Join(home, ".pi/agent/auth.json"))
p, err := openai.New(openai.WithCredentials(store))
```

`authstore.NewFile` reads and writes pi's `auth.json` format
(`{"openai": {"type": "oauth", ...}, "anthropic": {"type": "api_key", ...}}`,
bare or under a `"providers"` wrapper). `Modify` is the only write path: it
holds pi's own lock on `<path>.lock` (so pi and go-llm processes exclude each
other on a shared file), writes atomically with mode 0600, and preserves
fields it does not own.
`authstore.NewMemory` is the in-process equivalent.

OpenAI also accepts a Sign in with ChatGPT OAuth credential, which calls the
ordinary `api.openai.com` Responses API billed to a ChatGPT subscription. The
provider refreshes it five minutes before expiry inside `Modify` (re-checking
under the lock, so only one refresher wins), stores the rotated tokens before
using them, and lets a started refresh finish even if the caller's request is
cancelled. `openai.NewLoginFlow` mints the initial credential with a browser
PKCE flow; the host stores it:

```go
flow, err := openai.NewLoginFlow(openai.WithLoginDeviceID(deviceUUID))
authorization, err := flow.Begin(ctx)
showLoginURL(authorization.URL())
credential, err := flow.Complete(ctx) // or flow.Submit(ctx, pastedCallbackURL)
_, _, err = store.Modify(ctx, "openai", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
	return credential, true, nil
})
```

The callback binds only `127.0.0.1:1455`. A Sign in with ChatGPT token
reaches only the subscription's models (`Models` lists them) and a
streaming-only endpoint. So, as pi does, `Chat` drains a stream there, and
`MaxTokens`, `Temperature`, `TopP` and prompt-cache retention are not sent.
Anthropic and OpenRouter accept `api_key` credentials only; a stored
Anthropic subscription login (pi's Claude Pro/Max OAuth) fails with an
`ErrAuth` that says to use an API key, because Anthropic limits subscription
credentials to its own apps.

Provider-specific request extensions live in
each provider's `Options` type, passed through `Request.ProviderOptions`; the
raw SDK client is always reachable via each provider's `Client()`. Ordinary
`openai.Options` fields use go-llm and standard-library types; only advanced
escape hatches such as `Provider.Client`, `chatcompletions.Dialect`,
`chatcompletions.Config`, and `Provider.BuildParams` are vendor-coupled and
stability-exempt before v1.

## Self-hosted (vLLM, Ollama, any OpenAI-compatible server)

Self-hosted servers are first-class: constructors are **host-first** and the
API key is **optional** (no environment fallback). The `providers/vllm`
preset targets the current stable vLLM v0.26.0 protocol and knows its
dialect — `reasoning` output (parsed to
`llm.ReasoningPart`, streamed as `ReasoningDelta`), `Effort` →
`reasoning_effort`, choice-less mid-stream error events, `max_model_len` as
`ModelInfo.ContextWindow`, and typed extensions (`top_k`, `min_p`,
`stop_token_ids`, `chat_template_kwargs`, `thinking_token_budget`,
`vllm_xargs`, plus native
`structured_outputs` constraint modes — regex/choice/grammar/structural-tag
via `vllm.StructuredOutputs`; JSON schema stays on the unified
`ResponseFormat`):

```go
p, err := vllm.New("http://localhost:8000/v1") // keyless by default
if err != nil {
	panic(err)
}

model, err := p.ResolveModel(ctx, "qwen") // resolves deployment prefixes/suffixes
if err != nil {
	panic(err)
}

resp, err := p.Chat(ctx, &llm.Request{
	Model:    model.ID,
	Effort:   llm.EffortNone, // thinking-by-default models answer tersely
	Messages: []llm.Message{llm.UserText("hello")},
	ProviderOptions: vllm.Options{
		XArgs: map[string]any{"custom_engine_arg": "1"},
	},
})
```

For deployments that support it, `vllm.Options.ThinkingTokenBudget` caps
reasoning without adding a provider-neutral knob. When `MaxTokens` is set,
go-llm clamps the configured budget to reserve 1,024 tokens for the visible
answer and rejects thinking-disabled combinations before sending the request.

vLLM also gives you **exact token counting**: the preset exposes the
server's `/tokenize` endpoints as typed extension methods, so context
accounting is ground truth (server-rendered chat template, tools included)
instead of an estimate — live-verified to match a real request's
`prompt_tokens` exactly:

```go
result, err := p.Tokenize(ctx, req)     // same conversion + validation as Chat
usage := result.ContextUsage()          // exact count vs the model's max_model_len
fmt.Printf("prompt occupies %d of %d tokens (%.2f%%)\n",
	usage.UsedTokens, usage.Window, usage.UsedPercent)
```

`Detokenize` and `TokenizerInfo` round out the family. For Ollama and other
inference servers that expose an OpenAI Chat Completions-compatible API, use
`providers/chatcompletions` as the common entry point. `WithName` labels the
provider in responses, logs, and error metadata. For a local Ollama server:

```go
p, err := chatcompletions.New("http://localhost:11434/v1",
	chatcompletions.WithName("ollama"),
	chatcompletions.WithCompat(chatcompletions.Compat{StreamIncludeUsage: true}),
)
```

`StreamIncludeUsage` requests usage in the final stream chunk. For another
compatible endpoint, set its URL, optional key, and required quirks:

```go
p, err := chatcompletions.New("https://api.example.com/v1",
	chatcompletions.WithName("example"),
	chatcompletions.WithAPIKey(os.Getenv("EXAMPLE_API_KEY")),
	chatcompletions.WithCompat(chatcompletions.Compat{StreamIncludeUsage: true}),
)
```

Bonus recipe: current vLLM v0.26.0 also serves an
**Anthropic `/v1/messages` endpoint**, so the Anthropic provider can target a
vLLM box directly — `anthropic.New(anthropic.WithBaseURL("http://localhost:8000"),
anthropic.WithAPIKey("dummy"))` completes chats with usage and even maps
Qwen thinking output to reasoning parts (vLLM emits Anthropic-style thinking
blocks there).

## Persistence and cross-provider handoff

`llm.MarshalMessages` / `llm.UnmarshalMessages` give you a canonical,
versioned JSON envelope for conversation history — safe to store in a
database and reload across releases. Because every adapter re-encodes the
neutral history into its own wire shape (and drops what only the original
provider can accept, like signed thinking blocks), a saved conversation can
be continued on a *different* provider: mid-conversation failover,
cheap-model tool loops handed to a stronger model for synthesis, or histories
that simply outlive your vendor choice. Reasoning is replayed only to the
model that produced it. On a model switch, even within one provider (an
OpenRouter hop from `anthropic/*` to `openai/*`, or Opus to Haiku), the
previous model's reasoning payloads are dropped. Keep `Message.Model`
populated (`History.AddResponse` does) so go-llm can tell.

To persist a turn *while* it streams, collect it incrementally.
`llm.Collector` has exactly `Collect`'s semantics, and `Snapshot` returns an
owned copy of the partial response at any point:

```go
var c llm.Collector
for event, err := range p.ChatStream(ctx, req) {
	if err == nil {
		err = c.Add(event)
	}
	if err != nil {
		partial, finishErr := c.Finish(err) // partial survives, as with Collect
		return persistAborted(partial, finishErr)
	}
	forward(event)
	if time.Since(lastSave) > 100*time.Millisecond {
		savePartial(c.Snapshot())
		lastSave = time.Now()
	}
}
resp, _ := c.Finish(nil)
```

`Request.CacheRetention` (`none`, `short`, `long`) overrides every cache hint
on a request. Use `none` for one-off calls, such as summarization, that
shouldn't pollute a session's prompt cache.

For error handling across providers, `llm.ErrorClass`, `llm.Transient`, and
the log-safe `llm.SafeErrorInfo` classify through the typed sentinel chain.
Retry budgets and backoff stay your policy.

## Changing instructions and tools mid-conversation

A prompt cache is a prefix, so editing `System` or `Tools` between turns
throws the cache away from the top. Record the change where it happens
instead:

```go
history.Add(llm.Message{Role: llm.RoleSystem, Parts: []llm.Part{
	llm.SystemDeltaPart{Section: "skills", Text: "The deploy skill is active."},
	llm.ToolsDeltaPart{Added: []llm.Tool{deployTool}, Removed: []string{"draft"}},
}})
```

Each provider sends what the model accepts in place and folds the rest into
`System` and `Tools` (`llm.FoldPositional` does the same for your own
`Provider`):

| Provider | In place | Folded |
|---|---|---|
| Anthropic | `role: "system"` messages; tool changes as `tool_addition`/`tool_removal` (beta header added automatically) | models without the `behavior` flags |
| OpenAI | `developer` messages; on these models every request's tools travel in `additional_tools` items (hosted tools stay top-level), removals announced as text | models without the `behavior` flags |
| Chat Completions (OpenRouter, vLLM, Ollama) | — | always |

Which models qualify is per-model data in `models.json` (`behavior`),
generated by pi-style rules in `scripts/model-behavior.ts`.

## Context fitting

Measure a request before sending it, and guard against overflowing the
window:

```go
count, err := llm.CountTokens(ctx, p, req) // exact on vLLM/Anthropic, else an estimate
info, err := llm.ResolveModelInfo(ctx, p, req.Model)

guarded := llm.Wrap(p, llm.ContextGuard(llm.GuardConfig{
	MaxOccupancy:   0.85,
	ClampMaxTokens: true,
	OnNearFull: func(ctx context.Context, req *llm.Request, r llm.GuardReport) error {
		return compactHistory(ctx) // your policy; return an error to abort the call
	},
}))
```

`llm.EstimateRequestTokens` and `llm.EstimateContinuationTokens` are
deterministic, calibrated estimates (within roughly 0.95–1.17× of exact
counts, leaning high). `llm.FitMaxTokens` and `llm.EstimateCostBound` are
pure helpers. Nothing here runs implicitly.

## CLI

```sh
go install github.com/pkieltyka/go-llm/cmd/llm-cli@latest
```

```sh
llm-cli -p openai -m gpt-5.5 "write a short haiku about Go"
echo "long input" | llm-cli -p anthropic -m claude-opus-4-8 -s "summarize stdin"
llm-cli -p openrouter -m openai/gpt-5.5 --usage --json "return a JSON status"
llm-cli -p openai --auth-file ~/.pi/agent/auth.json -m gpt-5.5 "review this diff"

llm-cli models -p openrouter          # list models (table or --json)

llm-cli -p openai -m gpt-5.5 --save chat.json "Start a checklist"
llm-cli -p anthropic -m claude-opus-4-8 --load chat.json --save chat.json "Continue it"
```

`llm-cli models` explicitly performs provider model-discovery network I/O.
OpenRouter uses exactly its `/models` response and does not fall back to
models.dev. Listings include advisory supported/default reasoning effort,
whether the provider positively reports reasoning as required, and model
capabilities. Text output shows `EFFORTS`, `DEFAULT EFFORT`, and
`REASONING REQUIRED`; `--json` emits the corresponding optional fields.
`ReasoningRequired` is selection metadata only and never overrides the
caller's `Request.Effort`.

The last pair saves a conversation with one provider and continues it with
another — the handoff described above, from the shell.

`--auth-file` names a pi-compatible `auth.json` credential store used by
every provider (API keys, or Sign in with ChatGPT OAuth for `openai`); OAuth
refreshes are written back to it under its lock. Without it, providers read
their usual environment variable. `--api-key` is exclusive with
`--auth-file`; its value is visible in process arguments and often shell
history, so prefer the file or the environment.

## Testing your code

Use `llmtest` for unit tests that should never contact a provider — like
`net/http/httptest`, but for code that consumes go-llm:

```go
p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityJSONSchema))
p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(`{"ok":true}`)}})
```

Script responses, streams, and errors; assert on the requests your code made
via `p.Requests()`. See [`examples/testing`](examples/testing) for a complete
worked example.

Provider authors should run two complementary offline suites. `RunConformance`
checks the common lifecycle, streaming, and normalized-result contract.
`RunCapabilityConformance` checks that each reviewed advertised capability
activates its exact native request fields and returns the expected normalized
result. Its `CapabilityInvocation` is fixture control data passed to an
isolated provider factory; it never changes the real request or its model.
Explicit profile exemptions mean “offline evidence gap,” not “unsupported.”
Neither suite proves live account/model availability, quota, cache admission,
or service reliability; credentialed e2e scenarios remain the live evidence.
The advanced `chatcompletions.NewWithDialect` conformance suite intentionally
stays base-only; the public generic engine owns the reusable compatible-engine
activation profile.

Repository commands:

```sh
make build       # compile all packages and create bin/llm-cli
make test        # offline tests with the race detector
make check       # complete credential-free, network-dependent CI gate
make models      # refresh the embedded root models.json snapshot and provenance
```

Live end-to-end tests are behind the `live` build tag and read credentials
from `gollm-test.json` (copy `gollm-test.json.sample`; missing credentials
skip visibly, never fail):

```sh
make e2e-test
```

## Examples

Every program in [`examples/`](examples/) is **dual-mode**: it runs offline
out of the box against the scripted `llmtest` provider, and switches to the
real API when `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY` is set.

```sh
go run ./examples/chat                        # offline, scripted fallback
ANTHROPIC_API_KEY=... go run ./examples/chat  # same program, real API
```

| Example | Shows |
|---|---|
| [`chat`](examples/chat) | One blocking request/response |
| [`stream`](examples/stream) | Streaming deltas with `llm.StreamText` |
| [`tools`](examples/tools) | A full tool round trip: call → local execution → result → answer |
| [`parse`](examples/parse) | `llm.Parse[T]` structured output into a Go struct |
| [`history-replay`](examples/history-replay) | Serialize a conversation and continue it on another provider (a real Anthropic→OpenRouter handoff when both keys are set) |
| [`observability`](examples/observability) | `UsageTracker`, logging, and cost via `llm.Wrap` middleware |
| [`provider-selection`](examples/provider-selection) | Choosing a provider at runtime: env keys, an `LLM_AUTH_FILE` credential store (including Sign in with ChatGPT), or the offline fake |
| [`testing`](examples/testing) | Unit-testing *your* code with `llmtest`: request assertions, streaming, error paths (`go test ./examples/testing/`) |

Godoc examples in `example_test.go` run offline as part of the test suite.

## Status

Pre-1.0: public APIs are intended to be small and stable, but breaking
changes may happen before `v1.0.0` when provider behavior or the unified
model needs correction (the `chatcompletions.Dialect` interface is
explicitly an advanced, stability-exempt surface — prefer the declarative
`Compat`). After `v1.0.0`, standard Go module compatibility rules apply.

## License

MIT.
