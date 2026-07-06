# go-llm

One clean Go interface to chat LLMs — Anthropic, OpenAI, ChatGPT/Claude
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
- **Providers**: Anthropic (API key *or* Claude Pro/Max OAuth), OpenAI
  (Responses API), OpenAI Codex (ChatGPT Plus/Pro subscription OAuth),
  OpenRouter, and self-hosted vLLM — each live-tested against the real API —
  plus `chatcompletions.New(baseURL)` for any other OpenAI-compatible server
  (Ollama, llama.cpp, Groq, Together, ...).
- **Subscription auth**: consume and auto-refresh OAuth credentials minted by
  existing CLIs (pi, claude, codex). `llm.LoadAuthFile` reads the same
  credential format pi uses; renewed tokens are handed back to your code to
  persist.
- **Tools**: parallel calls, streamed arguments, and a defined contract for
  malformed tool calls — rescue what's rescuable, drop the rest *visibly*
  (`ToolCallDropped`), with an opt-in `llm.RetryDroppedToolCalls` middleware.
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
  price table — with provenance in `CostSource` (`native` vs `estimated`) —
  plus `ContextUsage` for context-window accounting.
- **Capabilities**: `Capabilities()` discovery with pre-flight request
  validation — unsupported features fail fast with `ErrUnsupported`, never
  silently.
- **Conveniences**: `Session` (auto-managed history, session tools with
  `AddToolResults` + `Continue`, cumulative usage), `History`,
  `PromptTemplate`, `llm.Ptr` for optional scalar fields, middleware via
  `llm.Wrap`, and observability built in but silent by default (`slog`
  logging, `UsageTracker`, wire capture — `WithWireCapture` — with always-on
  secret redaction).
- **Testing**: `llmtest` — like `net/http/httptest`, but for code that
  consumes go-llm.
- **CLI**: `llm-cli`, a curl-like frontend built entirely on the public API.

## Install

```sh
go get github.com/pkieltyka/go-llm
```

Requires Go 1.26+. Provider SDK dependencies are pulled only when importing
provider packages.

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
| `providers/anthropic` | `ANTHROPIC_API_KEY`, `WithAPIKey`, or `WithOAuth` (Claude Pro/Max) | Messages API |
| `providers/openai` | `OPENAI_API_KEY` or `WithAPIKey` | Responses API — reasoning survives across tool-call turns |
| `providers/openaicodex` | `WithOAuth` only (ChatGPT Plus/Pro) | Responses wire shape against the codex backend |
| `providers/openrouter` | `OPENROUTER_API_KEY` or `WithAPIKey` | Chat Completions; native per-request cost reporting |
| `providers/vllm` | optional `WithAPIKey` (vLLM `--api-key`) | Self-hosted vLLM preset: host-first, era-aware, live-tested |
| `providers/ollama` | none | Data-only local-Ollama preset over the engine below (community-verified) |
| `providers/chatcompletions` | optional `WithAPIKey` | Public engine for ANY OpenAI-compatible server: `New(baseURL, ...)` + declarative `Compat` quirks |

```go
anthropic.New()  // ANTHROPIC_API_KEY
openai.New()     // OPENAI_API_KEY
openrouter.New() // OPENROUTER_API_KEY
```

Subscription providers consume credentials minted by existing tools (pi,
claude, codex) — go-llm refreshes tokens automatically and hands renewals
back to your code to persist; it never writes credential files itself:

```go
auth, err := llm.LoadAuthFile("auth.json") // pi-compatible credential format
if err != nil {
	panic(err)
}

codex, err := openaicodex.New(openaicodex.WithOAuth(auth["openai-codex"], func(updated llm.AuthCredential) {
	// Persist rotated refresh credentials in your application.
}))
```

The same `WithOAuth(cred, onRefresh)` option exists on `providers/anthropic`
for Claude Pro/Max subscriptions. Provider-specific request extensions live
in each provider's `Options` type, passed through `Request.ProviderOptions`;
the raw SDK client is always reachable via each provider's `Client()`.

## Self-hosted (vLLM, Ollama, any OpenAI-compatible server)

Self-hosted servers are first-class: constructors are **host-first** and the
API key is **optional** (no environment fallback). The `providers/vllm`
preset knows vLLM's dialect — `reasoning` output (parsed to
`llm.ReasoningPart`, streamed as `ReasoningDelta`), `Effort` →
`reasoning_effort`, choice-less mid-stream error events, `max_model_len` as
`ModelInfo.ContextWindow`, and typed extensions (`top_k`, `min_p`,
`stop_token_ids`, `chat_template_kwargs`, `vllm_xargs`, plus native
`structured_outputs` constraint modes — regex/choice/grammar/structural-tag
via `vllm.StructuredOutputs`; JSON schema stays on the unified
`ResponseFormat`):

```go
p, err := vllm.New("http://localhost:8000/v1") // keyless by default
if err != nil {
	panic(err)
}

resp, err := p.Chat(ctx, &llm.Request{
	Model:    "Qwen/Qwen3.6-27B-FP8",
	Effort:   llm.EffortNone, // thinking-by-default models answer tersely
	Messages: []llm.Message{llm.UserText("hello")},
	ProviderOptions: vllm.Options{
		XArgs: map[string]any{"custom_engine_arg": "1"},
	},
})
```

Older vLLM deployments (pre-v0.12) use `vllm.WithLegacyEra()`; structured
output rides the era-portable `response_format: json_schema` spelling either
way.

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

`Detokenize` and `TokenizerInfo` round out the family. For everything else
that speaks the Chat Completions shape, the same engine is public:

```go
// Ollama's local convention, as a data-only preset:
p, err := ollama.New("") // http://localhost:11434/v1

// Or any OpenAI-compatible endpoint, with quirks declared as data:
p, err = chatcompletions.New("https://api.example.com/v1",
	chatcompletions.WithName("example"),
	chatcompletions.WithAPIKey(os.Getenv("EXAMPLE_API_KEY")),
	chatcompletions.WithCompat(chatcompletions.Compat{StreamIncludeUsage: true}),
)
```

Bonus recipe, verified live against vLLM 0.24: vLLM ≥0.11.1 also serves an
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
that simply outlive your vendor choice.

## CLI

```sh
go install github.com/pkieltyka/go-llm/cmd/llm-cli@latest
```

```sh
llm-cli -p openai -m gpt-5.5 "write a short haiku about Go"
echo "long input" | llm-cli -p anthropic -m claude-opus-4-8 -s "summarize stdin"
llm-cli -p openrouter -m openai/gpt-5.5 --usage --json "return a JSON status"

llm-cli models -p openrouter          # list models (table or --json)

llm-cli -p openai -m gpt-5.5 --save chat.json "Start a checklist"
llm-cli -p anthropic -m claude-opus-4-8 --load chat.json --save chat.json "Continue it"
```

The last pair saves a conversation with one provider and continues it with
another — the handoff described above, from the shell.

## Testing your code

Use `llmtest` for unit tests that should never contact a provider — like
`net/http/httptest`, but for code that consumes go-llm:

```go
p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityJSONSchema))
p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(`{"ok":true}`)}})
```

Script responses, streams, and errors; assert on the requests your code made
via `p.Requests()`. See [`examples/testing`](examples/testing) for a complete
worked example. Repository checks:

```sh
go vet ./...
go test -race ./...
```

Live end-to-end tests are behind the `live` build tag and read credentials
from `gollm-test.json` (copy `gollm-test.json.sample`; missing credentials
skip visibly, never fail):

```sh
go test ./internal/e2e -tags live
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
| [`provider-selection`](examples/provider-selection) | Choosing a provider at runtime: env keys, an `LLM_AUTH_FILE` credential file (including codex OAuth), or the offline fake |
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
