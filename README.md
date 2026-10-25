# go-llm

One clean Go interface to chat LLMs — Anthropic, OpenAI, ChatGPT
subscription plans, OpenRouter, and self-hosted servers (vLLM, Ollama, any
OpenAI-compatible endpoint) — with normalized streaming, tool calling,
structured output, reasoning, usage, cost, and errors. It also speaks to
classifier (decision) models such as Jev, Clef and GPT-6 Luna.

`go-llm` is a low-level **provider client library**: it does one layer well
and stops there. No agent loops, no prompt frameworks, no magic. The core
package has **zero third-party dependencies**; official vendor SDKs are used
where they exist and pulled in only by the provider package you import.

## Highlights

- **One `llm.Provider` interface**: blocking `Chat`, streaming `ChatStream` (`iter.Seq2`), one request/response model, and the raw SDK client when you need it.
- **Providers**: Anthropic, OpenAI (API key or Sign in with ChatGPT), OpenRouter, vLLM, and any Chat Completions server (Ollama, llama.cpp, ...).
- **Credentials the pi way**: a pi-compatible `auth.json` store, shared safely with pi, with refreshes done once across processes.
- **Classifier models** (Jev, Clef, GPT-6 Luna) behind `llm.Classify`: typed choice, score and bool questions answered with probabilities, through OpenRouter, TypeSafe, Cloudflare Workers AI, Vercel AI Gateway, OpenCode Zen, OpenAI or any System One server.
- **Tools** with parallel calls, streamed arguments, and malformed calls dropped *visibly*.
- **Structured output** into Go structs with `llm.Parse[T]`, and JSON Schema from Go types.
- **Cache-stable prompt changes**: change instructions or tools mid-conversation without breaking the prompt cache.
- **Reasoning** behind one `Effort` dial, with provider payloads round-tripped for replay.
- **Portable conversations**: versioned JSON you can save and continue on another provider.
- **Usage, cost and context fitting**: normalized tokens, native or estimated cost, token counting and a context guard.
- **Observability** built in and silent by default, plus OpenTelemetry middleware in `otelllm`.
- **Testing** with `llmtest`, like `net/http/httptest` for code that uses go-llm, and **`llm-cli`**, a curl-like CLI.

See [Details](#details) for the precise contracts behind each of these.

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
| `providers/anthropic` | `ANTHROPIC_API_KEY`, `WithAPIKey`, or `WithCredentials` (API key) | Messages API |
| `providers/openai` | `OPENAI_API_KEY`, `WithAPIKey`, or `WithCredentials` (API key or Sign in with ChatGPT) | Responses API — reasoning survives across tool-call turns |
| `providers/openrouter` | `OPENROUTER_API_KEY`, `WithAPIKey`, or `WithCredentials` (API key) | Chat Completions; routing/plugins, explicit rich model discovery, and native per-request cost reporting |
| `providers/vllm` | optional `WithAPIKey` (vLLM `--api-key`) | Self-hosted vLLM preset: host-first, era-aware, live-tested |
| `providers/chatcompletions` | optional `WithAPIKey` | Common entry point for OpenAI Chat Completions-compatible servers (Ollama, llama.cpp, Groq, Together, ...): `New(baseURL, ...)` + declarative `Compat` quirks |
| `providers/systemone` | `TYPESAFE_API_KEY`, `AI_GATEWAY_API_KEY`, `OPENCODE_API_KEY`, `WithAPIKey`, or `WithCredentials` | [Classifier models](#classifier-models) only, over TypeSafe's System One protocol: `NewTypeSafe`, `NewVercelAIGateway`, `NewOpenCode`, or `New(baseURL)` for any System One server (llama.cpp) |
| `providers/cloudflare` | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID`, options, or `WithCredentials` | Classifier models only, on Cloudflare Workers AI (Clef, Clef Flash, Clef Omni, Jev) |

```go
anthropic.New()  // ANTHROPIC_API_KEY
openai.New()     // OPENAI_API_KEY
openrouter.New() // OPENROUTER_API_KEY
```

OpenRouter and OpenAI (GPT-6 Luna over the Decisions API, API keys only) also
classify beside chat.

First-party presets run the shared offline conformance suite, and
credentialed presets have capability-driven live scenarios.

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
fields it does not own. An `api_key` may use pi's indirection:
`"$OPENAI_API_KEY"` or `"${VAR}"` reads an environment variable (or the
entry's `env` map), and `"!op read op://vault/key"` runs a shell command —
the latter only with `authstore.NewFile(path, authstore.WithKeyCommands())`,
since it runs code named by a file. Resolution happens on read; the file
keeps the reference.
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
Anthropic also accepts typed Pi-compatible `oauth` credentials through
`anthropic.WithCredentials(store)`, with locked refresh persistence. Its
`anthropic.NewLoginFlow()` supports browser login and
`anthropic.WithLoginMethod(anthropic.LoginCopyCode)` selects the copy-code page.
Both return only a private local launch URL: remote browsers require forwarding
that launch port, even in copy-code mode. A missing local listener is an error.
The host persists the returned credential. Pi's fixed client does not require
an issued `ClientID`; conflicting explicit client IDs are rejected.

Anthropic OAuth uses Pi's pinned request headers and system/tool adaptations.
Use the prepared `Chat`, `ChatStream`, `Models` and `CountTokens` methods;
raw `Client()` SDK requests with stored OAuth fail closed. API-key SDK access is
unchanged. Compatibility does not establish account eligibility or included-plan
allowances: eligible third-party usage may be charged to usage credits. See
[Claude authentication guidance](https://support.claude.com/en/articles/13189465-log-in-to-your-claude-account).
OpenRouter accepts stored `api_key` credentials only.

Provider-specific request extensions live in
each provider's `Options` type, passed through `Request.ProviderOptions`; the
raw SDK client is reachable via each provider's `Client()` (Anthropic OAuth
requires the prepared provider operations described above). Ordinary
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

## Classifier models

A classifier (decision) model, such as TypeSafe's Jev, Cloudflare's Clef or
OpenAI's GPT-6 Luna, answers typed questions about a JSON state with
probabilities instead of prose:

- a **choice** among labelled values (the choice, each value's probability,
  a confidence);
- a **score** on ordered levels (the expected level index, each level's
  probability, a confidence);
- a **bool**, the probability that the statement is true.

They are much cheaper and faster than asking a chat model for a label, and
the answer is calibrated. `llm.NewClassifierModel` binds the provider, the
model and the questions once; each call then passes only the state:

```go
p, _ := openrouter.New() // OpenRouter serves Jev, Clef, Luna and others
triage := llm.NewClassifierModel(p, "typesafe/jev-1.14", llm.WithClassifierQuestions(
	llm.ChoiceQuestion("category", "Classify the support message.",
		llm.Choice{Value: "bug", Description: "A defect report"},
		llm.Choice{Value: "question", Description: "A how-to question"}),
	llm.ScoreQuestion("urgency", "How urgent is it?", "low", "medium", "high"),
	llm.BoolQuestion("satisfied", "Is the user satisfied?", "Satisfied", "Not satisfied"),
))
result, err := triage.Classify(ctx, map[string]string{"message": "It crashes when I save!"})
// result.Answers["category"].Choice == "bug"; .Probabilities["bug"] ≈ 1
// result.Answers["urgency"].Score ≈ 1.99 (high); result.Usage has the cost
```

`llm.Classify(ctx, p, req)` is the one-shot form with a full
`ClassifyRequest`, which can also carry images. A provider classifies when it
implements `llm.Classifier`, found through middleware, and lists
`CapabilityClassify`; others fail with `ErrUnsupported`. Requests are
validated before any I/O, with each service's documented limits. Answers are
checked (declared choices, probabilities in [0, 1], scores within the levels),
and usage is recorded first, so a refusal or a malformed answer still returns
the result, its valid answers and its cost beside `ErrContentFiltered` or
`ErrServer`. Cost is native where reported (OpenRouter), else estimated only
from classifier pricing: your `WithPriceTable`, or a catalog row marked
`CapabilityClassify`; a chat model's price is never assumed.
Wrapped with `llm.Wrap`, a classifier stays one: `Middleware.Classify` sees
each call, so `UsageTracker` counts classification cost and `otelllm` records a
`classify {model}` span. OpenRouter's `Models` includes its classifier catalog (output modality
`decisions`), marked with `CapabilityClassify`, and the embedded model table
does the same for TypeSafe and Vercel AI Gateway. See
[`examples/classify`](examples/classify).

| Service | Constructor | Images |
|---|---|---|
| OpenRouter | `openrouter.New` | as image parts before the state |
| TypeSafe | `systemone.NewTypeSafe` | PNG/JPEG assets |
| Cloudflare Workers AI | `cloudflare.New` | Clef: up to 4 data URLs; Jev: assets |
| Vercel AI Gateway, OpenCode Zen | `systemone.NewVercelAIGateway`, `systemone.NewOpenCode` | text only |
| OpenAI Decisions (GPT-6 Luna) | `openai.New` (API key) | up to 128 images |
| Any System One server (llama.cpp) | `systemone.New(baseURL)` | data URLs (`images`); `WithImageForm` changes it |

## Persistence and cross-provider handoff

`llm.MarshalMessages` / `llm.UnmarshalMessages` give you a canonical,
versioned JSON envelope for conversation history — safe to store in a
database and reload across releases. Because every adapter re-encodes the
neutral history into its own wire shape (and drops what only the original
provider can accept, like signed thinking blocks), a saved conversation can
be continued on a *different* provider: mid-conversation failover,
cheap-model tool loops handed to a stronger model for synthesis, or histories
that simply outlive your vendor choice. Cross-provider handoff is part of
the live test matrix, not a hope. Reasoning is replayed only to the
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

## Details

### Tools

Parallel calls, streamed arguments, and a defined contract for malformed
tool calls: rescue what's rescuable, drop the rest *visibly*
(`ToolCallDropped`), with an opt-in `llm.RetryDroppedToolCalls` middleware.
No-argument tool schemas are sent with an explicit empty `properties: {}`
(OpenAI-compatible `parameters`, Anthropic `input_schema`); non-object
`properties` values fail locally.

### Structured output

`llm.Parse[T]` decodes model output straight into your struct (native
JSON-schema mode where supported, forced-tool or JSON-mode fallback
elsewhere), and `schema.For[T]` generates JSON Schema from Go types for tool
inputs, with `schema.ValidateArgs` for checking model-emitted arguments.

### Reasoning

One `Effort` dial across providers; reasoning output is normalized, and raw
provider payloads (signed thinking blocks, encrypted reasoning items)
round-trip so same-provider replay just works.

### Usage and cost

Normalized tokens (including cache reads/writes and reasoning tokens),
`CostUSD` from native provider reporting (OpenRouter) or estimated from an
embedded, refreshable [models.dev](https://models.dev) price table, including
request-wide long-context pricing tiers — with provenance in `CostSource`
(`native` vs `estimated`) — plus `ContextUsage` for context-window
accounting. Live model catalogs preserve independent
input/output/cache-read/cache-write availability, so unknown rates are not
displayed as free while explicit zero rates remain visible. Anthropic
one-hour cache writes are tracked separately (`CacheWrite1hTokens`); no price
table publishes their rate, so an estimated cost containing them is reported
as unknown rather than understated.

### Capabilities

Provider-wide `Capabilities()` discovery with pre-flight request validation,
plus advisory per-model capabilities from rich live catalogs such as
OpenRouter. Unsupported provider features fail fast with `ErrUnsupported`;
incomplete model metadata never blocks a call.

### Conveniences and observability

`Session` (auto-managed history, session tools with `AddToolResults` +
`Continue`, cumulative usage), `History`, `PromptTemplate`, `llm.Ptr` for
optional scalar fields, and middleware via `llm.Wrap`. Observability is built
in but silent by default (`slog` logging, `UsageTracker`, wire capture via
`WithWireCapture`), plus optional, content-free OpenTelemetry middleware in
the `otelllm` package (GenAI semantic conventions; OpenTelemetry is pulled in
only when you import it; see [`examples/otel`](examples/otel)). Sensitive
headers are always redacted; captured URLs and bodies remain
application-sensitive data. Built-in failure logs use safe provider-error
summaries and never emit provider-controlled bodies or metadata.

## CLI

`llm-cli` is a curl-like frontend built entirely on the public API.

```sh
go install github.com/pkieltyka/go-llm/cmd/llm-cli@latest
```

```sh
llm-cli -p openai -m gpt-5.5 "write a short haiku about Go"
echo "long input" | llm-cli -p anthropic -m claude-opus-4-8 -s "summarize stdin"
llm-cli -p openrouter -m openai/gpt-5.5 --usage --json "return a JSON status"
llm-cli -p openai --auth-file ~/.pi/agent/auth.json -m gpt-5.5 "review this diff"  # pi's credentials

llm-cli models -p openrouter          # list models (table or --json)

llm-cli classify -p openrouter -m typesafe/jev-1.14 --questions q.json \
  --state '{"message":"It crashes when I save!"}'      # classifier models

llm-cli -p vllm --base-url http://gpu-box:8000/v1 -m Qwen/Qwen3-8B "hello"
llm-cli -p chatcompletions --base-url http://localhost:11434/v1 -m llama3.2 "hello"

llm-cli login openai                  # Sign in with ChatGPT (subscription)
llm-cli login openai --auth api-key   # or store an OpenAI API key
llm-cli login openrouter              # API key: the only method it has
llm-cli login cloudflare-workers-ai   # API key, then the account ID
llm-cli auth                          # what is configured, per provider
llm-cli logout openai

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

`llm-cli classify` asks a [classifier model](#classifier-models) the
questions in `--questions`, a JSON file in pi's form
(`{"<name>": {"type": "choice"|"score"|"bool", "instructions": "...",
"criteria": ...}}`, where choice criteria map values to descriptions, score
criteria list the levels and bool criteria are `{"true": ..., "false": ...}`)
or a list of `llm.Question`. The state is `--state` (inline JSON, a file, or
`-`) or piped stdin; `--image` adds images and `--json` prints the full
result. Its providers are `openrouter`, `openai`, `typesafe`,
`cloudflare-workers-ai`, `vercel-ai-gateway`, `opencode` and `systemone`
(with `--base-url`); `llm-cli login` stores their keys, and for Cloudflare
also the account ID (in the entry's `env`, as pi stores it).

Self-hosted servers use `-p vllm` (the vLLM preset) or `-p chatcompletions`
(Ollama, llama.cpp, any Chat Completions server) with `--base-url` set to the
server's `/v1` root. Their API key is optional and never read from the
environment; it is the auth file's `api_key` entry named after the provider
(`"vllm"` or `"chatcompletions"`), if any.

Credentials work as in pi. `llm-cli login <provider>` stores one credential
per provider in a pi-compatible `auth.json`, and every command uses it; a
stored credential wins, and without one a provider reads its usual
environment variable. The auth file is `--auth-file`, else `LLM_AUTH_FILE`,
else `llm-cli/auth.json` in your user config directory; point it at
`~/.pi/agent/auth.json` to share credentials with pi.

- `login` uses the provider's account sign-in when it has one (`openai`:
  Sign in with ChatGPT, using your ChatGPT plan) and otherwise its API key.
  `--auth api-key` (or `api_key`) stores an API key instead; `--auth oauth`
  asks for the sign-in explicitly. A key is read at a hidden prompt, or from
  stdin when piped.
- The ChatGPT sign-in prints a URL to open. When the browser cannot reach
  the machine running the CLI (over SSH, say), paste the final redirect URL
  into the terminal. The installation's device ID is created once in your
  user config directory (`--device-id` overrides it).
- `logout <provider>` removes the stored credential; environment variables
  are unchanged. `auth` lists what is configured for each provider.

OAuth refreshes are written back to the auth file under its lock. `$VAR` key
references resolve as in pi; `!command` keys run only with
`--auth-file-commands`. `--api-key` bypasses the auth file; its value is
visible in process arguments and often shell history, so prefer `login`.

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
out of the box (against the scripted `llmtest` provider, or an in-process
server for `self-hosted` and `credentials`), and switches to the real API when its
credentials or server are set — `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`,
or `LLM_BASE_URL`.

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
| [`classify`](examples/classify) | A classifier model (Jev over OpenRouter) bound with `llm.NewClassifierModel`: choice, score and bool answers with probabilities |
| [`prompt-deltas`](examples/prompt-deltas) | Changing instructions and tools mid-conversation with `SystemDeltaPart`/`ToolsDeltaPart`, and what `llm.FoldPositional` sends models without in-place support |
| [`self-hosted`](examples/self-hosted) | vLLM or any Chat Completions server (Ollama, llama.cpp) via `LLM_BASE_URL`; offline it runs against an in-process server |
| [`history-replay`](examples/history-replay) | Serialize a conversation and continue it on another provider (a real Anthropic→OpenRouter handoff when both keys are set) |
| [`observability`](examples/observability) | `UsageTracker`, logging, and cost via `llm.Wrap` middleware |
| [`otel`](examples/otel) | OpenTelemetry spans and GenAI metrics with `otelllm.Middleware` |
| [`context-guard`](examples/context-guard) | Keeping a growing conversation inside the window with `llm.ContextGuard`: near-full hook, compaction, retry |
| [`credentials`](examples/credentials) | A pi-compatible `auth.json` store with a `$VAR` key reference, read by a provider on every request |
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
