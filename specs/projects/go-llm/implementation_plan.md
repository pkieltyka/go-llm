---
status: complete
---

# Implementation Plan: go-llm

Ordered by dependency; each phase is one reviewable unit ending green
(`go vet`, `golangci-lint`, `go test -race ./...`). Section references:
FS = functional_spec.md, ARCH = architecture.md.

**Human checkpoints — API keys.** Phases 4–8 end with live e2e runs. Before
each, the implementer must **pause and ask the user** to put that provider's
API key into `gollm-test.json` (copied from the committed
`gollm-test.json.sample`; file is gitignored — ARCH §9). Env vars work as
fallback. If the user declines or a key is unavailable, the e2e step skips
visibly and the phase still completes — never fail a phase on missing keys.

## Phases

- [ ] **Phase 1: Repo scaffolding + core vocabulary (package `llm`)**
  - `git init`, `go.mod` (`github.com/pkieltyka/go-llm`, Go 1.26), CI
    workflow (vet + lint + race tests + `govulncheck` + short fuzz),
    Dependabot config for SDK bumps (the wrapped SDKs move weekly),
    `.gitignore` (incl. `.specs_skill_state/` and `gollm-test.json`).
  - Core types per ARCH §2.1–2.6: message/parts + constructors,
    `Request`/`Tool`/`ToolChoice`/`ResponseFormat`/`Effort`/`CacheHint`,
    `Response`/`Usage`/`StopReason`, `Event` types + `Collect` +
    `StreamText`/`WithDebounce` (ARCH §2.5), error sentinels +
    `ProviderError` (ARCH §2.6),
    `Capability` constants + `CustomCapabilities` + `validateRequest`
    (ARCH §6, FS §12), `History` (ARCH §7), `Provider` interface +
    `ModelInfo` (ARCH §2.4).
  - Unit tests: Collect event-accumulation, validation → `ErrUnsupported`,
    error wrapping (`errors.Is`/`As`), History (incl. tool-result grouping).

- [ ] **Phase 2: Serialization + `schema` subpackage**
  - Canonical JSON per ARCH §2.7 / FS §10A: part discriminators,
    `RegisterPartType`, `UnknownPart`, versioned envelope helpers,
    `Response`/`Usage` marshaling (Raw excluded).
  - `schema/` per ARCH §7A / FS §7–8: `For[T]`, `MustFor[T]`,
    `ValidateArgs` (strict-mode subset; errors on unsupported types).
  - Tests: round-trip byte-identity property (incl. `ReasoningPart.Raw`),
    unknown-part preservation, schema generator goldens, `ValidateArgs`
    tables.

- [ ] **Phase 3: Core utilities — pricing, middleware, observability, `llmtest`, `Parse[T]`**
  - `pricing.go` + `pricing_table.go` snapshot (ARCH §5, FS §11) with
    `PriceTableDate`; prefix-fallback lookup; cost estimation helper.
  - `Middleware` + `Wrap` decorator (ARCH §2.8, FS §10B).
  - `observe.go` (ARCH §2.8A, FS §17B): `UsageTracker` + middleware,
    `WireCapture` + `NewWireTap` redacting transport, `DebugToLogger`.
    Tests: slog handler assertions, concurrent tracker aggregation,
    redaction guarantees, SSE tee.
  - `prompt.go` (ARCH §2.10, FS §10C): `PromptTemplate` — strict Format,
    immutable Partial. Tests: missing-var error, partial merge precedence,
    struct + map vars.
  - `llmtest` package (ARCH §7B, FS §17A) — FIFO scripted steps, request
    recording, real `iter.Seq2` streams, goroutine-safe.
  - `Parse[T]` (ARCH §2.9, FS §8): mode resolution (native json-schema →
    forced-tool extraction → json-mode + guidance), `WithParseMode`,
    `WithParseRetries`, `WithParseValidator`.
  - Tests: middleware ordering/pass-through, llmtest self-tests
    (= Provider conformance suite), Parse paths against llmtest, pricing
    lookup tables.

- [ ] **Phase 4: Anthropic provider**
  - `providers/anthropic` per ARCH §3.1/§3.3: request build (system +
    `SystemCache`, cache hints, effort mapping FS §9, MaxTokens default,
    tools/strict, structured output), response + stream mapping
    (FS §5, §6), reasoning replay (same-provider raw re-emit, foreign
    drop — FS §18), error mapping, `Models()`, `Client()` escape hatch,
    `anthropic.Options` extensions (FS §14), extension-part registration
    pattern established.
  - Wire `WithLogger` + `WithDebugCapture` (ARCH §2.8A) — pattern
    established here, repeated in every provider phase.
  - Tests: request-build goldens, response/stream fixtures (thinking +
    tool use + refusal + parallel tools), Collect-equivalence, error
    tables.
  - Build the **`internal/e2e` live harness** (ARCH §9): capability-driven
    scenarios, `-record` fixture capture, pinned cheap-model constants,
    `gollm-test.json` loader + committed `gollm-test.json.sample`.
  - ⏸ **Checkpoint: ask the user for their Anthropic API key**, then run
    the e2e suite vs Anthropic.

- [ ] **Phase 5: OpenAI provider (Responses API, direct wrap)**
  - `providers/openai` per ARCH §3.2: input-item request build
    (`instructions`, `max_output_tokens`, flattened tools,
    `text.format`, `reasoning: {effort, summary}`), stateless defaults
    (`store: false` + encrypted reasoning `include`), output-item →
    parts mapping (reasoning items → `ReasoningPart` w/ encrypted
    round-trip), semantic-event stream mapping, status → stop reasons
    (FS §5 note), `openai.Options`, `Models()`, `Client()`.
  - Tests: request-build goldens, response/stream fixtures (reasoning
    items + summaries, function calls, incomplete statuses),
    Collect-equivalence, reasoning-replay round-trip, error mapping.
  - ⏸ **Checkpoint: ask the user for their OpenAI API key**, then run the
    e2e suite vs OpenAI.

- [ ] **Phase 6: openaicompat adapter + OpenRouter provider**
  - `providers/internal/openaicompat` per ARCH §3.3: `Dialect` interface,
    shared message/tool/format conversion, streaming loop, tool-call index
    state machine (ARCH §8.1), extra-fields plumbing (ARCH §8.5),
    pipeline (ARCH §4).
  - `providers/openrouter` dialect per FS §14/§6/§16 + ARCH §3.3:
    attribution headers, `session_id`, routing/plugins/reasoning extras,
    `usage.cost` → `CostUSD`, typed `ResponseExtras` + accessor,
    mid-stream error chunks, warm-up empty-choices → `ErrServer` (FS §18),
    rich `Models()`.
  - **Resolve the flagged verification item**: fixture test that
    `openai-go` tolerates SSE comment keep-alives; add the RoundTripper
    strip mitigation only if it fails (ARCH §3.3).
  - Tests: fixtures incl. keep-alive lines + mid-stream error + fallback
    `model` echo; moderation-metadata error mapping.
  - ⏸ **Checkpoint: ask the user for their OpenRouter API key**, then run
    the e2e suite vs OpenRouter (incl. cost-reporting scenario).

- [ ] **Phase 7: ZAI provider**
  - `providers/zai` dialect per FS §14/§6/§16 + ARCH §3.3: thinking/
    effort/do_sample/request_id/user_id extras, `tool_stream` auto-set,
    `reasoning_content` (message + delta), business-code error table,
    base URL selector (General/CodingPlan), curated static `Models()`
    list, extension parts (`video_url`, `file_url`) with serialization
    registration (ARCH §2.7).
  - Tests: fixtures incl. reasoning_content + tool_stream, `tool-choice`
    capability gating (FS §7), error-code tables, extension-part
    round-trip.
  - ⏸ **Checkpoint: ask the user for their ZAI API key**, then run the
    e2e suite vs ZAI.

- [ ] **Phase 8: Docs, examples, release readiness**
  - README (sharply differentiated first line — naming collision note;
    pre-1.0 API-stability policy stated), godoc pass on all exported
    symbols, `example_test.go` runnable examples (chat, stream, tools,
    Parse, history, middleware, observability, provider switch/replay),
    `examples/` programs.
  - ⏸ **Checkpoint: confirm all four provider keys are present in
    `gollm-test.json`**, then run the full e2e matrix — including the
    cross-provider handoff scenario — plus a `-record` pass to refresh the
    offline fixture corpus; price-table snapshot refresh + date stamp;
    coverage check (≥85% adapters/mapping); LICENSE; tag `v0.1.0`.

## Deferred (explicitly not in these phases)

Batch APIs, fallback/router wrapper, token counting, stream tee, MCP —
per FS §3 out-of-scope and review decisions.

Response-cache middleware (exact-match; key = hash of provider name +
canonically-serialized `Request` per FS §10A; replay streams via collected
response): valuable for dev loops and resumable pipelines, near-zero hits
in conversational traffic — ship later as an example or v1.x subpackage on
top of `llm.Wrap`, never as core default behavior.

Generic public OpenAI-compatible provider (`openaicompatible.New(baseURL,
...)` over the openaicompat adapter, for Ollama/vLLM/Groq/Together/etc.):
v1.x candidate — the adapter already exists internally; exposing it is an
API-surface + testing-matrix decision, not an engineering one. Until then,
"other" providers implement the public `Provider` interface directly.
