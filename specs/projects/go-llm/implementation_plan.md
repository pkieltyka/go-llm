---
status: complete
---

# Implementation Plan: go-llm

Ordered by dependency; each phase is one reviewable unit ending green
(`go vet`, `golangci-lint`, `go test -race ./...`). Section references:
FS = functional_spec.md, ARCH = architecture.md.

> **Historical, non-normative record.** Phase bullets reflect phase-time API names; several
> were renamed or removed in the v0.2 cycle (`WithDebugCapture`→
> `WithWireCapture`, `DebugToLogger`→`WireCaptureToLogger`, `TokenSource`
> deleted, `ToolResultPart.Parts`→`Content`, generic `ParseOption[T]`→
> `ParseOption` — see reviews/1-first-release-review.md and historical commit
> identifier `09c85a2`, retained as provenance from before the history
> rewrite and not expected to resolve in the current repository). FS/ARCH are
> authoritative for current names and behavior.

**Human checkpoints — credentials.** Provider phases (4–7) and the release
phase (9) end with live e2e runs. Before
each, the implementer must **pause and ask the user** to put that provider's
credentials into `gollm-test.json` (copied from the committed
`gollm-test.json.sample`; file is gitignored — ARCH §9). Env vars work as
fallback. If the user declines or a key is unavailable, the e2e step skips
visibly and the phase still completes — never fail a phase on missing keys.

## Phases

- [x] **Phase 1: Repo scaffolding + core vocabulary (package `llm`)**
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

- [x] **Phase 2: Serialization + `schema` subpackage**
  - Canonical JSON per ARCH §2.7 / FS §10A: part discriminators,
    `RegisterPartType`, `UnknownPart`, versioned envelope helpers,
    `Response`/`Usage` marshaling (Raw excluded).
  - `schema/` per ARCH §7A / FS §7–8: `For[T]`, `MustFor[T]`,
    `ValidateArgs` (strict-mode subset; errors on unsupported types).
  - Tests: round-trip byte-identity property (incl. `ReasoningPart.Raw`
    and `Message.Provider/Model` provenance), unknown-part preservation,
    schema generator goldens, `ValidateArgs` tables.

- [x] **Phase 3: Core utilities — pricing, middleware, observability, `llmtest`, `Parse[T]`**
  - `pricing.go` + `models_table.go` + **snapshot pipeline** (ARCH §5,
    FS §11): `scripts/snapshot-models-table.ts` (tsx + package.json,
    dev-only) — fetches models.dev/api.json + OpenRouter models, trims to
    our providers/fields, applies `scripts/overrides.json`, writes
    `models.json` with `generated_at`; root package
    `go:embed`s it, lazy-parses via `sync.Once`; prefix + canonical-ID
    fallback lookup; cost estimation helper; add `scripts/node_modules/`
    to `.gitignore`. Tests: snapshot parse, lookup fallbacks, lazy-init
    race (via `-race`).
  - `Middleware` + `Wrap` decorator (ARCH §2.8, FS §10B).
  - `observe.go` (ARCH §2.8A, FS §17B): `UsageTracker` + middleware,
    `WireCapture` + `NewWireTap` redacting transport, `DebugToLogger`.
    Tests: slog handler assertions, concurrent tracker aggregation,
    redaction guarantees, SSE tee.
  - `prompt.go` (ARCH §2.10, FS §10C): `PromptTemplate` — strict Format,
    immutable Partial. Tests: missing-var error, partial merge precedence,
    struct + map vars.
  - `session.go` + context accounting (ARCH §7 Session, FS §10D/§13):
    `Session` wrapper (auto `SessionID`, cumulative usage, stream
    auto-append), `LookupModelInfo` (embedded snapshot),
    `Usage.ContextUsage`. Tests: session turn flow incl. tool results,
    context-usage math (cache tokens counted), unknown-model ok=false.
  - `llmtest` package (ARCH §7B, FS §17A) — FIFO scripted steps, request
    recording, real `iter.Seq2` streams, goroutine-safe.
  - `Parse[T]` (ARCH §2.9, FS §8): mode resolution (native json-schema →
    forced-tool extraction → json-mode + guidance), `WithParseMode`,
    `WithParseRetries`, `WithParseValidator`.
  - Tests: middleware ordering/pass-through, llmtest self-tests
    (= Provider conformance suite), Parse paths against llmtest, pricing
    lookup tables.

- [x] **Phase 4: Anthropic provider**
  - `providers/anthropic` per ARCH §3.1/§3.3: request build (system +
    `SystemCache`, cache hints, effort mapping FS §9, MaxTokens default,
    tools/strict, structured output), response + stream mapping
    (FS §5, §6), reasoning replay (same-provider raw re-emit, foreign
    drop — FS §18), error mapping, `Models()`, `Client()` escape hatch,
    `anthropic.Options` extensions (FS §14), extension-part registration
    pattern established.
  - Build `httpclient.go` — `llm.DefaultHTTPClient()` (ARCH §3.4, FS §17:
    tuned shared transport; darwin keep-alive workaround; `Timeout==0`
    guard test) and wire it as every constructor's default.
  - Add the malformed-tool-call core types (`ToolCallDropped` event,
    `Response.DroppedToolCalls` — FS §7, additive to phase 1 types) +
    `llm.RetryDroppedToolCalls(n)` middleware (ARCH §2.8). Tests: rescue
    vs drop fixtures in the Anthropic adapter; middleware retry loop +
    UsageTracker-ordering test against `llmtest`.
  - Wire `WithLogger` + `WithDebugCapture` (ARCH §2.8A) and
    `WithAPIKeyFunc` (ARCH §3.4) — pattern established here, repeated in
    every provider phase.
  - Tests: request-build goldens, response/stream fixtures (thinking +
    tool use + refusal + parallel tools), Collect-equivalence, error
    tables.
  - Build the **`internal/e2e` live harness** (ARCH §9): capability-driven
    scenarios, `-record` fixture capture, pinned cheap-model constants,
    public `llm.LoadAuthFile` (pi-compatible credential format, ARCH
    §3.4) with the `gollm-test.json` loader built on it + committed
    `gollm-test.json.sample`.
  - ⏸ **Checkpoint: ask the user for their Anthropic API key**, then run
    the e2e suite vs Anthropic.

- [x] **Phase 5: OpenAI provider (Responses API, direct wrap)**
  - `providers/openai` per ARCH §3.2: input-item request build
    (`instructions`, `max_output_tokens`, flattened tools with fail-open
    strict-schema sanitization — FS §8, `text.format`,
    `reasoning: {effort, summary}`), stateless defaults
    (`store: false` + encrypted reasoning `include`), output-item →
    parts mapping (reasoning items → `ReasoningPart` w/ encrypted
    round-trip), semantic-event stream mapping, status → stop reasons
    (FS §5 note), `openai.Options`, `Models()`, `Client()`.
  - Tests: request-build goldens, response/stream fixtures (reasoning
    items + summaries, function calls, incomplete statuses),
    Collect-equivalence, reasoning-replay round-trip, error mapping.
  - ⏸ **Checkpoint: ask the user for their OpenAI API key**, then run the
    e2e suite vs OpenAI.

- [x] **Phase 6: Subscription auth — Anthropic OAuth mode + `openai-codex` provider**
  - Core (`auth.go`): OAuth consumption per ARCH §3.4 — `TokenSource`
    (per-provider refresh, single-flight, goroutine-safe),
    `WithOAuth(cred, persist)` option pattern: bearer from
    `AuthCredential{type: "oauth"}`, refresh before expiry + one
    forced-refresh retry on 401, renewals → context-aware, error-returning
    durable persistence callback (required when a refresh token is present;
    go-llm never writes credential files).
  - `providers/anthropic` OAuth mode (ARCH §3.1, FS §17C): SDK auth-token
    option + `anthropic-beta: oauth-2025-04-20` header + Anthropic OAuth
    token-endpoint refresh (reference: pi `oauth/anthropic.ts`).
  - `providers/openaicodex` (ARCH §3.2A, FS §17C): extract the phase-5
    Responses mapping into a shared internal package; codex base URL,
    bearer + `chatgpt-account-id` (re-resolved per request) +
    `originator` headers; OpenAI OAuth token-endpoint refresh
    (references: pi `oauth/openai-codex.ts`, zero's codex client);
    curated static `Models()`.
  - Reconcile the phase-4 e2e loader with the pi-compatible credential
    format (`LoadAuthFile`, ARCH §3.4/§9) if it predates it; e2e
    constructs providers by credential `type` (api_key → `WithAPIKey`,
    oauth → `WithOAuth`).
  - NOT in scope: interactive login flows (PKCE/device-code minting) —
    deferred (`llm-cli auth login` candidate); credentials come from
    pi/claude/codex CLIs.
  - Tests: token-refresh state machine vs `httptest` (expiry, 401
    retry-once, durable persistence, single-flight under -race), codex
    request-build goldens (headers, account id), Responses fixtures
    reused from phase 5.
  - ⏸ **Checkpoint: ask the user to populate `oauth` entries** in
    `gollm-test.json` (paste from `~/.pi/agent/auth.json`) for
    `anthropic` and/or `openai-codex`, then run the e2e suite over both
    subscription paths.

- [x] **Phase 7: chatcompletions adapter + OpenRouter provider**
  - `providers/internal/chatcompletions` per ARCH §3.3: `Dialect` interface,
    quirks expressed as the declarative `Compat` struct where practical
    (positions the deferred public `chatcompletions.New`), shared
    message/tool/format conversion with fail-open schema adaptation,
    streaming loop, tool-call index state machine (ARCH §8.1),
    extra-fields plumbing (ARCH §8.5), pipeline (ARCH §4).
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

- [x] **Phase 8: `cmd/llm-cli`**
  - CLI per FS §19 / ARCH §7C: chat (positional + stdin prompts,
    streaming by default, `--no-stream`/`--json` canonical output),
    `models` subcommand, attachments (`--image`/`--file`), `--schema`
    structured output, `--tool` (print tool calls, never execute),
    **`--load`/`--save` conversation files** (canonical envelope;
    cross-provider replay from the shell), `--reasoning`,
    `--cache-system`, `--session-id`, `--usage` + `--debug` (dogfoods
    `UsageTracker` fields + `DebugToLogger`), Ctrl-C cancellation via
    `signal.NotifyContext`.
  - Constraints enforced: stdlib `flag` only; public go-llm API only
    (needing an internal import = spec bug to surface, not work around).
  - Tests: flag→`Request` construction tables against `llmtest`; manual
    smoke against any provider whose credentials are already in
    `gollm-test.json` (no new checkpoint — reuse credentials provided in
    phases 4–7).

- [x] **Phase 9: Docs, examples, release readiness**
  - README (sharply differentiated first line — naming collision note;
    pre-1.0 API-stability policy stated; `llm-cli` install + usage
    section), godoc pass on all exported symbols, `example_test.go`
    runnable examples (chat, stream, tools, Parse, history, middleware,
    observability, provider switch/replay), `examples/` programs.
  - ⏸ **Checkpoint: confirm configured provider credentials are present in
    `gollm-test.json`**, then run the full e2e matrix — including the
    cross-provider handoff scenario — plus a `-record` pass to refresh the
    offline fixture corpus; price-table snapshot refresh + date stamp;
    fixture-driven offline mapping tests wired over the recorded corpus
    (replay harness in `internal/e2e` + per-adapter replay suites), with
    per-package coverage floors enforced at the baseline recorded in
    `docs/release.md`; LICENSE; tag `v0.1.0`.

- [x] **Phase 10 (v0.3): public `chatcompletions.New` + vLLM provider**
  (shipped 2026-07-05; live-verified against vLLM 0.24.0 at the checkpoint
  host and re-verified on 0.23.1rc1 — see phase_plans/phase_10.md for wire
  findings; the two queued vLLM increments — `/tokenize` extension and
  native `structured_outputs` modes — shipped in phase 11)
  - **Promote** `providers/internal/chatcompletions` → public
    `providers/chatcompletions` (one layer, one name — decision recorded
    2026-07-05): add `New(baseURL string, opts ...Option)` with
    **key-optional** construction (self-hosted servers are commonly
    keyless); export `Compat` as the declarative quirk surface; keep the
    `Dialect` hooks as restrained as practical (fold into `Compat` funcs
    where possible — a public interface is forever-ish). `openrouter`
    updates its import to the public path; behavior identical.
  - `providers/vllm` preset per FS §2 row + `vllm_research.md` +
    ARCH §3.3: era-aware defaults (host verified: **vLLM 0.24.0** →
    `structured_outputs` param, `reasoning` output field; portable
    `response_format: json_schema` baseline), `reasoning` delta parsing,
    choice-less mid-stream error sniff, `vllm_xargs` passthrough
    extension, live `Models()` (max_model_len surfaced), Qwen-style
    thinking toggles (`chat_template_kwargs.enable_thinking`) as typed
    options. Responses-API opt-in surface stays deferred (flip criteria
    in Future Work). Ollama preset ships as data-only convenience
    (documented community-verified).
  - e2e: `vllm` live suite (scenario-per-capability; host has reasoning
    parser + auto tool-choice enabled — verified by probe 2026-07-05);
    loader gains the **base_url-presence = configured** rule for keyless
    entries; Effort-none defaulting wrapper for thinking-by-default Qwen.
  - Docs: README provider row + self-hosted section; the
    `anthropic.New(WithBaseURL(vllmHost))` `/v1/messages` recipe —
    live-verify it against the host as a bonus scenario if cheap.
  - Specs: FS §2 vLLM row → in scope; FS §3 deferred bullet removed;
    ARCH §3.3/§1 updated to the public package.
  - ⏸ **Checkpoint: satisfied** — user-provided vLLM host
    `http://pax.local:8000` serving `Qwen/Qwen3.6-27B-FP8` (2026-07-05).

- [x] **Phase 11 (post-v0.3.1): vLLM increments — `/tokenize` extensions +
  native `structured_outputs` modes** (shipped 2026-07-05, live-verified
  against the checkpoint host on 0.23.1rc1 — see
  phase_plans/phase_11.md for wire findings)
  - Typed tokenizer extension methods on `vllm.Provider` (escape hatch,
    not `llm.Provider`): `Tokenize` (chat-shaped `/tokenize` body reusing
    the engine's `BuildParams` conversion; endpoints live at the SERVER
    ROOT — probed: `/v1/tokenize` is 404 — reached via the engine's new
    `DoJSONURL`), `Detokenize`, `TokenizerInfo` (raw; flag-gated
    server-side). `TokenizeResult.ContextUsage()` bridges Count +
    MaxModelLen to exact `llm.ContextUsage`; live parity diff 0 vs a real
    chat's prompt_tokens (the extension mirrors the server-side
    `reasoning_effort`→`enable_thinking` injection).
  - `vllm.Options.StructuredOutputs` (Regex/Choice/Grammar/StructuralTag +
    WhitespacePattern; JSON schema stays on unified `ResponseFormat`) with
    fail-loud conflict rules (ResponseFormat conflict → ErrBadRequest,
    legacy era → ErrUnsupported, exactly-one-mode → ErrBadRequest); live
    finding: constraint modes corrupt output while thinking is active
    (doubled choice text), so the e2e scenarios pin Effort none.
  - e2e: `tokenize`, `structured_choice`, `structured_regex` scenarios in
    `TestLiveVLLM`; offline goldens + conflict/era table tests; vllm
    coverage floor ratcheted 76→84; FS §14 + README updated.

## Future Work (explicitly not in these phases)

Batch APIs, fallback/router wrapper, token counting, stream tee, MCP —
per FS §3 out-of-scope and review decisions.

ZAI provider (`providers/zai` dialect per FS §14/§6/§16 + ARCH §3.3):
thinking/effort/do_sample/request_id/user_id extras, `tool_stream`
auto-set, `reasoning_content` (message + delta), business-code error
table, base URL selector (General/CodingPlan), curated static `Models()`
list, extension parts (`video_url`, `file_url`) with serialization
registration (ARCH §2.7). Future tests should cover reasoning_content +
tool_stream fixtures, `tool-choice` capability gating (FS §7),
error-code tables, extension-part round-trip, and ZAI live e2e when a key
is available.

Response-cache middleware (exact-match; key = hash of provider name +
canonically-serialized `Request` per FS §10A; replay streams via collected
response): valuable for dev loops and resumable pipelines, near-zero hits
in conversational traffic — ship later as an example or v1.x subpackage on
top of `llm.Wrap`, never as core default behavior.

**vLLM increments beyond the v0.3 preset** (phase 10 shipped the public
`chatcompletions.New` engine + `providers/vllm`; phase 11 shipped the
`/tokenize`+`/detokenize`+`/tokenizer_info` typed extensions and the native
`structured_outputs` modes; research: `vllm_research.md`). Still deferred:

- **Responses API opt-in second surface** (e.g. `vllm.WithResponsesAPI()`)
  riding the existing `internal/responsesapi` mapping. Chat completions
  stays the default: it works on every deployed vLLM (~v0.8+), carries the
  deepest feature support, and the OpenAI-platform argument doesn't
  transfer (vLLM returns plain-text reasoning on CC, and open-model chat
  templates drop prior thinking on replay by design). **Flip-the-default
  criteria**: vLLM closes multimodal (#32685), truncation (#38132), and
  statefulness (#24603) gaps / "Open Responses" spec alignment stabilizes.
- **`vllm.ContextGuard` middleware** — see the dedicated Future Work
  entry below. Decision recorded 2026-07-05: the opt-in `llm.Wrap`
  middleware over the `Tokenize` extension is the PREFERRED integration
  for exact context accounting between calls; tokenize is never wired
  implicitly into `Chat`/`ChatStream` (would double every request, buys
  no correctness — the server validates anyway — and hides a second wire
  call from WireCapture/retry/timeout semantics).
- Remaining typed extras not yet modeled: `priority` scheduling,
  `request_id`, `cache_salt`, `thinking_token_budget`,
  `stream_options.continuous_usage_stats`, `bad_words`, `min_tokens`,
  `ignore_eos`, `prompt_logprobs`; first-class LoRA surfacing beyond
  Raw (`parent` field is preserved in `ModelInfo.Raw` today).

Multi-API-key round-robin with per-key backoff (oh-my-pi pattern): a
middleware candidate for v1.x — per-provider key pools sit naturally on
the `llm.Wrap` seam; typed `ErrRateLimited.RetryAfter` gives it clean
signals.

**`vllm.ContextGuard` middleware** (designed 2026-07-05, deliberately NOT
implemented yet): an opt-in `llm.Wrap` middleware over the concrete vllm
provider's `Tokenize` giving ground-truth pre-flight context management —
`GuardConfig{MaxOccupancy, ClampMaxTokens, OnNearFull func(...)}` (fail or
notify before the server 400s; deterministically clamp MaxTokens to fit
MaxModelLen; caller-supplied compaction hook). **Decision recorded**:
tokenize must never be wired implicitly into Chat/ChatStream — it would
double every request, buy no correctness (the server validates anyway;
TOCTOU-racy), leak a hidden call into WireCapture/retry semantics, and
make vllm behave unlike every other Provider. The Wrap seam keeps it
explicit, per-request-visible, and free for non-users. The `OnNearFull`
hook is the go-llm/go-agent compaction boundary: trigger primitive here,
summarization policy in go-agent. ~1 day incl. live clamp + near-full
scenarios; natural to build alongside go-agent's context loop.

From the zero (Gitlawb/zero) research pass, July 2026 — three v1.x
candidates (two further zero learnings were promoted straight into the
specs: the malformed-tool-call contract → FS §7 / ARCH §8, and the usage
normalization invariant → FS §11):

- **Content-stall stream watchdog**: SDK timeouts catch dead sockets but
  not heartbeat-without-output stalls (keep-alives flowing, no content —
  zero runs a second watchdog at 1.2× its idle timeout for exactly this).
  Natural fit as a stream middleware or `WithStallTimeout` option.
- **Auto prompt-cache option** (`WithAutoCache` on Anthropic-family
  providers): zero unconditionally sets `cache_control` on the system
  block + last tool — a good opinionated default; ours would be opt-in
  atop the existing `CacheHint` machinery.
- **Interactive OAuth login flows** (PKCE/device-code credential
  *minting*): the subscription auth paths themselves were **promoted into
  scope** (phase 6, FS §17C) — consuming + refreshing credentials minted
  by pi/claude/codex CLIs. Minting new ones (`llm-cli auth login`-style
  loopback PKCE; references: zero's provideroauth, pi's oauth/) stays
  deferred.

**Connection prewarm helper** (deferred, gated on TTFT data
from plan 2 phase 4): an explicit helper issuing one unauthenticated,
single-attempt, bounded (~3s) HEAD to the provider base URL to prime the
TCP+TLS handshake into the shared pool before a first request — a
401/404/405 still warms the connection. Constraints if built: never
called implicitly by `Chat`/`ChatStream`, never authenticated, never
retried, invisible to retry/WireCapture semantics, and skipped entirely
when the transport disables keep-alives (a non-reusable warmed connection
is pure waste). Revisit only if measured time-to-first-token shows
cold-start handshake cost worth optimizing; otherwise drop.
