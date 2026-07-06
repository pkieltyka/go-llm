---
status: complete
---

# Review 1: First Release Review (v0.1.1)

Conducted 2026-07-04 against HEAD `d0ce423` (v0.1.1 + README rewrite).
Method: four independent clean-context reviewers — API surface & DX, code
structure & simplicity, test suite, features-vs-field — with local
comparisons against **gollem** (fugue-labs) and **zero** (Gitlawb),
synthesized with editorial judgment. All findings cite files. Purpose: catch
API regrets while the pre-1.0 breaking window is open, and set the v0.2
agenda. Companion decision list at the end requires owner sign-off.

## 1. Executive summary

**Verdict: the library is genuinely tight — keep its shape.** All four
reviewers independently rated the core abstraction (Provider interface,
sealed Part/Event taxonomy, sentinel error chain, capability gating,
partial-`Collect`, versioned serialization) above both comparison projects.
Dependency arrows are strictly one-directional, the thin-provider /
shared-engine ratio is right, and the Dialect+Compat split was called "the
best idea in the provider layer."

The findings cluster into four themes:

1. **One early design decision is the biggest tax**: accepting both value
   and pointer `Part`/`Event` types everywhere has metastasized into ~73
   duplicate switch arms across nine files (~400 lines of noise that every
   future part type doubles). Fix pre-v0.2.
2. **A handful of stranded or dead public APIs** shipped in v0.1:
   `Session.AddToolResults` (Session can't actually do tools),
   `History.WithForeignReasoningAsText` (a no-op knob), `llm.TokenSource`
   (orphaned). Dead API is worse than no API.
3. **Cross-engine consistency drift** in error-sentinel mapping undermines
   the taxonomy's core promise (`errors.Is(err, ErrOverloaded)` is
   provider-dependent today).
4. **The test suite is high-trust exactly where the code is oldest** and
   weakest where it's newest: CLI credential wiring is 0% covered, the
   replay harness never inspects outbound requests, and the documented
   Provider contract has no exported conformance suite.

Six bugs found (§8) — none critical, four moderate. go-agent readiness was
assessed at ~95%; the notable strategic gap vs the field is **Gemini**,
absent from both the shipped set and the deferred queue.

## 2. Interfaces & API surface

**Strengths to protect** (consensus across reviewers):

- The **error taxonomy** — sentinels + `ProviderError.Unwrap() → Kind` —
  gives `errors.Is` everywhere with full detail underneath. Neither
  comparison has an equivalent.
- **`iter.Seq2` streaming + `Collect` returning the partial Response on
  error** — aborted turns are persistable. Better than zero's channels and
  gollem's `Next()/Close()`.
- Consistent **zero-value semantics** (`Effort("")` = default,
  `Temperature *float64` vs `MaxTokens int` is principled).
- `llmtest` as recorded-request fake with the httptest framing.
- StopReason coverage (`paused`, `context_overflow`, `refusal` + raw).

**Issues, ranked** (details in the API report; B# = bug list §8, D# =
decision list §10):

1. `Session` pretends to do tools but can't — stranded `AddToolResults`,
   no `WithSessionTools`, `Chat` force-appends a user turn
   (session.go:149,196-205). The first wall an agent-builder hits. →
   Tier 1 fix.
2. `WithForeignReasoningAsText` is a dead knob (history.go:22-31; nothing
   reads it). Ship the behavior or cut both symbols. → Tier 1.
3. `ParseOption[T]` forces `[T]` on every option
   (`llm.WithParseRetries[Recipe](2)`) when only the validator needs it.
   Flagship API; fix while breaking is cheap. → Tier 1.
4. `Usage` normalization semantics (additive: `InputTokens` excludes cache)
   are undocumented on the type — the one thing zero does better. → Tier 1
   doc fix, pairs with B6 (cost provenance).
5. `Request.ProviderOptions` is singular — router/failover layers must
   strip/swap per target. → D3.
6. `Models(ctx)` on the core interface vs optional-by-assertion. → D2.
7. `Middleware.bind` unexported — third-party middleware can't be
   provider-aware like `UsageTracker` is. Additive export.
8. Missing `llm.Ptr[T]` helper for `*float64` fields. Additive, trivial.
9. Root-package scope creep: `PromptTemplate` (zero consumers; README says
   "no prompt frameworks"), `TokenSource` (orphaned), auth types
   (pi-ecosystem-specific). → D4/D5.

## 3. Naming & domain fit

Mostly clean and domain-true. Fix pre-v0.2 (all breaking-but-cheap):

- **Wire-capture vocabulary is three-way inconsistent**: `NewWireTap` /
  `WireCapture` / `WithDebugCapture` / `DebugToLogger`. Standardize on
  "wire": `WithWireCapture`, `WireCaptureToLogger`.
- Option-prefix inconsistency (`WithSessionSystem` target-prefixed vs bare
  `WithDebounce`, `WithForeignReasoningAsText`). Prefix per family or not
  at all.
- `ToolResultPart.Parts` → `Content []Part` (a Part containing `Parts`
  reads badly).
- Four `ToolResult*` constructors are a combinatorial smell; collapse.
- `phase3_test.go` (905 lines) is implementation history as a filename —
  split by subsystem.
- `providerutil` doc comment still claims it serves only openai/openaicodex.

## 4. Feature coverage & the field

Full matrix in the features report; the compact read:

| | go-llm | gollem | zero (provider layer) |
|---|---|---|---|
| Shape | **normalization library** | framework | app client layer |
| Unique strengths | offline model/price catalog, capability gating, versioned serialization + live-tested handoff, malformed-call contract as public API, redacted wire capture, Effort dial, subscription auth as library API | Vertex/Gemini, audio-in, MCP, fallback/router wrappers, usage caps, PKCE login | native Gemini, stall watchdogs, think-tag splitting, battle-tested transport hygiene |

**They have / we lack (and it's not already queued):** Gemini (the word
appears nowhere in our specs — both comparisons carry it), audio input,
images in tool results on OpenAI (see B5), rate-limit header surfacing,
error-string redaction. Everything else missing is consciously queued
(vLLM, ZAI, `chatcompletions.New`, login flows, watchdog, multi-key) or
consciously out of scope (MCP, agent machinery, usage-cap enforcement).

**Goal-readiness:** go-agent ~95% (walls: B5 tool-result images on OpenAI;
no Gemini). Gateway: weaker by known deferral; genuinely unqueued gaps are
cost provenance (B6), rate-limit state, and the embeddings question (a
chat-only gateway is a legitimate v1 answer — D6).

## 5. Code organization & package structure

**Keep the shape.** Root `llm` (32 files / 4.3k lines) → stdlib +
`internal/schemajson` only; providers never import each other; `openrouter`
is a 565-line skin over `chatcompletions` (1.8k), `openai` a 564-line skin
over `responsesapi` (1.3k); no cycles; globals limited to two `sync.Once`
caches + the part registry. gollem's 11.9k-line `core/` is the explicit
anti-pattern we avoided; zero's types-leaf role is already played by our
root. **Rule going forward: root is closed to framework features.**

Ranked simplifications (sizes are reviewer estimates):

1. Kill pointer-part/event dual handling (~-400 lines, 9 files). The one
   big one. → D1 for the mechanism choice.
2. `Collect` vs `session.applyCollectEvent` duplicate switch — already
   diverged (B2). Make Collect loop over one shared apply. (~-40)
3. Unify the two error-kind heuristic tables into a `providerutil`
   classifier with per-family hooks — fixes B3's drift. (~-60)
4. providerutil sweep: `uniqueSyntheticToolCallID` ×3, `schemaAsMap` ×3,
   `toolResultText` ×2, `contextWithTimeout` ×4, single-use-stream guard ×6
   (`providerutil.SingleUse`), anthropic's duplicate log helpers, generic
   `OptionsOf[T]` for the ×3 `requestOptions`. (~-200)
5. Split `observe.go` (526 lines, three unrelated tools) into
   `wiretap.go` / `usage.go`. serialize.go stays as-is (one concern,
   justified hand-rolled writer).
6. Merge `cloneRequestForParse`/`cloneRequestForRetry`. (~-25)
7. Flatten `openai`'s pure-delegation shim files. (~-60)
8. Longer-term: one transport stack in `chatcompletions` (SDK for blocking,
   hand-rolled for streaming today — three retry policies exist across the
   codebase). zero's `providerio` is the model. Defer until it hurts, but
   don't add a fourth policy.

## 6. Developer experience

The walkthrough (construct → chat → stream → tools → Parse → session →
persist) is smooth except: the Session/tools wall (§2.1), the Parse option
noise (§2.3), the missing `Ptr` helper, and pkg.go.dev's thin front door —
`doc.go` is 7 lines while the README carries the narrative; fold the
Highlights into the package doc where godoc readers actually land.
`examples/` post-overhaul were rated genuinely good.

## 7. Tests & confidence

~10,100 test lines / 44 files. **Trust a green suite for:** stream/
non-stream Collect-equivalence (the suite's signature pattern, better than
both comparisons), request-build goldens with negative assertions, error
tables and OAuth flows against real `httptest` servers, core stream
semantics, `schema` (100%). **Don't trust it for:**

1. `cmd/llm-cli/provider.go` — `newProvider` (credential resolution,
   provider selection, OAuth persistence wiring) is **0% covered**.
2. The replay harness discards outbound request bodies — its "replays every
   recorded exchange through the full mapping paths" is response-side only.
   Even a cheap request-side invariant closes half the gap.
3. No exported conformance suite: the single-use-stream test is copy-pasted
   into three providers, **openrouter has none**, and no provider tests ctx
   cancellation or concurrent use. `llmtest.RunConformance(t, factory)`
   turns the documented contract into a checked one.
4. Redaction guard is shape-based (fixed header list + four regexes) — fine,
   but say so in release.md; novel-shaped tokens in unlisted headers would
   commit.
5. Floors are manual, exactly-at-current-value, and absent for root/cmd/
   llmtest/provideroauth. Adopt gollem's codecov config (project auto ±2% +
   **80% patch gate on new code**) to make this self-executing.

Readability: consolidate ×4/×5 duplicated helpers into an internal testutil
package; split `phase3_test.go`; replace ordered-`strings.Contains` wire
assertions in `chatcompletions/convert_replay_test.go` with the semantic
golden used elsewhere; `FuzzCollectTextDeltas` has near-zero power — fuzz
event sequences or drop it. One honesty flag: `providers/openai`'s replay
test replays the *codex* corpus (disclosed in-file; keep the disclosure
loud).

## 8. Bugs found

| # | Severity | What | Where |
|---|---|---|---|
| B1 | Moderate | `RetryDroppedToolCalls` swallows retry-attempt errors incl. `ctx.Err()` — returns prior response with nil error; cancellation doesn't propagate | retry.go:19-25 |
| B2 | Moderate | `Collect` installs `MessageEnd.Raw`; Session's duplicated switch drops it — live divergence between the two paths | stream.go:159-161 vs session.go:228-256 |
| B3 | Moderate | Sentinel drift across engines: Anthropic 503→`ErrServer` vs chatcompletions 503→`ErrOverloaded`; 402→`ErrInsufficientCredits` only in chatcompletions; 529→`ErrOverloaded` only in responsesapi; responsesapi matches bare "context" in codes (the exact false-positive chatcompletions' own comment warns against) | providers/*/errors.go, providers/internal/*/errors.go |
| B4 | Mild | Empty-but-200 SSE stream → zero events → `Collect` returns empty Response with nil error; should surface `ErrServer` | chatcompletions/stream.go:488-493 |
| B5 | Moderate | Tool-result modality asymmetry: images in `ToolResultPart` work on Anthropic, hard-`ErrUnsupported` on OpenAI Responses (the API accepts them — adapter gap) and chatcompletions (wire genuinely text-only there — correct); `image-input` capability can't express position-dependence. Blocks screenshot-returning agent tools off-Anthropic | responsesapi/adapter.go:378-392 |
| B6 | Mild (spec drift) | FS §11 promises "estimates are marked as estimates" — `Usage` has no native-vs-estimated cost discriminator; `UsageTracker` aggregates mix billing-grade and estimated dollars | response.go:105, pricing.go |

Dead-API defects (§2.1/2.2 + `TokenSource`) are tracked in the plan rather
than this table.

## 9. Improvement plan

**Tier 1 — pre-v0.2, inside the breaking window** (correctness + regrets):
value-only parts/events (D1); Session tools support or `AddToolResults`
removal; Parse option de-genericization; fix B1–B4 with the unified error
classifier; delete dead knobs (`WithForeignReasoningAsText`, `TokenSource`);
document Usage invariant on the type + add cost provenance (B6); wire-capture
naming unification; `llm.Ptr`; export `Middleware.Bind`.

**Tier 2 — quality, mostly non-breaking**: `llmtest.RunConformance` applied
to all providers; replay-harness request-side assertions; CLI `newProvider`
tests; testutil consolidation + `phase3_test.go` split + `observe.go` split;
codecov patch gate; providerutil dedup sweep; `cloneRequest` merge; openai
shim flattening; B5 fix (OpenAI tool-result images); doc.go expansion.

**Tier 3 — features (post-cleanup), in priority order**: Gemini (cheap
first step: chatcompletions preset via `chatcompletions.New` when it lands;
native provider later — skip Vertex/Bedrock credential plumbing for now);
vLLM per the existing plan; rate-limit header surfacing (nil-when-absent,
no behavior); think-tag splitter as explicit `Compat` option;
`SystemParts` multi-block caching **only if go-agent proves the need**.

**Considered and rejected as over-engineering** (recorded so we don't
relitigate): `StreamParse[T]` partial-JSON streaming (app concern);
OpenAI-wire ingest helpers (the gateway project's job, not the client's);
bundled tokenizers (deps + drift vs zero-dep core); token-bucket rate-limit
middleware (composable by callers); `llm.RunTools` mini-loop (parked — it's
the top of go-agent's territory; revisit only if go-agent finds the loop
trivial enough that a helper here serves everyone).

## 10. Decisions requested (owner)

- **D1 — pointer parts**: value-only Parts/Events (breaking, cleanest) vs
  deref-once-at-boundary helper (non-breaking, keeps both forms working)?
  Recommendation: value-only, now.
- **D2 — `Models()`**: keep on the core interface (uniformity; my lean) or
  split to optional interface discovered by assertion (smaller contract)?
- **D3 — `ProviderOptions`**: keep singular + document the routing story
  (my lean for v0.2) or change to a slice adapters filter?
- **D4 — `PromptTemplate`**: delete (my lean — zero consumers, contradicts
  README), move to a subpackage, or keep?
- **D5 — auth types**: keep in root (my lean — auth is core domain for this
  library; just unexport `TokenSource`) or move to `llm/auth` subpackage?
  **Resolved (July 2026): keep in root** — the public surface is two
  functions + two types; the machinery is already in
  `providers/internal/provideroauth`. **Revisit trigger**: if interactive
  login flows (deferred `llm-cli auth login` PKCE/device-code item) are
  scheduled before v1.0, introduce `llm/auth` then and move the credential
  types into it while breaking is still free. Decision deadline: v1.0.
- **D6 — gateway/embeddings**: accept a chat-only gateway story for now
  (my lean) or queue an embeddings surface?

## 11. Scorecard vs the field

go-llm wins on: normalization completeness, error taxonomy, persistence &
handoff, testing story, subscription auth as library API, docs honesty.
gollem wins on: provider breadth (Gemini/Vertex), audio, resilience
wrappers. zero wins on: transport battle-scars (watchdogs, single shared
retry layer — adopt when transport work next opens), Usage documentation
discipline (adopt in Tier 1). Both comparisons validate the go-llm/go-agent
boundary: gollem shows the cost of not drawing it; zero shows the provider
layer wants to be a library — which is what go-llm is.
