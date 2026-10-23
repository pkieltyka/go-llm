---
status: complete
planned_at: e94fcb3
planned_at_date: 2026-10-22
implemented_at_date: 2026-10-22
---

# Context Fitting Plan

This plan gives callers provider-neutral tools to answer: "how big is this
request, and does it fit?" It covers a deterministic estimate, an exact count
where the provider offers one, merged model metadata, and an explicit
max-tokens fitter. It also generalizes the recorded `vllm.ContextGuard`
design.

Sources:

- go-agent `077b363`, which reimplements all of this:
  - `truncation.go:estimateTokens`: `len/4`, images at 1000, files at a
    constant;
  - `budget_model.go:requestTokenMessages`: renders tools and the response
    format as text;
  - `budget_model.go:resolveModelBudgetInfo`: merges `Models()` with
    `LookupModelInfo` field by field;
  - `budget_model.go:upperCostBound`;
  - `token_estimator.go`: per-binding EMA calibration.
- pi v1.4.0:
  - `packages/ai/src/utils/estimate.ts`: 3.5 chars/token, about 4800 chars per
    image, tools priced as JSON, and the last reported usage plus an estimate
    of the trailing messages;
  - `clampMaxTokensToContext`: a 4096-token safety margin;
  - pi-durable's compaction thresholds, which use both.
- go-llm `implementation_plan.md` Future Work, the
  "`vllm.ContextGuard` middleware" entry and its 2026-07-05 decision:
  tokenize is never implicit in Chat/ChatStream.

ARCH §10 forbids silent parameter clamping. Every helper here is explicit and
opt-in. Nothing changes the default request path.

## Evidence baseline and dispositions

| Finding | Disposition |
|---|---|
| Every consumer needs a cheap request-size estimate, and each invents its own constants. | **Implement** `llm.EstimateRequestTokens(*Request) int64` with documented constants (Phase 1). |
| Exact counts exist on vLLM (`Tokenize`, live-verified exact) and Anthropic (`Messages.CountTokens` in anthropic-sdk-go v1.84). | **Implement** an optional `llm.TokenCounter` interface, implemented by those two providers (Phase 2). |
| Usage from the last turn is ground truth for the prefix. Only the trailing messages need estimating. | **Implement** `llm.EstimateContinuationTokens(lastUsage, trailing []Message)` (Phase 1). |
| Live catalog metadata and the embedded table are merged ad hoc. | **Implement** `llm.ResolveModelInfo(ctx, p, model)` with documented precedence (Phase 3). |
| Callers need an explicit "fit MaxTokens into the remaining window". | **Implement** `llm.FitMaxTokens(promptTokens, window, requested, margin) (int, error)`, a pure function (Phase 4). |
| `vllm.ContextGuard` was designed as vLLM-only. | **Generalize** to `llm.ContextGuard` middleware over any `TokenCounter`, falling back to the estimate. This supersedes the vLLM-only Future Work entry (Phase 5). |
| go-agent's EMA calibration learns per binding from usage. | **Stays in go-agent.** It is adaptive policy. go-llm supplies the base estimate it calibrates. |

## Confirmed decisions

1. Estimates are documented heuristics, never presented as exact. The
   `TokenCount` result type carries `Exact bool`.
2. Constants are chosen once and documented in FS §13:
   - 3.5 chars per token for text, matching pi 1.1;
   - images at a fixed per-image cost;
   - tools and `ResponseFormat` counted as their canonical JSON;
   - a small per-message overhead.
   Executors calibrate the constants against vLLM `Tokenize` and Anthropic
   `CountTokens` on a fixed corpus and record the error bands. The goal is a
   documented ±band, not precision.
3. `TokenCounter` is an optional interface discovered by type assertion, like
   other extensions. It is never called implicitly. `llm.Wrap` providers
   forward it only through an explicit unwrap mechanism; the executor decides
   this in Phase 2 and documents it.
4. `FitMaxTokens` returns an error rather than a non-positive budget.
   `ContextGuard` fails with `ErrContextTooLong` before the network call, or
   invokes `OnNearFull`; it never rewrites messages.

## Phases

| Phase | Priority | Effort | Risk | Depends on |
|---|---|---:|---|---|
| 1. Estimators | P1 | S | LOW | none |
| 2. `TokenCounter` (vLLM, Anthropic) | P1 | M | MEDIUM | 1 |
| 3. `ResolveModelInfo` | P2 | S | LOW | none |
| 4. `FitMaxTokens` and a cost upper bound | P2 | XS | LOW | 1 |
| 5. `llm.ContextGuard` middleware | P2 | M | MEDIUM | 2, 4 |
| 6. Docs and verification | P1 | S | LOW | 1–5 |

### Phase 1: estimators

- `EstimateRequestTokens(req *Request) int64` covers system, messages (text,
  tool calls, tool results, and reasoning text where the provider replays it),
  tools, and response format.
- `EstimateContinuationTokens(last Usage, trailing []Message) int64` returns
  `last.ContextUsage(0).UsedTokens` plus the estimate of `trailing`.
- Table tests pin the constants. The calibration corpus result is recorded in
  FS §13.

### Phase 2: TokenCounter

```go
type TokenCount struct {
	Tokens int64
	Exact  bool
}
type TokenCounter interface {
	CountTokens(ctx context.Context, req *Request) (TokenCount, error)
}
```

- **vLLM:** implemented over `Tokenize` with the same conversion and
  validation as Chat. Existing exact parity tests extend to it.
- **Anthropic:** implemented over `client.Messages.CountTokens` with the
  request built by the same conversion as Chat. A live scenario asserts the
  count is within ±1 of a real request's input tokens, or records the measured
  difference.
- Others do not implement it. Callers fall back to `EstimateRequestTokens`.
- STOP if Anthropic's count endpoint differs materially from billed input for
  cached or tool-bearing requests. Record the evidence and ship it
  vLLM-only.

### Phase 3: ResolveModelInfo

`ResolveModelInfo(ctx, p, model) (ModelInfo, error)` uses the live
`Models()` row first. Missing context window, max output, pricing, and
efforts are filled from `LookupModelInfo(p.Name(), model)`. A field is never
invented, per the "not curated" decision of 2026-09-24. A `Models()` error
degrades to the table, and the error is surfaced in the second result only
when both sources are empty.

### Phase 4: FitMaxTokens and cost bound

- `FitMaxTokens(prompt, window int64, requested, margin int) (int, error)`.
- `EstimateCostBound(promptTokens, maxOutput int64, pricing ModelPricing) (float64, bool)`
  takes the max of the input, cache-read, and cache-write interpretations
  times output, honoring pricing tiers and availability. It is lifted from
  go-agent `upperCostBound`.

### Phase 5: llm.ContextGuard middleware

`llm.ContextGuard(GuardConfig{MaxOccupancy float64, ClampMaxTokens bool, OnNearFull func(ctx, req, TokenCount, window) error})`
returns a `Middleware` (with `Bind`) that counts through the bound provider's
`TokenCounter` when present, and the estimate otherwise.

- The window comes from `ResolveModelInfo`.
- It fails or notifies before the call, and clamps `MaxTokens` only when
  `ClampMaxTokens` is set.
- The recorded rationale (no implicit tokenize; the WireCapture, retry, and
  timeout visibility of the extra call) carries over into FS §10B.
- Update `implementation_plan.md` Future Work to point here.

### Phase 6: docs and verification

FS §10B, §13, and §14 (vLLM, Anthropic) are amended, and the README gains a
"Context fitting" section. `make check` and the live e2e suite are green.

## STOP conditions

- An estimator constant can only be justified by per-model curated data.
- `TokenCounter` would have to be invoked implicitly by Chat or ChatStream.
- `ContextGuard` would need to mutate messages. That is compaction, which
  belongs to go-agent.

## Implementation notes (2026-10-22)

- **Estimator calibration replaced the sketched constants.** The original
  3.5 bytes/token, 1600 per image, and per-message overhead underestimated
  badly where it matters, measured against vLLM `/tokenize` on
  `nvidia/Qwen3.8-Flash-Next-NVFP4`:

  | Content | Estimate vs exact |
  |---|---|
  | Tool loop | 0.36× |
  | Indented JSON | 0.73× |
  | Short chat | 0.63× |

  The causes were a fixed per-request template framing, the tool-use
  instruction block that chat templates (and Anthropic) inject when tools
  are offered, and punctuation- and indentation-heavy text.
- **The shipped estimator** uses character classes: ASCII word characters at
  4 per token, punctuation at 0.6 tokens each, each newline and each
  whitespace character after the first in a run at 0.5, and non-ASCII runes
  at 1 each. Overheads: 8 per request, 4 per message, a one-time 300-token
  tools preamble, and 8 per tool plus its JSON.
- **Re-measured:**

  | Content | Estimate vs exact |
  |---|---|
  | Short chat | 0.95× |
  | README 4k | 1.11× |
  | README full | 1.05× |
  | stream.go | 1.01× |
  | Indented JSON | 1.10× |
  | Small JSON | 1.00× |
  | Tool loop | 1.17× |
  | 20-message conversation | 1.02× |
  | Chinese paragraph | 1.49× (Qwen packs about 1.5 CJK characters per token) |
  | Live `tokenize` scenario (one-tool request) | 1.36× (the tools preamble dominates a tiny prompt) |

  The constants were fitted offline, with least squares as a reference only:
  it overfits eight points and goes negative on short inputs. The fit used a
  run-aware whitespace rule and favored leaning high.
- **ClampMaxTokens (revised after review):** only lowers an explicit
  `MaxTokens`; see "Review" below.
- **Wrap transparency:** `wrappedProvider.Unwrap()` was added, and
  `llm.CountTokens` follows `Unwrap() Provider` chains (depth-bounded), so
  counters stay reachable through `llm.Wrap`. `ContextGuard` is middleware
  with `Bind`, so it counts on the provider it wraps.
- **Counter errors:** `CountTokens` returns them unchanged, and
  `ContextGuard` falls back to the estimate on a counter error, never
  failing the call because of it. Context cancellation still aborts.
- **`ContextGuard` decisions beyond the sketch:**
  - Only an exact count can fail a request; estimates only notify or clamp.
  - Clamping never exceeds the model's `MaxOutputTokens`. Superseded after
    review: it also never sets an unset `MaxTokens`.
  - A window that resolves as unknown passes through unguarded. Superseded
    after review: only successful lookups and `ErrNotFound` are cached, and
    transient lookup failures are retried.
  - An unknown window skips counting entirely, so a deployment without
    metadata pays no extra calls.
- **Anthropic `CountTokens`** builds the Chat params, keeps only `model`,
  `messages`, `system`, `tools`, `tool_choice`, `thinking`, `output_config`,
  `mcp_servers`, and `context_management`, and sends them through
  `sdkparam.Override`, with beta headers forwarded. It is offline-tested
  against an httptest server for path, body fields, beta headers, error
  mapping, and Chat-equivalent validation. It is **not live-verified**: the
  configured Anthropic credential is OAuth and is rejected by the
  API-key-only provider.
- **vLLM `CountTokens`** wraps `Tokenize`. The live `tokenize` scenario now
  asserts parity through `llm.Wrap`, and passes.
- **Docs:** FS §13A (new), the FS §10B middleware notes, the
  implementation-plan Future Work entries (`vllm.ContextGuard` marked
  shipped as `llm.ContextGuard`), and a README "Context fitting" section.
- **Verification:** vet (including the live tag), `make test` (race), the
  manifest test, coverage floors, lint (0 issues), and go-agent build, vet,
  and tests all pass. Live vLLM: 17 passing scenarios, with only the
  credential-blocked `cross_provider_handoff` failing.

## Review (2026-10-22)

A clean-context adversarial review found three majors, all fixed before
commit:

- **`ClampMaxTokens` raised an unset `MaxTokens`.** It filled
  `MaxOutputTokens`, for example 64,000 instead of Anthropic's 16,384
  default. anthropic-sdk-go then refuses the non-streaming call ("streaming
  is required"). The clamp now only lowers an explicit value, to the window
  and to a known `MaxOutputTokens`.
- **A transient `Models()` error disabled the guard for the process
  lifetime.** The guard now caches only successes and `ErrNotFound`, and
  retries other errors on the next request. A regression test covers a
  lookup that fails, then recovers.
- **PDFs were estimated at a flat 2000 tokens.** A 100-page PDF read as
  about 2k versus roughly 250k real. Inline PDFs now count 2500 per
  `/Type /Page` object (a `bytes.Index` scan), with 2000 as the minimum. The
  per-image 1600 is documented as Anthropic's cap, and therefore a floor on
  providers that tile large images.

Minors and nits:

- **`ResolveModelInfo`** prefers the lexically latest snapshot among
  inexact matches.
- **`OnNearFull`** must not mutate the request; this is now documented,
  since the guard doesn't re-measure.
- **Middleware ordering** is documented: put the guard innermost.
- **`MaxOccupancy`** above 1 now counts as 1, instead of silently becoming
  0.9.
- **`ChatStream`** measures lazily, at iteration.
- **Tool-call IDs** are now estimated.
- **Doc numbers** were reconciled (bands, the §13A references, the `Window`
  doc, and `EstimateContinuationTokens` excluding the response's own
  message).
- **`FitMaxTokens`** saturates `prompt + margin`.
- **JSON mode** without a schema adds nothing.
- **Dead count-tokens fields** `mcp_servers` and `context_management` were
  removed.
- **New tests** cover: stream clamp and laziness, concurrent first use, a
  cancelled count aborting, `MaxOccupancy > 1`, tab, CR, and emoji
  estimation, multi-page PDFs, tool IDs, and the `count_tokens`
  `tool_choice` and `output_config` fields.

Out of scope and fixed separately: the reviewer found that the Anthropic
adapter omits `input_schema` for a bare `{"type":"object"}` tool schema.
This is pre-existing, and the API requires the field. It gets its own
commit.

## Open follow-ups

- **Live-verify Anthropic `CountTokens`** once an API-key credential is
  configured. The plan's ±1 parity check against a real request's input
  tokens, including cached and tool-bearing requests, is still open.
- **go-agent plan 016, step 8,** can now adopt `EstimateRequestTokens`,
  `ResolveModelInfo`, and `EstimateCostBound`. Budget numbers will shift
  with the new estimator, so call it out there.

## Downstream

go-agent plan 016 can replace `estimateTokens`, `requestTokenMessages`,
`resolveModelBudgetInfo`, and `upperCostBound` with these helpers. Its EMA
calibration stays and wraps `EstimateRequestTokens`.
