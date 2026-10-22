---
status: ready
planned_at: e94fcb3
planned_at_date: 2026-10-22
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

## Downstream

go-agent plan 016 can replace `estimateTokens`, `requestTokenMessages`,
`resolveModelBudgetInfo`, and `upperCostBound` with these helpers. Its EMA
calibration stays and wraps `EstimateRequestTokens`.
