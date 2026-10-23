---
status: complete
planned_at: e94fcb3
planned_at_date: 2026-10-22
implemented_at_date: 2026-10-22
---

# Replay, Overflow, and Cache-Accounting Correctness Plan

This plan fixes three correctness gaps found while comparing go-llm with pi
v1.4.0 (`/home/peter/Dev/other/pi`, HEAD `476a303`) and go-agent `077b363`:

1. Reasoning replay is gated on provider only, never on model.
2. Context-overflow classification is narrower than real provider behavior.
   HTTP 413 maps to `ErrBadRequest`.
3. Anthropic 1-hour cache writes are counted and priced as 5-minute writes.

An executor must read the whole plan, run the live probes before changing
replay behavior, preserve unrelated worktree changes, and never add curated
model or pricing data. Model data comes from upstream only; see the decision
recorded with PRs #33 and #37.

## Evidence baseline and dispositions

| Finding | Evidence | Disposition |
|---|---|---|
| Raw reasoning is replayed whenever `ReasoningPart.Provider` equals the adapter's provider, whatever model produced it. | `providers/anthropic/convert.go:reasoningReplayBlock`; `providers/internal/responsesapi/adapter.go:reasoningInput`; `providers/chatcompletions/convert.go` (~L158) | **Fix** after live probes (Phase 1). |
| OpenRouter is one provider fronting many model families. `reasoning_details` from `anthropic/*` is replayed to `openai/*` or `google/*` after a model switch. | Same CC path; `Message.Model` already records the producer | The highest-risk case for Phase 1. |
| pi 1.3 drops thinking from a different provider, API, or model on replay. It no longer converts it to text. Same-model signed or redacted thinking is kept even when its text is empty. | pi `packages/ai/src/api/transform-messages.ts`; `CHANGELOG.md` 1.3.0 | **Adopt** the drop rule. go-llm already drops foreign-provider reasoning silently (FS §18). |
| HTTP 413 maps to `ErrBadRequest`. Anthropic `request_too_large` and gateway 413s are context overflow in practice. | `providers/internal/providerutil/errclass.go:canonicalStatusKind`; FS §16 table | **Amend** FS §16: 413 becomes `ErrContextTooLong` (Phase 2). |
| Overflow heuristics are split: Anthropic has `isContextOverflowMessage`; the OpenAI family has `heuristicErrorKind`. Common shapes are missed: "prompt is too long" on non-Anthropic gateways, "exceeds the available context size" (llama.cpp), "greater than the context length" (LM Studio), "reduce the length of the messages" (Groq), "is longer than the model's context length" (Together), "range of input length should be" (Qwen). | pi `packages/ai/src/utils/overflow.ts` (~27 patterns, with rate-limit exclusions and provider attributions) | **Consolidate** into one scoped matcher with the corpus as fixtures (Phase 2). |
| Overflow and content filtering each reach consumers through two channels (error vs `StopReason`). go-agent handles both overflow forms but treats `ErrContentFiltered` as a generic error rather than a refusal. | go-agent `loop.go` (~L528, ~L587, ~L821); go-llm `errclass.go:127` vs `responsesapi/response.go:138` | **Document** one rule per condition in FS §5/§16 (Phase 2). The go-agent fix is in its plan 016. |
| `CacheHint.TTL > 5m` sends Anthropic `ttl: 1h`, but usage records only `cache_creation_input_tokens`. The 1h split (`cache_creation.ephemeral_1h_input_tokens`) is discarded. | `providers/anthropic/convert.go` (~L461); `response.go` (~L87); `stream.go` (~L438) | **Fix** usage truthfulness (Phase 3). |
| models.dev publishes a single `cache_write` rate, so 1h writes are priced at the 5m rate (Anthropic charges 2× input for 1h versus 1.25× for 5m). | `scripts/model-sources/2026-08-26/models-dev.json.gz` cost keys: `cache_read`, `cache_write`, `context_over_200k`, `input`, `output`, `reasoning`, `tiers`, plus audio | **Ratified: option A** (Phase 3b). |
| pi detects "silent overflow" (a success with `input + cacheRead > contextWindow`) and length-stop overflow (`length` with zero output and a full window). | pi `overflow.ts:isContextOverflow` | **Defer.** This is agent-side policy over `ContextUsage`. A helper can come later if go-agent asks for one. |

## Confirmed decisions

1. Replay keeps a raw reasoning payload only when the provider matches **and**
   the producing model matches the request model. When the producing model is
   unknown (`Message.Model == ""`, i.e. legacy history), today's behavior is
   kept, so old transcripts don't regress.
2. Model identity comparison is exact after one documented normalization:
   trim whitespace, then remove a trailing date snapshot using the existing
   `model_matching.go` date-recognition helpers. It is not family matching. A
   dated response ID (`…-2026-08-15`) must match its undated request alias.
3. Cross-model reasoning is **dropped, never converted to text**. This
   matches pi 1.3 and FS §18's silent drop.
4. Overflow matchers stay scoped: no bare "context" or "tokens" substrings,
   and rate-limit or "too many requests" text never classifies as overflow.
5. No model or pricing curation. Missing upstream rates stay unknown.

## Phases

| Phase | Priority | Effort | Risk | Depends on |
|---|---|---:|---|---|
| 1. Model-aware reasoning replay | P0 | M | MEDIUM | live probes |
| 2. Overflow classification and one-channel rule | P1 | S–M | LOW | none |
| 3. 1h cache-write usage and cost | P1 | S | LOW | none (3b ratified: option A) |
| 4. Docs and verification | P1 | S | LOW | 1–3 |

### Phase 1: model-aware reasoning replay

**1a. Live probes (required before any code change).** Write temporary
probes in the session scratchpad. Credentials load in-process through
`llm.LoadAuthFile` (never print or read `gollm-test.json`), and the probes are
deleted afterwards. For each case, record status, error class, and whether the
continuation succeeds:

| Case | Producer → consumer |
|---|---|
| OpenAI Responses | encrypted reasoning from model A → model B (e.g. `gpt-5.5` → `gpt-5.6`) |
| Codex | same shape as OpenAI Responses, if credentials allow |
| Anthropic | signed thinking from Opus → Sonnet or Haiku |
| OpenRouter | `anthropic/*` `reasoning_details` → `openai/*`, and → another `anthropic/*` model |
| vLLM | plain-text `reasoning` replay across two served models, if more than one is served |

Record the results in this plan under "Probe evidence".

- A consumer that **rejects** foreign payloads confirms the fix.
- A consumer that **silently accepts** them is still covered by the drop rule:
  accepting does not mean the reasoning is meaningful, and pi drops for the
  same reason.
- STOP if a consumer **requires** a foreign payload to continue, for example
  a provider that rejects a tool-result turn without its preceding reasoning
  item. Report it with evidence.

**1b. Implementation.**
- Adapters receive the containing `Message` (or its `Model`) when converting a
  `ReasoningPart`. Add one shared helper in `providerutil`, for example
  `ReplayableReasoning(part, msg, reqModel, provider) bool`, so all three
  engines apply an identical rule.
- Tests cover a table per engine: same model, dated vs undated, different
  model, empty `Message.Model`, foreign provider, and OpenRouter
  cross-family.
- Existing same-provider replay goldens must stay byte-identical.

**1c. Live scenario.** Add a `reasoning_model_switch` scenario to the live
e2e registry for every provider that claims `CapabilityReasoning` and exposes
two reasoning models. It runs turn 1 on A and turn 2 on B and asserts that the
continuation succeeds with no foreign raw payload on the wire (checked through
WireCapture).

### Phase 2: overflow classification and one-channel rule

1. FS §16 table: `413 → ErrContextTooLong`. Update `canonicalStatusKind` and
   its tests.
2. Replace `anthropic.isContextOverflowMessage` and the overflow branch of
   `heuristicErrorKind` with one exported-internal matcher,
   `providerutil.IsContextOverflow(code, message string) bool`.
3. The matcher corpus is a table of real provider messages with attribution
   comments, taken from pi's `overflow.ts` and limited to backends go-llm can
   reach: OpenAI, Anthropic, OpenRouter backends, vLLM, Ollama, llama.cpp,
   LM Studio, Groq, Together, Mistral, Qwen/DashScope, and generic
   `context_length_exceeded`.
4. A negative table covers rate-limit and throttling texts ("Too many tokens,
   please wait"), quota errors, and messages that only mention "context".
5. In-stream errors use the same matcher (OpenRouter `finish_reason: error`,
   vLLM choice-less errors).
6. **One-channel rule** (FS §5.3/§16):
   - When a provider **rejects the request** before generating, the error
     sentinel (`ErrContextTooLong`, `ErrContentFiltered`) is the only signal.
   - When generation **starts and the provider stops it**, the `StopReason`
     (`context_overflow`, `content_filter`, `refusal`) is the only signal, on
     a successful `MessageEnd`.
   - Audit every adapter for cases that emit both, and pin the behavior with
     tests.

### Phase 3: 1h cache-write usage and cost

**3a. Usage (no decision needed).**
- Add `Usage.CacheWrite1hTokens int64`. It is a subset of `CacheWriteTokens`,
  following the `ReasoningTokens` precedent, and is documented in FS §11.
- Populate it from Anthropic `cache_creation.ephemeral_1h_input_tokens`, both
  streaming and blocking.
- Update the serialization (`usageJSON`) additively and keep it
  backward-compatible.
- Update the aggregation in `UsageTracker`.

**3b. Cost: ratified 2026-10-22, option A.** models.dev has no 1h rate. Options:

| Option | Behavior |
|---|---|
| A (recommended) | When `CacheWrite1hTokens > 0` and the pricing table has no 1h rate, price the 1h portion as unknown: `CostUSD` covers the known components only and reports it through the existing availability/partial-cost mechanism, if one exists. If none exists, add a `CostComplete`-style signal and amend FS §11 explicitly. Open a models.dev issue or PR requesting `cache_write_1h`. |
| B | Price 1h writes at 2× input as a documented provider rule. This is curated data and conflicts with the 2026-09-24 decision, so it needs explicit sign-off. |

The maintainer ratified **option A** on 2026-10-22: the 1h portion is
priced as unknown, never at the 5m rate and never by a curated multiplier,
and a models.dev request for a `cache_write_1h` rate is filed. Option B is
rejected.

### Phase 4: docs and verification

- FS §5.3, §11, §16, and §18 are amended. The README "Persistence and
  cross-provider handoff" section states the model-switch rule.
- `make check` is green, and the live e2e suite passes for every configured
  provider (including the new switch scenario).

## STOP conditions

- A live probe shows a consumer requires a foreign-model reasoning payload.
- Overflow detection needs an unscoped substring that matches non-overflow
  errors in the negative corpus.
- Cost handling for 1h writes would need curated pricing data (option B is rejected).

## Probe evidence (2026-10-22)

**Provider coverage.** Only OpenRouter and vLLM had working credentials. The
`anthropic` entry in `gollm-test.json` is an OAuth credential, which the
API-key-only provider rejects; no `openai` key is configured; the Codex
refresh token is stale. The direct OpenAI, Anthropic, and Codex cases are
therefore not probed. They are covered offline by the engine tests, and the
live scenario is reusable when credentials return.

**OpenRouter.** Each case: turn 1 makes a tool call at `Effort: low` and
returns a `ReasoningPart` with raw `reasoning_details`; turn 2 continues on
another model with the reasoning kept, then with it dropped. All 16
continuations succeeded:

| Producer → consumer | Keep raw | Drop raw |
|---|---|---|
| claude-haiku-4.5 → claude-haiku-4.5 (baseline) | OK | OK |
| claude-haiku-4.5 → claude-haiku-5.5 | OK | OK |
| claude-haiku-4.5 → openai/gpt-5-mini | OK | OK |
| openai/gpt-5-mini → claude-haiku-4.5 | OK | OK |
| openai/gpt-5-mini → openai/gpt-5-nano | OK | OK |
| google/gemini-3-flash-preview → claude-haiku-4.5 | OK | OK |
| claude-haiku-4.5 → google/gemini-3-flash-preview | OK | OK |
| qwen/qwen3.6-27b → claude-haiku-4.5 | OK | OK |

No consumer rejected foreign reasoning, and none required it. That includes
Anthropic with thinking enabled continuing a tool loop whose assistant turn
lacked a thinking block; OpenRouter reconciles this. The STOP condition did
not trigger, and the drop rule is safe.

**vLLM.** The deployment serves one model (`nvidia/Qwen3.8-Flash-Next-NVFP4`),
so a cross-model case was not applicable.

**Overflow wording (vLLM, live).**
- Prompt over the window: HTTP 400, "This model's maximum context length is
  262144 tokens. However, you requested 16 output tokens and your prompt
  contains at least 262129 input tokens …". This was already
  `ErrContextTooLong`.
- `max_tokens` above the window: "max_tokens=300000 cannot be greater than
  max_model_len=max_total_tokens=262144 …". This intentionally stays
  `ErrBadRequest`.

## Implementation notes (2026-10-22)

- **Phase 1: model-aware replay.**
  - `llm.SameModel(a, b)` is exported from `model_matching.go`, reusing
    `plausibleModelDate`. go-agent's switch detection needs it too.
  - One shared pre-pass, `providerutil.ReplayableHistory(messages, model)`,
    runs ahead of conversion in all three engines: Anthropic
    `buildParamsAfterValidation`, responsesapi `buildParamsAfterValidation`,
    and chatcompletions `buildMessages`. This replaced threading the request
    model into each converter.
  - The pre-pass drops whole `ReasoningPart`s, both `Raw` and text, from
    foreign-model assistant messages, and drops an assistant message left
    empty.
  - `responsesapi.Adapter.BuildInput` has no request model, so it stays
    unfiltered. It has no production callers, only tests.
- **Phase 1c: live scenario.** `reasoning_model_switch` is registered as an
  OpenRouter extension scenario, not under `CapabilityReasoning`. The
  registry is exhaustive, and runners for Anthropic, OpenAI, and Codex
  could not be live-verified without credentials.
  `liveReasoningModelSwitchScenario` is generic.
  `e2e.RecordingCaptures(ctx)` exposes the capture log so the scenario can
  assert on the consumer's request body. Live result: PASS
  (haiku-4.5 → gpt-5-mini, producer raw absent from the consumer request).
- **Phase 2: overflow.**
  - `providerutil.IsContextOverflow(code, message)` replaces Anthropic's
    `isContextOverflowMessage` and the overflow branch of
    `heuristicErrorKind`.
  - 413 maps to `ErrContextTooLong` in the canonical table.
  - Anthropic classifies unnamed overflow types (`request_too_large`,
    including in-stream) before its status fallback.
  - The corpus test has 21 attributed positives and 7 negatives.
  - The one-channel audit found no adapter emitting both an error and a stop
    reason. The FS states the rule as "exactly one channel per event".
- **Phase 3: one-hour cache writes.**
  - `Usage.CacheWrite1hTokens` comes from Anthropic
    `cache_creation.ephemeral_1h_input_tokens` on blocking responses and
    `message_start`.
  - It is serialized as `cache_write_1h_tokens` (omitempty) and summed by
    `UsageTracker`.
  - `pricingCoversUsage` treats any one-hour writes as unpriced, so an
    estimated `CostUSD` stays nil, in line with the existing all-or-nothing
    rule; no new signal was needed. Native costs are untouched.
- **Docs:** FS §11, §16, and §18, and the README handoff paragraph.
- **Verification:**
  - Offline: `go vet` (including the live tag), `make test` (race),
    coverage floors, lint (0 issues), the Collector fuzz test, and the
    go-agent build and vet all pass.
  - Live: OpenRouter and vLLM ran 34 passing scenarios. The only failures
    were `cross_provider_handoff` in both suites, caused by the pre-existing
    Anthropic OAuth credential problem.

## Review (2026-10-22)

A clean-context adversarial review gave "ship after fixes". All findings
were addressed before commit:

- **BLOCKER: `SameModel` treated an alias and its own response as
  different models.**
  - `Message.Model` is the server-reported ID, and Anthropic answers its
    aliases with different dated IDs: `claude-opus-4-0` becomes
    `claude-opus-4-20250514`, and `claude-3-7-sonnet-latest` becomes a
    dated ID.
  - Date-only normalization therefore stripped the signed thinking from a
    same-model tool loop, which direct Anthropic rejects when thinking is
    enabled.
  - Fix: `modelIdentityKey` now also removes one `:variant` suffix, one
    `-latest`, and one trailing `-0`, and leans toward matching, because a
    false match reproduces the previous behavior. The three alias pairs,
    routing and Bedrock variants, and non-match guards (`gpt-5` vs
    `gpt-5-mini`, `claude-opus-4-1` vs `claude-opus-4-20250514`) are now
    regression cases.
  - This intentionally reverses the earlier "`:variant` differs" decision.
- **MAJOR: the live replay scenarios could not catch the blocker.** The
  Anthropic, OpenAI, OpenRouter, and vLLM `reasoning_replay` scenarios
  replayed under `first.Model`. They now replay under the requested
  alias while the assistant message carries the response model, which is
  the shape real callers produce.
- **MAJOR: direct-Anthropic mid-tool-loop model switch is unverified.**
  FS §18 now states this and advises disabling thinking for the switch
  turn until it is live-verified. It stays an open follow-up below.
- **Minors and nits:**
  - The model-switch marker prefers opaque values (signature, data,
    encrypted content) and is also matched JSON-escaped.
  - The README notes that cost is unknown when one-hour writes are present.
  - FS §5 points to the §16 one-channel rule.
  - FS §16 notes that 413 can be request size (attachments), not only
    context.
  - `assertReplayUsage` checks `CacheWrite1hTokens <= CacheWriteTokens`.
  - llm-cli prints `cache_write_1h=` when the value is nonzero.
  - `TestReplayableHistory` covers a foreign first message.

Re-verified after the fixes: vet (including the live tag), `make test`
(race), the manifest test, coverage floors, lint, and the go-agent build
all pass. The OpenRouter and vLLM live suites ran 34 passing scenarios;
the only failures were the credential-blocked `cross_provider_handoff`
scenarios.

## Open follow-ups

- **Request a one-hour rate from models.dev.** File an issue or PR asking for
  a `cache_write_1h` rate. This is outward-facing, so the maintainer files
  it.
- **Probe the direct providers once credentials are fixed.** Run Phase 1a
  against OpenAI Responses, Anthropic, and Codex, and register
  `reasoning_model_switch` runners for them.
  - This includes the direct-Anthropic switch mid tool loop with thinking
    enabled. If Anthropic rejects a final assistant turn without thinking,
    disable thinking for the switch turn, either in go-agent or as an
    adapter rule, and record which.

## Downstream

go-agent plan 016 must keep `Message.Provider`/`Model` on persisted assistant
envelopes. Without them, Phase 1's rule falls back to legacy behavior for
go-agent histories.
