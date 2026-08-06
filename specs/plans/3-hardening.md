---
status: complete
planned_at: 46436aa
implemented_at: 2026-08-06
---

# Hardening, Model Metadata, and Tooling Plan

This plan turns the recent comparative audit into an implementation sequence for
go-llm. It is written against `hardening` commit `46436aa` on 2026-08-05. An
executor must read this file completely before changing code, run each phase's
focused tests, and stop rather than improvising when a STOP condition is met.

The plan is deliberately go-llm-native. Do not mention the source libraries that
inspired the audit in code, documentation, tests, commit messages, or public API
names. Preserve the current dependency upgrades in `go.mod`, `go.sum`, and
`scripts/pnpm-lock.yaml`; this work does not authorize downgrading or replacing
dependencies.

## Outcomes

- [x] Branch baseline: safe provider-error summaries, safe built-in failure
  logging, bounded model-source downloads, shared retry ownership for SDK-backed
  calls, and expanded provider conformance coverage
- [x] Phase 1: finish model snapshot safety and deterministic CI coverage
- [x] Phase 2: refine operational-error safety and make retry guarantees honest
- [x] Phase 3: strict stream termination with an explicit compatibility escape
  hatch
- [x] Phase 4: validated model metadata, upstream efforts, and tiered pricing
- [x] Phase 5: bounded prompt-cache keys and cacheable tool results
- [x] Phase 6: typed vLLM thinking-token budgets
- [x] Phase 7: reproducible CLI builds and tagged CI actions
- [x] Phase 8: documentation synchronization and full verification

## Priority, effort, and dependencies

| Phase | Priority | Effort | Risk | Depends on |
|---|---|---:|---|---|
| 1. Snapshot safety and CI | P1 | S | LOW | none |
| 2. Safe errors and retries | P1 | M | MED-HIGH | none |
| 3. Stream termination | P1 | M | MED | none |
| 4. Catalog validation and pricing tiers | P1 | M | MED | Phase 1 |
| 5. Cache keys and tool-result cache hints | P2 | M | MED | none; coordinate Compat edits with Phase 3 |
| 6. vLLM thinking-token budget | P2 | S-M | LOW-MED | none |
| 7. Build and CI supply-chain cleanup | P2 | S | LOW | Phase 1 |
| 8. Docs and final gates | P1 | S | LOW | all implementation phases |

## Recent-commit reconciliation

The current branch is `hardening` at `46436aa` (`Harden provider reliability and
conformance`), the head of PR #18. It is based on `master` at `94408a1` and now
contains the recovered hardening tree that was previously reachable as
`2e347e3`. Do not cherry-pick or reapply that old object.

The following branch work is real, tested, and must be preserved while it is
refined:

- `ProviderError.SafeSummary`, wrapper-aware `SafeError`, and safe default
  provider failure logging;
- one shared retrying `RoundTripper` for Anthropic, OpenAI, generic Chat
  Completions, and Codex SDK calls, with vendor SDK retries disabled;
- bounded, cancellable, size-limited model-source downloads;
- expanded blocking/streaming semantics, tool-call, deadline, cancellation,
  and partial-response conformance coverage.

The branch is not the final design. Review found concrete follow-up work:

1. Response-status retries are not provably billing-safe merely because the
   response is 429, 503, or 529. Split the guarantee between default,
   provably-pre-send transport replay and explicitly opted-in, at-least-once
   response replay. Remove claims that a status alone proves no upstream work.
2. Remove error-message matching from pre-send classification, apply the
   redirect guard to both error and status paths, and make an excessive
   `Retry-After` return the original response untouched.
3. Route Codex direct streaming through the shared policy instead of retaining
   a second hand-written status loop.
4. Narrow the public safe-error wording to its actual contract, sanitize the
   adapter label included in safe summaries, and stop logging raw
   provider-controlled retry header values.
5. Keep the existing downloader, but close its uncovered cleanup/test branches.
   The snapshot's `supported_efforts` readback bug is still present and must be
   fixed explicitly.

Recent model-matching work is also complete and must not be reimplemented:
`MatchModel` scans the already ordered embedded rows, uses natural numeric
versions, and makes `gpt-5.10` newer than `gpt-5.6` without sorting per call.

Other reviewed ideas are deliberately excluded because go-llm already has the
stronger or safer local contract: OAuth refresh work is singleflight,
context-bound, durably persisted, and time-bounded; Anthropic initial text,
thinking, and tool-block content is preserved; ambiguous early EOF is not a
transport retry signal; and runtime ETag-backed model refresh remains outside
an offline embedded core.

## Repository conventions and boundaries

- Core package API uses standard-library types and has zero third-party
  dependencies. Provider-specific extensions live in each provider's `Options`
  and implement `ProviderOptions`.
- Provider errors retain complete upstream detail for programmatic callers, but
  default operational logging must not copy provider-controlled content.
- Retry documentation must distinguish provably pre-send replay from
  at-least-once response replay; do not call both billing-safe.
- Streaming success means a grammar-valid stream with one `MessageStart` and one
  meaningful `MessageEnd`. Partial output may accompany an error; consumer early
  break is not provider truncation.
- `models.json` is generated, sorted, embedded, and never hand-edited. Runtime
  lookup remains offline and lazily initialized with `sync.Once`.
- Use typed compatibility fields for genuine wire quirks. Never silently make a
  non-standard behavior universal.
- Use `apply_patch` for edits. Do not run dependency upgrades, format unrelated
  files, or change generated fixtures outside the phase being executed.
- Keep runtime remote model catalogs, ETag caches, and startup networking out of
  the core package. Freshness belongs in a reviewable generated snapshot.
- Keep the current natural numeric `MatchModel` resolver. Do not replace it with
  lexical ordering or make it sort model slices per call.
- Do not add generic arbitrary request-parameter maps where a typed provider
  option suffices.

## Baseline and drift check

Before implementation:

```sh
PLAN_BASE=46436aa05b651ccc636b0bc8da7830270ff9337c
git status --short --branch
git diff --stat "$PLAN_BASE" -- \
  errors.go errors_test.go pricing.go pricing_test.go llm.go models_table.go models.json \
  Makefile README.md docs/release.md .github/workflows/ci.yml \
  providers scripts specs/projects/go-llm
make test
pnpm --dir scripts test
go test -count=1 -tags=live -run '^$' ./...
```

Expected baseline: only the understood plan file may be untracked; offline Go
tests, snapshot tests, and credential-free live-tag compilation all exit 0. PR
#18's two CI checks were green at planning time. If an in-scope file has changed
since `46436aa`, compare the live implementation with this plan and stop if the
planned behavior is already present or the API assumptions no longer hold.

## Phase 1: Make model snapshot maintenance fail safely

### Current state

- `scripts/snapshot-models-table.ts:165-175` treats `supported_efforts` as
  protected metadata, but `readSnapshot` at lines 611-638 reconstructs previous
  rows without reading it. A refresh can therefore remove all curated effort
  metadata without tripping the destructive-change guard.
- `readJSON` now has injected fetch support, a 15-second timeout, a 32 MiB
  declared/incremental body limit, and tests for success, timeout, and both
  oversize paths. Preserve that implementation. Non-2xx, missing-body,
  malformed-JSON, stalled-body, local-file, and all-path cleanup behavior still
  need focused coverage.
- `scripts/package.json` defines deterministic fixture tests, but
  `.github/workflows/ci.yml` runs only Go tooling.

### Scope

In scope:

- `scripts/snapshot-models-table.ts`
- `scripts/snapshot-models-table.test.ts`
- `scripts/package.json` only if a deterministic package-manager declaration is
  required for CI
- `.github/workflows/ci.yml`

Out of scope:

- live changes to `models.json` in this phase;
- dependency version changes or lockfile regeneration;
- scheduled refresh PRs or runtime catalog fetching.

### Steps

1. Parse `supported_efforts` in `readSnapshot` using the existing
   `optionalEffortList` validator. Add a **file-backed** regression: persist a
   previous snapshot containing multiple effort lists, build a replacement
   without them, and assert that the normal persistence path rejects it. A test
   that calls `assertNonDestructive` only with in-memory rows is insufficient.
2. Complete, rather than rewrite, the bounded reader:
   - cancel an available response body before returning a non-2xx or declared
     oversize error;
   - ensure the timeout also terminates a stalled body read in injected tests,
     not only a fetch promise that observes the abort signal;
   - release/cancel the reader on every rejection path and always clear the
     timer;
   - keep parsing after the complete bounded body and local-file behavior
     unchanged.
3. Extend the existing injected-fetch tests with non-2xx, missing body,
   malformed JSON, stalled body, cleanup/cancel assertions, and local-file
   behavior. No test may reach the public network.
4. Add the script fixture suite to CI. Use Node `26.7.0`, add
   `"packageManager": "pnpm@11.20.0"` as the local/CI source of truth, and use
   the exact setup Action versions recorded in Phase 7. Run
   `pnpm --dir scripts install --frozen-lockfile`, then run
   `pnpm --dir scripts test`. Do not run network-backed `make models` in ordinary
   CI.

### Verification

```sh
pnpm --dir scripts install --frozen-lockfile
pnpm --dir scripts test
git diff --exit-code 46436aa05b651ccc636b0bc8da7830270ff9337c -- scripts/pnpm-lock.yaml
```

Expected: all snapshot tests pass; the lockfile has not changed; no test reaches
the public network.

## Phase 2: Refine safe errors and retry ownership already on the branch

### Current state

- `errors.go` already preserves detailed `ProviderError` values while exposing
  `SafeSummary`/`SafeError`; `providerutil.LogFailure` already uses the safe
  representation. The implementation excludes upstream code, message,
  metadata, raw body, and wrapper text.
- Anthropic, OpenAI, Chat Completions, and Codex SDK paths already install one
  shared retrying transport and force vendor SDK retries to zero. Caller-owned
  clients are shallow-copied, bodies replay through `GetBody`, and wire
  capture/logging observes each attempt. Retain this architecture.
- The current decision boundary still uses `err.Error()` substrings, retries a
  redirected request when the target returns 429/503/529, clamps an excessive
  `Retry-After`, and calls response-status replay billing-safe without proof.
- Codex direct streaming still owns a separate status loop. It omits retry-count
  ordinals and duplicates the excessive-delay bug.
- `retrylog.go` emits the raw provider-controlled `Retry-After` header even
  though its parsed duration is sufficient for operations.

### Required contract

1. Keep detailed `ProviderError.Error()` behavior. Keep `SafeSummary` and
   `SafeError`, but make their public contract exact:
   - code, message, metadata, raw body, and wrapper text are excluded;
   - an included adapter/provider label must be a bounded ASCII identifier
     (`[A-Za-z0-9._-]`, at most 64 bytes); otherwise omit it;
   - numeric HTTP status and known normalized sentinel names are safe;
   - non-`ProviderError` values remain verbatim by design, so `SafeError` is not
     advertised as a general-purpose secret scrubber.
2. Remove raw retry-header values from built-in Warn logs. Keep status, parsed
   duration, provider, and attempt ordinal. Wire capture retains its separate
   structural redaction behavior.
3. Keep one shared retrying `http.RoundTripper`/client wrapper for blocking and
   pre-stream requests. Route Codex direct streaming through it and delete its
   duplicate loop, sleep hook, delay constants, drain helper, and status helper.
   The one-time OAuth 401 refresh remains a separate authentication flow, not a
   transport retry loop. Vendor SDK retries remain forced to zero.
4. Split retry semantics explicitly:
   - **default/provably pre-send** replay requires a replayable body and typed
     evidence that no HTTP request bytes were sent: transient/timeout DNS,
     dial, or proxy-connect failures through `net.DNSError`, `net.OpError`, and
     platform errno chains;
   - **response replay** for 429/503/529 is at-least-once behavior and is
     disabled by default. Expose a typed provider option such as
     `WithResponseRetries(bool)` for callers who accept possible duplicate work
     or billing. OpenRouter, vLLM, and Ollama forward the Chat Completions option;
   - `WithMaxRetries` continues to mean additional attempts (default 2) and
     bounds whichever enabled classes apply; negative values remain invalid;
   - `X-Should-Retry: false` always vetoes response replay. It never enables a
     status outside the narrow 429/503/529 set, and `true` does not bypass the
     caller's explicit response-retry opt-in.
5. Remove all message-text classification. NXDOMAIN, EOF, unexpected EOF,
   connection reset, broken pipe, read/write timeout, untyped TLS errors, and
   unknown/custom-transport errors receive one attempt. A bare string such as
   `"connection refused"` or `"TLS handshake timeout"` is never proof.
6. Apply the redirect guard (`req.Response != nil`) before both transport-error
   and response-status replay. A 307/308-followed POST is never retried by this
   layer even if the target returns 429/503/529.
7. Preserve context cancellation before attempts and during backoff. Retry
   response bodies are drained through a 1 MiB limit and closed. Preserve body
   bytes and set `X-Stainless-Retry-Count` consistently on all retry attempts.
8. Keep one 30-second maximum provider-requested delay. Honor a positive
   `Retry-After` within the bound. If it exceeds the bound, return the original
   response immediately so normal error mapping preserves `RetryAfter`; do not
   sleep, drain, close, or retry early. Backoff without a header remains bounded
   and context-aware.

### Scope

Expected files include:

- `errors.go`, `errors_test.go`, `observe_test.go`, `doc.go`
- `retrylog.go`, `retrylog_test.go`
- `providers/internal/providerutil/retry*.go`, `providerutil.go`, and tests
- provider construction/options files for Anthropic, OpenAI, Codex, generic
  Chat Completions, OpenRouter, vLLM, and Ollama
- `providers/openaicodex/direct.go` and obsolete direct-loop tests/helpers
- retry/error portions of `functional_spec.md` and `architecture.md`

Do not change application-level `Parse` or dropped-tool retry middleware; those
are model-output correction loops, not transport replay.

### Test matrix

- Safe summaries: nil, each known normalized kind, unknown kind, wrapped
  provider error, local error, invalid/oversize provider labels, and
  credential-shaped marker strings in every excluded field.
- Statuses: default one attempt for every status; with explicit response replay,
  retry 429/503/529 and make one attempt for 408/409/500/502/504 and ordinary
  4xx. `X-Should-Retry: false` vetoes every otherwise eligible status.
- Transport: retry typed temporary DNS/dial/proxy-connect errors; one attempt
  for NXDOMAIN, EOF, unexpected EOF, reset, broken pipe, read timeout, write
  timeout, untyped TLS text, matching forged error strings, and unknown errors.
- Mechanics: body replay, non-replayable body, retry-count header, transport and
  response redirect guards, bounded drain, cancellation before an attempt and
  during backoff, in-bound `Retry-After`, excessive `Retry-After` with the body
  still readable by normal mapping, and exact attempt counts.
- Provider construction: SDK retries are zero while the shared wrapper receives
  the configured policy; direct Codex and SDK Codex have identical attempt
  counts; caller-owned clients/transports are not mutated.
- Composition: define and test the exact bound across the separate one-time
  OAuth refresh. Without a 401, the maximum is `1 + maxRetries` transport
  attempts. If the first auth epoch ends in 401 and refresh succeeds, the second
  epoch receives one fresh transport budget, so the absolute maximum is
  `2 * (1 + maxRetries)`. One deterministic sequence covers a response retry,
  401 refresh, per-attempt capture/logging, final success, and exact ordinals.

### Verification

```sh
go test -count=1 . ./providers/internal/providerutil ./providers/chatcompletions ./providers/anthropic ./providers/openai ./providers/openaicodex ./providers/openrouter ./providers/vllm ./providers/ollama
go test -race -count=1 ./providers/internal/providerutil ./providers/...
```

Expected: all tests pass under race detection; default ambiguous/status
failures record exactly one attempt; explicitly enabled response replay never
exceeds `1 + maxRetries` attempts per auth epoch or
`2 * (1 + maxRetries)` across the sole permitted 401 refresh; excessive delays
preserve the original response; secret-shaped markers and raw retry headers
never appear in built-in log output.

## Phase 3: Require meaningful stream termination

### Current state

- Chat Completions accepts `[DONE]` after any real choice and calls `finish()`
  even if `state.sawFinish` is false (`providers/chatcompletions/stream.go:60-70`).
  `finishEvents` can therefore emit a successful `MessageEnd` with empty
  `StopReason` and `StopReasonRaw` (`stream.go:627-639`).
- A clean EOF without a finish reason currently settles partial blocks without a
  terminal event; the outer stream contract may reject it, but the provider
  adapter does not distinguish strict and compatible endpoints.
- Anthropic emits the zero-value terminal state when `message_stop` arrives
  without a preceding stop-bearing `message_delta`
  (`providers/anthropic/stream.go:64-80`).

### Steps

1. Add a declarative Chat Completions compatibility field named for the
   exception, e.g. `InferMissingFinishReason bool`. Default false means strict.
2. In strict mode, `[DONE]` or exhausted EOF without an observed choice-0
   `finish_reason` must preserve any valid partial events and then yield a
   normalized `ProviderError` wrapping `ErrServer`; it must not emit
   `MessageEnd`.
3. In opt-in compatibility mode, infer `StopReasonToolUse` when one or more
   valid tool calls survived, otherwise `StopReasonEndTurn`. Keep
   `StopReasonRaw` empty because there was no wire value. Permit inference only
   on a clean `[DONE]` or clean EOF, never after malformed JSON, an SSE reader
   error, a mid-stream provider error, or context cancellation.
4. Require Anthropic's terminal stop reason before accepting `message_stop`.
   Missing `message_delta.stop_reason` becomes a normalized partial-stream
   `ErrServer`; Anthropic has no compatibility opt-out.
5. Enforce the invariant in `providerutil.StreamContract`, not only in adapter
   tests: reject `MessageEnd{StopReason: ""}` before yielding that terminal
   event, preserve previously yielded partial output, and return normalized
   `ErrServer`. Provider compatibility inference must happen before this guard.
6. Extend shared conformance so successful exhaustion requires a non-empty
   normalized stop reason. Preserve the consumer-early-break exception: stopping
   iteration early must not synthesize a provider error.
7. Add focused fixtures for strict `[DONE]`, strict EOF, compatible `[DONE]`,
   compatible EOF, inferred tool use, malformed termination, Anthropic missing
   delta, direct shared-contract empty end, usage-only tails, partial collection,
   and consumer early break.

### Scope

- `providers/chatcompletions/config.go`, `stream.go`, and focused tests
- preset dialects only if a currently supported server demonstrably needs the
  opt-out; do not enable it speculatively
- `providers/anthropic/stream.go` and tests
- `providers/internal/providerutil/stream.go` and tests
- `llmtest/conformance.go` and its tests
- stream-contract documentation

### Verification

```sh
go test -count=1 ./llmtest ./providers/chatcompletions ./providers/anthropic ./providers/openrouter ./providers/vllm ./providers/ollama
go test -race -count=1 ./llmtest ./providers/chatcompletions ./providers/anthropic
```

Expected: strict missing-reason fixtures fail with `errors.Is(err,
llm.ErrServer)` and no `MessageEnd`; opt-in fixtures finish exactly once with an
inferred normalized reason and empty raw reason; valid existing streams remain
unchanged.

## Phase 4: Validate the embedded catalog and support tiered pricing

### Current state

- `ModelPricing` contains one flat set of rates (`llm.go:44-50`), and
  `EstimateCost` always applies those rates (`pricing.go:57-70`). Long-context
  request-wide pricing cannot be represented.
- `models_table.go:72-89` silently skips rows missing provider/ID and lets later
  duplicate keys overwrite earlier rows. It does not validate timestamps,
  ordering, limits, prices, or effort values.
- `normalizeModelsDevModel` reads identity, limits, and pricing but ignores
  upstream `reasoning_options`; `supported_efforts` comes only from overrides.
- The generator already sorts rows once. Runtime `MatchModel` and table lookup
  must continue to consume preordered data without per-call sorting.

### Public data model

Add a public tier type without recursively embedding `ModelPricing`:

```go
type ModelPricingTier struct {
    InputTokensAbove  int64   `json:"input_tokens_above"`
    InputPerMTok      float64 `json:"input_per_mtok"`
    OutputPerMTok     float64 `json:"output_per_mtok"`
    CacheReadPerMTok  float64 `json:"cache_read_per_mtok"`
    CacheWritePerMTok float64 `json:"cache_write_per_mtok"`
}

type ModelPricing struct {
    // existing rate fields unchanged
    Tiers []ModelPricingTier `json:"tiers,omitempty"`
}
```

The highest threshold strictly exceeded by
`InputTokens + CacheReadTokens + CacheWriteTokens` applies to the **entire**
request. Exact equality uses the lower/base tier. Tier selection must scan for
the highest matching threshold so caller-provided `PriceTable` values behave
correctly even if their slices are not sorted. Ignore invalid caller-provided
tier entries (non-positive threshold or negative/non-finite rate) and fall back
to the highest valid tier/base; the embedded generator and parser reject the
same invalid data. Native provider-reported cost continues to win unchanged.

### Steps

1. Add the tier types and calculation behavior. Deep-copy tier slices anywhere
   `ModelPricing`/`ModelInfo` is cloned or inherited through canonical fallback.
   Add base, exact-threshold, above-threshold, multiple-unsorted-tier,
   cache-read/write occupancy, invalid caller-tier fallback, caller-supplied
   `PriceTable`, canonical fallback, and native-cost-preservation tests.
2. Extend the snapshot schema and fixtures for context pricing tiers. Read only
   upstream tiers whose discriminator is `type: "context"` and whose threshold
   and rates are valid. Treat an upstream tier as rate overrides: each omitted
   tier rate inherits its corresponding flat/base rate, then embed a complete
   four-rate tier. Every completed rate must come from an explicitly present
   tier value or an explicitly present base value; if both are absent, fail
   generation with the provider/model/rate identity rather than silently making
   it free. Explicit override tiers must already contain all four finite
   non-negative rates and replace upstream tiers as one unit; do not merge tier
   arrays by index.
   Preserve ordinary flat pricing when no tiers exist. Add fixtures for omitted
   tier cache rates, explicit zero rates, inheritance, missing tier-and-base
   rates, and incomplete overrides.
3. Read models.dev effort metadata deliberately:
   - inspect `reasoning_options` only;
   - accept effort options with values from
     `none|minimal|low|medium|high|xhigh|max`;
   - omit `default` and null values;
   - ignore unrelated `toggle` and `budget_tokens` options;
   - fail generation with the provider/model identity when an option explicitly
     has `type: "effort"` but names an unknown non-null effort;
   - normalize to the existing weakest-to-strongest order without duplicates;
   - let an explicit `supported_efforts` override replace source-derived values.
   Add fixture cases for mixed option types, unknown known-shape values, empty
   effort sets, duplicates, ordering, and override precedence.
4. Phase 1 already makes `readSnapshot` preserve/protect effort lists. Extend the
   same readback/destructive-check path only for pricing tiers and add its
   file-backed metadata-loss regression; do not duplicate the effort fix.
5. Make `parseModelTable` fail closed. Use `json.Decoder` with
   `DisallowUnknownFields`, decode exactly one document, then require EOF on a
   second decode. Reject an empty model list; empty or invalid RFC3339
   `generated_at`; empty provider/ID; duplicate
   provider/ID keys; rows not in canonical `provider/id` order; non-positive
   provided limits; negative/non-finite prices; non-positive, duplicate, or
   non-ascending tier thresholds; invalid/duplicate/out-of-order effort values;
   unknown schema fields; concatenated JSON documents; and trailing
   non-whitespace. Add an internal-package test that validates the actual
   embedded bytes, not only spot lookups.
6. Regenerate `models.json` only through the snapshot script. Review semantic
   row changes separately from `generated_at`. Never use
   `--allow-destructive` merely to make the command pass; use it only after a
   human reviews a legitimate upstream removal.

### Scope

- `llm.go`, `pricing.go`, `models_table.go`, their tests
- `scripts/snapshot-models-table.ts`, fixture data, and tests
- `models.json` generated output
- pricing/catalog sections of README, functional spec, architecture, and release
  docs

Out of scope: runtime catalog refresh, ETag caching, provider-network calls from
`LookupModelInfo`, and changes to `MatchModel` ranking.

### Verification

```sh
pnpm --dir scripts test
make models
pnpm --dir scripts test
go test -count=1 . -run 'Pricing|Cost|Model|Effort'
go test -race -count=1 .
git diff --check 46436aa05b651ccc636b0bc8da7830270ff9337c -- models.json scripts llm.go pricing.go models_table.go
```

Expected: tier tests prove exact boundary semantics and full-request rate
selection; embedded catalog validation passes; generated rows stay sorted;
source-derived efforts are present where supported and explicit overrides win.

## Phase 5: Bound prompt-cache keys and preserve tool-result cache hints

### Part A: prompt-cache/session key

`Request.SessionID` is application-owned and may be arbitrarily long.
`providers/internal/responsesapi/adapter.go:85-90` forwards it verbatim as
`prompt_cache_key`, whose OpenAI/Codex wire limit is 64 Unicode characters.

Implement provider-boundary canonicalization in `providers/internal/responsesapi`:

- Leave values of 64 runes or fewer byte-for-byte unchanged.
- For longer values, retain a rune-safe readable prefix, append `-`, then append
  the first 16 lowercase hex characters of SHA-256 over the original bytes. Use
  exactly 47 prefix runes so the final value is 64 runes.
- Keep `Request.SessionID`, `Session.ID`, serialized history, and OpenRouter's
  separate `session_id` untouched. Only the Responses API prompt-cache key is
  canonicalized.
- Use the same helper for OpenAI and Codex through the shared adapter.
- Test empty, 63, 64, 65, long multibyte, deterministic output, distinct long
  IDs sharing a prefix, and exact wire bodies for both providers.

### Part B: tool-result cache hints

`TextPart.Cache` promises an explicit cache breakpoint where supported, but the
Anthropic tool-result conversion at `providers/anthropic/convert.go:327-344`
rebuilds text blocks without the hint. Chat Completions flattens tool content to
a string at `providers/chatcompletions/convert.go:224-244`, so OpenRouter cannot
forward a tool-result breakpoint.

1. Anthropic: construct nested tool-result text through the same cache-aware
   mapping used for ordinary text. Preserve image/file cache behavior and exact
   wire output when no hint is set.
2. Chat Completions: add an explicit compatibility field such as
   `ToolMessageContentBlocks bool`; enable it only for OpenRouter. When a tool
   result contains at least one cache hint, emit its text parts as content blocks
   with per-part `cache_control`, preserving boundaries and order. When no hint
   exists, retain the current string content byte-for-byte. Other compatible
   providers remain string-only, and image/file tool results remain unsupported
   on this surface.
3. Add golden tests for default and one-hour TTLs, multiple text parts, pointer
   parts, no-hint wire stability, OpenRouter enablement, generic-provider
   non-enablement, and unsupported nested parts.
4. Before enabling the OpenRouter preset, require either a maintainer-run redacted
   live probe or a reviewed recorded fixture showing that role-tool content
   blocks with `cache_control` are accepted and cache telemetry remains
   parseable. Offline golden tests alone do not authorize the wire-shape change.
   If credentials are unavailable or the probe is rejected, ship the Anthropic
   preservation fix and compatibility field/tests but leave OpenRouter preset
   enablement off.

Scope for the conditional OpenRouter acceptance work includes its narrow live
scenario/recorded fixture under `internal/e2e` and the capability matrix in
`specs/projects/go-llm/provider_capabilities.md`. Do not record unrelated live
traffic or refresh the full fixture set. Name the focused live test
`TestLiveOpenRouterToolResultCache` so the acceptance command is stable.

### Verification

```sh
go test -count=1 ./providers/internal/responsesapi ./providers/openai ./providers/openaicodex
go test -count=1 ./providers/anthropic ./providers/chatcompletions ./providers/openrouter
go test -race -count=1 ./providers/anthropic ./providers/chatcompletions ./providers/openrouter
# Required before enabling the OpenRouter preset; a credential-missing SKIP is
# not acceptance evidence.
go test -count=1 -tags=live -run '^TestLiveOpenRouterToolResultCache$' ./internal/e2e
```

Expected: no Responses key exceeds 64 runes; two long IDs with a common prefix
remain distinct; cache hints survive tool-result conversion only on supported
wires; no-hint requests remain byte-identical. OpenRouter support is advertised
only when the acceptance evidence exists.

## Phase 6: Add a typed vLLM thinking-token budget

### Current state and decision

`providers/vllm.Options` exposes typed sampling fields and thinking enablement,
while `mapEffort` emits `reasoning_effort`. Some vLLM-compatible reasoning
models share one `max_tokens` ceiling between reasoning and the visible answer;
reasoning can consume the entire allowance and produce neither an answer nor a
tool call.

Add an opt-in typed field to `vllm.Options`:

```go
ThinkingTokenBudget *int
```

Do not add it to the provider-neutral `Request`, generic Chat Completions
options, `ChatTemplateKwargs`, or `XArgs`. Do not send it by default because
support varies by vLLM deployment.

### Behavior

1. Emit a top-level `thinking_token_budget` only when explicitly configured.
2. Require a positive value. Resolve thinking enablement with the existing
   precedence before validating: typed `EnableThinking` wins over
   `ChatTemplateKwargs["enable_thinking"]`; when the typed field is nil, a
   generic value must be a boolean. Reject a budget combined with `EffortNone`,
   effective `enable_thinking=false`, or an invalid generic enablement value as
   `ErrBadRequest` rather than sending contradictory controls. Typed true may
   continue to override a generic false, matching existing option precedence.
3. When `Request.MaxTokens > 0`, reserve 1,024 tokens for the visible answer:
   send `min(configuredBudget, MaxTokens-1024)`. If `MaxTokens <= 1024`, reject
   the combination because no positive safe budget exists. When MaxTokens is
   unset, send the explicit budget unchanged; the request builder does not know
   every live server's model cap.
4. Keep effort mapping independent: the option caps reasoning but does not infer
   a budget from `Effort`, and `xhigh`/`max` mapping remains unchanged.
5. Add golden tests for absent, ordinary, clamped, no-ceiling, invalid,
   `EffortNone`, typed thinking-disabled, generic thinking-disabled, generic
   non-boolean, typed-precedence, and structured-output coexistence cases.
   Confirm tokenized prompt helpers remain unaffected because this is a
   generation limit, not a chat-template input.
6. Document the option in `providers/vllm/doc.go`, README's self-hosted section,
   and the provider-options architecture table.

### Verification

```sh
go test -count=1 ./providers/vllm ./providers/chatcompletions
go test -race -count=1 ./providers/vllm
```

Expected: the field is absent by default; explicit budgets use the exact
top-level spelling; clamping always leaves 1,024 answer tokens; contradictions
fail before any request.

## Phase 7: Produce a CLI artifact and use reviewed action releases

### Steps

1. Change `make build` so it continues compiling all packages and also creates
   `bin/llm-cli` with:

   ```sh
   go build ./...
   mkdir -p bin
   go build -o bin/llm-cli ./cmd/llm-cli
   ```

   `bin/` is already ignored. Do not add cross-compilation, packaging, release
   uploads, or a destructive clean target in this phase.
2. Add a CI build/smoke gate that runs `make build`, asserts the artifact is
   executable, and runs both `--help` and `--version` without credentials or
   network access.
3. Use explicit current release tags for every GitHub Action reference in
   `.github/workflows/*.yml`. These releases were reviewed on 2026-08-06:

   | Action | Release tag |
   |---|---|
   | `actions/checkout` | `v7.0.1` |
   | `actions/setup-go` | `v7.0.0` |
   | `actions/setup-node` | `v7.0.0` |
   | `pnpm/action-setup` | `v6.0.10` |
   | `golangci/golangci-lint-action` | `v9.3.0` |

   Include every action introduced in Phase 1. Dependabot already manages
   GitHub Actions; retain it and review future tagged release upgrades.
4. Do not change Go or npm dependency versions while updating Action references.

### Verification

```sh
make build
test -x bin/llm-cli
./bin/llm-cli --help >/dev/null
./bin/llm-cli --version >/dev/null
test -z "$(rg -n 'uses:[[:space:]].*@[0-9a-f]{40}([[:space:]]|$)' .github/workflows || true)"
git diff --exit-code 46436aa05b651ccc636b0bc8da7830270ff9337c -- go.mod go.sum scripts/pnpm-lock.yaml
```

Expected: `bin/llm-cli` exists and runs; workflow actions use the reviewed
explicit release tags; dependency manifests are unchanged.

## Phase 8: Synchronize documentation and run all gates

Update authoritative documentation in the same logical commits as the behavior
it describes, then do a final consistency pass:

- `README.md`: tier-aware estimated pricing, safe default logging, strict stream
  termination/compatibility, vLLM budget, tool-result cache support, and the
  `make build` artifact path.
- `specs/projects/go-llm/functional_spec.md`: safe versus detailed errors,
  default pre-send versus opt-in at-least-once response retry classes,
  excessive-delay behavior, meaningful terminal reasons, request-wide price
  tiers, prompt-cache key canonicalization, and provider-specific cache/budget
  behavior.
- `specs/projects/go-llm/architecture.md`: shared retry transport ownership,
  SDK retries disabled, catalog validation and tier ingestion, new Compat fields,
  and clone/immutability requirements.
- `specs/projects/go-llm/provider_capabilities.md`: advertise tool-result cache
  hints only for providers with implemented and accepted wire support.
- `docs/release.md`: script tests as an offline gate, generated-catalog invariant
  checks, CLI artifact smoke test, reviewed Action tags, and conditional live
  model refresh review.

Run:

```sh
PLAN_BASE=46436aa05b651ccc636b0bc8da7830270ff9337c
go version
go mod verify
unformatted="$(git ls-files -co --exclude-standard -z -- '*.go' | xargs -0 -r gofmt -l)"
test -z "$unformatted"
git diff --check "$PLAN_BASE"
go vet ./...
go test -count=1 ./...
go test -race -count=1 ./...
go test -shuffle=on -count=10 ./...
pnpm --dir scripts install --frozen-lockfile
pnpm --dir scripts test
go test -count=1 -tags=live -run '^$' ./...
./scripts/check-coverage.sh
go run golang.org/x/vuln/cmd/govulncheck@v1.5.0 ./...
make build
test -x bin/llm-cli
./bin/llm-cli --help >/dev/null
./bin/llm-cli --version >/dev/null
git diff --exit-code "$PLAN_BASE" -- go.mod go.sum scripts/pnpm-lock.yaml
{
  git diff --name-only "$PLAN_BASE"
  git ls-files --others --exclude-standard
} | sort -u | while IFS= read -r path; do
  case "$path" in
    .github/workflows/*.yml|.github/workflows/*.yaml|Makefile|README.md|doc.go|errors.go|errors*_test.go|observe_test.go|retrylog.go|retrylog_test.go|llm.go|pricing.go|pricing*_test.go|models.json|models_table.go|models*_test.go|efforts_test.go|llmtest/*|providers/*|internal/e2e/*|scripts/package.json|scripts/overrides.json|scripts/snapshot-models-table.ts|scripts/snapshot-models-table.test.ts|scripts/fixtures/*|specs/projects/go-llm/functional_spec.md|specs/projects/go-llm/architecture.md|specs/projects/go-llm/provider_capabilities.md|docs/release.md|specs/plans/3-hardening.md) ;;
    *) echo "out-of-scope path: $path" >&2; exit 1 ;;
  esac
done
git status --short
```

Expected: every command exits 0; no dependency manifest changed relative to the
plan baseline; only files authorized by the phases are modified. A credentialed
live test is required only for OpenRouter tool-result block enablement and any
maintainer-requested `models.json` wire/catalog verification.

## Suggested commit sequence

Use the repository's concise imperative/conventional style. Keep commits
reviewable and do not combine generated catalog churn with retry changes.

1. `scripts: finish model snapshot safeguards`
2. `providers: refine retry safety and ownership`
3. `llm: tighten safe error logging contracts`
4. `providers: require terminal stream reasons`
5. `llm: support tiered model pricing`
6. `models: derive efforts and validate embedded catalog`
7. `providers: bound prompt cache keys`
8. `providers: preserve tool result cache hints`
9. `vllm: support thinking token budgets`
10. `build: emit cli artifact and pin ci actions`
11. `docs: synchronize hardening contracts`

Do not push or open a PR unless the operator explicitly asks.

## Done criteria

- [x] The branch contains detailed provider errors plus safe summaries/default
  failure logging, bounded model-source downloads, and expanded provider
  conformance coverage.
- [x] One shared transport owns provider retries; SDK and direct loops do not
  stack, and the separate OAuth 401 refresh path remains bounded.
- [x] Default retries are limited to typed, provably pre-send failures.
  Response-status replay is explicit at-least-once opt-in, never described as a
  billing guarantee, and excessive `Retry-After` never causes an early retry.
- [x] Redirected requests, ambiguous errors, and forged matching error strings
  are never replayed.
- [x] Successful streams always carry a meaningful terminal reason unless an
  explicit provider compatibility flag infers one.
- [x] The snapshot guard reads every field it claims to protect, existing remote
  bounds clean up on all paths, and fixture tests run in deterministic CI.
- [x] Embedded model data fails closed on malformed rows, duplicates, invalid
  metadata, and ordering drift.
- [x] Cost estimates select request-wide pricing tiers from total input
  occupancy while preserving native provider cost.
- [x] Supported reasoning efforts are derived from known upstream effort
  metadata with overrides authoritative.
- [x] OpenAI/Codex prompt-cache keys are at most 64 runes and collision-resistant
  for long application session IDs.
- [x] Anthropic preserves cache hints on tool-result text. OpenRouter does so
  only if the focused live/recorded acceptance gate passes; otherwise its preset
  remains disabled for content blocks and the capability docs say so.
- [x] vLLM can opt into a bounded reasoning budget that reserves visible-answer
  capacity.
- [x] `make build` produces an executable `bin/llm-cli`.
- [x] CI runs Go and snapshot tests and every external Action uses the reviewed
  explicit release tag.
- [x] Go/module dependency upgrades present at `94408a1` remain intact, and
  `go.mod`, `go.sum`, and `scripts/pnpm-lock.yaml` are unchanged from `46436aa`.
- [x] Documentation matches the implemented behavior and all Phase 8 commands
  pass.

## STOP conditions

Stop and report instead of improvising if any of these occur:

- The in-scope code has drifted enough from `46436aa` that the current-state descriptions or
  public API assumptions are false.
- A vendor SDK cannot have its retries disabled while retaining required auth,
  middleware, or request-body replay behavior.
- A proposed pre-send retry class cannot be proven through typed Go errors. Treat
  it as ambiguous and do not retry it; if a supported provider requires more,
  report the evidence for a design decision.
- A supported live endpoint requires missing-finish inference but no narrow
  preset-specific compatibility setting can express it.
- OpenRouter acceptance evidence for cache-aware role-tool content blocks is
  unavailable or the endpoint rejects them. Keep the Anthropic fix and offline
  compatibility work, leave OpenRouter enablement off, and report the result.
- The live models.dev `reasoning_options` or pricing-tier shape differs from the
  recorded fixtures. Preserve the raw fixture, update the plan/schema explicitly,
  and do not silently accept arbitrary shapes.
- Regenerating `models.json` triggers destructive-change protection. Do not use
  the override without human review.
- Any phase appears to require downgrading dependencies, editing credentials,
  weakening fixture redaction, adding runtime catalog networking, or changing
  `MatchModel` freshness ordering.
- A focused verification command fails twice after a reasonable correction.

## Maintenance notes

- Review retry policy whenever provider SDKs change their default behavior, but
  keep go-llm's public guarantee independent of SDK internals.
- Review the 64-character prompt-cache limit if the provider contract changes;
  do not apply it globally to unrelated providers.
- Treat pricing and effort metadata as advisory generated data. Public request
  forwarding remains server-validated and must not reject a request solely
  because the embedded snapshot is stale.
- New pricing fields require updates in four places: public structs, snapshot
  parser/merger, embedded validator, and deep-copy paths.
- New Chat Completions quirks should be explicit `Compat` data with a focused
  preset enablement and golden test, not hostname checks or universal behavior.
- A future scheduled catalog-refresh workflow may open reviewable PRs, but it
  must never auto-merge and is intentionally outside this plan.
