---
status: ready
---

# Provider Improvements Plan

Work items for the v0.1.x/v0.2 line: two correctness fixes
(Phases 1–2), one model-family upgrade (Phase 3, with the subscription
surface run as a live-verified spike), and two capability/observability
additions (Phases 4–5). Keep the existing architecture; no unrelated
provider work.

Dependencies: Phases 1, 2, 3a, 4a, 4b, and 5 have none and may run in any
order; the Phase 3b *implementation* depends on 3a (cache-options surface
and usage math), though its exploratory live calls may happen earlier.
Baseline: commit `7f486df` already includes the SDK upgrades —
`openai-go/v3 v3.44.0` and `anthropic-sdk-go v1.59.0` (`PromptCacheOptions`
with mode+ttl, `prompt_cache_breakpoint`, `cache_write_tokens`; no escape
hatch needed).

After external review (two rounds): error-assertion and wording
corrections in Phases 1–2; Phase 3a made deterministic (library-owned
cache-options types, no model-name gate, no snapshot regeneration);
Phase 3b converted to a spike with STOP conditions and an explicit 3a
dependency; Phase 4 split into a concrete stats-only TTFT telemetry API
and a separately-scoped conformance pass; Phase 5 given an exact public
contract and curated data source; prewarm demoted to Future Work.

When a phase is handed to an executor agent, extract it into its own plan
file carrying the planned-at SHA, its dependencies, and its STOP
conditions; this document is the roadmap.

## Phase checklist

- [x] Phase 1: OAuth credential-POST redirect hardening
- [x] Phase 2: in-stream error-code classification
- [x] Phase 3a: GPT-5.6 on the platform Responses API
- [x] Phase 3b (spike): GPT-5.6 on the ChatGPT subscription backend — live-verified PASS 2026-07-22, findings below
- [x] Phase 4a: TTFT usage telemetry
- [x] Phase 4b: append-only prefix conformance
- [x] Phase 5: per-model reasoning-effort enumeration

## Phase 1: OAuth credential-POST redirect hardening

Threat: Go's `http.Client` replays a POST body on 307/308. A redirecting
token endpoint (compromised, intercepted, or misconfigured) would cause a
refresh token, authorization code, or PKCE verifier to be re-POSTed to an
attacker-chosen origin. Our token URLs are hardcoded first-party HTTPS, but
the HTTP client is caller-injectable, and the fix is ~10 lines.

1. In `providers/internal/provideroauth`, add a helper that shallow-copies
   an `*http.Client` and sets `CheckRedirect` to return an exported
   sentinel (`ErrUnsafeRedirect`) unconditionally. Never mutate the
   caller's client.
2. Route every credential-bearing POST through it: the Anthropic refresh
   path (`providers/anthropic/oauth.go`) and the Codex refresh path
   (`providers/openaicodex/oauth.go`). Legitimate token endpoints respond
   200 directly, so behavior is unchanged for valid flows.
3. Test with a redirect-trap: a test server that redirects to a second
   server which fails the test if hit. Cover both 307 and 308. Assert
   `errors.Is(err, provideroauth.ErrUnsafeRedirect)` — the sentinel
   surfaces through the `*url.Error` chain that `http.Client.Do` returns
   (the refresh paths return transport errors unwrapped; do not assert a
   library-typed error). Assert the trap records zero hits and that the
   caller's original client — including any caller-set `CheckRedirect` —
   is unchanged after the call.
4. Record the companion rules in the deferred interactive-login entry of
   `specs/projects/go-llm/implementation_plan.md` Future Work (design
   notes only, no implementation now): a single exported endpoint
   validator (https-or-loopback) applied when merging any discovered
   endpoint plus a backstop at the authorize-URL choke point, and no
   network discovery when endpoints are fully preconfigured.

Acceptance:

- No credential POST in the repo follows a redirect; grep-level check that
  every token/refresh request uses the wrapped client.
- Redirect-trap tests (307 and 308) pass for both providers via
  `errors.Is` on the sentinel; caller client verifiably unmutated;
  valid-refresh fixtures unchanged.

```sh
go test -count=1 ./providers/internal/provideroauth ./providers/anthropic ./providers/openaicodex
go test -race -count=1 ./providers/internal/provideroauth
```

## Phase 2: In-stream error-code classification

Gap: an OpenAI-compatible server that returns HTTP 200 and then an SSE
payload like `{"error":{"code":"429"}}` mid-stream classifies as
`ErrServer` through the generic Chat Completions dialect, because
numeric-string codes are only handled by the vLLM dialect
(`providers/vllm/dialect.go`) and OpenRouter's special-cased `402`.
Mid-stream errors are never internally retried (streams are not resumed
after a 2xx begins), so the harm is that callers making their own retry
decisions off the typed error classes are misled.

1. Lift the numeric-code fallback into the shared classifier in
   `providers/internal/providerutil`: when HTTP status is 0/unknown and
   the error `code` parses as an integer in 400–599 (accept string,
   float64, and int JSON forms), classify by that status before falling
   back to heuristics. Non-integral codes must never classify: the
   current float64 formatting in `providers/chatcompletions/errors.go`
   uses `%.0f`, which rounds — a `429.5` code must be rejected, not
   rounded into a valid status.
2. Keep existing semantic mappings intact — `insufficient_quota` and
   quota/rate phrasing continue to map to `ErrInsufficientCredits` /
   `ErrRateLimited` as today; the numeric fallback runs only when
   semantics didn't already classify.
3. Remove the now-redundant vLLM-local copy; confirm OpenRouter's `402`
   handling still wins where it should.
4. Fixtures: in-stream error objects with `"429"`, `429`, `429.0`,
   `429.5` (must not classify), `"401"`, `"503"`, and a non-numeric code,
   asserted through the generic dialect, vLLM, and OpenRouter; assert the
   resulting typed error classes (`ErrRateLimited`, `ErrAuth`,
   `ErrServer`).

Acceptance:

- A status-0 in-stream `{"error":{"code":"429"}}` classifies as
  `ErrRateLimited` through every Chat Completions dialect; fractional
  codes do not classify; no existing error-classification fixture changes
  result.

```sh
go test -count=1 ./providers/internal/providerutil ./providers/chatcompletions ./providers/vllm ./providers/openrouter ./providers/ollama
go test -race -count=1 ./providers/internal/providerutil ./providers/chatcompletions
```

## Phase 3a: GPT-5.6 on the platform Responses API (`providers/openai`)

Baseline `7f486df` carries SDK v3.44.0; the models table already carries
the 5.6 family rows with cache-write rates. This phase adds the
cache-options surface and the usage math.

1. Cache options — library-owned types on `openai.Options`
   (`providers/openai/request_options.go`), mirroring the SDK contract
   (`ResponsePromptCacheOptions`: mode ∈ implicit|explicit, ttl ∈ 30m):

   ```go
   type PromptCacheMode string

   const (
       PromptCacheModeImplicit PromptCacheMode = "implicit"
       PromptCacheModeExplicit PromptCacheMode = "explicit"
   )

   type PromptCacheTTL string

   const PromptCacheTTL30m PromptCacheTTL = "30m"

   type PromptCacheOptions struct {
       Mode PromptCacheMode
       TTL  PromptCacheTTL
   }
   ```

   Add `PromptCacheOptions *PromptCacheOptions` to `openai.Options`.
   Rules: emit the wire field only when explicitly supplied; validate
   enum values in preflight; preflight-reject a request that sets both
   `PromptCacheRetention` and `PromptCacheOptions` (the SDK deprecates
   retention and documents ttl as a *minimum cache lifetime* — different
   semantics, never silently translate). No model-name preflight gate:
   the option stays forward-compatible and the server owns model
   acceptance (docs say gpt-5.6+; a client-side gate would wrongly
   reject future models).
2. Usage: map `usage.input_tokens_details.cache_write_tokens` additively
   per the FS §11 contract (`response.go` — `InputTokens` excludes cache
   reads AND writes): extend `mapUsage` in
   `providers/internal/responsesapi/response.go` to subtract both
   `cached_tokens` and `cache_write_tokens` from provider `input_tokens`
   with underflow protection (the cache-read subtraction already exists —
   mirror it), preserve provider `total_tokens` when reported, and
   reconstruct it as input + read + write + output when absent.
3. Pricing: no snapshot regeneration (it would pull live catalog churn
   unrelated to this phase). Add focused assertions against the embedded
   table: the 5.6 family rows exist with `cache_write_per_mtok`, and
   cache-write tokens are costed at the cache-write rate, not the input
   rate.
4. Cache breakpoints — defer. v3.44 exposes `prompt_cache_breakpoint`
   ("marks the exact end of a reusable prompt prefix", TTL inherited
   from `prompt_cache_options.ttl`), so mapping `TextPart.Cache` hints
   is now representable. FS §15 remains accurate about go-llm's current
   behavior (hints are not sent to OpenAI) until a mapping ships — do
   not rewrite the normative statement; add a Future Work entry in
   `implementation_plan.md` recording the mapping as known, deferred
   work.
5. Guard note at `providers/internal/responsesapi/adapter.go:86`:
   `prompt_cache_key` is Responses-surface-only by design — never add it
   to Chat Completions dialects without a provider-kind gate (strict
   compatible gateways reject unknown fields with 400).

Acceptance:

- Requests that set `PromptCacheOptions` carry `prompt_cache_options` on
  the wire; requests that don't, don't; setting both retention and
  options fails preflight; invalid enum values fail preflight; requests
  without either option are byte-identical to before.
- Usage remains additive: InputTokens + CacheReadTokens +
  CacheWriteTokens + OutputTokens equals TotalTokens in fixtures with
  cache writes; cache writes are costed at the cache-write rate.
- Future Work entry for breakpoint mapping recorded; FS §15 untouched.

```sh
go test -count=1 ./providers/openai ./providers/internal/responsesapi
go test -count=1 . -run 'Pricing|Cost|ModelsTable|Usage'
```

## Phase 3b (spike): GPT-5.6 on the ChatGPT subscription backend (`providers/openaicodex`)

Depends on Phase 3a for implementation (cache-options surface and usage
math); the exploratory live calls below may run before 3a lands.

The subscription backend gates GPT-5.6 ("Responses Lite") on client
compatibility. The expected wire, mirroring current Codex CLI (0.144.0):
`X-OpenAI-Internal-Codex-Responses-Lite: true` on 5.6-family requests;
User-Agent `codex_cli_rs/0.144.0` on all subscription requests; tools
moved from top-level `tools` into the input array as a leading
`{"type":"additional_tools","role":"developer","tools":[...]}` item;
system prompt as a `developer` role input message instead of top-level
`instructions`; forced `parallel_tool_calls: false` and
`reasoning.context: "all_turns"`. This is an undocumented upstream
contract — verify live before hardcoding any of it.

Spike protocol (requires subscription credentials):

1. Live checks, in order: (a) 5.6 without tools; (b) 5.6 with tools and
   system instructions; (c) stream decoding and cache-write usage
   reporting; (d) a pre-5.6 control request confirming existing behavior
   is unaffected.
2. STOP conditions: if the 5.6 family is unavailable on the subscription,
   or the observed wire differs from the shape above, stop — record the
   observed contract in this plan and do not improvise a variant.
3. On PASS: implement as one isolated request-transform function gated on
   the 5.6 family, with exact-wire fixture tests (item order included)
   and a pre-5.6 control fixture; extend the live e2e matrix with a 5.6
   scenario. Pre-5.6 requests stay **body-identical** — only the
   User-Agent header changes globally, and that change is part of the
   contract under test.
4. Model discovery: the static catalog ends at gpt-5.5 and its source
   comment ("the backend has no public models endpoint",
   `providers/openaicodex/provider.go:30`) is outdated — an authenticated
   models endpoint now exists. The spike determines whether to adopt live
   discovery with the static list as fallback, and updates the stale
   comment either way. Invariant if adopted: auxiliary requests must
   share the runtime path's header decoration and auth-retry behavior.

Acceptance (post-spike, if implemented):

- 5.6 subscription requests match the live-verified Lite shape
  byte-for-byte in fixtures; pre-5.6 request bodies byte-identical to
  before; decode path unchanged; cache-write usage flows per Phase 3a
  math.

```sh
go test -count=1 ./providers/openaicodex ./providers/internal/responsesapi
```

### Spike findings (2026-07-22) — PASS

Live run: `go test -tags=live ./internal/e2e -run TestLiveOpenAICodexGPT56`
(subscription credentials from `gollm-test.json`; rotated tokens persisted
back).

- The bare `gpt-5.6` id is REJECTED on subscription (`400 "The 'gpt-5.6'
  model is not supported when using Codex with a ChatGPT account."`). The
  family is served as `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`.
- The Lite shape as specified was accepted end-to-end on `gpt-5.6-sol`:
  plain text turn; tools + system via the `additional_tools` +
  `developer`-message input items (top-level `tool_choice:"auto"` retained
  and accepted); stream decode unchanged; usage additive, with the wire now
  reporting `input_tokens_details.cache_write_tokens` on subscription.
- Pre-5.6 control (gpt-5.4-mini) passed under the global
  `codex_cli_rs/0.144.0` User-Agent — the bump is safe; the server answered
  with dated snapshot `gpt-5.4-mini-2026-03-17`.
- Model discovery: the authenticated endpoint EXISTS —
  `GET {base}/models?client_version=<compat>` (400 without the query
  param; the old "no public models endpoint" comment was wrong and has
  been corrected). Per-model metadata includes `use_responses_lite: true`
  (the authoritative Lite gate — our name-based family gate is a stand-in),
  `context_window: 272000`, `prefer_websockets: true`,
  `minimal_client_version: 0.142.2`, and `supported_reasoning_levels`
  low/medium/high/xhigh/max plus **`ultra`** ("maximum reasoning with
  automatic task delegation") — `ultra` is outside go-llm's `Effort`
  vocabulary and is recorded here, not adopted.
- Decision: keep the static catalog (refreshed from the live snapshot,
  now including the three 5.6 ids with the low→max dial;
  `codex-auto-review` excluded) and defer runtime discovery-with-fallback
  to follow-up work — it needs the shared auth-retry path, the compat
  query, and schema modeling. An `EffortUltra` vocabulary decision is
  likewise deferred.

## Phase 4a: TTFT usage telemetry

Measurement first: latency and cache work (including the deferred prewarm
idea) needs data we do not collect. `UsageTracker` is stats-only — no
logger, no callbacks (`usage.go`) — and stays that way; this phase adds
aggregate fields only.

1. Extend the accumulator and `UsageStats` with:

   ```go
   StreamCalls             int64
   MessageStartSamples     int64
   TotalTimeToMessageStart time.Duration
   FirstContentSamples     int64
   TotalTimeToFirstContent time.Duration
   ```

   recorded in `trackStream` (which already stamps a per-stream start
   time), aggregated per provider/model bucket like the existing fields.
2. Definitions: time-to-MessageStart is start → the MessageStart event.
   First content is the first non-empty TextDelta, non-empty
   ReasoningDelta, or ToolCallStart. Empty or error-only streams produce
   no first-content sample (and no MessageStart sample if none arrived).
   Blocking calls leave all stream fields zero.
3. No slog, no callbacks, no new option types in this phase — callers
   compute averages from samples/totals in the `UsageStats` snapshot.

Acceptance:

- Streaming calls on every first-party preset produce the new fields with
  correct sample counts; blocking calls leave them zero; error-only and
  empty streams produce no first-content sample; existing `UsageStats`
  consumers compile unchanged.

```sh
go test -count=1 . -run 'Usage'
go test -race -count=1 .
```

## Phase 4b: Append-only prefix conformance

Pins the property provider prompt caches rely on: within a session, each
turn's serialized request extends the previous one — it never rewrites it.

1. Offline conformance test run from each provider package — Anthropic,
   OpenAI, Codex, OpenRouter, vLLM, and Ollama: run a two-turn tool
   session, capture both serialized wire requests at the provider
   boundary (via the existing wire-capture seam), and assert the turn-2
   serialized message prefix deep-equals turn 1's full serialized
   messages (append-only), with tools and cache/session keys unchanged.
2. Prefix-fingerprint middleware — design spike only, not part of this
   phase's checklist, tied to the queued `WithAutoCache` Future Work item
   (placement and drift detection ship as siblings). The spike must
   answer before any implementation: where the middleware obtains its
   logger; how per-SessionID state is bounded and evicted; whether
   component hashes are exposed to callers (hashes are not
   anonymization); how cache hints are compared as history grows; and
   how tests inspect provider-serialized requests.

Acceptance:

- Append-only prefix conformance passes for Anthropic, OpenAI, Codex,
  OpenRouter, vLLM, and Ollama, running from those packages.

```sh
go test -count=1 ./providers/anthropic ./providers/openai ./providers/openaicodex ./providers/openrouter ./providers/vllm ./providers/ollama -run 'Conformance|Prefix'
```

## Phase 5: Per-model reasoning-effort enumeration

Gap: capabilities are provider-level flags and `ModelInfo` carries
context/output/pricing, but nothing enumerates which `Effort` levels a
model actually supports — callers guess.

1. Public contract:
   - `ModelInfo.SupportedEfforts []Effort`, ordered weakest → strongest.
   - The embedded-table lookup clones the slice and applies the existing
     canonical-ID fallback.
   - Each first-party `Models()` copies embedded effort metadata into its
     returned rows.
   - `SupportedEffortsForModel(provider, modelID string) []Effort` —
     curated metadata first, family inference second.
   - `llmtest`'s `cloneModels` (llmtest/provider.go) clones the new
     slice.
2. Data source: models.dev supplies no effort information, so values are
   ours — curated in `scripts/overrides.json` with provenance comments,
   flowing through `modelTableRow`, the snapshot TypeScript
   types/validation, and `models.json`. Initial curated matrix (verify
   each row against vendor documentation at execution time and record
   the source in the override comment):
   - OpenAI `gpt-5*` reasoning models: `minimal, low, medium, high`
     (Responses API `reasoning.effort` enum; 5.6-family additions such
     as `xhigh` to be confirmed against current docs).
   - OpenAI o-series (`o3*`, `o4*`): `low, medium, high`.
   - Codex subscription models (`gpt-5.3-codex-spark` … `gpt-5.5`,
     plus 5.6 post-spike): `low, medium, high`.
   - Anthropic extended-thinking models: the full dial `minimal … max`
     plus `none` (efforts map to thinking budgets per FS §9, so any
     level is expressible); non-thinking models: empty.
   - Aggregator/self-hosted rows (OpenRouter, vLLM, Ollama): no curated
     values — canonical-ID fallback or family inference only.
3. Family inference for models absent from the table: a dedicated
   `modelFamilyName` helper used ONLY for effort inference — normalize by
   stripping a `provider:` prefix and a leading vendor `/` segment
   (`openai/o3-mini` → `o3-mini`) before family matching. Explicitly NOT
   shared with pricing lookups: the price table is keyed
   `provider/model-id` with longest-prefix matching and deliberately
   preserves aggregator namespaces (`pricing.go`).
4. Policy: the metadata is advisory — request forwarding and
   server-side validation behavior are unchanged (some gateways ignore
   an unsupported effort, others 400; both remain the server's call).
   Preflight stays exactly as today: `ErrUnsupported` only where the
   provider contract is authoritative.

Acceptance:

- Catalogued models return curated efforts; recognized-but-uncatalogued
  families may return inferred efforts; unrecognized models return empty.
- `SupportedEffortsForModel` resolves curated-before-inferred; returned
  slices are clones (mutating a result never affects the table).
- No preflight or request-forwarding behavior change; pricing lookups
  byte-identical before and after (family normalization provably not
  applied to them).

```sh
go test -count=1 . ./providers/... ./llmtest
```

## Deferred out of this plan

- **Connection prewarm helper** — moved to Future Work in
  `specs/projects/go-llm/implementation_plan.md`, gated on Phase 4a TTFT
  data showing cold-start handshake cost is worth optimizing. The design
  constraints worth keeping: explicit-only (never implicit in
  Chat/ChatStream), unauthenticated single-attempt bounded probe, and
  skip when the transport disables keep-alives.
- **`TextPart.Cache` → `prompt_cache_breakpoint` mapping** — recorded as
  Future Work by Phase 3a item 4; FS §15 stays as-is until it ships.
