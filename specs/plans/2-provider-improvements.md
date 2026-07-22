---
status: ready
---

# Provider Improvements Plan

Five work items for the v0.1.x/v0.2 line: two correctness/security fixes
(Phases 1–2), one model-family upgrade (Phase 3, with the subscription
surface run as a live-verified spike), and two capability/observability
additions (Phases 4–5). Phases are independent of each other; 1 and 2 are
small and should land first. Keep the existing architecture; no unrelated
provider work.

After external review: error-assertion and wording
corrections in Phases 1–2, explicit SDK prerequisite and additive usage
math in Phase 3a, Phase 3b converted to a spike with STOP conditions,
Phase 4 restructured around TTFT-first, Phase 5 metadata sourcing fixed,
prewarm demoted to Future Work.

When a phase is handed to an executor agent, extract it into its own plan
file carrying the planned-at SHA, its dependencies, and its STOP
conditions; this document is the roadmap.

## Phase checklist

- [ ] Phase 1: OAuth credential-POST redirect hardening
- [ ] Phase 2: in-stream error-code classification
- [ ] Phase 3a: GPT-5.6 on the platform Responses API
- [ ] Phase 3b (spike): GPT-5.6 on the ChatGPT subscription backend
- [ ] Phase 4: TTFT observability + append-only prefix conformance
- [ ] Phase 5: per-model reasoning-effort enumeration

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

Prerequisite (part of this phase, currently sitting uncommitted in the
working tree): commit the SDK bumps — `openai-go/v3 v3.44.0` and
`anthropic-sdk-go v1.59.0`. v3.44 natively carries `PromptCacheOptions`,
`prompt_cache_breakpoint`, and `cache_write_tokens`; no escape hatch is
needed.

1. Model IDs: add the `gpt-5.6` family (`gpt-5.6`, `-sol`, `-terra`,
   `-luna`) with a family-match helper covering `gpt-5.6` and `gpt-5.6-*`
   including dated snapshots.
2. Cache options: add a library-owned `PromptCacheTTL` option to
   `openai.Options` alongside the existing `PromptCacheRetention`
   (`providers/openai/request_options.go`). Never silently translate one
   into the other — the SDK deprecates `prompt_cache_retention` and
   documents that `prompt_cache_options.ttl` expresses a *minimum cache
   lifetime*, a different semantic. Preflight-reject a request that sets
   both. Do not hard-gate the new field on the 5.6 family name: the
   API-wide deprecation suggests broader acceptance — verify actual
   server behavior per model family live and gate on what is observed.
3. Usage: map `usage.input_tokens_details.cache_write_tokens` additively
   per the FS §11 contract (`response.go` — `InputTokens` excludes cache
   reads AND writes): extend `mapUsage` in
   `providers/internal/responsesapi/response.go` to subtract both
   `cached_tokens` and `cache_write_tokens` from provider `input_tokens`
   with underflow protection (the cache-read subtraction already exists —
   mirror it), preserve provider `total_tokens` when reported, and
   reconstruct it as input + read + write + output when absent.
4. Pricing: the models snapshot already carries the 5.6 family with
   cache-write rates (`models.json`) — verify by regenerating the
   snapshot and updating only if it differs. Test that cache-write tokens
   are costed at the cache-write rate, not the input rate.
5. Cache breakpoints — explicit decision, default defer: v3.44 exposes
   `prompt_cache_breakpoint` ("marks the exact end of a reusable prompt
   prefix", TTL inherited from `prompt_cache_options.ttl`), so mapping
   `TextPart.Cache` hints (currently discarded by the Responses adapter)
   is now representable. Defer the mapping to its own item, and update FS
   §15 now — its claim that OpenAI ignores cache hints is no longer
   accurate for 5.6+ — recording the mapping as known, deferred work.
6. Guard note at `providers/internal/responsesapi/adapter.go:86`:
   `prompt_cache_key` is Responses-surface-only by design — never add it
   to Chat Completions dialects without a provider-kind gate (strict
   compatible gateways reject unknown fields with 400).

Acceptance:

- 5.6 requests carry `prompt_cache_options` per the observed acceptance
  gate; setting both TTL and retention fails preflight; pre-5.6 request
  bodies are unchanged.
- Usage remains additive: InputTokens + CacheReadTokens +
  CacheWriteTokens + OutputTokens equals TotalTokens in fixtures with
  cache writes; cache writes are costed at the cache-write rate.
- FS §15 updated; snapshot verified against regeneration.

```sh
go test -count=1 ./providers/openai ./providers/internal/responsesapi
go test -count=1 . -run 'Pricing|Cost|ModelsTable|Usage'
```

## Phase 3b (spike): GPT-5.6 on the ChatGPT subscription backend (`providers/openaicodex`)

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

## Phase 4: TTFT observability + append-only prefix conformance

Measurement first: latency and cache behavior work (including any future
prewarm or cache-placement features) needs data we do not currently
collect.

1. TTFT metrics: `UsageTracker.trackStream` (`usage.go`) already stamps a
   start time per stream — record time-to-MessageStart and
   time-to-first-content-delta and expose them through the existing
   observability surface (`slog` fields / tracker callbacks, silent by
   default, following the current UsageTracker conventions).
2. Append-only prefix conformance test (offline suite, all first-party
   presets): run a two-turn tool session, capture both serialized wire
   requests at the provider boundary (via the existing wire-capture
   seam), and assert the turn-2 serialized message prefix deep-equals
   turn 1's full serialized messages (append-only), with tools and
   cache/session keys unchanged. This pins the property provider prompt
   caches rely on.
3. Prefix-fingerprint middleware — design spike only, tied to the queued
   `WithAutoCache` Future Work item (placement and drift detection ship
   as siblings). The spike must answer before any implementation: where
   the middleware obtains its logger; how per-SessionID state is bounded
   and evicted; whether component hashes are exposed to callers (hashes
   are not anonymization); how cache hints are compared as history grows;
   and how tests inspect provider-serialized requests.

Acceptance:

- TTFT fields appear for streaming calls on every first-party preset and
  are absent/zero for blocking calls; zero overhead when no tracker is
  installed.
- Append-only prefix conformance passes for every first-party preset.

```sh
go test -count=1 . ./llmtest ./internal/e2e ./providers/...
go test -race -count=1 . ./llmtest
```

## Phase 5: Per-model reasoning-effort enumeration

Gap: capabilities are provider-level flags and `ModelInfo` carries
context/output/pricing, but nothing enumerates which `Effort` levels a
model actually supports — callers guess.

1. Source of truth: models.dev does not publish effort levels, so the
   data is ours. Extend the snapshot pipeline end to end: `ModelInfo`
   gains an ordered efforts field (weakest → strongest, cloned on read),
   `modelTableRow` and the snapshot TypeScript types/validation gain the
   column, and values come from curated overrides in the snapshot script.
   Canonical-ID fallback applies as for other metadata.
2. Name-family inference for models absent from the table: a dedicated
   `modelFamilyName` helper used ONLY for effort inference — normalize by
   stripping a `provider:` prefix and a leading vendor `/` segment
   (`openai/o3-mini` → `o3-mini`) before family matching. Explicitly NOT
   shared with pricing lookups: the price table is keyed
   `provider/model-id` with longest-prefix matching and deliberately
   preserves aggregator namespaces (`pricing.go`).
3. Policy: the enumeration is advisory — preflight validation stays
   exactly as today (`ErrUnsupported` only where the provider contract is
   authoritative). Rationale: forwarding an effort a gateway ignores is a
   silent no-op, not an error; do not turn unknown into a hard failure.

Acceptance:

- Catalogued models return curated efforts; recognized-but-uncatalogued
  families may return inferred efforts; unrecognized models return empty.
- No preflight behavior change; pricing lookups byte-identical before and
  after (family normalization provably not applied to them).

```sh
go test -count=1 . ./providers/...
```

## Deferred out of this plan

- **Connection prewarm helper** — moved to Future Work in
  `specs/projects/go-llm/implementation_plan.md`, gated on Phase 4 TTFT
  data showing cold-start handshake cost is worth optimizing. The design
  constraints worth keeping: explicit-only (never implicit in
  Chat/ChatStream), unauthenticated single-attempt bounded probe, and
  skip when the transport disables keep-alives.
