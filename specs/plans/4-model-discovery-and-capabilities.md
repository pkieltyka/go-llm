---
status: complete
planned_at: 32cd8b3
planned_at_date: 2026-08-20
---

# Model Discovery and Advisory Capabilities Plan

This plan implements the focused go-llm follow-up selected after reviewing
recent model-catalog work in comparable projects. It is written against
go-llm commit `32cd8b3` on 2026-08-20.

The work stays within go-llm's provider-mechanics and model-metadata boundary:
explicit, cached Codex model discovery and advisory per-model capabilities.

An executor must read this plan completely before changing the repository.
Use the live trees as authoritative when line numbers drift, preserve unrelated
worktree changes, and stop rather than broadening the public API when a STOP
condition below is met.

## Confirmed decisions

1. **Keep `PromptTemplate`.** It is an intentional go-llm convenience API. This
   plan must not remove, deprecate, move, or redesign it.
2. **Keep conversation handoff and reasoning replay unchanged.** This plan does
   not alter history serialization, provider/model switching, `ReasoningPart`,
   or foreign-reasoning filtering.
3. **Codex discovery is explicit-only.** Remote catalog traffic may occur only
   when a caller invokes `(*openaicodex.Provider).Models(ctx)`. Provider
   construction, `Chat`, `ChatStream`, session continuation, and model
   switching must never trigger discovery.
4. **Model capabilities are advisory.** Per-model metadata helps discovery,
   selection, and UI. It does not replace provider-level `Capabilities()` and
   must not introduce model-name-based request rejection.

## Outcomes

- [x] `ModelInfo` can report advisory per-model capabilities without changing
  request validation.
- [x] Codex `Models(ctx)` uses the authenticated live catalog when available.
- [x] Codex discovery performs no background/startup/chat traffic, caches a
  successful result for 30 minutes, suppresses repeated failed refreshes for
  five minutes, and coalesces concurrent callers per provider instance.
- [x] A stale successful catalog or curated static catalog remains usable for
  transient discovery failures; authentication and context errors remain
  visible.
- [x] Codex and OpenRouter populate only capabilities their live catalog data
  positively advertises.
- [x] Offline tests cover caching, fallback, cloning, concurrency, capability
  mapping, and documentation contracts without reaching the public network.

## Priority, effort, and dependencies

| Phase | Priority | Effort | Risk | Depends on |
|---|---|---:|---|---|
| 1. Advisory model capability contract | P1 | S-M | LOW-MED | none |
| 2. Explicit cached Codex discovery | P1 | M | MED | Phase 1 |
| 3. Live capability mapping | P2 | S-M | LOW-MED | Phases 1-2 |
| 4. Documentation and final verification | P1 | S | LOW | all phases |

All phases are scoped to go-llm.

## Repository state and drift check

At planning time:

- go-llm is at `32cd8b3285e6cb7b7231d5b0ba310e21708a6037`.

Before implementation:

```sh
cd /home/peter/Dev/pkieltyka/go-llm
git status --short --branch
git diff --stat 32cd8b3285e6cb7b7231d5b0ba310e21708a6037
make test
```

Expected: go-llm differs only by this plan unless the maintainer has added work
after planning. If an in-scope API or provider path has changed, reconcile the
plan with the live source before editing.

## Boundaries and invariants

- Keep the root go-llm package standard-library-only.
- Do not add a background goroutine, global model cache, disk cache, ETag
  subsystem, scheduler, or implicit model-list request.
- Cache Codex catalogs per `Provider` instance because catalogs can differ by
  OAuth account, base URL, and custom headers.
- Do not add a public cache-control option in this phase. The fixed cache
  contract keeps the API tight; revisit only with a concrete caller requiring
  a different freshness policy.
- `Models(ctx)` remains the explicit network boundary for providers whose
  catalog is remote. `LookupModelInfo` and the embedded `models.json` table
  remain offline.
- Provider-level `Capabilities()` remains the authoritative preflight input.
  `ModelInfo.Capabilities` is positive, best-effort metadata; an absent value
  means unknown, not unsupported.
- Do not infer capabilities from a model name. Map only explicit catalog
  fields or established embedded metadata.
- Do not add `EffortUltra`, reasoning-summary controls, service-tier fields, or
  request rejection in this plan. Live raw catalog payloads may retain those
  upstream fields for a later vocabulary decision.
- Preserve model-row ordering unless this plan explicitly requires
  deterministic normalization.

## Phase 1: Add advisory per-model capabilities

### Current state

`Provider.Capabilities()` is provider-wide and drives request validation.
`ModelInfo` contains identity, token limits, pricing, efforts, and raw provider
data, but cannot express that one model supports tools, reasoning, or image
input while another model from the same provider does not.

### Public contract

Add the following field to `llm.ModelInfo`:

```go
// Capabilities lists capabilities positively advertised for this model.
// The metadata is advisory: empty means unknown, and request validation
// continues to use Provider.Capabilities().
Capabilities []Capability
```

Contract details:

1. Entries use the existing `Capability` vocabulary. Do not introduce a
   second model-capability enum.
2. The slice contains positive claims only. Missing entries and an empty slice
   must never be interpreted as a negative capability assertion.
3. Preserve stable source order while removing duplicates. First-party
   mappings should emit standard capabilities in the declaration order used
   by `capability.go` so fixture output is deterministic.
4. Every API returning `ModelInfo` must return an independently mutable
   capability slice. Mutating one result must not alter a provider cache,
   embedded table, configured `llmtest` provider, or later call.
5. Do not add model-aware checks to `ValidateRequest`,
   `ValidateStreamRequest`, or `Session` in this phase.

### Steps

1. Extend `ModelInfo` and its godoc in `llm.go`.
2. Update all existing model-cloning paths, especially:
   - `models_table.go`'s `cloneModelInfo`;
   - `llmtest/provider.go`'s `cloneModels`;
   - provider `Models()` implementations that enrich or copy embedded rows;
   - the new Codex discovery cache in Phase 2.
3. Add root and `llmtest` mutation-isolation tests for both
   `SupportedEfforts` and `Capabilities`.
4. Update architecture and functional-spec `ModelInfo` definitions. State
   explicitly that provider capabilities remain the request-validation
   authority.
5. Do not extend `models.json` or the snapshot script merely to make the new
   field non-empty. Embedded capability ingestion requires a source with
   sufficiently precise semantics and is deferred.

### Verification

```sh
go test -race -count=1 . ./llmtest ./providers/...
```

Acceptance:

- Existing provider validation behavior is byte-for-byte and error-for-error
  unchanged.
- Capability slices are cloned defensively.
- Existing model providers compile without being required to populate the
  field.

## Phase 2: Add explicit cached Codex model discovery

### Current state

`openaicodex.Models(ctx)` ignores its context and returns a curated list last
refreshed from the authenticated backend on 2026-07-22. Plan 2's live spike
confirmed this endpoint:

```text
GET {codex-base}/models?client_version=<compat-version>
```

The request requires Codex OAuth/header behavior, and the response contains
model visibility, context, supported reasoning levels, service tiers, and
other evolving metadata. The static list remains valuable as a resilient
fallback.

### Network and cache contract

1. **Trigger:** only an explicit `p.Models(ctx)` call may fetch.
2. **Success TTL:** cache a valid live catalog for 30 minutes per provider
   instance.
3. **Failure suppression:** after a transient transport/server/protocol
   failure, do not attempt another refresh for five minutes. During that
   interval return a stale successful catalog when one exists, otherwise the
   curated static catalog.
4. **No hidden auth fallback:** normalized authentication failures are returned
   to the caller and do not silently become a successful static response.
5. **Cancellation:** a canceled/deadline-exceeded caller receives its context
   error. Cancellation does not install or extend a fallback cache entry.
6. **Coalescing:** concurrent cache misses on one provider instance share one
   logical discovery operation. Waiting callers can still abandon the wait via
   their own context.
7. **Isolation:** no cache sharing across provider instances or accounts.
8. **No background refresh:** expired entries refresh only on a later explicit
   `Models` call.
9. **Defensive results:** every return is a deep-enough clone of mutable
   `ModelInfo` fields, including pricing tiers, efforts, capabilities, and raw
   JSON bytes.
10. **Empty success is failure:** an HTTP success containing no usable visible
    models is a protocol failure and follows stale/static fallback behavior.

The cache implementation must use the standard library. A mutex plus an
in-flight completion channel is sufficient; do not add `x/sync/singleflight`
solely for this feature. Tests may inject an unexported clock into a provider
instance; do not expose a clock or TTL option publicly.

### Request contract

1. Build the models URL from `codexBaseURL`, never from a string replacement
   against the `/responses` endpoint.
2. Supply exactly one `client_version` query parameter. Derive the value from a
   single named Codex compatibility constant so model discovery and the
   declared client protocol cannot drift accidentally.
3. Retain the currently supported compatibility version unless a focused live
   check proves the endpoint requires a newer minimum. A version bump that
   changes ordinary chat headers is a separate wire change and requires its
   existing control fixtures.
4. Reuse the configured HTTP client, observation/wire-capture layer, custom
   headers, originator/account headers, OAuth source, and one-refresh auth retry
   behavior. Do not create an unobserved default client or duplicate token
   refresh logic.
5. The catalog request is `GET` with JSON accept semantics. Do not copy
   streaming-only `Accept: text/event-stream` or a request body onto it.
6. Apply the provider's configured timeout and caller context.

### Response and merge contract

Parse both top-level `models` and `data` arrays, accepting the endpoint fields
needed for the unified surface while preserving each complete row as copied
`json.RawMessage` in `ModelInfo.Raw`.

For each row:

- identity: first non-empty `id` or `slug`;
- display: first non-empty `display_name`, `name`, or description;
- include only rows with empty visibility or `visibility: "list"`;
- continue explicitly excluding the internal `codex-auto-review` model;
- deduplicate by ID while retaining first appearance;
- map positive context/output limits and supported reasoning efforts;
- retain upstream values outside go-llm's current vocabulary, including
  `ultra`, in `Raw` but omit them from `SupportedEfforts` for now;
- enrich missing display, context, output, efforts, and pricing from the
  curated/embedded row with the same ID;
- do not append static-only rows after a successful live response: the
  account's visible live catalog is authoritative for membership.

Static fallback keeps the present curated order and exclusions. A transient
fallback may emit one safe operational log through the configured logger, but
must not include response bodies, bearer tokens, account IDs, or arbitrary
provider-controlled messages.

### Implementation shape

1. Move static model cloning/enrichment into a helper used by both fallback and
   live merge paths.
2. Add a small catalog request/parser file under `providers/openaicodex` rather
   than growing request/stream conversion files.
3. Extend the existing `codexTransport` or factor a shared authenticated
   request helper so discovery and streaming share auth/header ownership.
4. Add private cache state to `Provider`; keep zero-value test providers safe.
5. Keep `Models(ctx)` small: check cache/coalescing, perform one refresh, then
   clone the selected result.

### Offline tests

Use `httptest.Server` and fake OAuth credentials. Cover:

- exact GET path, query, JSON accept header, user agent, originator, account
  header, custom headers, and absence of a body;
- auth refresh retry uses the existing source and persistence path;
- `models` and `data` response shapes;
- visibility filtering, explicit internal-model exclusion, duplicate IDs,
  malformed rows, and empty-catalog rejection;
- live-only membership with static/embedded metadata enrichment;
- unknown reasoning efforts retained in `Raw` but omitted from typed efforts;
- first call fetches; repeated calls inside 30 minutes do not;
- expiry causes one refresh using an injected clock, without sleeps;
- concurrent callers cause one server request and receive isolated clones;
- a waiting caller can cancel without canceling the shared refresh needed by
  another active caller;
- stale-success fallback and initial static fallback on transient/protocol
  failures, with five-minute retry suppression;
- auth and context errors remain visible and are not negative-cached;
- `New`, `Chat`, and `ChatStream` never call `/models`.

### Verification

```sh
go test -race -count=1 ./providers/openaicodex
go test -race -count=1 ./providers/internal/responsesapi ./llmtest
```

Optional credentialed verification, never ordinary CI:

```sh
go test -count=1 -tags=live ./internal/e2e -run 'OpenAICodex.*Models'
```

STOP if the live endpoint requires a materially different authentication flow,
returns account-sensitive fields that cannot safely be retained in `Raw`, or
requires changing the ordinary chat wire merely to list models. Record the
observed contract before revising this plan.

## Phase 3: Populate live model capabilities conservatively

### Codex

Map only explicit positive catalog evidence:

- tool support from an explicit tool/tool-call flag or supported parameter;
- reasoning from supported reasoning levels or an explicit reasoning flag;
- image input, prompt caching, stop sequences, structured output, and other
  capabilities only when the row explicitly advertises the corresponding
  feature.

Do not copy every provider-wide capability onto every live row, and do not
infer capabilities from a `gpt-*` family name. Keep unmodeled catalog fields in
`Raw`.

### OpenRouter

Extend its existing `/models` row decoder while retaining the full raw row:

- `supported_parameters` containing `tools` maps to `CapabilityTools`;
- explicit reasoning parameters map to `CapabilityReasoning`;
- an explicit stop-sequence parameter maps to `CapabilityStopSequences`;
- an input modality of `image` maps to `CapabilityImageInput`;
- structured-output and PDF capabilities map only when the upstream field is
  unambiguous and fixture-backed. Do not guess from generic `response_format`
  or `file` spellings.

Add table-driven mapping tests with unknown fields and parameters. Unknown
values must remain in `Raw` and must not fail the entire model list.

### Other providers

Anthropic, public OpenAI, vLLM, Ollama, and generic Chat Completions may leave
`ModelInfo.Capabilities` empty when their model-list response lacks reliable
per-model data. Empty is the designed unknown state. Do not add name inference
or a manually curated capability matrix in this plan.

### Acceptance

- Catalog fields produce deterministic, deduplicated capability slices.
- Unknown or absent catalog fields do not create negative claims.
- `ValidateRequest` still consults only provider-level capabilities.
- Existing `ModelInfo.Raw` fidelity tests continue to pass.

```sh
go test -race -count=1 ./providers/openaicodex ./providers/openrouter ./llmtest .
```

## Phase 4: Synchronize docs and run final gates

Update:

- `README.md`: model listing may carry advisory model capabilities; Codex
  listing is explicit live discovery with caching and static fallback.
- `specs/projects/go-llm/functional_spec.md`: replace the stale statement that
  Codex has no model-list endpoint; specify the explicit-only cache contract.
- `specs/projects/go-llm/architecture.md`: update `ModelInfo`, Codex provider
  layout, cache ownership, and advisory-vs-authoritative capability semantics.
- `specs/projects/go-llm/provider_capabilities.md`: record which providers
  currently expose positive per-model metadata.

Keep the existing `PromptTemplate` documentation. Do not rewrite history replay
or provider switching as part of this plan.

### Final verification

```sh
cd /home/peter/Dev/pkieltyka/go-llm
gofmt -w <only changed Go files>
make build
make test
go vet ./...
go test -count=1 -tags=live -run '^$' ./...
```

Review the diff for secrets and scope:

```sh
cd /home/peter/Dev/pkieltyka/go-llm
git diff --check
git status --short
```

## Suggested commit sequence

If the work is committed in phases, use reviewable boundaries:

1. `models: add advisory per-model capabilities`
2. `openaicodex: discover and cache account model catalog`
3. `models: map live Codex and OpenRouter capabilities`
4. `docs: synchronize model discovery and capability contracts`

Do not combine unrelated dependency changes with these commits.

## Deferred work

- `EffortUltra` and adapter mappings.
- OpenAI reasoning-summary selection.
- Service-tier metadata in the unified model surface.
- Model-aware fail-fast request validation.
- Embedded model capability ingestion and snapshot generation.
- Runtime ETag/disk caches, background refresh, and cross-process cache
  sharing.
- Conversation replay/history behavior changes.
- Any change to `PromptTemplate`.

## STOP conditions

Stop and ask for a design decision if:

- correct capability semantics require treating absence as authoritative
  “unsupported” rather than advisory unknown;
- the Codex models endpoint cannot share the existing OAuth source and header
  ownership without duplicating refresh logic;
- catalog discovery requires startup/background networking or a global cache;
- a remote response contains secrets/account-private fields that would be
  exposed through `ModelInfo.Raw`;
- normal offline verification would require live credentials or public network
  access.

## Definition of done

- [x] All confirmed decisions and boundaries above remain true.
- [x] Explicit repeated `Models` calls stay within the cache/failure-suppression
  request bounds, including under concurrency.
- [x] Authentication and cancellation failures remain observable.
- [x] All returned mutable model metadata is defensively copied.
- [x] Model capabilities are useful positive metadata but do not affect request
  validation.
- [x] Public docs and authoritative specs describe the shipped behavior.
- [x] go-llm passes its full offline build, race-test, vet, and live-tag compile
  gates.
- [x] No unrelated worktree changes, dependency upgrades, generated snapshots,
  credentials, or public-network fixtures are included.
