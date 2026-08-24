---
status: complete
planned_at: c983a92
planned_at_date: 2026-08-23
---

# OpenRouter and Compatible-Endpoint Correctness Plan

This plan implements the localized correctness work found during the August
2026 comparison of go-llm with other provider libraries. It is written against
go-llm commit `c983a92addf6f1b738e969c16f8b4f4b079fd0a5`.

The work stays inside go-llm's provider-mechanics and model-metadata boundary:
safe tool-schema serialization for OpenAI-compatible endpoints, truthful
OpenRouter pricing, and richer advisory reasoning metadata from an explicit
OpenRouter `Models(ctx)` call.

An executor must read this plan completely before changing code. Use the live
tree as authoritative when line numbers drift, preserve unrelated worktree
changes, and stop rather than broadening request policy or network behavior
when a STOP condition below is met.

At planning time `models.json` contains user-owned staged changes from a model
snapshot refresh. Those changes are outside this plan. Do not unstage,
regenerate, rewrite, or otherwise absorb them into this work unless the
maintainer separately authorizes it.

## Evidence baseline and dispositions

The comparison evidence for this plan is pinned so later source drift cannot
change what this plan claims to address:

| Finding | Pinned evidence | Disposition |
|---|---|---|
| Compatible servers may require object-valued `parameters.properties` for no-argument tools. | Zero `ad34dc8d81daa6e2c171df4c237b14aff8561ff9`, `internal/providers/openai/provider.go`, no-argument tool normalization | Implement in Phase 1 at the shared Chat Completions boundary. |
| Negative catalog sentinels are unknown prices rather than billable negative USD rates. | Zero `ad34dc8d81daa6e2c171df4c237b14aff8561ff9`, `internal/providermodelcatalog/remote.go`, remote price normalization | Implement in Phase 2 while preserving valid zero/free pricing. |
| OpenRouter exposes supported/default/mandatory reasoning metadata. | oh-my-pi `160ed439ac0df594347e7d7018b813a7ffdb5e81`, `packages/catalog/src/provider-models/openai-compat.ts`, OpenRouter reasoning mapping | Implement live advisory mapping in Phase 3. |
| OpenRouter can fall back to another remote catalog. | Zero comparison behavior | Rejected: `Models(ctx)` remains one explicit OpenRouter request; no models.dev fallback or hidden refresh. |
| Embedded snapshots could also carry default/mandatory reasoning metadata. | go-llm embedded model table | Deferred: no authoritative snapshot source or complete generator/readback contract is part of this work. Do not add dormant table fields. |
| Content-stall watchdogs can bound silent streams. | Zero `ad34dc8d81daa6e2c171df4c237b14aff8561ff9`, `internal/providers/providerio/providerio.go` | Deferred: reasoning streams can be legitimately silent and the existing implementation plan already records this separately. |

PromptTemplate, trace replay/model switching, agent orchestration, routing,
fallback policy, and persistent caches were reviewed but are not defects
addressed by this plan. PromptTemplate remains part of go-llm.

## Confirmed decisions

1. **OpenRouter discovery remains explicit-only.** This plan may use metadata
   already returned by `(*openrouter.Provider).Models(ctx)`. It must not add a
   background refresh, implicit fetch from `New`, `Chat`, or `ChatStream`, or a
   second network source such as models.dev.
2. **Per-model metadata remains advisory.** Reasoning efforts, defaults, and
   mandatory-reasoning state help selection and UI. They must not become
   model-aware preflight rejection, automatic effort clamping, or silent
   request rewriting.
3. **Zero-dollar prices are valid.** OpenRouter has free models. Only negative,
   malformed, or non-finite catalog prices are invalid; explicit zero must not
   be discarded as though it were missing.
4. **Unknown provider data remains reachable.** Preserve the complete copied
   OpenRouter row in `ModelInfo.Raw`, including reasoning fields go-llm does not
   normalize.
5. **No-argument tool normalization is structural, not heuristic.** Missing or
   JSON `null` `properties` becomes `{}`. A present non-object `properties`
   value is a bad schema and returns `ErrBadRequest`; do not silently replace
   arrays, strings, or numbers.
6. **Do not redesign the common reasoning request surface.** `Request.Effort`
   and OpenRouter's existing effort wire mapping stay unchanged. In particular,
   an explicit `EffortNone` still maps as it does today and may be rejected by
   a mandatory-reasoning model; the newly exposed metadata lets the caller
   avoid that request.

## Outcomes

- [x] OpenAI-compatible no-argument tools serialize
  `parameters.properties` as an empty JSON object, never omitted or `null`.
- [x] Malformed non-object tool `properties` fails locally as `ErrBadRequest`
  without a network call.
- [x] OpenRouter model discovery never exposes negative or non-finite pricing,
  preserves explicitly free pricing, and maps cache-read/cache-write rates.
- [x] OpenRouter model discovery maps advertised reasoning effort ladders,
  default effort, and mandatory-reasoning state into advisory `ModelInfo`
  fields while retaining the original row in `Raw`.
- [x] `llm-cli models` exposes the new typed reasoning metadata in text and JSON
  output without making an additional request.
- [x] Offline tests cover wire shape, invalid schemas, pricing sentinels,
  reasoning metadata normalization, caller isolation, and the no-hidden-fetch
  boundary.

## Priority, effort, and dependencies

| Phase | Priority | Effort | Risk | Depends on |
|---|---|---:|---|---|
| 1. Compatible no-argument tool schemas | P1 | S | LOW-MED | none |
| 2. Truthful OpenRouter pricing | P1 | S | LOW | none |
| 3. Advisory OpenRouter reasoning metadata | P1 | S-M | MED | none; coordinate public fields with Phase 4 docs |
| 4. CLI, docs, and final verification | P1 | S | LOW | Phases 1-3 |

Phases 1 and 2 are independent quick fixes. Phase 3 changes the additive public
`ModelInfo` surface and should receive an API-focused review before Phase 4 is
declared complete.

## Repository state and drift check

Before implementation:

```sh
cd /home/peter/Dev/pkieltyka/go-llm
git status --short --branch
git diff --cached --stat
git diff --stat c983a92addf6f1b738e969c16f8b4f4b079fd0a5
make test
```

Expected at planning time: `models.json` is staged by the maintainer. Treat it
as unrelated, pre-existing work. If an implementation path overlaps that file,
stop and isolate the code change without regenerating or rewriting the
snapshot.

Record and preserve these planning-time object IDs:

```text
index  models.json: dd552b022cd493159bbaaf7f69ccfb6f25e64ded
worktree models.json: dd552b022cd493159bbaaf7f69ccfb6f25e64ded
```

Before and after every implementation commit, re-run:

```sh
git rev-parse :models.json
git hash-object models.json
git diff --cached -- models.json
```

Both object IDs must remain unchanged. While this unrelated path is staged,
never use a plain `git commit`: use `git commit --only <explicit changed paths>`
with any necessary intent-to-add step for new files, or use a dedicated
temporary index. If that cannot be done safely, leave implementation changes
uncommitted for the maintainer.

### Maintainer disposition recorded 2026-08-24

Commit `1815a64` is already published and combined the planning artifacts with
the then-staged model snapshot despite the isolation rule above. The
maintainer's disposition is to preserve published branch history rather than
rewrite or split that commit. Treat it as a documented historical exception,
not a precedent: every future generated snapshot refresh must land in a
dedicated commit with its captured source inputs and provenance manifest.

## Boundaries and invariants

- Keep the root `llm` package standard-library-only.
- Do not add a provider or dependency.
- Do not add remote catalog fallback, disk persistence, ETags, background
  goroutines, or a global/per-provider OpenRouter model cache.
- Do not infer reasoning properties from model names or canonical slugs.
- Do not add embedded table or snapshot-generator support for the new live
  OpenRouter fields in this plan.
- Do not add model-aware behavior to `ValidateRequest`,
  `ValidateStreamRequest`, `Session`, or the OpenRouter request mapper.
- Preserve provider-level `Capabilities()` as the request-validation authority.
- Preserve unknown reasoning efforts and fields in `ModelInfo.Raw`; only known
  unified effort values enter typed metadata.
- Keep `ModelInfo.SupportedEfforts` ordered weakest to strongest regardless of
  OpenRouter's source order.
- Preserve defensive-copy guarantees for every mutable model field.
- Preserve caller-owned schemas and maps; normalization must operate on the
  decoded copy returned by `SchemaAsMap`.
- Do not change strict-schema fail-open behavior except for rejecting a
  structurally invalid non-object `properties` member.
- Do not change PromptTemplate, replay, history, switching, retry, OAuth, or
  stream behavior in this plan.

## Phase 1: Compatible no-argument tool schemas

### Current state

`providers/chatcompletions.buildTools` decodes `Tool.InputSchema` through
`providerutil.SchemaAsMap` and forwards the resulting map unchanged. A schema
such as:

```go
map[string]any{
    "type":       "object",
    "properties": map[string]any(nil),
}
```

therefore serializes `"properties": null`. Some otherwise compatible local
servers, notably LM Studio, reject that shape for a no-argument function.

### Contract

For tool parameter schemas sent through the shared Chat Completions adapter:

1. Decode through `SchemaAsMap` exactly as today.
2. Require the decoded root to be a non-nil JSON object. A typed-nil map or
   JSON `null` root is `ErrBadRequest`, not an empty schema.
3. The root `type` may be absent or the string `"object"`. Any present type
   that does not select an object schema is `ErrBadRequest`; do not add
   object-only keywords to a string, array, numeric, or boolean schema.
4. When `properties` is absent or its decoded value is nil, install a new empty
   `map[string]any{}` so the wire contains `"properties": {}`.
5. When `properties` is already a JSON object, preserve its contents.
6. When `properties` is present and not a JSON object, return a wrapped
   `ErrBadRequest` naming the provider and tool. Issue no HTTP request.
7. Preserve every other schema keyword and value exactly.
8. Never mutate the caller's original map, `json.RawMessage`, or nested
   `properties` object.
9. Apply the normalization once in the shared Chat Completions tool builder so
   generic endpoints, OpenRouter, vLLM, and the Ollama preset receive the same
   structural guarantee.

Do not silently coerce a generally invalid tool schema into a different
schema. This phase fixes absent/null no-argument properties only.

### Tests

Add exact `BuildParams` and wire tests covering:

- omitted `properties` -> `{}`;
- typed-nil `map[string]any` -> `{}` and never `null`;
- `json.RawMessage` containing `"properties": null` -> `{}`;
- typed-nil root maps and `json.RawMessage("null")` return `ErrBadRequest`;
- absent root `type` is eligible, `type: "object"` is eligible, and explicit
  non-object root types return `ErrBadRequest`;
- an already-empty object remains an object;
- a populated properties object is unchanged;
- array/string/number/bool `properties` returns `ErrBadRequest` before the
  fixture sees a request;
- the caller's original map and nested property map remain unchanged;
- blocking and streaming requests use the same normalized tool definition;
- every streaming case drains the lazy iterator with `llm.Collect` (or ranges
  it to termination) before asserting wire shape or request counts.

Run:

```sh
go test -race -count=1 ./providers/chatcompletions ./providers/openrouter ./providers/vllm ./providers/ollama
```

### STOP conditions

- If the OpenAI SDK transforms an explicit empty map back into `null`, stop and
  use the existing raw-parameter override seam; do not fork or patch the SDK.
- If fixing the shape requires changing the public `Tool` type, stop. The
  decoded schema is already sufficient for this fix.
- If a test demonstrates a documented compatible endpoint requires a
  non-object `properties` value, stop and document the dialect-specific case
  rather than weakening the structural check globally.

## Phase 2: Truthful OpenRouter pricing

### Current state

`providers/openrouter.parsePricing` accepts every JSON-decodable float and
multiplies it by one million. OpenRouter uses negative catalog values on
dynamic/router rows, so `-1` currently becomes `-1000000` dollars per million
tokens. The live catalog also exposes `input_cache_read` and
`input_cache_write`, but the parser leaves the existing `ModelPricing` cache
fields empty.

### Parsing contract

Extend the private OpenRouter pricing row and parser for:

- `prompt` -> `InputPerMTok`;
- `completion` -> `OutputPerMTok`;
- `input_cache_read` -> `CacheReadPerMTok`;
- `input_cache_write` -> `CacheWritePerMTok`.

The parser must distinguish missing, valid, and invalid values:

1. Trim surrounding whitespace.
2. Parse only finite numeric values, and after multiplying by one million
   require the converted per-MTok value to remain finite and non-negative.
3. Explicit zero is valid and represents free pricing.
4. A negative prompt or completion value marks only that component as
   non-fixed/unknown; it must never become a negative rate or be presented as
   free. Preserve independently valid sibling components.
5. Apply the same independent degradation to cache-read and cache-write rates.
   JSON `null`, non-numeric JSON types, malformed strings, overflow, and
   non-finite values are unknown for that component and do not fail the row or
   complete catalog.
6. Record per-component availability in `ModelPricing.Availability`. This
   additive metadata distinguishes an explicit zero/free component from the
   zero value used for an unavailable component while retaining legacy pricing
   values and cost-estimation behavior. Return nil only when no component is
   valid.
7. Keep the complete wire pricing object in `ModelInfo.Raw`.

Do not add per-request, per-image, internal-reasoning, or tiered OpenRouter
pricing fields in this phase: the common `ModelPricing` type has no lossless
home for them. Raw remains the escape hatch.

### Tests

Cover at least:

- ordinary positive prompt/completion conversion;
- explicit zero/free prompt and completion returns non-nil zero pricing;
- `-1` prompt/completion returns nil pricing;
- positive token rates plus `-1` cache rate never expose a negative cache rate;
- cache-read and cache-write conversion;
- absent, whitespace, malformed, parse overflow, NaN, and infinity spellings;
- a value that parses as finite per-token but overflows during the per-million
  conversion;
- no negative field reaches `llm-cli` model rows or JSON;
- the copied raw row still contains every original pricing key.

Run:

```sh
go test -race -count=1 ./providers/openrouter ./cmd/llm-cli
```

### STOP conditions

- If current OpenRouter documentation assigns a billable negative meaning to a
  token price, stop and retain it only in `Raw`; common USD cost estimation
  cannot represent a negative billable rate safely.
- If tiered pricing appears in the live `/models` response during
  implementation, record a focused follow-up instead of silently flattening
  tiers in this phase.

## Phase 3: Advisory OpenRouter reasoning metadata

### Public metadata contract

Extend `llm.ModelInfo` additively with:

```go
// DefaultEffort is the provider-advertised default reasoning effort.
// Empty means unknown. It is advisory and never changes request forwarding.
DefaultEffort Effort

// ReasoningRequired reports a positive provider claim that reasoning cannot
// be disabled for this model. False means false or unknown; it never causes
// client-side request rejection or rewriting.
ReasoningRequired bool
```

Keep `SupportedEfforts` as the canonical effort ladder rather than introducing
a second nested reasoning-policy type. These fields are populated only from the
explicit live OpenRouter `Models(ctx)` response in this plan. Do not change the
embedded table, `models.json`, its generator/readback path, canonical fallback,
or other providers' model enrichment. Snapshot ingestion requires a separate
plan that can land its data source and end-to-end preservation contract
together.

### OpenRouter live mapping

Parse the optional model-row `reasoning` object:

- `supported_efforts`;
- `default_effort`;
- `mandatory`.

Mapping rules:

1. Omitted `reasoning` or omitted `supported_efforts` means unknown; expose a
   nil typed effort slice.
2. An explicit `supported_efforts: null` means all gateway effort values are
   accepted. Populate all known unified levels in weakest-to-strongest order;
   when `mandatory` is true, omit `none`.
3. An explicit empty array means the provider supplied a ladder but no known
   efforts are available; expose a non-nil empty typed slice.
4. A non-empty array is filtered to known `Effort` values, deduplicated, and
   sorted by the unified weakest-to-strongest order. Unknown-only arrays become
   a non-nil empty slice. Unknown future values remain in `Raw`.
5. `mandatory: true` sets `ReasoningRequired` and removes `none` from every
   typed effort ladder, including explicit arrays. False, null, omitted, or a
   malformed value leaves the public bool false/unknown.
6. Populate a known `default_effort` only when it is consistent: it must be in
   a non-empty typed ladder when one is available, and it must never be `none`
   when reasoning is mandatory. When the ladder is omitted/unknown, a known
   non-`none` default may still be exposed; an explicit empty or unknown-only
   ladder leaves the default empty. Contradictory and unknown defaults remain
   only in `Raw`.
7. Treat the complete `reasoning` value and its children as advisory. A scalar
   `reasoning`, wrong-shaped effort list, mixed-type effort array, non-string
   default, or non-boolean mandatory value must not fail the row or the catalog.
   Preserve the copied row in `Raw`, omit only the malformed typed field, and
   continue mapping other independently valid advisory fields.
8. Positive live data is authoritative. Do not infer from the model name and
   do not issue an additional call to fill gaps.
9. Preserve catalog row order.

### CLI surface

Extend `llm-cli models`:

- JSON rows add optional `default_effort` and `reasoning_required` fields.
- JSON pricing fields are emitted only for known components, so explicit zero
  remains visible while unavailable prices are omitted. Cache-read and
  cache-write pricing use their own fields.
- Text output adds compact `DEFAULT EFFORT` and `REASONING REQUIRED` columns
  after `EFFORTS` and before `CAPABILITIES`.
- Text pricing has separate input, output, cache-read, and cache-write columns;
  unavailable components remain blank and explicit zero prints as free.
- Empty/false metadata remains visually empty rather than printing a misleading
  negative assertion.

### Tests

Add table and provider fixtures covering:

- effort arrays arriving in descending, ascending, duplicate, and mixed-known
  order;
- omitted, explicit null, explicit empty, and unknown-only effort lists;
- mandatory arrays that contain `none`, plus optional reasoning;
- known defaults with an omitted ladder, known defaults with an explicit empty
  ladder, unknown, off-by-default, and contradictory defaults;
- scalar reasoning objects and wrong-shaped/mixed-type child fields preserve
  the row while omitting only invalid typed metadata;
- unknown future effort values retained in Raw but omitted from typed fields;
- defensive copies of effort slices;
- CLI text and JSON fields;
- `New`, `Chat`, and a fully consumed `ChatStream` never call `/models`; the
  stream fixture records one chat request and zero `/models` requests;
- explicit `Models(ctx)` makes exactly the existing one catalog request;
- `EffortNone` request mapping remains unchanged and no model-aware preflight
  is introduced.

Run:

```sh
go test -race -count=1 . ./providers/openrouter ./cmd/llm-cli
```

### STOP conditions

- If naming either public field reveals a conflict with an already released
  API or documented external consumer, stop for maintainer review instead of
  adding aliases.
- If representing OpenRouter's current response requires transport mode,
  provider-specific wire routing, or automatic effort selection, keep those
  details in Raw and stop before expanding this plan's public API.
- If the only available source for mandatory/default metadata becomes a second
  remote catalog, stop. One explicit OpenRouter request is the network budget.

## Phase 4: Documentation and final verification

Update:

- `README.md` model-discovery/CLI examples;
- `specs/projects/go-llm/functional_spec.md` ModelInfo and advisory-metadata
  contracts;
- `specs/projects/go-llm/architecture.md` model discovery, pricing, and
  normalization details;
- `specs/projects/go-llm/provider_capabilities.md` only if its OpenRouter model
  discovery description needs the new fields;
- package comments for Chat Completions/OpenRouter where they describe tool or
  catalog behavior.

Document explicitly:

- model discovery is caller-triggered network I/O;
- no models.dev fallback is performed by `Provider.Models`;
- `ReasoningRequired` is advisory and does not override `Request.Effort`;
- free pricing is represented by a non-nil zero-rate `ModelPricing`;
- invalid/dynamic pricing is unknown, not negative and not free;
- no-argument compatible tool schemas are normalized to an object-valued
  `properties` field.

Final commands:

```sh
gofmt -w <changed-go-files>
go vet ./...
make test
make check
git diff --check
git status --short
```

Review the final diff specifically for:

- accidental `models.json` changes beyond the maintainer's pre-existing staged
  snapshot;
- hidden network calls or caches;
- zero/free pricing accidentally treated as unknown;
- negative price leakage;
- caller schema mutation;
- model-aware request gating;
- docs claiming metadata is authoritative enforcement.

Before each suggested commit and once more at final verification, confirm the
recorded index/worktree object IDs for `models.json` are unchanged. Use only
path-limited commits or a temporary index; a plain `git commit` is prohibited
while the unrelated snapshot remains staged.

## Acceptance criteria

- [x] Compatible no-argument tools have exact `{}` properties on blocking and
  streaming wire paths.
- [x] Malformed properties fail before network I/O.
- [x] OpenRouter catalog pricing is never negative; explicit free rows remain
  distinguishable from unknown pricing by a non-nil `Pricing` pointer.
- [x] Cache read/write rates populate the existing common fields when valid.
- [x] OpenRouter reasoning efforts, default, and mandatory state are accurately
  normalized from the single explicit catalog response.
- [x] Unknown future OpenRouter data remains available in Raw.
- [x] No request is rejected or rewritten because of model metadata.
- [x] No new background, startup, chat-time, or fallback catalog traffic exists.
- [x] Focused tests, `make test`, `make check`, vet, and diff checks pass.
- [x] User-owned staged snapshot work remains intact and separate.

## Suggested implementation commits

1. `Normalize compatible no-argument tool schemas`
2. `Harden OpenRouter model pricing`
3. `Expose advisory OpenRouter reasoning metadata`
4. `Document provider correctness contracts`

Keep commits separate from the maintainer's staged `models.json` snapshot
unless the maintainer explicitly asks for one combined commit. Each commit must
name its paths explicitly (or use a dedicated temporary index) and must be
followed by the object-ID and cached-diff checks above.
