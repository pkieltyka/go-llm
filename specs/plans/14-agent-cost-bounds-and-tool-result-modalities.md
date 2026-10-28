---
status: implemented
ratified_date: 2026-10-28
planned_at: c1915884b931e2ea11eb6b07b266af5072341685
planned_at_date: 2026-10-28
---

# Agent cost bounds and tool-result modalities

## Outcome and priority

Address the two Go-LLM boundaries exposed by Go-Agent Plan 030's test harness:

1. **First: make strict cost preflight usable where the complete applicable
   pricing is actually known.** Preserve rejection when a possible charge is
   unknown. Investigate model/request-specific billing semantics before adding
   an estimator or changing catalog normalization.
2. **Second: make tool-result modality support precise and testable.** Extend
   an adapter only when its remote protocol supports the representation.
   Image input alone does not establish image-bearing tool-result support.

Neither item blocks the completed deterministic Go-Agent harness. The first
improves strict estimated-cost admission; the second is conditional work for
screenshot/image-returning tools on additional transports. A documented,
tested unsupported outcome is a valid resolution of the second item.

This is a planning document, not authorization to change existing accounting
contracts or a claim that remote support has been verified by live probes.
Keep the library small: reuse pricing, provider adapters, optional provider
extensions, compatibility settings, and existing conformance tests.

## Evidence and corrections to the initial diagnosis

Baseline: Go-LLM v0.6.2 at the SHA above. Go-Agent evidence is in
[PR #5](https://github.com/pkieltyka/go-agent/pull/5), especially
`internal/e2e/README.md`, `TestStrictPreflightRejectsUnknownPriceComponents`,
and `TestOpenRouterEscapedMarkerAndImageBoundary`. The Go-Agent report records
actual OpenRouter `openai/gpt-4.1-mini` execution; its image-tool-result attempt
failed locally with `ErrUnsupported`, before the continuation HTTP request.
No new paid requests or credential resolution were performed to write this plan.

### Cost preflight

- `context_fit.go:EstimateCostBound` evaluates uncached-input, cache-read, and
  cache-write interpretations and takes the maximum. An unknown needed rate
  returns `ok=false`; this is the current documented contract, not a proven
  arithmetic defect.
- Go-Agent reproduced an unknown bound for the selected GPT-4.1-mini pricing
  with an 8,192-token prompt and 256-token output allowance. Its strict mode
  therefore fails preflight. Explicit call/HTTP/time mode remains available
  without claiming a dollar bound.
- **Do not generalize this to "OpenAI has no cache writes."** Existing
  `TestOpenAICacheWriteUsageFixture` maps OpenAI cache writes, and
  `TestEmbeddedTableGPT56CacheWritePricing` requires a real write rate.
- Current [OpenAI prompt-caching documentation](https://developers.openai.com/api/docs/guides/prompt-caching)
  distinguishes earlier models from GPT-5.6-and-later billing and describes
  request-dependent cache lifetimes. [OpenRouter's caching explanation](https://openrouter.ai/blog/tutorials/prompt-caching-sticky-routing/)
  also distinguishes model generations. These sources support investigating
  exact model/request applicability; they do not justify a provider-wide
  exemption, a made-up rate, or treating absent metadata as known zero.

### Rich tool results

- FS §7 explicitly supports native rich tool results on Anthropic and OpenAI
  Responses. It explicitly rejects image/file tool results on Chat Completions.
- `providers/chatcompletions/convert.go:toolResultContent` and
  `providers/internal/providerutil/helpers.go:ToolResultText` enforce that
  boundary. `ToolMessageContentBlocks` preserves text/cache blocks; it is not
  an image-support flag.
- `TestToolResultImageStaysUnsupported` deliberately pins this rejection.
  `providers/internal/responsesapi/adapter_replay_test.go` already proves a
  native Responses content-array mapping. Do not implement that support twice.
- The inspected [OpenRouter API overview](https://openrouter.ai/docs/api_reference/overview)
  gives `role: "tool"` string content and distinguishes it from image-bearing
  user content. This is evidence against enabling images through the existing
  Chat Completions preset without further protocol proof. Recheck its current
  OpenAPI schema and endpoint-specific documentation during implementation.

The initial follow-up description was too broad if interpreted as a confirmed
OpenRouter mapping bug or as permission to ignore all OpenAI cache-write prices.
This plan narrows both claims to the evidence above.

## Existing contracts to preserve

Read these before implementing:

- [Functional specification](../projects/go-llm/functional_spec.md): §7 tools,
  §11 usage/pricing, §12 capabilities, §13 catalog, §13A context/cost helpers,
  and §15 prompt caching.
- [Architecture](../projects/go-llm/architecture.md): provider conversion,
  pricing/catalog provenance, and context-management primitives.
- [Cost completeness amendment](../amendments/2026-10-27-cost-completeness-openai-image.md).
- Plans [9](9-replay-overflow-and-cache-accounting.md),
  [10](10-context-fitting.md), and
  [11](11-positional-instructions-cache-retention-telemetry.md).

In particular:

1. Known zero, unavailable price, native cost, estimated cost, and incomplete
   subtotal remain distinct. Preserve `Usage.CostIncomplete` and its aggregation
   and persistence behavior. A projected bound does not repair actual usage.
2. Keep the existing `EstimateCostBound` signature and conservative semantics.
   A pricing-only function cannot safely infer endpoint or authentication mode.
3. Preserve tier selection, cache-write subsets, and retention-specific unknown
   charges. No generic multiplier may be substituted for missing source pricing.
4. A cost estimate over estimated prompt tokens is not a billing guarantee.
   An unenforced output cap, unknown extra charge, or unsupported endpoint
   invalidates a claim of a complete request bound.
5. Preserve caller-owned request data, transcript roles/order, tool-call IDs,
   error sentinels, credential isolation, and silent defaults.

## Part A — applicable cost-bound calculation

### A0. Establish the exact billing domain

Create a small checked-in evidence table for representative requests:

| Case | Evidence required | Safe initial disposition |
| --- | --- | --- |
| Official OpenAI API-key GPT-4.1-mini | Exact model/snapshot, normal input/read charging, applicability of a separately reported write category | Unknown until the documented exclusion and catalog meaning are established. |
| Official OpenAI newer model with writes | Default and explicitly selected retention, relevant write rates and usage fields | Include every applicable write category; missing rate remains unknown. |
| OpenRouter exact model | Published pricing, allowed provider routes, relevant request overrides and extra charges | No inheritance from the direct OpenAI endpoint by model-name substring. |
| Anthropic default/long retention | Applicable default and one-hour write semantics | Preserve rejection of an unpriced possible one-hour write. |
| OpenAI subscription OAuth | Actual output-cap behavior and available billing authority | No complete token/dollar-bound claim from a requested cap the transport omits. |
| Custom gateway, vLLM, unknown model | Explicit endpoint billing authority or caller-supplied documented pricing | Unknown by default; an API-compatible URL does not prove billing equivalence. |

Record source URLs and retrieval dates, endpoint, auth kind, exact model match,
request options, catalog availability, and assumptions. Do not record keys or
private endpoint values. A missing SDK field is not proof that no charge exists.
Check request-level and part-level cache settings and provider options.

Use the existing snapshot/model-source pipeline for any supported catalog
correction. Do not curate a parallel price table. In particular, "no additional
write fee" must not be encoded as free prompt processing or as a zero rate for
tokens that belong in the ordinary input category.

**A0 exit:** classify each candidate as a catalog defect, a request-applicability
gap, or genuinely insufficient pricing. Freeze the smallest design justified
by that table before shipping a new exported API.

### A1. Implement only the supported correction

Preferred order:

1. Fix source ingestion/normalization if authoritative data is being lost.
   Preserve provenance and unknown-versus-zero semantics in fixtures.
2. If catalog correction cannot express applicability, prove a private
   provider-owned calculation first. It must consider the effective request,
   exact model, endpoint mode, applicable prices, and effective output cap.
3. Expose a narrow additive provider-aware operation only if Go-Agent needs
   to consume that proven calculation. Follow the existing optional-extension
   and wrapper-discovery style used by `TokenCounter`; do not require every
   Provider to implement a new method or add a general billing-policy engine.
   Specify the public signature, unknown/error behavior, and wrapper discovery
   in an FS/architecture amendment before implementing that public surface.

The operation must not dispatch a generation, execute key commands, refresh
credentials, or silently initiate catalog discovery just to calculate a bound.
Use already-resolved inputs or explicitly caller-requested metadata. Unknown
models, ambiguous routes, unsupported overrides, or missing applicable rates
remain unknown. The result is scoped to the request assumptions used; changing
those assumptions requires a fresh estimate.

If A0 cannot establish a safe useful correction, retain current behavior and
close Part A with a precise limitation and tests. Do not add an unused public
extension merely to obtain a green Go-Agent strict-mode test.

### A2. TDD and acceptance matrix

Add the smallest failing regression before a justified runtime correction:

- Synthetic known input/output/read pricing plus unknown write pricing remains
  unknown through the existing generic helper.
- A documented inapplicable category is excluded only by the new exact scoped
  path; an otherwise identical request where writes apply remains unknown or
  includes their full price.
- Preserve the existing newer-OpenAI write-usage and pricing fixtures. Include
  cache-hit, miss, mixed usage, and retention overrides where supported.
- Missing/negative/dynamic/non-finite applicable rates reject the bound;
  explicit known zero remains usable. Exercise negative token counts, zero
  counts, large finite counts, overflow and tier boundaries.
- Across supported input/read/write mixtures, the projected value covers each
  applicable `EstimateCost` result at the same prompt/output ceilings and tier.
  Do not double-count informational subsets or assume one-hour pricing exists.
- Unknown routed/custom endpoints and OAuth with an omitted output cap cannot
  inherit an official API-key bound. Estimation sends zero generation requests.
- Neither estimation nor aggregation loses an existing incomplete-cost signal.
- If a provider extension is added, test wrapped providers and unchanged
  providers, and retain source compatibility of the old helper.

Prefer synthetic tables and provider-owned fixtures over volatile catalog-ID
expectations for arithmetic. Retain one captured catalog reproduction to prove
the actual Go-Agent limitation and its corrected or intentionally unknown result.

## Part B — explicit tool-result modality boundaries

### B0. Verify protocols and improve the support explanation

Build a concise endpoint-specific table: text tool results, text/cache blocks,
native image/PDF tool results, and unknown custom endpoints. Reuse
`specs/projects/go-llm/provider_capabilities.md`; model `image-input` remains a
separate fact from tool-result serialization support.

For OpenRouter, inspect the exact Chat Completions request schema and supported
provider-route behavior. Generic vLLM compatibility is not proof of equivalent
support on every served model/version. Record the wire shape and source for
each positive claim. Existing Anthropic/Responses support is the reference for
preserving rich results, not evidence that another endpoint accepts it.

For the inspected text-only Chat Completions path, retain `ErrUnsupported`
before HTTP, document the distinction prominently, and test the OpenRouter
preset itself in addition to the shared adapter. Keep the existing generic
negative test. This is useful completion even if B1 stays deferred.

### B1. Conditional provider-native implementation

Proceed only for an endpoint with authoritative protocol support. Amend FS §7
and architecture for that exact endpoint before relaxing a rejection. Reuse
the current adapter or a narrow explicit compatibility setting if one is
needed; the default for generic Chat Completions stays unchanged.

Preserve all tool results, their source order, IDs, text/image ordering, and
supported cache hints. Honor existing URL/data/MIME handling and ownership.
Unsupported mixed parts fail atomically before dispatch. Test both blocking
and streaming request construction where the adapter has separate paths.

Do not flatten image bytes into prose, silently drop them, invent a user
message, switch endpoint/provider, or add a new Responses preset as a shortcut.
Such behavior changes require a separate justified design. No broad capability
registry, arbitrary JSON escape hatch, or SDK dependency is needed here.

### B2. TDD and acceptance matrix

- Text-only tool results preserve their current wire shape byte-for-byte
  where identity matters. Existing text/cache-block replay remains valid.
- The OpenRouter preset and generic Chat Completions path reject unsupported
  image/PDF and mixed content with `errors.Is(err, llm.ErrUnsupported)` and
  zero HTTP dispatch, regardless of advisory model image capability.
- Existing Anthropic and Responses rich-result fixtures continue to pass.
- For any newly supported endpoint, add exact wire fixtures for text plus
  inline image, URL image and supported file forms; multiple results and IDs;
  value/pointer parts; unsupported MIME/parts; and caller-data ownership.
- Positive fixtures prove the image is present in the follow-up request with
  the original tool association. A model answering from prior text is not an
  image-consumption oracle.
- Use the existing conformance/replay suites and live harness. No new general
  test framework or duplicate live loop.

An optional live probe follows offline proof, using explicit opt-in and one
selected API-key target with finite generation/HTTP/time limits. No credentials
or raw provider payloads in tracked evidence; no automatic whole-probe retries.
Unsupported or unavailable targets remain explicit non-passes. No live request
is required for the documentation/negative-conformance resolution of B0.

## Phase 0 findings (2026-10-28)

No paid requests and no credentials were used. The sources were read on
2026-10-28.

### A0: cost applicability

Reproduced from the embedded catalog: `EstimateCostBound(8192, 256, p)`
returns `ok=false` for `openrouter/openai/gpt-4.1-mini`,
`openai/gpt-4.1-mini` and `openai/gpt-5.5`. Their input, output and
cache-read rates are known, but the cache-write rate is not.
`openai/gpt-5.6-luna` publishes a write rate, so its bound is known
(0.0023552 USD).

| Case | Evidence | Classification |
| --- | --- | --- |
| Official OpenAI, models before GPT-5.6 (e.g. gpt-4.1-mini) | [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching): "No additional cache-write charge" for "GPT-5.5 and GPT-5.5 Pro" and "Other earlier models"; "Cache-write pricing is not an additive fee". Retention (`in_memory`/`24h`) has no stated price difference. | **Request-applicability gap.** The write category does not exist for these models, so a written token bills as uncached input. The fact depends on the model and endpoint, not on request options. |
| Official OpenAI, GPT-5.6 and later | Same page: writes cost 1.25× (30m) and 2.5× (12h, Responses only). | Writes apply. A missing write rate stays unknown. The 12h rate is not in any catalog, so it remains unknown (unchanged). |
| OpenRouter `openai/*` | [OpenRouter prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching): writes have "no cost on models before the GPT-5.6 family"; GPT-5.6+ charges 1.25×. | Same gap, now documented by OpenRouter itself, so nothing is inherited from the direct endpoint. `openai/gpt-oss-*` is served by third parties and is excluded. |
| Catalog data | models.dev, OpenRouter and the snapshot carry no write rate for these models and correctly mark it unavailable. 41 `openai` and 78 `openrouter/openai/*` rows lack a write rate. Some of those must not be exempted (e.g. `openai/gpt-5.6-luna:batch`, image/audio/realtime/embedding models). | **Not a catalog defect.** Nothing is lost in ingestion. |
| Anthropic, OAuth, custom gateways | Unchanged. | Out of scope; they keep today's behavior. |

Missing a write rate is never treated as proof that writes are free.
Exemption needs a documented rule for the exact endpoint and model family.

### B0: tool-result modalities

| Endpoint | Text tool results | Text + cache blocks | Image/PDF tool results |
| --- | --- | --- | --- |
| Anthropic Messages | yes | yes | yes (native) |
| OpenAI Responses | yes | n/a | yes (`function_call_output` array) |
| OpenAI Chat Completions | yes | n/a | **no**. The official tool message content is a string or text parts only (openai-go `ChatCompletionToolMessageParamContentUnion`). |
| OpenRouter Chat Completions | yes | yes (live-verified) | **unproven**. The [API overview](https://openrouter.ai/docs/api_reference/overview) gives `role: 'tool'; content: string` and says content parts "are only for the user role". The current machine schema (`openrouter.ai/openapi.json`, `ChatToolMessage`) does accept a `ChatContentItems` array, including `image_url` and `file`. No source says the parts reach the model on a given upstream route. |
| vLLM / generic Chat Completions | yes | opt-in | no (model- and version-dependent) |

**B0 resolution:** keep `ErrUnsupported` before HTTP. Add the
OpenRouter-preset negative test, `TestOpenRouterRichToolResultsAreUnsupported`
(image, image URL, PDF, mixed and cache-hinted mixed; Chat and ChatStream;
zero requests). Record the source conflict in the docs. B1 is deferred: the
schema shows the request would be accepted, not that the image is used. A
B1 would need an opt-in live probe whose answer depends on the image itself,
run on each route it claims to support.

### A1 design (ratified 2026-10-28)

The smallest correction is one additive catalog fact. It needs no new
provider operation, because applicability for these models does not depend
on the request.

```go
type ModelPricing struct {
	// ...existing fields...

	// CacheWritesAsInput records a documented endpoint rule: the model has
	// no separate cache-write charge, so tokens written to the cache bill at
	// the (tier) input rate. It applies only while the write rate is
	// unavailable; a published write rate always wins. Absence keeps
	// writes unknown.
	CacheWritesAsInput bool `json:"cache_writes_as_input,omitempty"`
}
```

- `EstimateCost`: when the flag is set and the write rate is unavailable,
  `CacheWriteTokens` are priced at the applicable tier input rate and do
  not mark the cost incomplete. Without the flag the behavior is unchanged.
- `EstimateCostBound`: same signature. Its write interpretation then equals
  the input one, so the bound becomes known.
- The source is a snapshot-pipeline rule beside `scripts/model-behavior.ts`,
  citing both documents above. It is an exact allowlist of base families
  (`gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`,
  `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5.1`, `gpt-5.2`, `gpt-5.4`,
  `gpt-5.4-mini`, `gpt-5.4-nano`, `gpt-5.5`, plus dated snapshots), for
  `openai` and `openrouter` `openai/…`. Pro, codex, `:batch`,
  image/audio/realtime/embedding models, `gpt-oss` and all GPT-5.6+ models
  are excluded. The build fails if a matched row publishes a write rate.
- Like `behavior`, this is a documented rule rather than captured catalog
  data. It is a deliberate, narrow exception to "models.json holds captured
  catalogs only".

**Rejected alternatives:**

- A provider-aware `CostBounder` extension: a larger API, and these models
  don't need request-level applicability.
- Setting `cache_write_per_mtok` equal to the input rate: it invents a
  published price.
- A zero write rate: it underprices reported writes.

If the exception is declined, close Part A as a documented limitation and
add a test that pins it.

### What landed

- **Part A:** `ModelPricing.CacheWritesAsInput`, applied in `pricingForUsage`
  after tier selection without mutating the caller's pricing.
  `scripts/model-pricing-rules.ts` matches 35 rows: 17 `openai` and 18
  `openrouter` `openai/…`. The snapshot fails on a matched row that publishes
  a write rate or lacks an input rate. models.json was regenerated from the
  2026-10-27 sources; the only change is the new flag. FS §11/§13A and the
  architecture doc are amended.
- **Tests:** `TestCacheWritesAsInput` covers unknown without the rule, a
  published or explicit-zero write rate winning, an unknown input rate,
  one-hour writes, tier boundaries, and a sweep showing the bound covers
  every input/read/write mixture. `TestEmbeddedCacheWritesAsInputRule`
  covers the catalog allowlist and Go-Agent's case: OpenRouter GPT-4.1-mini
  at 8,192 + 256 tokens is now known, while GPT-5.6 `:batch` stays unknown.
  The snapshot script has rule and contradiction tests.
- **Part B:** B0 as recorded above, with FS §7 and provider_capabilities
  noting the OpenRouter case.
- **Independent review:** found no pricing-math bug. It caught that
  Go-Agent resolves pricing through `ResolveModelInfo`, where OpenRouter's
  live pricing wins and never carried the flag. `mergeModelInfo` now applies
  the table's rule to live pricing that publishes no write rate (without
  mutating the provider's value), tested by
  `TestResolveModelInfoCarriesCacheWritesAsInput`. Smaller follow-ups:
  - the FS wording was corrected;
  - the generator's loss guard now covers the field;
  - the mixture sweep now runs on tiered pricing too;
  - a test shows overrides can't set the flag;
  - a dead strip step was removed.
- **Second review (PR #55):** the flag leaked through fuzzy model
  resolution. `LookupModelInfo` gave it to
  `gpt-4.1-mini-2025-04-14:batch` and `gpt-4.1-mini-20250414-preview`
  through prefix matching. Separator spellings, canonical fills from
  another endpoint's row, and a `SameModel`-matched live `:batch` row in
  `ResolveModelInfo` leaked it too. The flag now follows only the row's
  own ID or that ID plus one `-YYYY-MM-DD` date, mirroring the generator
  rule, while fuzzy metadata lookup is unchanged. Tests:
  `TestCacheWritesAsInputStaysOnAuthorizedIDs`,
  `TestResolveModelInfoKeepsRuleOffExcludedLiveRows` and
  `TestCanonicalFallbackDropsCacheWritesAsInput`.

## Delivery order and integration handoff

- [x] **Phase 0: Evidence and design.** Complete A0/B0 tables; record whether
  either runtime change is actually justified. Keep catalog, protocol support,
  and request policy distinct. Ratify any necessary additive API/FS amendment.
- [x] **Phase 1: Cost preflight.** Deliver A1/A2, or documented retained
  unknown behavior if no safe correction is possible. Independent review.
- [x] **Phase 2: Tool-result boundary.** Deliver B0 docs/preset negatives;
  B1/B2 positive mapping only for a proved protocol. Independent review.
- [ ] **Phase 3: Consumer verification.** Publish the actual outcomes and
  remaining limitations. After a Go-LLM release, update Go-Agent's consuming
  tests in a separate change; no local `replace` dependency for release checks.

For Go-Agent, remove the L11 transport guard only after the released adapter
supports the exact path and its consuming regression passes. Enable strict
cost preflight only for proved targets. Keep call/HTTP/time fallback explicit,
and retain original failed live evidence. API-key improvements do not imply
OAuth output-cap enforcement. Nothing here reopens Anthropic subscription OAuth.

## Verification and scope limits

Implementation gates use the repository's current Makefile and pinned tools:

```sh
GOWORK=off make check
```

This includes build, vet, race, lint, model/source checks, live-tag offline
checks, coverage, vulnerability checks and fuzzing. Run focused red/green
regressions first. Live credentials and paid tests stay outside ordinary gates.
For a docs-only phase, validate links, contract consistency and whitespace;
do not claim runtime tests or live compatibility that were not executed.

Expected files, only as justified: `context_fit.go`/tests, provider-owned cost
or conversion helpers/tests, snapshot pipeline fixtures if ingestion is wrong,
provider docs/capabilities, and the owning FS/architecture amendment. Preserve
the existing cost-completeness amendment and historical plans. No Go-Agent
runtime/harness migration, monorepo restructuring, auth redesign, dependency
upgrade, or unrelated provider feature is part of this plan.

Completion means each reported limitation has an evidence-backed resolution:
a tested improvement or an explicit retained restriction. It does not require
manufacturing support where the protocol or pricing data cannot establish it.
