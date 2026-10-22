---
status: ready
ratified_date: 2026-10-22
planned_at: e94fcb3
planned_at_date: 2026-10-22
---

# Positional Instructions, Cache Retention, and Telemetry

**Ratified 2026-10-22.** The maintainer approved the recommended direction
for all three parts; the decisions are recorded under "Ratified decisions" in
each part. Part A remains gated by its live probes: if a probe contradicts
the chosen representation, STOP and report rather than improvising. The
parts came out of the pi v1.4.0 comparison (`/home/peter/Dev/other/pi`,
HEAD `476a303`) and all three change public surface. Execute them as
separate phases in the order C, B, A: C is additive, B is a small request
field, and A is the large request-model change.

## Part A: mid-conversation system prompt and tool-set changes

### Why

The prompt cache is a prefix. Today a `Request` carries one `System` and one
`Tools` set. When the system prompt or tool set changes mid-conversation, the
cached prefix is invalidated from the top. go-agent hits this whenever a skill
activates tools or an MCP catalog refreshes. go-agent `077b363`
`prompt.go:buildRequest` sends the filtered `catalog.defs()` even though
`ActiveToolsEntry` is positional in its log, and that conflicts with go-agent
FS tenet 10 (cache-stable prompts).

pi 0.86 made this its biggest new abstraction:

- **The transcript type.** Providers receive a `TranscriptContext` whose
  `SystemMessage`s carry `sections` (named, replaceable, null to delete),
  `toolsAdded`, and `toolsRemoved`. Replaying them reconstructs the current
  prompt and tools. See `packages/ai/src/types.ts` (SystemMessage) and
  `packages/ai/src/utils/transcript.ts` (`normalizeContext`,
  `getCurrentTools`, `collapseSystemMessages`, `resolveTranscriptTools`,
  `getToolStateChanges`).
- **Native wire support where it exists:**
  - Anthropic uses the `inline-tools-2026-09-15` beta, with
    `tool_addition`/`tool_removal` blocks and a stable placeholder tool
    (`packages/ai/src/api/anthropic-messages.ts`).
  - OpenAI Responses uses `additional_tools` items
    (`api/openai-responses-shared.ts`).
  - GPT-6 changes per-turn effort with `configuration_update` items.
- **Elsewhere,** everything is collapsed into one leading system message.
  Per-model compat flags decide: `supportsMidConvoSystemMessages`,
  `supportsMidConvoToolChanges`, `supportsMidConvoToolAdditions`, and
  `supportsMidConvoEffort` (`providers/compat-schema.ts`).

go-llm today has `RoleSystem`, documented as "provider-dependent". Anthropic
rejects it with `ErrUnsupported` (`providers/anthropic/convert.go:messageRole`).

### Design questions

1. **Representation.** Choose between:
   - (a) an extended `RoleSystem` message with typed `SystemDeltaPart`
     (sections) and `ToolsDeltaPart` (added and removed); or
   - (b) keeping `Request.System`/`Tools` as the baseline and adding an
     ordered `Request.Deltas` list anchored to message indexes.

   Option (a) matches pi and makes the transcript self-describing for
   go-agent persistence. Option (b) is additive and less invasive.
2. **Capability.** A new `CapabilityPositionalInstructions`, plus per-model
   advisory metadata. Folding is the universal fallback and is lossless for
   meaning, losing only cache benefit. It must be byte-deterministic.
3. **Serialization.** New part types join the canonical envelope (FS §10A)
   with a version bump policy.
4. **Validation.** A tool call naming a tool absent from the effective set at
   that position returns `ErrBadRequest` before send, or is left to the
   provider. pi validates against the replayed offered set.
5. **Interaction with `CacheHint`** breakpoints and with `llmtest`'s
   append-only prefix conformance (plan 2, phase 4b).

### Ratified decisions

1. **Representation: option (a).** `RoleSystem` messages become positional
   transcript entries carrying typed `SystemDeltaPart` (named sections;
   empty text deletes a section) and `ToolsDeltaPart` (added and removed
   tools). `Request.System` and `Request.Tools` remain the baseline, so
   existing callers are unchanged. This matches pi and keeps go-agent
   transcripts self-describing.
2. **Capability.** Add `CapabilityPositionalInstructions`. Folding into the
   leading system prompt is the universal, byte-deterministic fallback, so
   providers without native support still accept the messages; this
   replaces today's Anthropic `ErrUnsupported`.
3. **Serialization.** The new part types join the canonical envelope with an
   additive version bump.
4. **Validation.** A tool call naming a tool outside the effective set at
   its position is rejected locally with `ErrBadRequest` before send (pi
   validates against the replayed offered set the same way).
5. **CacheHint and prefix conformance.** Breakpoints on delta parts are
   honored where native, and `llmtest` prefix conformance gains
   delta-append cases.

### Evidence to gather before implementing (gate)

Probe live:

- Anthropic inline-tools beta availability, the exact wire shape, and
  cache-read tokens before and after a tool addition;
- OpenAI `additional_tools` on `gpt-5.6`, and on Codex if accepted.

Measure cache-read deltas with the existing prompt-cache live scenario
harness.

## Part B: cache retention modes

pi's `cacheRetention: none | short | long` maps to:

- Anthropic: no `cache_control`, 5m, or 1h;
- OpenAI: omitting or setting `prompt_cache_key`, plus
  `prompt_cache_options.ttl`;
- Codex: suppressing session identity.

pi-durable forces `none` for summarization calls so throwaway prompts don't
evict or pollute the cache.

go-llm has per-part `CacheHint{TTL}` and `SessionID`, but no way to say "do
not cache this request". OpenAI's automatic caching keys on
`prompt_cache_key`/`SessionID`.

**Ratified decision.** Add `Request.CacheRetention` (`""` meaning the
provider default, then `none`, `short`, `long`). It maps per provider as
listed above and composes with per-part `CacheHint`s (`none` suppresses
them). Model-level cache lifetime metadata is added only if upstream
(models.dev) publishes it; no curation.

## Part C: OpenTelemetry integration package

### Why now

anthropic-sdk-go v1.84 already brings OpenTelemetry into go-llm's module
graph, and go-agent defers its OTel exporter (go-agent FS §19). pi's
vendor-neutral, content-free vocabulary is a well-tested shape to copy
(`packages/ai/src/telemetry.ts`, `AI_TELEMETRY_SCHEMA`):

- **Span** `llm.request` per logical call.
  - Start attributes: provider, model, and effort.
  - End attributes: stop reason, token counts (input, output, cache read,
    cache write, reasoning), cost, response id.
- **Events:** `first_chunk`, plus one `block` event per content block,
  carrying only type, index, character count, and tool name. Never one
  per delta, and never content.
- **Metrics:** request duration, time to first chunk, tokens by kind, and
  cost.
- **Error status** is a fixed category plus a message truncated to 500 bytes,
  using `llm.SafeError` (go-llm is stricter than pi here). An aborted or
  cancelled request is not an error.

### Constraints

FS §10B and ARCH §10 keep integrations out of core.

- **Proposal:** a separate nested module,
  `github.com/pkieltyka/go-llm/contrib/otelllm`, with its own `go.mod`. It
  exposes `otelllm.Middleware(tracer, meter)` on the `llm.Wrap` seam, and the
  parent span comes from `context.Context` (Go already carries it; pi needed
  a field).
- **Rule:** no prompt, completion, reasoning, or tool-argument bytes ever
  appear in attributes or events. A secret-canary test enforces it.

### Ratified decisions

- **A separate nested module**, `github.com/pkieltyka/go-llm/contrib/otelllm`,
  with its own `go.mod`. The main module gains no direct OpenTelemetry
  requirement.
- **OTel GenAI semantic convention names** (`gen_ai.*`) wherever a
  convention exists, falling back to `go_llm.*` names only for concepts the
  conventions lack (for example cost and the per-block events). pi's own
  vocabulary is the content model, not the naming.

## Recorded non-adoptions from the same study

| pi feature | Reason |
|---|---|
| Deferred/background responses (`DeferredHandle`, `fetchDeferred`) | Only pi's fake provider implements it. Revisit alongside Batch APIs (FS §3 deferred). |
| Image-generation and classifier model types | Out of scope per FS §3. |
| Regex retry and overflow corpora as a consumer API | Typed sentinels are the API. The overflow corpus is used only inside adapters (plan 9). |
| Grammar-constrained tools (OpenAI Lark/regex custom tools), strict prefer/require per tool | Candidate for a later plan once there is a consumer. Today `Tool.Strict` plus `StrictSchemaSupported` covers JSON schema. |
| The `pi-messages` wire protocol | A reference for the possible inference gateway, not go-llm core. |

## Implementation status (2026-10-22)

**Part C: shipped** as `contrib/otelllm`, a nested module with its own
`go.mod`.

- **Naming source.** The OTel Go semconv packages carry the GenAI
  vocabulary only up to v1.41.0, which ships in OTel v1.47.0; v1.42 and
  v1.43 drop it. So the names are string constants following v1.41.0 GenAI
  (development status), with no semconv import. The plan's "`gen_ai.*`
  names where they exist" covered:
  - request: operation, provider, model, max tokens, temperature, top-p,
    and stream;
  - response: id, model, finish reasons, and time to first chunk;
  - usage: input (including cache, per the conventions), output, cache
    read, cache creation, and reasoning tokens;
  - `error.type`, and the client duration, token-usage, and
    time-to-first-chunk metrics.

  The `go_llm.*` names cover cost and its source, the raw provider,
  effort, tool count, dropped calls, canceled, abandoned, and the
  first-chunk and per-block events.
- **Behavior.** `openai-codex` reports as `gen_ai.provider.name=openai`.
  Cancellation and stream abandonment are not error status.
- **Tests.** The tests use the SDK in-memory exporter and a manual reader:
  attributes and metrics, streaming first-chunk, errors, cancellation, and
  abandonment, parent/child context propagation, and a canary test that
  content never appears in spans, events, status, or metrics.
- **Build integration.** The Makefile gained `CONTRIB_MODULES`, and build,
  test, vet, tidy, lint, and govulncheck all run inside the module.
  docs/release.md describes bumping the contrib `require` and tagging
  `contrib/otelllm/vX.Y.Z`.

**Part B: shipped** as `Request.CacheRetention` (`""`, `none`, `short`,
`long`).

- One shared `providerutil.ApplyCacheRetention` runs before conversion in
  the Anthropic, Responses, and chat-completions engines. It is
  copy-on-write and reaches nested tool-result hints.
- `none` also omits the Responses `prompt_cache_key`.
- OpenAI `prompt_cache_options` is deliberately not auto-set by `long`: it
  is documented for gpt-5.6 and later only, and server acceptance varies.
- Wire tests cover Anthropic, OpenRouter, and Responses.

**Part A: STOPPED at its gate.** The required live probes need a direct
Anthropic API key (inline-tools beta) and an OpenAI key (`additional_tools`
on gpt-5.6), and neither is configured: the `anthropic` entry is OAuth,
and there is no `openai` entry.

Implementing the ratified representation with only the fold fallback would
fix a public API shape before the wire formats it must map to are seen,
which is what the gate exists to prevent. Options for the maintainer:

- **(a) Wait.** Supply API-key credentials, then run the probes and build
  Part A.
- **(b) Ship the representation, validation, and byte-deterministic fold
  now.** Native mappings follow behind the probes, accepting that the
  representation might need a revision.

### Review of Parts C and B (2026-10-22)

A clean-context adversarial review gave "approve with changes". It found
one major and four minors, all fixed before commit:

- **Major: a collector rejection became the stream's error.** The
  `contrib/otelllm` stream wrapper made the collector's own rejection of an
  event the call's error. The span showed Error, and usage was lost, even
  though the consumer had seen a clean stream. The collector now supplies
  only the response summary, and the error is exactly what the consumer
  saw. A regression test covers it.
- **Minors:**
  - The contrib `go.mod` comment states that the `require` is a
    placeholder below the real minimum.
  - docs/release.md lists the gates that actually loop over contrib
    modules, and adds a check that the module resolves without the
    `replace`.
  - Status descriptions are content-free for non-provider errors too:
    the class only, so text authored by caller middleware is never
    recorded. There is a canary test for this.
  - The `ApplyCacheRetention` doc now matches its copy behavior.
- **Nits:**
  - Unmeasured usage is no longer recorded as zero.
  - `error.type` appears on the duration metric only.
  - `gen_ai.response.model` is omitted when unknown.
  - FS §15 notes that OpenRouter session affinity is unaffected by
    `none`.
