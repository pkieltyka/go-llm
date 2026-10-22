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

### Probe evidence (2026-10-22)

Probed live with a throwaway program that sent raw bodies through the
go-llm providers' SDK clients (Anthropic API key; OpenAI under Sign in with
ChatGPT, the only OpenAI credential available). Each cache experiment used a
fresh ~25k-token system prefix. **Verdict: nothing contradicts ratified
option (a); the gate is passed**, with the design inputs below.

**Anthropic, mid-conversation `role: "system"` text** (no beta needed):

- Accepted by claude-haiku-5-5, sonnet-5-5, opus-5-5, fable-5-1, opus-5,
  sonnet-5, fable-5, and opus-4-8. Rejected by opus-4-7, sonnet-4-6, and
  haiku-4-5 with 400 "role 'system' is not supported on this model".
  pi's model regex omits sonnet-5, which does accept system text.
- Placement: a system message must follow a `user` message (or an assistant
  message ending in a server tool result); directly after an assistant turn
  it is a 400. pi's placement (held back until just before the next
  assistant turn, so after the user message) works.
- The model follows it, and the whole cached prefix survives: turn 2 read
  all 25,122 tokens turn 1 wrote.

**Anthropic, `inline-tools-2026-09-15`** (`tool_addition` with
`{"type":"tool_definition","definition":{...}}`, `tool_removal` with
`{"type":"tool_reference","name":...}`, inside `role: "system"` messages):

- Requires the beta header (400 otherwise) and a model with
  mid-conversation system content. Supported on the same models as system
  text except sonnet-5 (400 "tool_addition/tool_removal is not supported on
  this model"). So tool changes need their own capability, separate from
  system text.
- An added tool is called immediately; after `tool_removal` the model
  refuses it while remaining tools still work; a `tool_use` of the removed
  tool stays valid in history.
- Same-name `tool_addition` redefines a tool (the new schema is used), alone
  or together with a removal in the same message.
- `tool_removal` of a tool never declared is a 400 ("references unknown
  tool"), so local validation against the effective tool set is required.
- `tool_addition` works with no top-level `tools` at all; a top-level list
  of only deferred tools is a 400 ("At least one tool must have
  defer_loading=false").
- Cache: every turn read the full previous prefix (haiku-5-5 and
  sonnet-5-5), **with or without pi's deferred placeholder tool, and also
  when the beta header first appears at the first tool change**. This
  contradicts pi's code comment ("full miss without it"); go-llm should not
  need the placeholder. `cache_control` on a `tool_addition` block is
  accepted.
- Tool changes add roughly 460 tokens of hidden scaffolding to the input.

**OpenAI Responses under Sign in with ChatGPT** (gpt-5.6-luna and
gpt-6-luna):

- Mid-conversation `developer` (and `system`) messages are accepted and
  followed.
- `additional_tools` items (`{"type":"additional_tools","role":"developer",
  "tools":[...]}`) add callable tools at the head of the input, after the
  user message, or mid-conversation, with no top-level `tools`. There is no
  removal item; pi announces removals as developer text.
- A same-name redefinition in a later `additional_tools` item took effect.
  With a top-level tool of the same name the model used the redefinition in
  one sample; pi reports it cannot replace a top-level tool. Following pi
  (all tools via `additional_tools` once positional changes are in use)
  avoids the question.
- `configuration_update` items are rejected on this endpoint ("User
  subscription sharing currently supports only ... additional tools ...").
- Cache deltas cannot be measured here: `cached_tokens` is always 0 on
  this endpoint (see FS §17C). The API-key Responses path was not probed
  (no key configured).

**Follow-up probes during implementation (2026-10-22):**

- OpenAI under Sign in with ChatGPT, tools only in `additional_tools`:
  `tool_choice` named, `required`, and `none` and `parallel_tool_calls:
  false` all behave. A top-level hosted `web_search` next to in-place
  function tools is accepted, and the model uses each when appropriate.
- Anthropic: a named `tool_choice` for a tool that exists only through
  `tool_addition` is honored, as is `tool_choice: any` with no top-level
  `tools`.

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

**Part A: shipped 2026-10-22** after the gate passed (see "Probe
evidence" in Part A). FS §7A and §13B are normative; the shape follows the
ratified decisions with these refinements:

- **Parts.** `SystemDeltaPart{Section, Text, Cache}` and
  `ToolsDeltaPart{Added, Removed, Cache}` in `RoleSystem` messages, which
  also take `TextPart`. `llm.FoldPositional(req, support)` is the shared,
  deterministic fold; `ValidateRequest` replays the same transcript.
- **Per-model support** lives in `models.json` as a `behavior` object,
  written by pi-style rules in `scripts/model-behavior.ts` (maintainer
  decision: track model behavior the way pi does) and exposed as
  `ModelInfo.Behavior`. Two flags, not one capability, because
  claude-sonnet-5 takes system text but not tool changes.
  `positional-instructions` and `positional-tools` are advisory
  provider capabilities.
- **Anthropic:** `role: "system"` messages, held until before the next
  assistant turn, with `tool_addition`/`tool_removal` under the inline-tools
  beta header (added automatically). No placeholder tool.
- **OpenAI Responses:** `developer` messages; on supported models every
  tool travels in `additional_tools` items from the first turn, so the
  prefix never switches layout (the `AppendDelta` prefix conformance case
  caught the one-time rewrite a lazy switch would cause). `tool_choice`
  works with them (probed).
- **Chat Completions** always folds.
- **Serialization:** `system_delta`/`tools_delta`; envelope version 2 only
  when a message carries them.
- **Tests:** root fold/validation/serialization tests, adapter wire tests,
  `llmtest` `AppendDelta` prefix cases for Anthropic and OpenAI, and the
  live `positional_changes` scenario (cache read across a tool addition on
  Anthropic; in-place behavior on OpenAI under Sign in with ChatGPT).
- Also fixed on the way: Anthropic streams are now read to EOF after
  `message_stop` (shared `providerutil.DrainStream`), so its fixture
  records completely again.

The paragraph below is the earlier stop record.

**Part A (earlier): STOPPED at its gate.** The required live probes need a direct
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
