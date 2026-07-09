---
status: complete
---

# Review 2: Sol Max Whole-Repository Review

Conducted 2026-07-09 against HEAD 2508d35b70558a16f76efee60dfbabd7a14735ae.

Scope: every document under specs/, all production Go packages, tests,
examples, internal/e2e, release scripts, CI, generated model data, and
recorded fixtures. Four independent clean-context review passes covered the
core/schema/test API, provider engines, auth/security/e2e/CLI, and
docs/examples/tooling. Their findings were rechecked against the source and
the binding functional and architecture specs before inclusion here.

The review standard is the stated goal: simple, strongly functional, clean,
well named, modular, expressive, and balanced. A finding is included only
when it has a concrete behavioral, security, maintenance, or contract cost.
This is not a request for maximal abstraction.

## 1. Executive summary

**Verdict: preserve the architecture, but do not cut another release until
the two credential-boundary defects in section 3 are fixed.**

The library has grown substantially without losing its center. The Provider
interface remains small; the root package remains stdlib-only; provider
dependencies point inward through two coherent shared engines; parts,
events, errors, serialization, usage, and capability gating form a useful
normalization layer rather than an agent framework. Most recommendations
from Review 1 were implemented well.

The review found two release-blocking credential leaks:

1. Ambient OpenAI SDK Authorization headers can be sent to nominally
   keyless custom, vLLM, or Ollama endpoints.
2. OpenRouter's documented WithAPIKey("") opt-out can silently reload the
   environment key and send it to a caller-supplied base URL.

The remaining important work is concentrated rather than architectural:

- stream lifecycle and event grammar;
- preflight validation and structured-output mode isolation;
- OAuth refresh ordering;
- a few provider-specific equivalence defects;
- fixture recording that must fail closed;
- conformance/e2e tests that currently promise more than they prove;
- stale normative documentation after phases 10 and 11.

The correct response is a focused hardening release. It is not a provider
rewrite, a new framework layer, or a feature expansion.

## 2. What is working and should be protected

### Core shape

- Provider is the right-sized interface: Name, Capabilities, Models, Chat,
  and ChatStream. It has enough uniformity to be useful without hiding
  provider identity.
- The sealed Part and Event taxonomies are expressive. Value-type doctrine
  plus boundary normalization sharply reduced the pointer/value duplication
  identified in Review 1.
- iter.Seq2 streaming, single-use enforcement, and partial-response Collect
  are a strong Go-native combination.
- Sentinel errors plus ProviderError preserve both portable branching and
  provider detail.
- Canonical serialization preserves unknown parts and provider reasoning
  payloads, including byte-sensitive replay data.
- The schema public facade over internal/schemajson is a good boundary. It
  avoids an import cycle and keeps the root package independent.

### Modularity

- Root llm depends only on the standard library and internal/schemajson.
- providers/openai and providers/openaicodex share responsesapi rather than
  duplicating mappings.
- OpenRouter, vLLM, Ollama, and arbitrary compatible servers share the
  chatcompletions engine while retaining declarative Compat and focused
  Dialect hooks.
- Provider packages do not import each other. Shared code lives under
  providers/internal instead of becoming public by accident.
- The package split reflects real behavioral ownership. There is no reason
  to merge the Responses and Chat Completions engines.

### Functionality and tests

- Tool calls, malformed-call visibility, structured output, reasoning
  replay, multimodal input, prompt caching, usage/cost provenance, sessions,
  OAuth refresh, model listing/resolution, and observability are all real,
  integrated features.
- The offline, race, shuffle, lint, vet, coverage-floor, module-verification,
  live-tag compile, fixture-redaction, and secret scans pass.
- Request-build goldens are generally semantic, and replay now checks
  outbound requests instead of validating only response mapping.
- The present tracked repository contains no credential found by gitleaks.
  The recorded fixtures also pass the repository's current redaction guard.

## 3. Release blockers

### R1. Ambient Authorization leaks into keyless compatible providers

Severity: **Critical**

providerutil.AmbientCustomHeaderDeleteOptions deliberately skips
Authorization because authenticated providers overwrite it
(providers/internal/providerutil/providerutil.go:24-35). That assumption is
false for key-optional chat-completions providers. sdkOptions supplies an
empty API key and therefore emits no replacement Authorization header
(providers/chatcompletions/config.go:177-208).

With OPENAI_CUSTOM_HEADERS set to an Authorization value, blocking Chat or
SDK-backed Models calls can send that ambient credential to an arbitrary
custom base URL, including vLLM and Ollama endpoints. The direct streaming
path does not have the same leak, making behavior path-dependent.

The defect is reproduced by:

    OPENAI_CUSTOM_HEADERS='Authorization: Bearer ambient-secret' \
      go test -count=1 ./providers/chatcompletions -run '^TestGenericNew$'

The test fails because the keyless fixture server receives the ambient
bearer token.

**Required fix:** delete ambient Authorization unconditionally, then apply
only provider-owned authentication after ambient cleanup. Add explicit
ambient-environment tests for generic chatcompletions, vLLM, Ollama,
OpenRouter, OpenAI, and Codex, covering both Chat and Models where
applicable.

### R2. WithAPIKey("") can re-enable OPENROUTER_API_KEY

Severity: **Critical**

OpenRouter reads OPENROUTER_API_KEY in defaultConfig, and WithAPIKey("")
documents that an empty value disables environment fallback
(providers/openrouter/options.go:56-68). The shared NewWithDialect constructor
then performs environment fallback a second time whenever APIKey is empty
(providers/chatcompletions/config.go:127-137).

A caller trying to suppress ambient credentials, especially while pointing
WithBaseURL at a proxy or test endpoint, can therefore send the environment
OpenRouter key to that endpoint instead of receiving ErrAuth.

**Required fix:** model authentication selection as three states: unset,
explicitly empty, and configured. Resolve environment fallback once in the
public preset constructor; the shared engine should consume the resolved
decision without rereading process environment.

## 4. Correctness and lifecycle findings

### C1. Request preflight can panic and is provider-dependent

Severity: **Moderate**

ValidateProviderOptions checks only whether the interface is nil before
calling ForProvider twice (validate.go:52-59). An interface containing a
typed nil pointer is non-nil, and every current Options type uses a value
receiver. A request carrying (*openai.Options)(nil), for example, panics
instead of returning ErrBadRequest, contrary to functional_spec.md:139-147.

The same preflight accepts negative MaxTokens. Anthropic sends the negative
value, while Responses and Chat Completions silently omit it. Forced
ToolChoice also does not verify that its named tool exists, and duplicate
tool names are accepted.

**Fix:** reject typed-nil provider options without invoking their method,
read ForProvider once, reject MaxTokens below zero, require unique tool
names, and require a named forced choice to identify one declared tool.

### C2. Three direct stream engines accept empty or truncated success

Severity: **Moderate**

OpenAI and Anthropic return normally when their SDK stream reaches clean EOF
without a MessageEnd (providers/openai/provider.go:123-162 and
providers/anthropic/provider.go:91-131). Codex does the same in its direct SSE
decoder (providers/openaicodex/direct.go:173-233). A clean EOF is not a
decoder error.

Consequences:

- a 2xx response with zero events is a silent success;
- a response that emits deltas and then truncates returns a partial response
  with nil error;
- Codex Chat, implemented by Collect over the same stream
  (providers/openaicodex/provider.go:111-129), can return an empty Response
  with nil error.

This violates functional_spec.md:790-792. Chat Completions already rejects
an empty stream, but it still does not provide a shared terminal-event
contract for the other engines.

Codex also accepts output deltas before response.created. StreamState emits
the delta immediately and synthesizes MessageStart only at the terminal
response (providers/internal/responsesapi/stream.go:75-88,276-299). Its
normal fixture therefore produces TextDelta, MessageStart, MessageEnd,
contrary to architecture.md:1099-1102.

**Fix:** add one providerutil stream-contract wrapper that tracks start,
terminal error, and MessageEnd. On a fully drained upstream sequence without
MessageEnd, yield ProviderError wrapping ErrServer. Preserve early consumer
break semantics. Buffer pre-start Responses events and emit MessageStart
first, using request model identity as the fallback when the backend omits
response.created.

### C3. Session.ChatStream mutates history before use

Severity: **Moderate**

Session.ChatStream appends the user turn and constructs the provider stream
before the returned iterator is ranged (session.go:121-134). Rollback begins
only inside collectingStream (session.go:136-178).

An abandoned iterator leaves a phantom user turn. Multiple streams created
before consumption can also record stale rollback indexes; a later failed
stream can truncate history committed by an earlier one. Session is
documented as not goroutine-safe, but these cases require no concurrency.

**Fix:** move history mutation, request construction, provider stream
creation, and rollback installation into the first-range closure. Add tests
for a never-ranged stream and two created-before-consumption streams.

### C4. Parse modes are not isolated from the caller request

Severity: **Moderate**

applyParseMode preserves any non-nil ResponseFormat in ModeNative
(parse.go:159-169). A request carrying FormatJSONMode can therefore resolve
to ModeNative while still sending JSON mode. ModeTool appends parse_result
to all existing tools rather than installing the spec's single synthetic
tool (parse.go:170-178), and decodeParseResponse blindly consumes the first
returned call rather than the call named parse_result (parse.go:190-203).

**Fix:** ModeNative must force FormatJSONSchema while retaining only
deliberately supported schema/name overrides. ModeTool must install one
collision-free synthetic tool and decode exactly one matching call. Add
tests with a preexisting JSON-mode format, existing tools, a parse_result
name collision, and an unrelated first returned call.

### C5. Cost provenance aggregation depends on call order

Severity: **Moderate**

sumUsage marks a token-bearing unknown-cost call as estimated only when a
cost already exists (usage.go:186-213). Unknown-cost then native-cost yields
CostSourceNative; native-cost then unknown-cost yields CostSourceEstimated.
Identical usage sets therefore get different provenance, and the first
ordering labels an incomplete dollar total as billing-grade.

**Fix:** track whether every token-bearing component had a cost independently
of whether the first cost has appeared. Add commutativity tests for native,
estimated, and missing-cost permutations.

### C6. Middleware factories are rebuilt on every request

Severity: **Moderate**

Wrap binds middleware once, but wrappedProvider.Chat and ChatStream invoke
every Chat/Stream factory again for every call (middleware.go:31-79).
Factory-local state is therefore reset per request, setup work repeats, and
the normal Go middleware expectation that wrapping occurs once is violated.

**Fix:** compose ChatFunc and StreamFunc once in Wrap and store the resulting
handlers. Keep Bind's current once-only semantics. Test factory invocation
count across multiple calls.

### C7. Chat Completions violates its single-choice and warm-up contracts

Severity: **Moderate**

Blocking mapping consumes choice zero, but stream mapping loops over every
choice (providers/chatcompletions/response.go:22-35 and
providers/chatcompletions/stream.go:346-371). A server that ignores n:1 can
mix text/tools from several choices and overwrite the finish reason.
functional_spec.md:953-955 explicitly requires consuming one choice.

The same state marks a stream started before observing any choice. A stream
containing only choices:[] consequently produces MessageStart and
MessageEnd instead of the ErrServer required for OpenRouter warm-up empty
choices (functional_spec.md:936-937). A trailing usage-only chunk remains
valid after at least one real choice.

**Fix:** process only choice index zero, track whether any actual choice was
seen, and reject choice-less completion while still accepting trailing
usage.

### C8. Anthropic has three local contract drifts

Severity: **Moderate**

1. Any invalid-request message containing the bare word "context" maps to
   ErrContextTooLong (providers/anthropic/errors.go:63-98), despite the
   binding scoped-phrase rule in functional_spec.md:750-780.
2. An input_json_delta received before tool metadata is buffered, but a
   later content_block_start replaces that state and loses the arguments
   (providers/anthropic/stream.go:93-124,158-176).
3. Foreign reasoning is correctly dropped, but buildMessages still emits an
   assistant message whose content became empty
   (providers/anthropic/convert.go:182-194,241-250). Cross-provider
   reasoning-only turns can therefore become provider-invalid empty
   messages.

**Fix:** use only scoped overflow phrases, merge late tool metadata into
existing state, and omit messages that become empty solely through
documented foreign/unknown-part filtering.

### C9. Schema generation and validation disagree at valid Go boundaries

Severity: **Moderate**

- uint64 is generated as JSON Schema integer
  (internal/schemajson/schema.go:124-141), but ValidateArgs requires
  json.Number.Int64 and rejects values above MaxInt64
  (internal/schemajson/validate.go:127-134).
- Integer enum parsing also uses ParseInt, making unsigned enum values above
  MaxInt64 unrepresentable (internal/schemajson/schema.go:380-415).
- Go 1.26's json omitzero option is ignored, so fields omitted by
  encoding/json may be marked required
  (internal/schemajson/schema.go:418-452).
- malformed required constraints fail open: a non-array value or non-string
  member is silently discarded (internal/schemajson/validate.go:241-253).

**Fix:** validate exact arbitrary-precision JSON integer syntax, preserve
unsigned enum values without float conversion, recognize omitzero, and
validate the supported schema shape before validating arguments.

### C10. OAuth single-flight does not order refresh and persistence safely

Severity: **Moderate**

provideroauth.Source publishes the refreshed credential, clears inflight,
and wakes waiters before invoking onRefresh
(providers/internal/provideroauth/source.go:137-154). A second forced refresh
can complete and persist credential B while callback A is blocked, after
which A persists the older credential last.

The refresh operation also uses the leader caller's context. Cancellation of
that one request causes otherwise healthy waiters to inherit its failure.
Finally, an access-only credential whose expiry is within the five-minute
margin fails immediately for a missing refresh token, even though Refresh
is explicitly optional in architecture.md:863-870.

**Fix:** serialize callback completion in refresh-generation order, make the
shared refresh lifetime independent of one waiter while preserving waiter
cancellation, and use an unrefreshable access token until its actual expiry.

### C11. Unknown provider I/O failures escape the two-layer error contract

Severity: **Moderate**

Non-timeout transport and decoder failures are often returned raw instead of
as ProviderError wrapping ErrServer. Examples include
providers/chatcompletions/errors.go:18-50,
providers/anthropic/errors.go:15-34, and
providers/internal/responsesapi/errors.go:20-38. Codex malformed SSE wraps
ErrServer but not ProviderError (providers/openaicodex/direct.go:203-213).

Callers can therefore rely on errors.Is or errors.As depending on which
failure and engine they hit, contrary to functional_spec.md:750-758.

**Fix:** centralize provider I/O/decode normalization in providerutil.
Preserve context cancellation/deadlines and typed provider errors; wrap
unclassified remote transport/parse failures as ProviderError with a stable
sentinel.

### C12. Capability narrowing does not narrow the wire request

Severity: **Moderate**

The public compatible-provider constructor allows callers to declare a
narrow capability set, but BuildParams sends parallel_tool_calls:true for
every tool request regardless of CapabilityParallelTools
(providers/chatcompletions/convert.go:18-57). A deliberately narrowed server
can reject a request that validation declared supported.

**Fix:** gate or omit parallel_tool_calls according to the provider's
declared capability and add a custom-provider request golden.

## 5. Security, recording, and operational findings

### S1. Fixture recording is not fail-closed

Severity: **Moderate**

The recorder has careful known-secret replacement, sensitive header
handling, token patterns, and MOCK identifiers
(internal/e2e/record.go:18-61,78-120,219-280). Current credentials are passed
to WriteFixture by the live suites. That is good.

The final safety boundary is still weaker than its "safe to commit" claim:

- all live suites write from t.Cleanup even after a failed or filtered run
  (for example internal/e2e/anthropic_live_test.go:79-85);
- WriteFixture truncates the tracked file in place
  (internal/e2e/record.go:283-297);
- it does not run the fixture guard before replacement;
- test ordering can run TestRecordedFixturesAreRedacted before the cleanup
  writes the new fixture;
- redaction uses finite field/header/regex lists, so a novel provider field
  can pass through;
- the vLLM fixture currently preserves the private endpoint hostname
  pax.local throughout internal/e2e/fixtures/vllm/live.json.

No tracked credential was found in this review. The defect is that future
recording cannot make that guarantee by construction.

**Fix:** write to a temporary file, structurally sanitize URL/query,
headers, JSON, and SSE JSON payloads, run the same guard plus gitleaks-style
high-entropy checks on the staged bytes, require a complete expected
scenario set, and atomically rename only when the test passed. Replace
private hosts with a stable fixture host. Prefer per-fixture sequential
MOCK identifiers over unsalted truncated hashes where correlation is
needed.

### S2. WireTap's request cap does not cap memory

Severity: **Moderate**

WireTap reads the complete request body into memory before applying its
configured capture limit (wiretap.go:104-127). Large inline images and PDFs
are duplicated in full even when capture is capped at a few bytes. A read
failure also returns before closing the original body.

**Fix:** tee a bounded prefix while the transport consumes the original
request, analogous to the response wrapper. Preserve retryability through
GetBody when available and close every failure path.

The core WireTap and e2e recorder also maintain different sensitive-header
lists (wiretap.go:14-22 versus internal/e2e/record.go:36-61). Centralize the
classification or make the core capture conservative enough that logging a
WireCapture cannot expose organization/account identifiers later scrubbed
only by the fixture layer.

### S3. CLI Codex auth is forced through a command-line secret

Severity: **Moderate**

The CLI interprets --api-key as a Codex OAuth access token
(cmd/llm-cli/provider.go:49-61). It offers no Codex-specific environment or
explicit auth-file path even though llm.LoadAuthFile exists. Passing a
bearer token on argv exposes it to shell history and process inspection.

**Fix:** add an explicit --auth-file or OPENAI_CODEX_ACCESS_TOKEN input with
clear precedence, and keep file loading opt-in. Retain --api-key for
compatibility but document it as lower-priority and unsafe for shared
systems.

### S4. The model snapshot can publish an empty or partial table

Severity: **Moderate**

scripts/snapshot-models-table.ts casts unknown upstream JSON, treats schema
drift as empty collections, and writes models.json directly
without provider/count invariants or atomic replacement
(scripts/snapshot-models-table.ts:26-50,68-130). A successful HTTP response
with a changed shape can erase most of the embedded catalog.

The script also uses an unpinned npm execution dependency without a lockfile
(scripts/package.json:1-9).

**Fix:** validate upstream schemas, require expected providers and sane row
counts, reject destructive deltas unless explicitly approved, write
atomically, and add recorded source fixtures plus a committed lockfile.

## 6. First-party consumer and test-confidence findings

### T1. The conformance suite does not prove cancellation or event grammar

Severity: **Moderate**

RunConformance cancels after the first event but does not require
context.Canceled, a terminal error, or even evidence that the provider
noticed cancellation (llmtest/conformance.go:80-124). A finite provider
that ignores context passes. The normal stream test requires only a
non-empty first range, not MessageStart-first, exactly one MessageEnd, or
successful completion (llmtest/conformance.go:50-78).

**Fix:** use a controllable fixture that blocks after its first event until
context cancellation. Assert prompt termination, context error identity,
no post-error events, MessageStart-first, exactly one MessageEnd on success,
and ErrServer on empty/truncated EOF. Apply the suite to the Ollama preset
as well as its shared engine.

### T2. llmtest's defensive-copy contract is incomplete

Severity: **Moderate**

llmtest.Provider.Requests promises defensive copies
(llmtest/provider.go:164-172), but Temperature and TopP pointers remain
shared, and cloneJSONLike does not clone common map[string]any or []any
schemas (llmtest/provider.go:189-203,439-447). Pointer parts/events are also
cloned as pointers rather than normalized to the value form emitted by
built-in providers (llmtest/provider.go:240-393).

This can hide mutation and pointer-shape bugs in downstream code certified
against the designated fake.

**Fix:** clone scalar pointers and JSON-compatible containers recursively,
normalize core parts/events to values, and explicitly document opaque
ProviderOptions as immutable or shallow-copied.

### T3. The live harness is scenario-list-driven, not capability-driven

Severity: **Moderate**

Architecture promises one scenario per declared capability and a runner
that iterates provider.Capabilities (architecture.md:1183-1210).
RunScenarios instead iterates each provider's hand-written scenario list and
skips unsupported entries (internal/e2e/scenario.go:25-49). It cannot detect
a declared capability with no scenario. OpenAI, for example, declares PDF
input, prompt caching, and session affinity but its list omits those cases
(providers/openai/provider.go:13-28 and
internal/e2e/openai_live_test.go:70-88).

The shared stream live check Collects without directly asserting the event
grammar promised by the spec. Missing tested models can be logged rather
than failed. Codex treats a configured but invalid credential as a skip
(internal/e2e/openaicodex_live_test.go:70-76), allowing a credentialed live
job to appear green without running scenarios.

**Fix:** maintain a capability-to-scenario coverage registry with explicit
exemptions for semantic capabilities covered by another scenario. Fail on
uncovered declarations, inspect raw stream events, fail when the configured
model is absent, and distinguish missing credentials (skip) from invalid
configured credentials (fail).

### T4. Coverage reports 100 percent for the schema facade, not the engine

Severity: **Mild**

scripts/check-coverage.sh runs go test -cover ./schema and reports 100
percent (scripts/check-coverage.sh:11-32). That instruments only the thin
public facade while its tests execute the real engine in
internal/schemajson as an uninstrumented dependency. A cross-package
measurement during this review put internal/schemajson at 82.1 percent.

The reverse issue exists for providerutil: its own-package number is low
because shared behavior is primarily exercised by provider tests; combined
provider coverage measured 76.9 percent.

**Fix:** make floors represent owned engines with -coverpkg or grouped
profiles. Do not advertise the facade's 100 percent as schema-engine
coverage.

### T5. The CLI buffers streams and ignores output failures

Severity: **Moderate**

runStreaming stores every event, prints deltas, and then replays the slice
through Collect (cmd/llm-cli/run.go:92-121). Memory grows with the full
response and an in-stream error discards the partial response. Most output
writes throughout run.go and printUsage also ignore errors
(cmd/llm-cli/run.go:26-71,92-145), so broken pipes or full disks can produce
truncated output with a successful exit.

**Fix:** wrap the provider iterator with print side effects and pass that
single sequence directly to Collect. Return write errors immediately while
retaining the partial response/error contract. Add a failing-writer test.

### T6. The canonical tool example teaches an invalid parallel-tool loop

Severity: **Moderate**

examples/tools/main.go calls AddToolResults once per returned tool call
(examples/tools/main.go:48-55), creating consecutive tool-role messages
instead of one grouped result message. It ignores argument JSON errors and
dispatches every call as weather regardless of name
(examples/tools/main.go:68-75).

**Fix:** dispatch by tool name, validate arguments, convert execution or
validation failures into IsError results, collect all parallel results, and
call AddToolResults once.

### T7. CI does not compile the live-tag surface

Severity: **Mild**

Every CI command uses default build tags (.github/workflows/ci.yml:22-50).
The live suites currently compile, but a future build-tag-only regression
can merge unnoticed.

**Fix:** add go test -tags live -run '^$' ./... without credentials.

## 7. API, naming, and simplicity assessment

### Keep

- Keep schema as the public facade over internal/schemajson.
- Keep ProviderOptions singular. A request targets one provider, and routing
  layers should replace provider options when dispatching.
- Keep Models on Provider unless a real implementation cannot supply it;
  uniform discovery is useful and existing providers already satisfy it.
- Keep the two provider engines. Their wire semantics are genuinely
  different.
- Keep value parts/events with pointer tolerance at boundaries. The current
  centralized dereference helpers capture the compatibility benefit without
  Review 1's switch-arm explosion.
- Keep raw Client and Raw payload escape hatches. They prevent the unified
  surface from expanding to every provider feature.

### Decide before v1

The spec calls vendor SDKs implementation details
(functional_spec.md:34-43), but openai.Options publicly exposes several
openai-go types (providers/openai/options.go:29-43). The advanced
chatcompletions.Dialect and BuildParams also expose SDK types, although that
surface is explicitly documented as unstable
(providers/chatcompletions/config.go:17-50).

Recommendation: keep the advanced Dialect explicitly unstable, but replace
SDK types in ordinary provider Options with library-owned values or raw JSON
extension fields before v1. Otherwise revise the stability statement and
accept that routine SDK upgrades can break downstream builds.

### Small API cleanup

- DefaultHTTPClient returns a shared mutable pointer while asking callers to
  treat it as immutable (httpclient.go:10-25). Return a fresh client sharing
  a private transport, or make mutation consequences explicit.
- RetryAfter is exported but absent from the normative API design. Either
  document it as supported utility or keep retry parsing internal.
- RetryAfter should reject negative numeric values and clamp past HTTP dates
  instead of returning negative durations (retrylog.go:52-74).
- AuthCredential.UnmarshalJSON leaves a previous Expires value intact when
  a reused destination receives JSON without expires (auth.go:65-104).
- UnknownPart can be constructed with a built-in discriminator such as
  "text"; marshaling succeeds but unmarshaling interprets it as TextPart
  (serialize.go:600-620,780-810). Reject reserved type names.
- Blocking and streaming Chat Completions use different block indexes for
  dropped tool calls (providers/chatcompletions/response.go:101-131 versus
  providers/chatcompletions/stream.go:453-480). Define index semantics and
  make Chat and Collect(ChatStream) agree.

These are focused contract cleanups, not reasons to redesign the core.

## 8. Specification and documentation review

The functional spec remains a strong statement of product boundaries, but
the documentation set now mixes three kinds of truth without labeling them:

1. current normative behavior;
2. historical phase plans and reviews;
3. provider research for deferred work.

That is the main source of drift.

Concrete inconsistencies:

- architecture.md still lists removed observe.go and an obsolete dependency
  table near the opening package layout.
- architecture.md:746-762 says vLLM /tokenize is deferred although phase 11
  shipped it.
- architecture.md:1237-1243 describes OAuth runnable-state as future work
  even though it is implemented.
- architecture coverage floors are older than scripts/check-coverage.sh and
  docs/release.md.
- project_overview.md:9-11 says the project supports ZAI even though the
  user explicitly moved it to future work.
- provider_capabilities.md analyzes only the original four researched
  providers, foregrounds deferred ZAI, and omits shipped vLLM/Ollama. It is
  useful research, but not a current capability matrix.
- doc.go:11-21 omits chatcompletions, vLLM, and Ollama from the shipped
  provider list.
- docs/release.md:1-103 remains a v0.1.0 tagging checklist while describing
  v0.3/v0.4-era coverage and a repository that already has a v0.1.0 tag.
- README's broad live-tested and "always-on secret redaction" language is
  stronger than the current OpenAI evidence and finite redaction policy.
- implementation_plan.md and Review 1 cite commits from the history that was
  intentionally replaced. Those references are useful provenance only if
  labeled pre-rewrite/historical.
- gollm-test.json.sample still includes deferred ZAI.
- scripts/snapshot-models-table.ts still includes ZAI in its current
  provider set even though ZAI is future work.

**Recommendation:** make functional_spec.md and architecture.md the two
authoritative current-state documents. Add an explicit "historical,
non-normative" banner to phase plans and old reviews. Mark provider research
as research. Then perform one mechanical synchronization pass across
project_overview.md, doc.go, README, release.md, samples, dependency tables,
coverage floors, and shipped provider lists.

Do not delete the phase history; label it so it stops competing with the
current contract.

## 9. Review 1 disposition

Most of the first review was acted on:

| Review 1 item | Current disposition |
|---|---|
| Pointer/value switch explosion | Resolved pragmatically through centralized dereference/normalization helpers |
| Session tools stranded | Resolved with session tools, AddToolResults, and Continue |
| Generic Parse options | Resolved; options are non-generic except the typed validator |
| RetryDroppedToolCalls swallowed retry errors | Resolved |
| Collect/Session event-fold duplication | Resolved through applyCollectEvent |
| Cross-engine status table drift | Mostly resolved through providerutil; Anthropic's bare-context heuristic remains |
| Empty 2xx streams | Fixed for Chat Completions; still open for Anthropic, OpenAI, and Codex |
| OpenAI tool-result images/files | Resolved in responsesapi |
| Cost provenance absent | API added; aggregation order bug remains |
| Middleware.Bind hidden | Resolved and documented |
| Wire naming / ToolResultPart.Content / Ptr | Resolved |
| Replay ignored requests | Resolved with outbound capture and invariants |
| No provider conformance suite | Suite added; cancellation/grammar assertions remain too weak |
| CLI provider factory untested | Tests added, but the option-forwarding test does not exercise the configured behavior |
| Coverage floors only documented | Enforced in CI; cross-package ownership measurement needs correction |

This is meaningful progress. Review 2 findings are mostly second-order edge
cases exposed by the larger, stronger implementation rather than evidence
that Review 1 was ignored.

## 10. Prioritized implementation plan

### P0: before any release or fixture recording

1. Fix ambient Authorization isolation for all SDK-backed providers.
2. Fix OpenRouter's explicit-empty auth state and add leak regressions.
3. Make fixture writes staged, validated, complete, and atomic; anonymize the
   tracked vLLM host.
4. Reject empty/truncated direct streams and enforce MessageStart/MessageEnd
   grammar.
5. Upgrade release verification to Go 1.26.5 or newer.

### P1: correctness hardening

1. Fix typed-nil/request/tool validation.
2. Make Session.ChatStream lazy and rollback-safe.
3. Isolate Parse modes and synthetic tools.
4. Make usage provenance commutative.
5. Compose middleware once.
6. Fix Chat Completions choice handling and capability narrowing.
7. Fix the three Anthropic mapping drifts.
8. Fix schema unsigned/omitzero/malformed-keyword behavior.
9. Order OAuth refresh callbacks and handle cancellation/access-only tokens.
10. Normalize unknown provider I/O errors.

### P2: make tests prove the contract

1. Strengthen llmtest conformance for cancellation, terminal events, and
   successful concurrency.
2. Make llmtest clones genuinely defensive and value-normalized.
3. Make live scenarios capability-covered, fail configured auth failures,
   and assert raw event grammar.
4. Measure internal engines with grouped cover profiles.
5. Add live-tag compilation to CI.
6. Fix the CLI stream/output path, its behavioral provider test, and the
   canonical parallel-tool example.
7. Harden and test the model snapshot script.

### P3: pre-v1 surface and documentation

1. Decide whether ordinary provider options may expose vendor SDK types.
2. Resolve the small API cleanup list in section 7.
3. Synchronize the normative specs, package docs, README, release guide,
   samples, and historical labels.

After P0-P2, resume feature work. Adding another provider before these
contracts are pinned would multiply the same edge cases.

## 11. Balance: what not to do

- Do not add agent loops, MCP orchestration, fallback routing, prompt
  frameworks, or policy engines to root llm.
- Do not replace iter.Seq2 with channels or a custom stream object.
- Do not merge provider engines merely to reduce file count.
- Do not generalize every provider quirk into the public Request. Keep
  provider options and raw clients as escape hatches.
- Do not replace the focused schema subset with a full JSON Schema runtime
  unless actual callers require broader dialect support.
- Do not add Gemini, ZAI, or other providers until lifecycle, auth, and
  conformance work above is complete.
- Do not chase a single universal retry abstraction until the concrete
  blocking/direct-stream/idempotent-operation semantics are specified.

The project is already modular. The next simplicity gain comes from
enforcing a few shared contracts once, not adding more layers.

## 12. Verification performed

| Check | Result |
|---|---|
| go test ./... -count=1 | Pass |
| go test -race ./... -count=1 | Pass |
| go test -shuffle=on -count=10 ./... | Pass |
| go vet ./... | Pass |
| golangci-lint v2.12.2 with CI's govet/ineffassign/unused set | Pass, zero issues |
| go mod verify | Pass |
| scripts/check-coverage.sh | Pass all configured floors |
| Cross-package schema engine measurement | internal/schemajson 82.1 percent |
| Cross-provider shared utility measurement | providerutil 76.9 percent |
| Short fuzz smoke | Pass; one 3-second FuzzFor runner timeout, isolated 5-second rerun passed about 854k executions |
| go test -tags live -run '^$' ./... | Pass |
| Fixture redaction guard | Pass |
| gitleaks over worktree and two reachable commits | Pass, no leaks |
| git diff --check before review artifact | Pass |
| Targeted ambient Authorization regression | Fails as described in R1 |
| govulncheck v1.5.0 | Reports GO-2026-5856 in local Go 1.26.4; fixed in Go 1.26.5 |

The reported standard-library vulnerability concerns ECH handshakes; the
default go-llm transport does not configure ECH, and CI's 1.26.x selector
should install the latest patch. Nevertheless, release verification should
run on 1.26.5 or newer so the scan is clean.

Credentialed live API scenarios were not rerun during this review. This
review therefore relies on the repository's recorded fixtures and prior
documented live runs for provider behavior, while all offline and compile
checks above were rerun against the reviewed HEAD.

## 13. Final assessment

**Simplicity:** strong. The abstractions map to real domain boundaries.

**Functionality:** strong and unusually complete for the code size.

**Modularity:** very strong. Keep the dependency direction and shared-engine
shape.

**Naming and expressiveness:** strong, with small stale-doc/API exceptions.

**Correctness:** good on normal paths; lifecycle and malformed/partial-path
contracts need the focused P0/P1 pass.

**Security posture:** current tracked credentials are clean, but the two
ambient-auth defects and non-fail-closed recorder must be fixed before the
project can claim credential-safe operation.

**Test confidence:** broad and disciplined, but conformance, live capability
coverage, and cross-package coverage measurement currently overstate what
they prove.

The codebase does not need a new direction. It needs one hardening cycle
that makes its strongest ideas -- normalized contracts, explicit provider
boundaries, safe replay, and executable conformance -- true at the edges as
well as on the happy path.
