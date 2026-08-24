---
status: complete
planned_at: c983a92
planned_at_date: 2026-08-23
---

# Provider Capability Conformance Plan

This plan adds credential-free proof that selected capabilities advertised by
first-party providers activate the intended native request fields and produce
the expected normalized result. It complements the existing base
`llmtest.RunConformance` suite and the credentialed `internal/e2e` registry; it
does not replace either one.

The plan was written against go-llm commit
`c983a92addf6f1b738e969c16f8b4f4b079fd0a5` and is implemented against the
authoritative clean baseline
`d776f149cc62bba96653838b5bb202eefcf148c4`. Plans 5 and 6 are complete. In
particular, OpenAI reasoning-summary selection remains fully covered by Plan 6
and is not an extension case in this conformance framework.

An executor must read this plan fully, preserve unrelated worktree changes,
and stop rather than adding provider policy, model-name inference, live
credentials, or application/agent behavior.

At implementation start `models.json` is clean at object
`ff84f2c6eed559d9bbf1192fe792fb5fc614f602` in HEAD, index, and worktree. This
plan does not modify or regenerate that file. The unrelated untracked
`reviews/` directory is also preserved and excluded.

## Evidence baseline and dispositions

| Finding | Pinned evidence | Disposition |
|---|---|---|
| Capability claims benefit from executable, provider-specific activation fixtures. | Gollem `6e024b44eea2c15dab9576c226ed18a586f8704f`, `provider/conformance/conformance.go` | Add an offline activation runner without copying Gollem's provider API. |
| go-llm already has a strong common provider-contract suite. | go-llm `llmtest/conformance.go` at `c983a92addf6f1b738e969c16f8b4f4b079fd0a5` | Preserve `RunConformance` unchanged and run activation checks beside it. |
| go-llm already has exhaustive credentialed scenario/exemption coverage. | go-llm `internal/e2e/coverage.go` at the planned commit | Keep it authoritative for live availability; do not create a duplicate live registry. |
| Normalized tool-call success does not prove that native `tools` and required tool-choice fields were sent correctly. | Current first-party `ConformanceTools` fixture handlers; Plan 5's no-argument tool-schema defect | Include tools and required tool choice in the initial activation-sensitive set. |
| Synthetic model IDs are not inert for model-dependent providers. | go-llm `providers/openaicodex/direct.go` and `lite.go`; gpt-5.6 selects the Responses Lite rewrite | Preserve each case's real model and pass fixture identity out of band. |
| OpenAI summary selection could be the first provider extension case. | Earlier draft of Plan 6 | Rejected: focused Plan 6 tests already prove it, and empty/custom cases would broaden the public runner without a second need. |
| Cache admission can be nondeterministic even when request activation is deterministic. | OpenRouter cache-control mapping and Codex `prompt_cache_key` mapping; live e2e exemption text | Require deterministic wire proof; exempt only live cache effectiveness. |

Remote catalog fallback, runtime provider selection, model gates, appserver
catalogs, agent orchestration, persistent caches, PromptTemplate changes, and
stream watchdogs are outside this plan.

## Confirmed decisions

1. `RunConformance` remains source- and behavior-compatible.
2. The activation runner is additive and offline.
3. Case identity is passed to the provider factory as data; the runner never
   overwrites `Request.Model` and never parses an HTTP body to recover a case.
4. Initial cases cover only a reviewed set of standard `llm.Capability` values.
   No empty-capability or custom extension cases are accepted.
5. Exemptions are structured slice entries so duplicates remain observable and
   rejectable.
6. Provider fixtures own exact native request assertions. The common runner
   owns profile integrity, execution, stream collection, and normalized result
   assertions.
7. Offline activation does not prove live account/model availability, quota,
   cache admission, or service reliability.

## Outcomes

- [x] `llmtest` can run isolated Chat and ChatStream activation cases while
  preserving each request's real model.
- [x] Every targeted capability advertised by an included first-party provider
  has a deterministic case or a non-empty reviewed exemption.
- [x] Tools and required tool choice receive native wire proof rather than only
  normalized tool-response proof.
- [x] OpenRouter cache-control and Codex prompt-cache-key activation are proven
  offline.
- [x] Codex cases explicitly exercise both gpt-5.6 Responses Lite and legacy
  request branches where the shapes differ.
- [x] Malformed profiles fail with actionable test errors and no framework
  panic.
- [x] No ordinary test requires credentials or public network access.

## Priority, effort, and dependencies

| Phase | Priority | Effort | Risk | Depends on |
|---|---|---:|---|---|
| 1. Model-preserving activation runner | P1 | M | MED | none |
| 2. OpenAI and Anthropic profiles | P1 | M | MED | Phase 1 |
| 3. Compatible-provider and Codex profiles | P1 | M-L | MED | Phase 1; reuse Phase 2 fixture patterns |
| 4. Documentation and final verification | P1 | S | LOW | Phases 1-3 |

Implement provider profiles in small provider-focused commits. Do not combine
the framework and every provider fixture into one review unit.

## Repository state and safe commits

Before implementation:

```sh
cd /home/peter/Dev/pkieltyka/go-llm
git status --short --branch
git diff --cached --stat
git diff --stat d776f149cc62bba96653838b5bb202eefcf148c4
make test
```

Preserve these planning-time object IDs:

```text
HEAD/index/worktree models.json: ff84f2c6eed559d9bbf1192fe792fb5fc614f602
```

Before and after every implementation commit:

```sh
git rev-parse :models.json
git hash-object models.json
git diff --cached -- models.json
```

The ID must remain unchanged. Keep every implementation commit explicitly
path-limited so the unrelated `reviews/` directory stays excluded. If safe
isolation is unavailable, leave the changes uncommitted.

## Boundaries and invariants

- Keep the root `llm` package and existing capability vocabulary unchanged.
- Do not add dependencies.
- Do not change `RunConformance`, its sentinel models, or its public helper.
- Do not use sentinel models in the new activation runner.
- Do not put provider-specific wire structs, headers, or raw JSON in `llmtest`.
- Do not add empty-capability, provider-extension, or arbitrary custom cases in
  the first version.
- Do not make offline profiles a second credentialed e2e registry.
- Do not infer support from model names; an individual fixture may use an
  explicit model solely to exercise a documented model-dependent branch.
- Do not add real credentials, captured provider payloads, external URLs, or
  network calls to ordinary tests.
- Do not weaken provider `Capabilities()` to make a profile pass.
- Keep every stream bounded and fully consumed.
- Keep failure messages actionable: provider, case, capability, and path.

## Phase 1: Model-preserving activation runner

### Public `llmtest` shape

Add these APIs; names may receive small Go-style adjustments during
implementation, but the semantics are fixed:

```go
type ConformancePath string

const (
    ConformanceChat   ConformancePath = "chat"
    ConformanceStream ConformancePath = "stream"
)

type CapabilityInvocation struct {
    CaseName   string
    Capability llm.Capability
    Path       ConformancePath
}

type CapabilityCase struct {
    Name       string
    Capability llm.Capability
    Paths      []ConformancePath
    Request    func() *llm.Request
    Assert     func(*llm.Response) error
}

type CapabilityExemption struct {
    Capability llm.Capability
    Reason     string
}

type CapabilityProfile struct {
    Cases      []CapabilityCase
    Exemptions []CapabilityExemption
}

type CapabilityProviderFactory func(
    t *testing.T,
    invocation CapabilityInvocation,
) llm.Provider

func RunCapabilityConformance(
    t *testing.T,
    newProvider CapabilityProviderFactory,
    profile CapabilityProfile,
)
```

There is deliberately no `RunProviderConformance` wrapper in this version.
First-party tests call existing `RunConformance` with their base fixture and
`RunCapabilityConformance` with an invocation-aware fixture. This keeps the
existing base factory simple and prevents a wrapper from hiding two materially
different fixture contracts.

### Provider-factory and invocation contract

1. The runner first calls `newProvider(t, CapabilityInvocation{})` to construct
   an isolated probe provider and read `Name()` and `Capabilities()`. The
   factory must accept the zero invocation without issuing requests.
2. For each case/path, the runner calls the factory again with a fully populated
   immutable invocation before it calls `Request`. A local fixture handler can
   close over that value and assert the corresponding native body and headers.
3. The invocation is test control data only. It is never copied into the LLM
   request, model, headers, URL, or provider options by `llmtest`.
4. The request returned by `Request` retains its model byte-for-byte. The runner
   does not rewrite or normalize it.
5. Each case/path receives a fresh provider and a fresh `Request()` call.
6. The targeted capability subset returned by every case provider must equal
   the probe provider's targeted subset; fail if a factory silently changes
   advertised coverage by invocation.

### Reviewed activation-sensitive set

Define one unexported, tested list in `llmtest`:

- `CapabilityTools`;
- `CapabilityToolChoiceRequired`;
- `CapabilityJSONSchema`;
- `CapabilityImageInput`;
- `CapabilityPromptCaching`;
- `CapabilityStopSequences`;
- `CapabilityReasoning`.

For each of these advertised by the probe provider, require exactly one or more
cases or one exemption. A capability may have multiple cases when distinct
model branches or native features need proof. Cases and exemptions outside
this set are rejected rather than silently ignored.

Do not include streaming, tool streaming, parallel tools, strict tools, JSON
mode, PDF input, session affinity, cost reporting, or models listing in this
initial completeness set. Their existing focused tests and live coverage stay
in force; add them later only through a reviewed plan with deterministic
activation contracts.

### Validation and execution contract

Validate the complete profile before executing a request:

1. `newProvider` must be non-nil and its zero-invocation result must be a
   non-nil provider interface/value.
2. Case names must be non-empty and unique.
3. Every case capability must be non-empty, in the reviewed set, and advertised
   by the probe provider.
4. `Request` and `Assert` are both required. A nil request result is a profile
   failure before provider dispatch.
5. Paths must be non-empty, limited to Chat/Stream, and unique within a case.
6. Exemption capabilities must be non-empty, unique, reviewed, and advertised;
   reasons must remain non-empty after trimming.
7. A capability cannot have both any case and an exemption.
8. Every reviewed capability advertised by the provider must have case
   coverage or an exemption in this invocation.
9. A per-case factory result must be non-nil and must preserve the probe's
   reviewed capability claims.
10. Chat calls `Provider.Chat`; Stream calls `Provider.ChatStream` and always
    drains it with `llm.Collect` under the existing conformance watchdog.
11. Require a nil error and non-nil response before invoking `Assert`.
12. Wrap failures with provider name, case, capability, and path.

The framework promises not to panic on nil or malformed profile values listed
above. Panics raised inside caller-provided factories, request callbacks,
assertion callbacks, or provider implementations remain ordinary test panics;
the runner does not recover arbitrary user code.

### Framework tests

Use `llmtest.Provider` and small local factories to prove:

- zero-invocation probing and one fresh provider per case/path;
- Chat and fully collected Stream execution;
- real request models remain unchanged, including punctuation and model IDs
  that resemble the old conformance sentinels;
- the factory receives the exact immutable case/path invocation;
- capability-present, capability-missing, and per-invocation claim-drift
  behavior;
- nil factory, nil provider, nil `Request`, nil request result, and nil `Assert`
  fail through the test contract rather than framework panics;
- duplicate/empty names and duplicate/unknown/empty paths fail;
- empty, custom, out-of-target, stale, duplicate, blank, and contradictory
  exemptions fail;
- advertised targeted capabilities require a case or exemption;
- assertion errors carry provider/case/capability/path context;
- callback panic behavior is documented and not misleadingly covered by the
  malformed-profile guarantee.

No HTTP request-body restoration helper is added or tested: out-of-band
invocation makes it unnecessary.

Run:

```sh
go test -race -count=1 ./llmtest
```

### STOP conditions

- If case selection requires modifying `Request.Model` or parsing fixture HTTP
  bodies in the common runner, stop and redesign the factory boundary.
- If the runner needs provider-specific types or wire assertions, keep those in
  the provider package.
- If an additive implementation would change any existing `RunConformance`
  behavior, stop and preserve the current suite.

## Phase 2: OpenAI and Anthropic profiles

Add profiles incrementally in the existing provider conformance test packages.
Keep existing `RunConformance` calls. Add a separate
`RunCapabilityConformance` test/factory whose handler closes over the supplied
`CapabilityInvocation`, asserts the native request, and returns a minimal
deterministic response for the common normalized assertion.

### OpenAI Responses

Cover each advertised reviewed capability:

- tools: native function tool name, schema, and relevant strict setting;
- required tool choice: native required selection value;
- JSON schema: `text.format` name, schema, and strict value;
- image input: the expected Responses image content shape;
- prompt caching: the deterministic session/cache key request field;
- reasoning: unified effort in `reasoning.effort` and normalized reasoning
  output;
- stop sequences: not advertised, so no case or exemption.

Do not repeat Plan 6's reasoning-summary selector cases.

### Anthropic Messages

Cover:

- tools and required tool choice in the native tool configuration;
- JSON schema in the native structured-output configuration;
- image input content blocks;
- explicit cache-control blocks plus normalized cache read/write usage;
- native stop sequences;
- the selected unified effort/thinking configuration plus normalized visible
  reasoning.

If a reviewed advertised capability truly lacks a deterministic local request
or response contract, add one narrow non-empty exemption explaining that exact
gap. Do not exempt functionality merely because live service admission is
nondeterministic when the wire field itself is deterministic.

Run:

```sh
go test -race -count=1 ./providers/openai ./providers/internal/responsesapi
go test -race -count=1 ./providers/anthropic
go test -race -count=1 ./llmtest
```

## Phase 3: Compatible-provider and Codex profiles

### OpenRouter Chat Completions

Cover:

- tools and required tool choice in compatible native fields;
- JSON schema response-format fields;
- compatible image content;
- stop values;
- nested OpenRouter reasoning effort plus normalized reasoning;
- prompt caching through deterministic `cache_control` request blocks,
  including the enabled tool-result form where that path is distinct.

Do not exempt prompt caching merely because positive backend cache hits are
nondeterministic. That limitation belongs only to live-e2e evidence.

### vLLM preset

Cover every reviewed capability it advertises:

- tools and required tool choice;
- JSON schema using the configured documented vLLM wire shape;
- image input;
- stop sequences;
- reasoning effort/budget through a deterministic configured dialect fixture.

Prompt caching is not advertised and needs no entry. Document any case whose
meaning depends on a server flag, but keep the local configured wire assertion
deterministic.

### Generic Chat Completions and Ollama preset

Define one reusable compatible-engine profile constructor for tools, required
tool choice, JSON schema, image input, stop sequences, and reasoning request
construction. Apply it to:

- the public generic `chatcompletions.New` engine; and
- the data-only Ollama preset.

Keep `TestChatCompletionsConformance` for the advanced `NewWithDialect`/
`replayDialect` seam on base `RunConformance` only. Document this as the single
exception to first-party activation migration: the public generic engine owns
activation completeness, while the advanced suite retains distinct base-engine
coverage without duplicating the profile.

### OpenAI Codex subscription provider

Assign a real explicit model to every case. Never let the runner choose or
replace it.

- Use a gpt-5.6 model for tools, required tool choice, reasoning, and any other
  case whose native request is rewritten by Responses Lite. Assert the Lite
  header, moved `tools`/`instructions` fields, `parallel_tool_calls: false`, and
  `reasoning.context: "all_turns"` where applicable.
- Add at least one pre-gpt-5.6 case that pins the legacy Codex Responses shape.
- Where a capability's native shape differs between the two branches, add two
  cases rather than treating one branch as representative.
- Prove prompt caching with the deterministic `SessionID` to
  `prompt_cache_key` mapping. Exempt only positive cache admission/telemetry in
  credentialed live tests.
- Do not add OpenAI Plan 6's reasoning-summary option.

### Fixture rules

- Construct the fixture after receiving `CapabilityInvocation` and close over
  the value; do not use mutable globals.
- Decode requests into small local structs or `map[string]json.RawMessage`.
- Assert exact native field names, nesting, headers, and relevant values.
- Use structural JSON comparison unless byte equality is itself contractual.
- Use tiny synthetic image data and deterministic token counts.
- Run both paths when mapping or normalization differs. When a case declares
  Stream, drain it to completion before asserting.
- Do not infer support from the model name except inside Codex's explicit
  branch-coverage fixtures, where the production adapter itself is
  model-dependent.

Run provider groups after each profile:

```sh
go test -race -count=1 ./providers/chatcompletions ./providers/openrouter ./providers/vllm ./providers/ollama
go test -race -count=1 ./providers/openaicodex
go test -race -count=1 ./llmtest ./internal/e2e
```

### STOP conditions

- If activation cannot be proven without a live service, add a precise
  exemption; do not count fixture success as proof.
- If a setting is intentionally ignored, correct the capability claim or add
  an accurate exemption rather than weakening the assertion.
- If a provider needs new public request fields, split that feature into its
  own approved plan.
- If fixture work becomes a broad provider rewrite, stop after the runner plus
  OpenAI/Anthropic and split the remaining profiles.

## Phase 4: Documentation and final verification

Update:

- `llmtest` package docs and examples;
- `README.md` testing/provider-author guidance;
- `specs/projects/go-llm/functional_spec.md` conformance contract;
- `specs/projects/go-llm/architecture.md` offline activation design;
- `specs/projects/go-llm/provider_capabilities.md` distinction between an
  advertised claim, offline native activation, live testing, and exemption.

Document explicitly:

- `RunConformance` proves the common lifecycle and normalization contract;
- `RunCapabilityConformance` proves reviewed native feature activation plus a
  normalized result;
- `CapabilityInvocation` is fixture control data and never changes a request;
- offline wire proof is not live availability proof;
- exemptions are explicit evidence gaps, not capability denials;
- the advanced `NewWithDialect` suite intentionally remains base-only.

Final commands:

```sh
gofmt -w <changed-go-files>
go vet ./...
make test
make check
git diff --check
git status --short
```

Review the final diff for:

- any changed existing `RunConformance` behavior;
- request-model stamping or a body-parsing case helper;
- provider wire types leaking into `llmtest`;
- fixture success without native request assertions;
- missing tools or cache activation proof;
- Codex cases that exercise only one request branch;
- duplicate offline/live registries;
- lazy streams that were not consumed;
- external credentials, URLs, or network calls;
- accidental changes to the staged `models.json`.

Re-check the recorded `models.json` object IDs and cached diff before any
commit and at final verification.

## Acceptance criteria

- Existing `RunConformance` callers and behavior remain unchanged.
- The new runner preserves each request model and passes case identity only to
  the provider factory.
- Nil and malformed profiles fail without framework panics and with actionable
  context.
- Structured exemptions detect duplicates and cannot hide unreviewed or
  unadvertised capabilities.
- Every advertised reviewed capability in each included provider has native
  activation evidence or a reviewed exemption.
- Tools, required tool choice, OpenRouter cache-control, and Codex
  prompt-cache-key activation are explicitly proven.
- Codex gpt-5.6 Lite and legacy paths are both pinned where their shapes differ.
- All streams are collected and all ordinary tests remain offline.
- Focused tests, vet, `make test`, `make check`, and diff checks pass.
- The staged snapshot remains intact and outside every implementation commit.

## Suggested implementation commits

1. `Add model-preserving capability conformance runner`
2. `Prove OpenAI and Anthropic capability activation`
3. `Prove compatible-provider capability activation`
4. `Prove Codex capability activation by model branch`
5. `Document offline capability evidence`

Use explicit path-limited commits or a temporary index, and verify the staged
snapshot object IDs after each commit.
