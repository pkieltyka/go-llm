---
status: ready
planned_at: c983a92
planned_at_date: 2026-08-23
---

# OpenAI Reasoning Summary Plan

This plan adds bounded reasoning-summary selection to the OpenAI Responses
provider. It is intentionally small and provider-specific. Capability
conformance is independent work in Plan 7 and is not a prerequisite for this
feature.

The plan is written against go-llm commit
`c983a92addf6f1b738e969c16f8b4f4b079fd0a5`. An executor must read it fully,
preserve unrelated worktree changes, and stop rather than broadening the
provider-neutral request API or assuming protocol parity with Codex.

At planning time `models.json` contains unrelated user-owned staged changes.
This plan does not modify, regenerate, unstage, or commit that file.

## Evidence baseline and dispositions

| Finding | Pinned evidence | Disposition |
|---|---|---|
| OpenAI Responses accepts bounded `auto`, `concise`, and `detailed` reasoning-summary selectors. | Gollem `6e024b44eea2c15dab9576c226ed18a586f8704f`, `provider/openai/responses.go`, reasoning summary mapping; the installed OpenAI Go SDK constants are the implementation authority | Implement as an OpenAI-owned option. |
| go-llm currently requests `summary: auto` whenever unified effort is set. | go-llm `providers/internal/responsesapi/adapter.go`, effort mapping | Preserve byte-for-byte when the new option is empty. |
| Provider activation conformance can verify native request fields. | Gollem `6e024b44eea2c15dab9576c226ed18a586f8704f`, `provider/conformance/conformance.go` | Moved to independent Plan 7; do not duplicate the summary tests there. |
| The Codex subscription backend shares some Responses event shapes. | go-llm OpenAI and Codex adapters | Rejected as support evidence: do not add the option to Codex without a separately approved protocol contract. |
| Reasoning-summary selection could be provider-neutral. | Comparison discussion | Rejected: the current vocabulary and behavior are OpenAI-specific. |

PromptTemplate, replay/model switching, routing, agent orchestration, model
gates, and remote catalog behavior are outside this plan and remain unchanged.

## Confirmed decisions

1. Add the option to `openai.Options`, not `llm.Request`.
2. Empty option plus non-empty `Request.Effort` preserves today's automatic
   summary behavior exactly.
3. Validate the bounded vocabulary locally, but let the OpenAI server decide
   which models accept each value.
4. Do not trim, pass through, or silently downgrade unknown values.
5. Do not add the option to OpenRouter or `openaicodex`.
6. Keep request construction shared between Chat and ChatStream.

## Outcomes

- [ ] OpenAI callers can explicitly select `auto`, `concise`, or `detailed`.
- [ ] Existing effort-only requests still send `summary: "auto"` unchanged.
- [ ] Unknown or whitespace-padded values fail as `ErrBadRequest` before any
  network request.
- [ ] Blocking and fully consumed streaming requests serialize the same
  reasoning object.
- [ ] Existing normalized reasoning-summary output remains unchanged.
- [ ] No provider-neutral API, model-name gate, Codex support, or conformance
  framework is added by this plan.

## Priority, effort, and dependencies

| Phase | Priority | Effort | Risk | Depends on |
|---|---|---:|---|---|
| 1. OpenAI option, mapping, and tests | P1 | S | LOW | none |
| 2. Documentation and verification | P1 | XS | LOW | Phase 1 |

Plan 6 is independent of Plans 5 and 7 except for ordinary documentation-file
coordination.

## Repository state and safe commits

Before implementation:

```sh
cd /home/peter/Dev/pkieltyka/go-llm
git status --short --branch
git diff --cached --stat
git diff --stat c983a92addf6f1b738e969c16f8b4f4b079fd0a5
make test
```

Preserve these planning-time object IDs:

```text
index  models.json: dd552b022cd493159bbaaf7f69ccfb6f25e64ded
worktree models.json: dd552b022cd493159bbaaf7f69ccfb6f25e64ded
```

Before and after every implementation commit:

```sh
git rev-parse :models.json
git hash-object models.json
git diff --cached -- models.json
```

Both IDs must remain unchanged. Never use a plain `git commit` while this
unrelated path is staged. Use `git commit --only <explicit changed paths>` with
the necessary intent-to-add step for new files, or use a dedicated temporary
index. If safe isolation is unavailable, leave the changes uncommitted.

## Boundaries and invariants

- Keep the root `llm` package standard-library-only and unchanged by this
  feature.
- Do not add dependencies or upgrade the OpenAI SDK solely for this option.
- Do not add `ReasoningSummary` to `llm.Request`.
- Do not add summary selection to `openaicodex` without separate evidence.
- Do not add model-name inference or model-aware rejection.
- Do not change reasoning replay, encrypted-reasoning retention, or normalized
  `ReasoningPart` behavior.
- Do not change the existing effort-only `summary: "auto"` behavior.
- Do not add capability-conformance types or tests; Plan 7 owns that work.
- Do not change PromptTemplate, sessions, streaming contracts, OAuth, catalogs,
  routing, or agent-layer behavior.

## Phase 1: OpenAI option, mapping, and tests

### Public API

Add a provider-owned type and constants in `providers/openai`:

```go
// ReasoningSummary selects the detail level of OpenAI's user-visible
// reasoning summary. Empty preserves go-llm's existing behavior.
type ReasoningSummary string

const (
    ReasoningSummaryAuto     ReasoningSummary = "auto"
    ReasoningSummaryConcise  ReasoningSummary = "concise"
    ReasoningSummaryDetailed ReasoningSummary = "detailed"
)
```

Add this field to `openai.Options`:

```go
// ReasoningSummary selects OpenAI Responses reasoning.summary. Empty leaves
// the existing effort-driven automatic-summary behavior unchanged.
ReasoningSummary ReasoningSummary
```

String backing allows a future go-llm release to add another constant without
changing the field type. It is not a raw pass-through escape hatch: today the
request builder accepts only empty, `auto`, `concise`, and `detailed`. Unknown
non-empty values return a wrapped `ErrBadRequest` before network I/O. Callers
that need an unrecognized future wire value must use the raw client until the
library explicitly supports it.

### Mapping contract

Apply the provider option after the shared Responses adapter maps
`Request.Effort`:

1. No effort and no option: omit the complete `reasoning` object as today.
2. Effort set and no option: retain the current effort plus
   `summary: "auto"` byte-for-byte.
3. No effort and a valid option: send a reasoning object containing only the
   selected summary; do not invent an effort.
4. Effort and a valid option: preserve the mapped effort and replace the
   automatic summary with the selected value.
5. `EffortNone` plus a selected summary is forwarded without model-aware
   rewriting.
6. Do not mutate caller-owned `Options` or `Request` values.
7. Apply the option in the common OpenAI Responses parameter build so blocking
   and streaming paths cannot diverge.

### Tests

Assert exact request JSON for:

- empty option with empty effort;
- empty option with low and high effort, preserving `summary: "auto"`;
- each of `auto`, `concise`, and `detailed` with empty effort;
- concise and detailed overriding effort-driven auto;
- `EffortNone` plus a selected summary;
- invalid and whitespace-padded values returning `ErrBadRequest`;
- the existing wrong-provider-options concrete-type error;
- caller-owned request and options values remaining unchanged.

For both Chat and ChatStream:

- assert identical reasoning request objects;
- assert normalized reasoning summaries and terminal raw payloads remain
  unchanged;
- consume every stream with `llm.Collect` or range it to termination before
  checking the result or fixture request count;
- for invalid streaming options, collect the iterator, assert
  `errors.Is(err, llm.ErrBadRequest)`, and assert the fixture received zero
  requests.

Run:

```sh
go test -race -count=1 ./providers/openai ./providers/internal/responsesapi
```

### STOP conditions

- If the installed OpenAI SDK cannot represent one of the three values, use
  the existing raw-parameter override seam; do not upgrade solely for this.
- If the current official OpenAI reference has removed or renamed a value,
  stop and update this plan before implementation.
- If Codex support appears desirable, record a separate live-verified plan;
  shared event normalization is not protocol evidence.

## Phase 2: Documentation and verification

Update:

- `README.md` OpenAI provider-options example;
- `providers/openai` package documentation;
- `specs/projects/go-llm/functional_spec.md` provider-options contract;
- `specs/projects/go-llm/architecture.md` OpenAI Responses mapping.

Document that empty preserves existing behavior, the vocabulary is currently
closed, server/model acceptance remains authoritative, and Codex is not
included.

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

- a provider-neutral summary field or model-name gate;
- changed default `summary: "auto"` behavior;
- Codex or OpenRouter summary support;
- lazy streaming tests that were never consumed;
- capability-conformance machinery;
- accidental changes to the staged `models.json`.

Re-check the recorded `models.json` object IDs and cached diff before any
commit and at final verification.

## Acceptance criteria

- The three supported values serialize exactly on Chat and ChatStream.
- Empty option preserves all existing request behavior.
- Invalid values fail locally without a request.
- Summary output normalization remains unchanged.
- No common request type, capability, model gate, or Codex surface changes.
- Focused tests, vet, `make test`, `make check`, and diff checks pass.
- The staged snapshot remains intact and outside every implementation commit.

## Suggested implementation commits

1. `Add OpenAI reasoning summary selection`
2. `Document OpenAI reasoning summary selection`

Use explicit path-limited commits or a temporary index, and verify the staged
snapshot object IDs after each commit.
