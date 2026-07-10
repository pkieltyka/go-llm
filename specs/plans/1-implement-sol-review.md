---
status: complete
---

# Sol Max Review Implementation Plan

Source: `specs/projects/go-llm/reviews/2-sol-max-review.md`.

The plan artifact is complete; implementation checkboxes remain open. Execute phases in order. Keep the existing architecture and avoid unrelated feature or provider work.

## Phase checklist

- [x] Phase 1: credential boundaries and fail-closed fixture recording
- [x] Phase 2: stream grammar and provider equivalence
- [x] Phase 3: core lifecycle and API correctness
- [ ] Phase 4: schema and OAuth correctness
- [ ] Phase 5: first-party consumers and release tooling
- [ ] Phase 6: executable conformance and live/coverage confidence
- [ ] Phase 7: SDK surface, documentation synchronization, and final gates

## Phase 1: Credential boundaries and fixture recording

Findings: R1, R2, S1, V5, V7.

1. In `providerutil`, remove ambient `Authorization` unconditionally before applying provider-owned authentication. Verify Chat, ChatStream, and SDK-backed Models paths for generic Chat Completions, vLLM, Ollama, OpenRouter, OpenAI, and Codex; harmless ambient custom headers must remain supported.
2. Represent compatible-provider authentication as unset, explicitly empty, or configured. Resolve environment fallback once in each public preset constructor. The shared engine must never reread provider credential environment variables. `openrouter.WithAPIKey("")` must return `ErrAuth` without issuing a request.
3. Stage fixture output in the destination directory; structurally sanitize URL/query values, headers, JSON bodies, and JSON carried in SSE data; run the fixture guard and entropy scan on staged bytes; atomically rename only after the test and recording finalize successfully.
4. Make expected-scenario completeness a warning plus explicit `-record-allow-incomplete` acknowledgement, not an absolute gate, so intentional single-scenario recordings remain possible. Never silently replace a complete fixture with a partial run.
5. Finalize response captures on EOF or Close, track outstanding response bodies, and report an abandoned body as an incomplete exchange before fixture replacement. This must not turn a consumer early break into a transport error.
6. Feed all credentials produced by `PersistOnRefresh` into the recorder's secret set before staging. Replace private hosts with one stable fixture hostname and use per-fixture sequential `MOCK_*` identifiers where correlation is needed.
7. Add a narrow entropy allowlist for the exact committed red-pixel PNG in `internal/e2e/scenario.go`; do not allowlist a general image, base64, or entropy pattern.

Acceptance:

- No ambient bearer value reaches any caller-supplied or keyless endpoint, while configured provider credentials still do.
- Failed, incomplete, unsafe, or abandoned recordings leave the tracked fixture byte-for-byte unchanged unless incomplete replacement was explicitly acknowledged.
- Rotated secrets, URL credentials, sensitive metadata, private hosts, and high-entropy tokens cannot enter staged fixtures.

```sh
OPENAI_CUSTOM_HEADERS='Authorization: Bearer ambient-secret' go test -count=1 ./providers/chatcompletions -run '^TestGenericNew$'
OPENAI_CUSTOM_HEADERS='Authorization: Bearer ambient-secret' OPENROUTER_API_KEY='openrouter-env-secret' go test -count=1 ./providers/chatcompletions ./providers/openrouter ./providers/vllm ./providers/ollama ./providers/openai ./providers/openaicodex
go test -count=1 ./internal/e2e -run 'TestRecordedFixturesAreRedacted|TestWriteFixture|TestWireTap'
go test -race -count=1 ./providers/... ./internal/e2e
```

## Phase 2: Stream grammar and provider equivalence

Findings: C2, C7, C8, C11, C12, V1, V2, plus dropped block indexes.

1. Add one `providerutil` stream-contract wrapper used by Responses, Chat Completions, Anthropic, and Codex. A fully exhausted successful stream must start with exactly one MessageStart and end with exactly one MessageEnd; empty or truncated EOF yields a `ProviderError` wrapping `ErrServer`.
2. Track whether downstream returned `false` from `yield`. Do not synthesize a truncation error after consumer early break; enforce terminal grammar only when upstream was actually exhausted.
3. Buffer pre-start Responses events, emit MessageStart first, and use the request model when `response.created` is absent.
4. For Chat Completions, consume only entries whose `choice.Index == 0`; never use `Choices[0]` positionally for extras or finish reason. Require at least one real choice, allow trailing usage-only chunks, and reject choice-less warm-up completion.
5. Define BlockIndex as the stable provider/wire output position. Retained parts preserve that position, malformed dropped items may leave gaps, and already emitted events are never renumbered; blocking Chat and `Collect(ChatStream)` must expose equal retained parts and BlockIndex positions.
6. For Anthropic, use only scoped context-overflow phrases, merge late tool metadata into buffered argument state, omit assistant messages emptied by foreign/unknown-part filtering, and make `message_stop` without `message_delta` terminate correctly.
7. Centralize transport/decode normalization: preserve `context.Canceled`, `context.DeadlineExceeded`, and existing typed provider errors; wrap unknown remote I/O or parse failures in `ProviderError` with the stable sentinel.
8. Omit `parallel_tool_calls` entirely when `CapabilityParallelTools` is absent; do not send `false`.

Acceptance:

- Engine fixtures cover empty EOF, truncated EOF, pre-start deltas, early break, Anthropic `message_stop`, nonzero-only choices, usage-only tails, malformed tool drops, and unknown decoder/transport errors.
- Equivalent blocking and streaming inputs produce the same retained parts and stable provider/wire BlockIndex positions, including the same gaps after dropped items, plus the same finish reason, usage, and provider error classification.

```sh
go test -count=1 ./providers/internal/providerutil ./providers/internal/responsesapi ./providers/chatcompletions ./providers/anthropic ./providers/openai ./providers/openaicodex ./providers/openrouter ./providers/vllm ./providers/ollama
go test -race -count=1 ./providers/internal/... ./providers/chatcompletions ./providers/anthropic ./providers/openai ./providers/openaicodex
```

## Phase 3: Core lifecycle and API correctness

Findings: C1, C3, C4, C5, C6 and the valid small API cleanups.

1. Make request preflight reject typed nil ProviderOptions across every reflection nil-able kind (chan, func, interface, map, pointer, slice) before method calls. Call `ForProvider` once, reject `MaxTokens < 0`, duplicate tool names, and forced tool choices naming an undeclared tool.
2. Move Session.ChatStream history mutation, request construction, provider stream creation, and rollback setup into the first-range closure. Test never-ranged streams and two streams created before either is consumed.
3. Isolate Parse modes. Native mode must force JSON Schema and preserve only supported name/schema overrides. Tool mode must replace caller tools with one collision-free synthetic parse tool and decode exactly one call matching its name, regardless of call order.
4. Make usage/cost provenance commutative by tracking whether every token-bearing component supplied cost independently of aggregation order; test native, estimated, and missing-cost permutations.
5. Compose Chat and Stream middleware factories once in Wrap, retain Bind's once-only behavior, and test invocation counts over repeated calls.
6. Return a fresh DefaultHTTPClient while sharing only a private transport; reject negative RetryAfter seconds and clamp past HTTP dates to zero; clear stale `AuthCredential.Expires` on unmarshal; reject UnknownPart values using reserved built-in discriminators.
7. Do not add documentation work for the refuted claim that RetryAfter is absent from normative design. Its runtime edge-case fix above remains required.

```sh
go test -count=1 .
go test -race -count=1 .
go test -shuffle=on -count=10 .
```

## Phase 4: Schema and OAuth correctness

Findings: C9, C10, V3, V4 (with Parse mode selection implemented in Phase 3).

1. Introduce exact JSON-number canonicalization without float conversion. Validate arbitrary-precision integers, preserve unsigned enum values above MaxInt64, and compare numerically equivalent enum spellings such as `1`, `1.0`, and `1e0` as equal.
2. Recognize Go 1.26 `omitzero`. Validate the supported schema shape before arguments so malformed `required` keywords and non-string members fail closed.
3. Add native Parse wire tests proving the derived schema is always sent, never silently downgraded to JSON mode, and still enforced by client-side validation.
4. Keep one OAuth refresh generation in flight through ordered persistence callback completion before publishing/waking waiters. A waiter's cancellation must stop only that wait, and cancellation assertions must use `errors.Is`.
5. Let access-only credentials inside the refresh margin remain usable until actual expiry; after expiry, absence of a refresh token returns the appropriate auth failure.

```sh
go test -count=1 . ./schema ./internal/schemajson ./providers/internal/provideroauth
go test -race -count=1 ./providers/internal/provideroauth
```

## Phase 5: First-party consumers and release tooling

Findings: S2, S3, S4, T5, T6, V6. Skip lockfile work already completed.

1. Rework WireTap request capture as a bounded tee consumed by the transport. Close all failure paths, preserve GetBody retryability, centralize sensitive-header classification, and preserve the original ContentLength exactly, including unknown/chunked length.
2. For Codex CLI auth, use precedence `--auth-file`, then `OPENAI_CODEX_ACCESS_TOKEN`, then compatibility `--api-key`. Load files only when explicitly requested, document argv exposure, and test actual configured behavior rather than option forwarding alone.
3. Stream CLI print side effects through the same iterator passed directly to Collect. Return all output errors immediately while retaining the partial response/error contract; cover broken writers in response, event, usage, and error output.
4. Update the tools example to dispatch by tool name, validate argument JSON, convert validation/execution failures to IsError results, group all parallel results, and call AddToolResults once.
5. Validate model-source schemas, expected providers, sane row counts, and destructive deltas; require an explicit destructive-update override; test from recorded upstream fixtures; write `models.json` atomically; remove deferred ZAI from the current provider set.
6. Do not recreate or otherwise churn `scripts/pnpm-lock.yaml`; the S4 lockfile subclaim is already resolved.

```sh
go test -count=1 . ./cmd/llm-cli ./examples/...
go test -race -count=1 . ./cmd/llm-cli
pnpm --dir scripts install --frozen-lockfile
pnpm --dir scripts test
```

## Phase 6: Executable conformance and confidence

Findings: T1, T2, T3, T4, T7, V8.

1. Strengthen `llmtest.RunConformance` with a fixture that blocks after its first event until cancellation. Require prompt termination with `errors.Is(err, context.Canceled)`, no post-error events, MessageStart first, exactly one MessageEnd on success, and ErrServer for empty/truncated EOF. Cover successful independent concurrent streams.
2. Run conformance for every preset, adding the missing Ollama test as well as retaining shared-engine coverage.
3. Make llmtest copies genuinely defensive: clone Temperature/TopP pointers and JSON-compatible `map[string]any`/`[]any` recursively, normalize pointer parts/events to values, and document opaque ProviderOptions as immutable or shallow-copied.
4. Build a capability-to-live-scenario registry with explicit justified exemptions. Inspect raw stream grammar, fail invalid configured credentials, and distinguish them from missing credentials. Resolve configured model aliases before declaring a model absent.
5. Replace facade-only coverage claims with grouped `-coverpkg` profiles for `internal/schemajson` and `providerutil`; enforce owned-engine floors in `scripts/check-coverage.sh` and synchronize release docs.
6. Add credential-free live-tag compilation to CI.

```sh
go test -count=1 ./llmtest ./providers/...
go test -race -count=1 ./llmtest ./providers/...
go test -count=1 -tags=live -run '^$' ./...
./scripts/check-coverage.sh
go test -count=1 -coverpkg=./internal/schemajson -coverprofile=/tmp/schemajson.cover ./schema ./internal/schemajson
go test -count=1 -coverpkg=./providers/internal/providerutil -coverprofile=/tmp/providerutil.cover ./providers/...
go tool cover -func=/tmp/schemajson.cover
go tool cover -func=/tmp/providerutil.cover
```

## Phase 7: SDK surface, documentation, and all gates

1. Make and record the pre-v1 SDK-surface decision. Preferred outcome: replace vendor SDK types in ordinary OpenAI Options with library-owned values or raw JSON extension fields; keep advanced Chat Completions Dialect explicitly unstable. If vendor types remain, revise the stability contract and treat SDK upgrades as public breaking changes.
2. Keep `functional_spec.md` and `architecture.md` authoritative. Synchronize removed files/dependencies, shipped vLLM tokenize and OAuth state, coverage floors, provider lists, project overview, package docs, README claims, version-neutral release guidance, samples, model sources, and deferred ZAI status.
3. Mark phase plans and old reviews historical/non-normative, label provider capability research as research, and label pre-history-rewrite commit references as provenance rather than current links.
4. Require Go 1.26.5 or newer for release verification, add live-tag compilation to CI, and run every gate below. Credentialed live tests are a release gate; fixture recording and model refresh are conditional gates when those artifacts change.

```sh
go version
go mod tidy
go mod verify
test -z "$(gofmt -l $(git ls-files '*.go'))"
git diff --check
go vet ./...
go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.12.2 run --default=none --enable=govet --enable=ineffassign --enable=unused ./...
go test -count=1 ./...
go test -race -count=1 ./...
go test -shuffle=on -count=10 ./...
./scripts/check-coverage.sh
go test -count=1 -tags=live -run '^$' ./...
go run golang.org/x/vuln/cmd/govulncheck@v1.5.0 ./...
go test -count=1 ./internal/e2e -run '^TestRecordedFixturesAreRedacted$'
gitleaks detect --source . --no-git --redact --no-banner
gitleaks detect --source . --redact --no-banner
go test -count=1 -tags=live -v ./internal/e2e
```

When fixtures change:

```sh
go test -count=1 -tags=live -v ./internal/e2e -record
go test -count=1 ./...
go test -count=1 ./internal/e2e -run '^TestRecordedFixturesAreRedacted$'
```

When model sources change:

```sh
make models
pnpm --dir scripts test
```

## Finding-ID coverage matrix

| ID | Phase | Planned disposition |
|---|---:|---|
| R1 | 1 | Delete ambient Authorization before provider auth; test all paths. |
| R2 | 1 | Tri-state auth and one-time environment resolution. |
| C1 | 3 | Nil-safe, single-read request/tool preflight. |
| C2 | 2 | Shared exhausted-stream terminal contract and start ordering. |
| C3 | 3 | Lazy, rollback-safe Session.ChatStream. |
| C4 | 3 | Isolated native/tool Parse requests and named-call decoding. |
| C5 | 3 | Order-independent cost provenance. |
| C6 | 3 | Once-composed middleware factories. |
| C7 | 2 | Choice-index-zero handling and choice-less rejection. |
| C8 | 2 | Anthropic error, tool-delta, and empty-message fixes. |
| C9 | 4 | Exact integers, unsigned enums, omitzero, schema-shape validation. |
| C10 | 4 | Ordered OAuth persistence, independent waiters, access-only expiry. |
| C11 | 2 | Two-layer normalization for unknown provider I/O/decode errors. |
| C12 | 2 | Capability-aware omission of parallel_tool_calls. |
| S1 | 1 | Staged, sanitized, scanned, complete-aware atomic fixtures. |
| S2 | 5 | Bounded request tee and centralized sensitive headers. |
| S3 | 5 | Non-argv Codex auth with explicit precedence. |
| S4 | 5 | Validated, invariant-checked, atomic snapshots; lockfile skipped. |
| T1 | 6 | Executable cancellation and event grammar. |
| T2 | 6 | Defensive llmtest copies and value normalization. |
| T3 | 6 | Capability-driven live registry and alias-aware model checks. |
| T4 | 6 | Owned-engine grouped coverage floors. |
| T5 | 5 | Streaming collection without buffering and checked output errors. |
| T6 | 5 | Correct grouped parallel-tool example. |
| T7 | 6 | CI live-tag compile gate. |
| V1 | 2 | Select extras by choice.Index, never slice position. |
| V2 | 2 | Anthropic message_stop terminal fixture/handling. |
| V3 | 4 | Numeric enum equivalence across JSON spellings. |
| V4 | 3/4 | Force and wire-test derived native Parse schema. |
| V5 | 1 | Detect/finalize abandoned response captures. |
| V6 | 5 | Preserve request ContentLength semantics. |
| V7 | 1 | Redact credentials rotated during recording. |
| V8 | 6 | Add Ollama preset conformance. |

Excluded by verification addendum: no RetryAfter normative-documentation task; no S4 lockfile task. Still included: RetryAfter runtime edge cases, dropped-index equivalence, SDK-surface decision, documentation synchronization, and Go 1.26.5+ release verification.
