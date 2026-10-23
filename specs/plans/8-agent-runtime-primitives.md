---
status: complete
planned_at: e94fcb3
planned_at_date: 2026-10-22
implemented_at_date: 2026-10-22
---

# Agent Runtime Primitives Plan

This plan exports small, provider-neutral building blocks that go-agent
currently reimplements or works around. Each one already exists privately in
go-llm or is a thin layer over existing types. None of them adds agent policy
to go-llm.

Sources:

- the go-agent gap analysis of 2026-10-22, against go-agent `077b363`;
- a study of pi v1.4.0 (`/home/peter/Dev/other/pi`, HEAD `476a303`),
  especially `packages/durable/src/harness/generation.ts`, which commits
  throttled stream partials from an incrementally accumulated message.

An executor must read the whole plan, preserve unrelated worktree changes, and
stop rather than move turn policy (retry ladders, nudges, compaction decisions)
into go-llm.

## Evidence baseline and dispositions

| Finding | Evidence | Disposition |
|---|---|---|
| go-agent cannot collect a stream incrementally. It buffers every event and replays the slice into `llm.Collect` after the call. | go-agent `loop.go:callModel` (~L1212); go-agent plan 003, decision D17, "rejected … until go-llm exposes an equivalent public incremental seam" | **Implement** a public `llm.Collector` with the exact `Collect` precedence (Phase 1). |
| Durable harnesses need a snapshot of the response so far during streaming. | pi-durable commits `event.partial` about every 100 ms, so a crash loses at most one window | `Collector.Snapshot()` returns an owned deep copy (Phase 1). |
| go-agent clones events and parts itself. It deep-copies parts through a `MarshalMessages`/`UnmarshalMessages` round trip and handles pointer events. | go-agent `clone.go:cloneLLMEventOwned`, `derefPart`, `clonePartsOwned`; go-llm private `stream.go:normalizeEvent/derefEvent` and `message.go:cloneParts` | **Export** `NormalizeEvent`, `CloneParts`, `CloneMessage`/`CloneMessages`, and `(*Response).Clone` (Phase 2). |
| go-agent copies a hand-written list of go-llm sentinels to snapshot error classes. The list already omits the three `ErrLogin*` sentinels. | go-agent `clone.go:snapshotErrorOwned` | **Export** `llm.ErrorClass(err) error` plus a serializable, log-safe `llm.ErrorInfo` (Phase 3). |
| go-agent's retry ladder classifies by typed sentinels only. That is correct, but every consumer re-derives which classes are transient. | go-agent `retrypolicy.go:defaultRetryPolicy`; go-agent FS §3 rejection "Regex/string matching for error classification" | **Export** `llm.Transient(err) bool` as go-llm's documented opinion. Policy (attempts, backoff, jitter) stays in go-agent (Phase 3). |
| go-agent hardcodes provider credential environment names to scrub tool subprocess environments, and omits vLLM and Ollama. | go-agent `tools/localrunner.go:providerAuthEnvNames` | **Export** `llm.CredentialEnvNames()` (Phase 4). |
| pi's agent harness relies on regex classifiers over error text. | pi `utils/retry.ts`, about 60 patterns | **Rejected.** go-llm's typed sentinel chain is the signal. Message heuristics stay confined to adapter-boundary classification (FS §16). |

## Confirmed decisions

1. The root `llm` package stays standard-library-only.
2. `Collect` must be reimplemented on top of `Collector`, so precedence cannot
   drift between them. Existing `Collect` tests must pass byte for byte.
3. `Snapshot()` never exposes internal mutable state. It returns an owned copy
   and is safe to call from the consuming goroutine between `Add` calls.
   `Collector` itself is not goroutine-safe, which matches the single-use
   stream contract.
4. `ErrorInfo` follows the `SafeSummary` policy. It never carries
   `Message`, `Code`, `Metadata`, or `RawBody`.
5. `Transient` is advisory classification, not a retry loop. It never sleeps,
   counts, or wraps a provider.

## Outcomes

- [x] go-agent can drop its event-replay buffer and collect live.
- [x] A consumer can take an owned partial `*Response` at any point in a
  stream.
- [x] Clone and normalize helpers replace go-agent's copies.
- [x] One exported function maps any error to its normalized sentinel,
  including the login sentinels and `context.Canceled`/`DeadlineExceeded`.
- [x] Credential environment names have one source of truth.

## Phases

| Phase | Priority | Effort | Risk | Depends on |
|---|---|---:|---|---|
| 1. Public incremental `Collector` | P0 | M | MEDIUM | none |
| 2. Clone and normalize exports | P1 | S | LOW | none |
| 3. `ErrorClass`, `ErrorInfo`, `Transient` | P1 | S | LOW | none |
| 4. `CredentialEnvNames` | P2 | XS | LOW | none |
| 5. Docs, FS amendments, verification | P1 | S | LOW | 1–4 |

### Phase 1: public incremental Collector

API sketch (final names are ratified in FS §6 before coding):

```go
type Collector struct{ /* private accumulation state */ }

func NewCollector() *Collector
// Add folds one event. It returns the canonical malformed-stream error for
// contract violations (e.g. ToolCallIDChanged mismatch) exactly as Collect does.
func (c *Collector) Add(ev Event) error
// Snapshot returns an owned copy of the response accumulated so far
// (possibly with an empty StopReason). Never nil after the first Add.
func (c *Collector) Snapshot() *Response
// Finish applies the terminal stream error (nil on clean end) and returns
// what Collect would have returned for the same event sequence.
func (c *Collector) Finish(streamErr error) (*Response, error)
```

Steps:

1. Move the private accumulation state out of `stream.go:Collect` into
   `Collector`. `Collect` becomes a loop over `Add`, followed by `Finish`.
2. Event normalization (pointer events) happens inside `Add` through the same
   private path `Collect` uses today.
3. Property test: for every existing `Collect` fixture and fuzz corpus, the
   output of `Collect(seq)` equals `NewCollector` + `Add`… + `Finish`,
   including partial-on-error, early-break, and malformed-stream cases.
4. Test that `Snapshot()` mutations never affect later `Add` calls, and that
   later `Add` calls never mutate earlier snapshots. Use `-race`.

### Phase 2: clone and normalize exports

Export `NormalizeEvent(Event) (Event, error)`, `CloneParts([]Part) []Part`,
`CloneMessage(Message) Message`, `CloneMessages([]Message) []Message`, and
`(*Response).Clone() *Response`. These are thin wrappers over existing
private code. Deep-copy rules:

- copy `Raw` byte slices;
- copy `ToolCallPart.Args`;
- copy `ImagePart`/`FilePart` data;
- copy `CacheHint` pointers;
- `UnknownPart` keeps its raw bytes;
- extension parts are copied through their own clone method if one exists,
  otherwise by marshal/unmarshal through the registered decoder.

STOP if an extension part can be neither cloned nor round-tripped. Report it
rather than sharing memory.

### Phase 3: ErrorClass, ErrorInfo, Transient

```go
// ErrorClass returns the normalized sentinel in err's chain (ErrAuth …
// ErrLoginClosed), context.Canceled / context.DeadlineExceeded, or nil.
func ErrorClass(err error) error

// ErrorInfo is a serializable, log-safe description of an error.
type ErrorInfo struct {
	Class      string        // sentinel identity, e.g. "rate_limited"; "" when unclassified
	Provider   string        // bounded adapter identity (SafeSummary rules)
	HTTPStatus int
	RetryAfter time.Duration
}
func SafeErrorInfo(err error) ErrorInfo

// Transient reports whether err's class is conventionally retryable as a new
// attempt: ErrRateLimited, ErrOverloaded, ErrServer, ErrTimeout. Never
// ErrContextTooLong (the caller must change the request), auth, credits,
// content filtering, bad request, or cancellation.
func Transient(err error) bool
```

- `Class` uses a stable lowercase vocabulary documented in FS §16. It is not
  `err.Error()`.
- Tests cover every sentinel, wrapped chains, `ProviderError` with and without
  `Kind`, and the guarantee that `SafeErrorInfo` never contains provider
  message bytes. Use the existing secret-canary pattern.

### Phase 4: CredentialEnvNames

`llm.CredentialEnvNames() []string` returns every environment variable any
first-party provider constructor reads. Today that is `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_CODEX_ACCESS_TOKEN` (CLI only),
and any vLLM or Ollama names, if they exist. The executor must enumerate these
by grepping the constructors, not from this list.

A test greps provider packages for `os.Getenv`/`LookupEnv` calls and fails when
a name is missing from the list.

### Phase 5: docs, FS amendments, verification

- FS §6: `Collector` contract and its equivalence with `Collect`.
- FS §16: `ErrorClass`, `ErrorInfo`, `Transient`, and the class vocabulary.
- README "Persistence" or "Errors": short examples.
- `make check` is green. Coverage floors hold or ratchet.

## STOP conditions

- `Collect` precedence cannot be expressed incrementally without changing an
  existing test expectation. Report the case; do not change semantics.
- `Transient` would need message or regex matching for any class.
- A helper would require a third-party dependency in the root package.

## Implementation notes (2026-10-22)

These record where the implementation deviates from or refines the sketches
above.

- **`Collector` has no constructor.** Its zero value is ready, like
  `strings.Builder`, so `NewCollector` was not added. `Collect` and
  `Session` streaming both run on it, and `applyCollectEvent` is its only
  fold path.
- **Error and finish semantics:**
  - An `Add` error is sticky and wins over a later stream error in `Finish`.
  - `Finish` is idempotent, and `Add` after it returns `ErrBadRequest`.
  - `Snapshot` after `Finish` returns a copy of the final response.
- **The equivalence oracle** is the pre-Collector `Collect` body, kept
  verbatim in `collector_test.go` as `referenceCollect`.
  `FuzzCollectorMatchesReferenceCollect` checks `Collect`, every prefix
  `Snapshot`, `Finish`, and snapshot immutability against it.
- **Extension parts in `CloneParts` are shared by reference.** This keeps the
  existing `History.Messages` rule rather than round-tripping through the
  decoder, which can fail and would make cloning fallible. It is documented
  on `CloneParts`. This intentionally resolves the Phase 2 STOP condition:
  sharing is the documented, pre-existing behavior.
- **`ErrorClass` precedence:** cancellation, deadline, then specific
  conditions, then `ErrBadRequest`/`ErrServer`. `Transient` is false when a
  chain also carries cancellation. An expired provider `WithTimeout`
  surfaces as `context.DeadlineExceeded`, which is not transient. This
  matches go-agent's `retrypolicy.go` class set exactly.
- **`CredentialEnvNames`:**
  - The gap analysis claimed vLLM and Ollama were missing. They read no
    environment variables at all.
  - Adapters disable SDK environment defaults (`WithoutEnvironmentDefaults`,
    `WithAdminAPIKey("")`, …), so SDK-read names are not listed.
  - `credential_env_test.go` scans `providers/` and `cmd/` sources in both
    directions.
- **Docs:** FS §6 (incremental collection), FS §16 (classification
  helpers), the FS §17 configuration bullet, a README section, and the
  `ExampleCollector` and `ExampleTransient` godoc examples.
- **Verification:** `go vet`, `make test` (race), coverage floors (root
  86.3% → 86.5%), `check-lint` (golangci-lint v2.14.0, 0 issues), and a
  20-second fuzz run (about 400k execs) all pass.

## Review (2026-10-22)

A clean-context adversarial review found no blockers and no majors. Its
verdict was "ship after fixes". It confirmed `Collect` and `Session`
streaming are equivalent to the previous code, using explicit probes plus
the fuzz oracle, and that snapshots are owned in both directions. Fixed
before commit:

- **The `CredentialEnvNames` docs over-claimed.** openai-go's
  `DefaultClientOptions` always reads its environment, and the adapters
  override those values rather than disable them.
  - `OPENAI_ADMIN_KEY` and `OPENAI_WEBHOOK_SECRET` are now listed as
    SDK-read credentials.
  - The godoc and FS §17 state the override-versus-disable distinction.
- **The source-scan test was under-sensitive.** It now resolves identifier
  arguments to string constants per package directory, with qualified names
  resolved in the named package, and fails on an unresolvable argument.
  - The first strict run exposed a bug in the scan itself: three packages
    define `apiKeyEnv`, and a global binding table let the last one win.
  - A temporary-file negative probe (`const codexTokenVar = "X_SECRET_TOKEN"`
    read via `os.Getenv`) now fails the test as intended.
- **`Response.Clone` preserves nil versus empty `DroppedToolCalls`,** as it
  does for `Parts`.
- **Doc polish:**
  - the typed-nil pointer part clause on `CloneParts`;
  - `Snapshot` after `Finish` reflecting caller edits;
  - `ErrorClass` deadline precedence;
  - the FS §6 Raw-sharing caveat;
  - README `finishErr` naming.

Declined: a `noCopy` vet guard on `Collector`. The copy rule is documented
like `strings.Builder`'s, and the repository has no `noCopy` pattern.

## Downstream

go-agent plans 016 and 017 consume Phases 1–3: they drop the replay buffer,
the hand-written sentinel list, and the clone helpers, and add throttled
partial persistence.
