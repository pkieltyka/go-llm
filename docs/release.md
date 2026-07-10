# Release Checklist

This checklist is version-neutral. Replace `$VERSION` with the intended tag.
Release verification requires Go 1.26.5 or newer.

## Offline Gates

Run the same commands after any fixture or model snapshot refresh:

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
./scripts/check-coverage_test.sh
./scripts/check-coverage.sh
go test -count=1 -tags=live -run '^TestLiveRunnerManifests$' ./internal/e2e
go test -count=1 -tags=live -run '^$' ./...
go run golang.org/x/vuln/cmd/govulncheck@v1.5.0 ./...
go test -count=1 ./internal/e2e -run '^TestRecordedFixturesAreRedacted$'
gitleaks detect --source . --no-git --redact --no-banner
gitleaks detect --source . --redact --no-banner
```

The root-only entries in `.gitleaksignore` suppress the known rule findings
from the ignored local `gollm-test.json` by their no-git fingerprints. Git
history findings include a commit hash in the fingerprint, so committing that
file still fails the history scan. Keep the credential file at mode `0600`;
update the narrow fingerprints only when its line layout changes.

## Credentials

Copy `gollm-test.json.sample` to the gitignored `gollm-test.json` and fill
only the providers available for this release check:

- `anthropic`: API-key credential.
- `openai`: API-key credential.
- `openai-codex`: OAuth credential from a compatible ChatGPT/Codex login.
- `openrouter`: API-key credential.
- `vllm`: keyless self-hosted entry with `base_url`; add `key` only when the
  server uses `--api-key`. `model` is an optional preference, not an exact ID.

The harness loads credentials through `llm.LoadAuthFile`. Missing entries
skip visibly. A configured credential that is malformed or rejected is a
failure. Refreshable OAuth credentials use context-aware, error-returning
persistence; renewal is published only after the updated file is durably
written.

Never commit `gollm-test.json`.

## Credentialed Live Gate

```sh
go test -count=1 -tags=live -v ./internal/e2e
```

The capability-derived manifests cover the scenarios each configured
provider claims. This includes stream grammar, tools and tool streaming,
structured output, usage, error mapping, reasoning/replay where supported,
positive prompt-cache evidence where claimed, model listing, and
cross-provider canonical-history handoff. vLLM resolves the configured model
preference against `/v1/models`, so a preference such as `qwen` can select a
deployment-qualified Qwen model.

Missing credentials may skip; rejected configured credentials may not. Keep
the complete verbose output as release evidence.

## Fixture Recording

Record only when provider wire fixtures need refreshing:

```sh
go test -count=1 -tags=live -v ./internal/e2e -record
go test -count=1 ./...
go test -count=1 ./internal/e2e -run '^TestRecordedFixturesAreRedacted$'
```

Recording is staged in the destination directory, structurally sanitizes
headers, URLs, JSON, and SSE JSON, replaces correlatable identifiers with
deterministic `MOCK_*` values, scans staged bytes for known and high-entropy
secrets, and atomically replaces a fixture only after validation. Rotated
OAuth credentials are registered with the same secret set. Incomplete or
abandoned captures leave the existing fixture untouched unless an intentional
partial recording is explicitly acknowledged with
`-record-allow-incomplete`.

Review every fixture diff despite these controls. A new provider auth shape
must ship with matching redaction tests before its recordings are accepted.

## Model Snapshot

Refresh only when model sources or overrides change:

```sh
make models
pnpm --dir scripts test
```

The script validates the models.dev object-map and OpenRouter `data[]`
sources, provider presence and minimum counts, positive limits, nonnegative
pricing, identity loss, and material metadata loss. It merges deterministically
without replacing known values with `undefined`, writes `models.json`
atomically, and refuses destructive changes unless `--allow-destructive` is
explicitly supplied after review. ZAI is not in the current snapshot provider
set.

## Coverage Floors

`scripts/check-coverage.sh` enforces these current floors. Ordinary rows
measure one package; the owned groups instrument the implementation package
with `-coverpkg` through its real facade/provider consumers.

| Package | Floor |
| --- | --- |
| `.` | 81% |
| `llmtest` | 61% |
| `cmd/llm-cli` | 77% |
| `providers/anthropic` | 78% |
| `providers/openai` | 82% |
| `providers/openaicodex` | 69% |
| `providers/openrouter` | 75% |
| `providers/chatcompletions` | 77% |
| `providers/vllm` | 84% |
| `providers/ollama` | 100% |
| `providers/internal/responsesapi` | 77% |
| `providers/internal/provideroauth` | 71% |
| `internal/schemajson` (owned group) | 82% |
| `providers/internal/providerutil` (owned group) | 76% |

Ratchet a floor upward only after remeasuring and updating both this table and
the script in the same change.

## Tagging

After review approval and all applicable gates pass:

```sh
git tag "$VERSION"
git push origin "$VERSION"
```
