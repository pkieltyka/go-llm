# B2 stage 1 — local evidence, 2026-10-27

Go LLM baseline and unchanged HEAD:
`b62ba7b4cabad94e5c8d508fd2be3d35ba5d18ae` (v0.6.1).
Implementation passed independent review with no findings before commit.
Go Agent and Console source were not edited. Pre-existing untracked `reviews/` and
`specs/plans/13-monorepo-migration.md` were left untouched.

The [contract amendment](../../specs/amendments/2026-10-27-cost-completeness-openai-image.md)
defines `Usage.CostIncomplete`, public `Usage.Add`, legacy JSON compatibility,
and the bounded official OpenAI image mapping. Regression tests cover priced
and unpriced ordering, known zero versus unknown, mixed provenance, explicit
incomplete inputs, one-hour writes, cloned storage, tracker/session totals and
729 grouped aggregates with intermediate JSON. Image tests cover all 20 exact
IDs, unknown variants, independent results and API-key versus OAuth authority.

## Source fingerprints

SHA256 of the sorted `sha256  relative-path` manifests, covering every Go
source/test file, `go.mod`, `go.sum` and `models.json`:

| Point | Manifest SHA256 |
| --- | --- |
| Baseline, before implementation (284 files) | `7abd3c053c4f90818363c25074117d202040efd0e29401dab4c7f6c0dd4a605d` |
| Final sources, before gates (287 files) | `1f28666ae5087d1dda1b1af7e3acee52eb69e56bf163a2b4a61d54c8f730bb47` |
| Final sources, after gates (287 files) | `1f28666ae5087d1dda1b1af7e3acee52eb69e56bf163a2b4a61d54c8f730bb47` |

[sources.sha256](sources.sha256) records the final manifest for both Linux
and Windows gates. No module/dependency or embedded-price snapshot change.

## Local results

Go 1.27.3, Linux amd64. All final gates passed:

- `gofmt -w .`; `go mod tidy`; `git diff --check`.
- `go test . ./llmtest ./providers/openai -count=1`.
- `make check`: build and CLI smoke, vet, pinned lint, full race tests,
  25 model-snapshot tests, synthetic live-runner manifest and compile-only
  live-tag checks, all package/owned-engine coverage floors, vulnerability
  scan (none found), and all five short fuzz targets.
- `GOOS=windows GOARCH=amd64 go build ./...`.
- Exact correspondence between the 20 implemented IDs and the official
  source evidence was checked locally.

Full provider E2E tests require `-tags=live` and were not run. The ordinary
config test uses its own temporary synthetic `gollm-test.json`; no private
credential file was opened. The live-tag manifest test uses explicit synthetic
credentials without invoking providers; the other live-tag gate uses
`-run '^$'` and executes no tests. Local HTTP fixtures and official OpenAI
documentation were the only provider-facing verification inputs.

[openai-image-sources.json](openai-image-sources.json) records the nine
official model-page URLs, fetched-document SHA256s, explicit image modality,
Responses support and exact aliases/snapshots. The Models API itself has no
modality fields; unmapped API-key IDs remain unknown. These sources establish
documented capability, not current account entitlement, custom endpoint
conformance or successful inference. Native/live/hosted acceptance and later
Go Agent/Console integration remain open.
