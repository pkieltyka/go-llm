# Release Checklist

This checklist captures the credentialed steps for `v0.1.0` readiness. The ordinary CI checks must pass before these live checks.

## Local Checks

```sh
go vet ./...
go test ./...
go test -race ./...
./scripts/check-coverage.sh
```

Run the same commands after any generated fixture or model-table refresh.

## Credentials

Copy `gollm-test.json.sample` to `gollm-test.json` and fill any configured providers:

- `anthropic`: API key credential.
- `openai`: API key credential.
- `openai-codex`: OAuth credential from a compatible ChatGPT/Codex login flow.
- `openrouter`: API key credential.
- `vllm`: keyless self-hosted entry — `base_url` (e.g. `http://host:8000/v1`)
  plus `model`; a `base_url` alone marks the entry configured (add `key`
  only when the server runs with `--api-key`).

Missing entries should skip visibly in the live suite. Do not commit `gollm-test.json`.

## Live Matrix

```sh
go test ./internal/e2e -tags live
```

The matrix should include:

- Anthropic chat, stream, tools, tool streaming, structured output, reasoning, and cache scenarios when configured.
- OpenAI Responses chat, stream, tools, tool streaming, structured output, reasoning, and replay scenarios when configured.
- OpenAI Codex subscription OAuth chat, stream, tools, tool streaming, and refresh scenarios when configured.
- OpenRouter chat, stream, tools, tool streaming, cost reporting, routing metadata, and mid-stream error coverage when configured.
- vLLM (self-hosted) chat, stream, models (max_model_len), tools + tool-result round trip, auto-parser tool streaming, parallel tools, structured output, reasoning + replay-drop semantics, usage, error mapping, and the Anthropic `/v1/messages` recipe smoke when a host is configured.
- Cross-provider handoff (`cross_provider_handoff` in every provider's scenario list): a tool-using conversation is round-tripped through the canonical envelope and continued on the next configured provider. It skips visibly when fewer than two providers are configured.

## Fixture Recording

Refresh the offline live fixture corpus with:

```sh
go test ./internal/e2e -tags live -record
```

Review recorded payloads for secrets before committing (`TestRecordedFixturesAreRedacted` must pass). The redaction guard is **shape-based**: it checks a fixed header list plus a handful of token-shape regexes, so a credential in a novel header or with a novel token shape would slip through — extend the guard's rules alongside any re-record that introduces new auth surfaces, and still eyeball the payloads. Captures must preserve provider behavior needed by adapter tests while redacting credentials: the recorded corpus drives the offline mapping replay suites (`providers/*/replay_test.go`, `providers/chatcompletions`, and `providers/internal/responsesapi`), so a re-record must be followed by a full offline test run.

Fixtures are recorded for Anthropic, OpenAI Codex, OpenRouter (Phase 9), and vLLM (Phase 10). The OpenAI Responses fixture is intentionally absent until an OpenAI API key is configured; before tagging a release that claims OpenAI live fixture evidence, run:

```sh
go test ./internal/e2e -tags live -run TestLiveOpenAI -record
```

## Model And Price Snapshot

Refresh the generated model table before tagging:

```sh
make models
```

Commit `models.json` with the refreshed `generated_at` stamp and any
intentional `scripts/overrides.json` changes.

## Coverage

The recorded fixture corpus is wired into offline mapping tests: `internal/e2e/replay.go` replays every recorded exchange through each adapter's full mapping paths — response side asserting normalized invariants (parts, usage math, stop reasons, reasoning raw preservation, tool-call parsing, error kinds), and request side asserting invariant-level properties of the outbound body (valid JSON, recorded model echoed, tools present when the recorded scenario had tools, non-empty message/input list). The request-side checks are deliberately not byte goldens; wire-shape goldens live in the provider unit tests. The `llm.Provider` behavioral contract itself is machine-checked by `llmtest.RunConformance`, which every provider package runs against offline fixture servers.

The per-package floors below are enforced by `scripts/check-coverage.sh` (run locally and in CI; self-contained `go test -cover`, no external service); a drop below any floor fails the check and blocks the tag until explained or fixed. Floors only move up: after coverage-improving work, re-measure and ratchet both tables to the new actuals.

| Package | Floor |
| --- | --- |
| `.` (root llm) | 81% |
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
| `schema` | 100% |

(v0.3 actuals for the promoted `providers/chatcompletions` — public as of v0.3, previously floored at 75% under `providers/internal/` — plus the new `providers/vllm` — ratcheted 76→84 with the v0.4.0 tokenize/structured-outputs increments — and data-only `providers/ollama`; other rows are v0.2 actuals, ratcheted from the v0.1.0 floors of 71/72/67/78/66/71 for the six originally floored packages.) The floors sit below a full 85% mapping gate on purpose: the remaining uncovered statements are retry/backoff plumbing, OAuth refresh flows, and logging/transport error seams that recorded 200/4xx exchanges cannot reach. Raising the gate requires targeted transport-fault tests, not more recordings.

## Tagging

After review approval and green checks:

```sh
git tag v0.1.0
git push origin v0.1.0
```
