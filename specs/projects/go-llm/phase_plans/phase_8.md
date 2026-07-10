---
status: complete
---

# Phase 8: cmd/llm-cli

> Historical, non-normative execution record. `functional_spec.md` and
> `architecture.md` define the current CLI contract.

## Overview

Build the `cmd/llm-cli` command-line frontend described in FS §19 and ARCH §7C. The CLI is a thin public-API consumer: it uses stdlib `flag`, constructs providers through public provider packages, builds `llm.Request` values from flags/stdin/files, streams by default, supports canonical JSON/history persistence, and exposes a `models` subcommand.

## Steps

1. Add `cmd/llm-cli/main.go` with manual subcommand routing, `signal.NotifyContext`, shared exit handling, and `--version` via `runtime/debug.ReadBuildInfo`.
2. Add `cmd/llm-cli/flags.go` with stdlib `flag` parsing for chat and models flags, repeatable `--image`, `--file`, and `--tool`, and helpers for numeric/duration options.
3. Add `cmd/llm-cli/provider.go` with a provider factory for `anthropic`, `openai`, `openai-codex`, and `openrouter`, applying `--api-key`, `--base-url`, `--timeout`, `--debug`, and provider debug capture through public constructors only.
4. Add `cmd/llm-cli/request.go` with `buildRequest`: load canonical history via `llm.UnmarshalMessages`, append positional/stdin/image/file user parts, read `--schema` into `llm.ResponseFormat`, read `--tool` JSON declarations into `[]llm.Tool`, apply `--reasoning`, `--cache-system`, `--session-id`, and scalar options.
5. Add `cmd/llm-cli/run.go` to execute chat: stream text by default, optionally mirror reasoning deltas to stderr, collect for `--json`/`--no-stream`, print tool calls as JSON without executing them, print usage summaries to stderr, and save updated conversations with `llm.MarshalMessages` plus `History`.
6. Add `cmd/llm-cli/models.go` to list models as a compact table or JSON.
7. Add focused tests in `cmd/llm-cli` using `llmtest` for flag-to-request construction, stdin + attachment handling, schema/tool parsing, canonical load/save replay, JSON/no-stream behavior, and models output.
8. Run `gofmt`, `go vet ./...`, `go test -race ./...`, and any configured lints available in the repo.

## Tests

- `TestBuildRequestFromFlags`: verifies model, system, effort, cache hint, session, max tokens, temperature, stdin/positional prompt parts, and attachments map into `llm.Request`.
- `TestBuildRequestSchemaAndTools`: verifies JSON schema and tool declarations are parsed from files into `ResponseFormat` and `Tools`.
- `TestRunChatSavesConversation`: verifies `--load`/`--save` uses canonical message envelopes and appends assistant responses through `History`.
- `TestRunChatJSONAndNoStream`: verifies non-streaming and canonical response JSON output paths against `llmtest`.
- `TestRunModelsOutput`: verifies the models subcommand table and JSON output without contacting live providers.
