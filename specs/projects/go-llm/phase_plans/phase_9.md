---
status: complete
---

# Phase 9: Docs, examples, release readiness

> Historical, non-normative execution record. Use `docs/release.md` for the
> current version-neutral release checklist.

## Overview

Prepare the repository for the first public pre-1.0 release. This phase adds user-facing documentation, runnable examples, a license, and a release checklist that records the live verification steps that require local provider credentials.

## Steps

1. Add a README with a sharply differentiated first line, naming-collision note, pre-1.0 API-stability policy, install snippets, provider setup, `llm-cli` install and usage, common library examples, testing guidance, and release-readiness notes.
2. Add godoc comments for exported symbols that still lack useful package documentation, prioritizing public root-package APIs, `schema`, `llmtest`, and provider constructors/options.
3. Add root `example_test.go` runnable examples for chat, streaming, tools, `Parse`, history, middleware, observability, and provider switch/replay using `llmtest`.
4. Add `examples/` programs that demonstrate CLI-adjacent real usage: simple chat, streaming, tool calls, structured parsing, history/replay, observability, and provider selection.
5. Add a release checklist document covering full e2e matrix, cross-provider handoff, fixture recording, price-table snapshot refresh, coverage expectations, and tagging.
6. Add a repository `LICENSE` for the public release.
7. Run formatting, documentation/example tests, `go vet ./...`, and `go test -race ./...`; run live e2e and snapshot refresh only when local credentials/network access are available.

## Tests

- `go test ./...`: verifies unit tests and runnable examples compile and pass.
- `go test -race ./...`: verifies the full package test suite under the race detector.
- `go vet ./...`: verifies Go static checks and example documentation names.
- `go test ./internal/e2e -tags live`: verifies configured provider credentials against the live e2e matrix when `gollm-test.json` is populated.
- `go test ./internal/e2e -tags live -record`: refreshes the offline fixture corpus when live credentials are available.
