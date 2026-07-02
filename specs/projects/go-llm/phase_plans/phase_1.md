---
status: complete
---

# Phase 1: Repo Scaffolding + Core Vocabulary

## Overview

This phase creates the Go module and repository automation, then implements
the dependency-free `llm` root package vocabulary required by later provider
phases. It establishes request, message, response, stream, validation, error,
capability, provider-interface, and in-memory history primitives with focused
unit coverage.

## Steps

1. Add repository scaffolding:
   - `go.mod` with module path `github.com/pkieltyka/go-llm` and Go `1.26`.
   - `.gitignore` including `.specs_skill_state/` and `gollm-test.json`.
     Also ignore `scripts/node_modules/` for the planned model-table snapshot
     tooling.
   - `.github/workflows/ci.yml` running `go vet`, `golangci-lint`, race tests,
     `govulncheck`, and a short fuzz smoke.
   - `.github/dependabot.yml` for GitHub Actions and Go module updates.
2. Implement package surface in `llm.go`:
   - `Provider` interface with `Name`, `Capabilities`, `Models`, `Chat`, and
     `ChatStream`.
   - `ModelInfo` including `CanonicalID`, and `ModelPricing`.
3. Implement messages and parts in `message.go`:
   - `Role`, `Message` with optional provider/model provenance, `Part`,
     `ExtensionPart`, `CacheHint`.
   - Core part structs and constructors such as `UserText`, `AssistantText`,
     `UserParts`, `Text`, `ImageURL`, `ImageData`, and `ToolResult`.
4. Implement requests in `request.go`:
   - `Request`, `Tool`, `ToolAnnotations`, `ToolChoice`, `ResponseFormat`,
     `Effort`, and `ProviderOptions`.
5. Implement responses in `response.go`:
   - `Response`, `Usage`, `StopReason`, and accessors `Text`, `Reasoning`,
     and `ToolCalls`.
6. Implement streaming in `stream.go`:
   - Event structs with provider/model metadata and indexed content deltas,
     `Collect`, `StreamText`, and `WithDebounce`.
   - Accumulate text, reasoning, tool calls, stop reason, and usage from a
     stream into a normalized `Response`, including interleaved blocks and
     partial responses on stream errors.
7. Implement normalized errors in `errors.go`:
   - Sentinels and `ProviderError` with `Error` and `Unwrap`.
8. Implement capabilities and validation:
   - `capability.go` with standard capability constants and
     `CustomCapabilities`.
   - `validate.go` with exported `ValidateRequest` and
     `ValidateStreamRequest` / `ValidateProviderOptions` helpers for provider
     packages.
9. Implement `History` in `history.go`:
   - `Add`, `AddUserText`, `AddResponse` with provenance stamping,
     `AddToolResults`, defensive `Messages`, and the
     `WithForeignReasoningAsText` option flag for later replay logic.

## Tests

- `TestCollectAccumulatesResponse`: verifies event accumulation into text,
  reasoning, tool calls, stop reason, usage, id, and model.
- `TestCollectReturnsPartialResponseOnStreamError`: verifies stream errors
  return accumulated partial content.
- `TestStreamTextDebounce`: verifies text-only filtering and rate-limited
  debounce flush.
- `TestValidateRequestUnsupportedCapabilities`: verifies capability mismatches
  return `ErrUnsupported`.
- `TestProviderErrorWrapping`: verifies `errors.Is` and `errors.As`.
- `TestHistory`: verifies user turns, assistant response replay parts,
  defensive copies, and grouped tool-result messages.
