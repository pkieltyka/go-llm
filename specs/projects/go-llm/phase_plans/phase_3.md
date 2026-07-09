---
status: complete
---

# Phase 3: Core Utilities

## Overview

This phase fills in the provider-independent utility layer around the core
`llm.Provider` contract: embedded model/pricing lookup, middleware, stdlib
observability helpers, prompt templates, sessions with context accounting,
the `llmtest` fake provider, and generic structured-output parsing. These
features stay in the root package or test-only helper packages and keep the
root module dependency-free at runtime.

## Steps

1. Add model/pricing snapshot support:
   - Create `models.json` with the embedded snapshot shape:
     `generated_at` plus provider/model rows carrying ID, canonical ID,
     display name, context window, max output tokens, and MTok pricing.
   - Add `models_table.go` with `//go:embed`, lazy `sync.Once` parsing,
     `LookupModelInfo(provider, modelID) (ModelInfo, bool)`, prefix lookup,
     canonical-ID fallback, and `PriceTableDate() string`.
   - Add `pricing.go` with `type PriceTable map[string]ModelPricing`,
     `EstimateCost(usage Usage, pricing ModelPricing) Usage`, and
     `EstimateCostForModel(provider, modelID string, usage Usage) Usage`.
   - Add `Usage.ContextUsage(window int64) ContextUsage` to `response.go`.
2. Add the dev snapshot pipeline:
   - Create `scripts/package.json` with a `snapshot-models-table` script
     using `tsx`.
   - Create `scripts/overrides.json` for hand-maintained patches.
   - Create `scripts/snapshot-models-table.ts` that fetches models.dev and
     OpenRouter models, trims to go-llm fields/providers, applies overrides,
     and writes `models.json`.
3. Add provider middleware:
   - Create `middleware.go` with `ChatFunc`, `StreamFunc`, `Middleware`,
     and `Wrap(p Provider, mw ...Middleware) Provider`.
   - Compose middleware so the first middleware argument is outermost while
     `Name`, `Capabilities`, and `Models` delegate unchanged.
4. Add observability helpers:
   - Create `observe.go` with `WireCapture`, `DebugToLogger`,
     `NewWireTap`, `UsageTracker`, `UsageStats`, and tracker middleware for
     both blocking and streaming calls.
   - Redact sensitive headers in captures, cap buffered request/response
     bodies, tee streaming response bodies, and emit captures on response
     body close.
5. Add prompt templates:
   - Create `prompt.go` with `PromptTemplate`, `NewPromptTemplate`,
     `MustPromptTemplate`, `Name`, `Partial`, and strict `Format`.
   - Convert map/struct variables into an immutable merge map so call-time
     variables override partials.
6. Add sessions:
   - Create `session.go` with `NewSession`, options for system/effort/max
     tokens/session ID, `Chat`, `ChatText`, `ChatStream`, `AddToolResults`,
     `History`, `Messages`, cumulative `Usage`, and `ContextUsage`.
   - Generate a best-effort random session ID when none is supplied.
7. Add `llmtest`:
   - Create `llmtest/provider.go` with a goroutine-safe FIFO scripted
     provider, request recording, configurable name/capabilities/models,
     canned responses, canned streams, and canned errors.
   - Ensure returned requests and responses are defensive copies and streams
     are real `iter.Seq2[llm.Event,error]` iterators.
8. Add `Parse[T]`:
   - Create `parse.go` with `ParseMode`, generic parse options,
     schema generation via `schema.For[T]`, auto mode resolution
     (native schema, forced synthetic tool, JSON mode guidance), bounded
     retries, and typed semantic validation.
   - On parse failures, append the assistant response and a correction user
     turn before retrying.
9. Add focused tests:
   - Pricing/model lookup tests for snapshot parsing, prefix fallback,
     canonical fallback, cost estimation, context accounting, and lazy
     concurrent lookup.
   - Middleware tests for ordering, stream decoration, and pass-through
     provider methods.
   - Observability tests for slog output, usage aggregation under
     concurrency, redaction, response-body tee capture, and SSE capture.
   - Prompt tests for missing variables, partial precedence, struct vars,
     map vars, and immutability.
   - Session tests for turn flow, stream auto-append, tool results,
     cumulative usage, and unknown-model context lookup.
   - `llmtest` self-tests as a provider conformance suite.
   - Parse tests covering native schema, forced-tool extraction, JSON mode,
     retries, validator retries, mode override, and unsupported providers.

## Tests

- `TestLookupModelInfoFallbacks`: embedded table lookup, prefix matching, and
  canonical-ID fallback.
- `TestEstimateCostForModel`: estimated cost uses input, output, cache-read,
  and cache-write pricing without overriding provider-reported costs.
- `TestModelTableConcurrentLookup`: repeated concurrent lookups are race-safe.
- `TestUsageContextUsage`: cache tokens are counted in context occupancy.
- `TestWrapOrderingAndDelegation`: middleware ordering and provider method
  pass-through.
- `TestUsageTrackerAggregatesChatAndStream`: concurrent blocking and stream
  calls are aggregated by provider/model.
- `TestNewWireTapRedactsAndCapturesBody`: sensitive headers are redacted and
  response bodies remain readable while captured.
- `TestDebugToLogger`: wire captures log the expected debug fields.
- `TestPromptTemplate`: strict missing-variable errors, partial precedence,
  struct vars, map vars, and immutable partials.
- `TestSessionTurnFlow`: user turns, responses, tool results, usage, and
  context accounting are managed by `Session`.
- `TestSessionChatStreamAppendsOnCompletion`: streams collect and append the
  assistant turn when complete.
- `TestLLMTestProvider`: scripted steps are FIFO, requests are recorded, and
  streams exercise the real iterator contract.
- `TestParseModes`: native schema, forced-tool, JSON-mode fallback, retry,
  validator retry, and unsupported-mode behavior against `llmtest`.
