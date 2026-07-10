---
status: complete
---

# Phase 4: Anthropic Provider

> Historical, non-normative execution record. It intentionally preserves
> phase-time names and decisions; current behavior is in the main specs.

## Overview

This phase adds the first real provider package and the core primitives it
needs: a tuned shared HTTP client, malformed tool-call visibility and retry
middleware, Anthropic request/response/stream/error mapping, provider
configuration hooks, offline fixtures, and a live e2e harness that can record
redacted fixtures without exposing secrets.

## Steps

1. Add `httpclient.go` and `httpclient_test.go` in package `llm`.
   - Implement `DefaultHTTPClient() *http.Client` as a shared `sync.Once`
     singleton.
   - Factor transport construction into `defaultHTTPTransportForGOOS(goos string)
     *http.Transport` so tests can assert the darwin keep-alive branch.
   - Preserve SSE safety by setting `Client.Timeout == 0`.

2. Extend the core stream/response model.
   - Add `type DroppedToolCall struct { Index int; Reason string }`.
   - Add `DroppedToolCalls []DroppedToolCall` to `Response`.
   - Add `ToolCallDropped` as a stream event and teach `Collect`,
     `normalizeEvent`, canonical response serialization, and `llmtest`
     cloning to preserve it.

3. Add `llm.RetryDroppedToolCalls(n int) Middleware`.
   - Chat-only wrapper: when a response has dropped tool calls, retry up to
     `n` times by appending the assistant response and a fixed user correction
     message to the next request copy.
   - Return the final response even if drops remain.
   - Leave streaming unchanged.

4. Add `providers/anthropic`.
   - Files: `provider.go`, `options.go`, `convert.go`, `stream.go`,
     `errors.go`, and package tests.
   - Constructor:
     `func New(opts ...Option) (*Provider, error)` with `WithAPIKey`,
     `WithAPIKeyFunc`, `WithBaseURL`, `WithHTTPClient`, `WithMaxRetries`,
     `WithTimeout`, `WithPriceTable`, `WithLogger`, `WithDebugCapture`,
     `WithDefaultMaxTokens`, and Anthropic `Options`.
   - Expose `func (p *Provider) Client() *sdk.Client`, `Name`,
     `Capabilities`, `Models`, `Chat`, and `ChatStream`.
   - Build Anthropic Messages requests from `llm.Request`: system/cache,
     content parts, tool calls/results, tools/strict, tool choice,
     response format, effort/thinking, stop sequences, temperatures, and
     same-provider reasoning replay.
   - Map Anthropic responses and streams to normalized parts/events, stop
     reasons, usage, cost estimates, and malformed tool-call rescue/drop
     records.
   - Normalize Anthropic SDK/API errors to `*llm.ProviderError`.

5. Add offline Anthropic fixtures and tests.
   - Request-build goldens covering system cache, tools, response format,
     effort, defaults, and provider options.
   - Response and stream fixtures for text, thinking, tool use, refusal,
     parallel tools, and malformed tool calls.
   - Collect-equivalence and error mapping tables.
   - Debug-capture redaction and API-key-function behavior.

6. Add `internal/e2e` live harness.
   - `//go:build live` tests with provider-neutral scenario helpers and
     Anthropic registration.
   - Load gitignored `gollm-test.json` with env fallback and commit
     `gollm-test.json.sample`.
   - Add `-record` fixture capture with robust secret redaction for
     `x-api-key`, `authorization`, config contents, request metadata, and
     secret-looking key values.
   - Keep live Anthropic scenarios minimal and opt-in; pinned cheap default
     model comes from one constants file.

## Tests

- `TestDefaultHTTPClient`: verifies timeout, transport tuning, shared
  singleton, and darwin keep-alive behavior.
- `TestCollectToolCallDropped`: verifies stream dropped-call events become
  `Response.DroppedToolCalls`.
- `TestRetryDroppedToolCalls`: verifies retry request shaping and final
  response behavior.
- `TestRetryDroppedToolCallsUsageTrackerOrdering`: verifies usage tracker
  records each retry attempt when composed inside the retry middleware.
- `TestAnthropicBuildRequestGolden`: verifies request conversion goldens.
- `TestAnthropicMapResponseFixtures`: verifies response fixture mapping.
- `TestAnthropicStreamFixturesCollectEquivalent`: verifies stream event
  mapping and `Collect` equivalence.
- `TestAnthropicErrorMapping`: verifies HTTP/API errors map to sentinels.
- `TestAnthropicOptions`: verifies provider options, API key function,
  logger/debug capture, defaults, and `Client()` escape hatch.
- `TestE2ERedaction`: verifies recorded fixtures and config-derived values
  cannot contain Anthropic secrets.
