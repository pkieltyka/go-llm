---
status: complete
---

# Phase 7: OpenAI-Compatible Adapter and OpenRouter

> Historical, non-normative execution record. ZAI references describe future
> extensibility considered at the time, not a shipped provider.

## Overview

This phase adds the shared chat-completions-shaped adapter that OpenRouter and
future ZAI support build on, then ships the OpenRouter provider as the first
dialect. The adapter owns common request conversion, response mapping,
stream normalization, usage/cost normalization, error mapping, and provider
plumbing; OpenRouter contributes declarative request extras, attribution
headers, typed response extras, native cost reporting, and OpenRouter-specific
error/stream handling.

## Steps

1. Add `providers/internal/chatcompletions`:
   - define `type Dialect interface` (behavioral hooks: `ApplyRequest` with a
     mutable extras map, `ExtractParts` with a defer-to-default contract,
     `ExtractExtras`, stop-reason/error/usage mapping, `Models`) and
     `type Compat struct` for data-expressible quirks — the fields the
     OpenRouter dialect consumes: `StreamIncludeUsage` (include_usage on
     streams), `MapEffort` (unified `Effort` → wire `reasoning` object), and
     `DefaultHeaders` (construction-time attribution defaults; per-request
     values win on both Chat and ChatStream). `Compat` is positioned for the
     deferred public `openaicompatible.New(baseURL, compat)`.
   - implement `Provider` with `Name`, `Capabilities`, `Models`, `Chat`, and
     `ChatStream`.
   - expose config hooks for API key/env, base URL, HTTP client, retries,
     timeout, price table, logging, debug capture, and typed dialect options.
   - convert `llm.Request` to `chat.ChatCompletionNewParams`, including
     messages, tools, tool choice, response format, stop sequences,
     temperature/top_p/max_tokens, session affinity, Compat-mapped reasoning
     effort, Anthropic-style `cache_control` (with the `ttl: "1h"` tier for
     hints above five minutes), and strict schema fail-open behavior (shared
     keyword check in `providers/internal/providerutil`, with name carve-outs
     for `properties`/`$defs`/`definitions`).
   - map SDK/direct responses into `llm.Response`, including text, reasoning
     (`ReasoningPart.Raw` holds the wire `reasoning_details` ARRAY verbatim;
     replay splices array elements back rather than nesting), tool calls,
     malformed-call drops, stop reasons, usage, cost estimates, and raw
     extras.
   - implement stream mapping as a **direct SSE transport** (decision note in
     ARCH §3.3: mid-stream HTTP-200 error chunks, comment keep-alives, and
     billing-safe pre-stream retries are owned by the adapter; blocking Chat
     stays on openai-go): single-use iterators, keep-alive tolerant SSE
     consumption, tool-call index state with deterministic (sorted) pending
     flush, `ToolCallDropped`, mid-stream errors, per-block accumulation of
     `reasoning_details`/annotations with the merged array emitted once at
     block completion (ReasoningDelta.Raw REPLACE semantics), `MessageEnd`
     only at stream end ([DONE]/EOF) carrying usage, and partial-content
     preservation via ordinary `llm.Collect` compatibility.
2. Add `providers/openrouter`:
   - define `providerName = "openrouter"`, default base URL
     `https://openrouter.ai/api/v1`, API key env `OPENROUTER_API_KEY`, and
     capabilities including streaming, tools, strict tools, JSON schema,
     reasoning, image input, stop sequences, session affinity, cost reporting,
     and models listing.
   - define `Options` with fallback models, provider routing, plugins,
     prediction, reasoning extras, extra sampling params, and attribution
     headers; implement `ForProvider`.
   - implement `New`, common provider options, `Client`, `Models`, `Chat`, and
     `ChatStream` by composing `chatcompletions`; the dialect's `Compat` maps
     FS §9's OpenRouter effort column (levels pass through verbatim; `none` →
     `reasoning: {enabled: false}`).
   - implement typed `ResponseExtras` plus `Extras(resp *llm.Response)`;
     usage-accounting extras (`cost`, `cost_details`, `is_byok`) are parsed
     from INSIDE the wire `usage` object per the OpenRouter docs.
   - map OpenRouter `usage.cost` to `Usage.CostUSD`, model echo/fallbacks to
     `Response.Model`, response annotations/provider/native finish reason to
     extras, moderation metadata into `ProviderError.Metadata`, and warm-up
     empty choices to `ErrServer`.
3. Wire live e2e support:
   - add OpenRouter construction from `gollm-test.json`/env credentials.
   - include the provider in the live harness with scenario-per-capability
     coverage (ARCH §9): chat, stream, models, tools, parallel_tools, parse,
     reasoning, reasoning_replay, multimodal, prompt_cache (Anthropic-family
     model via cache_control passthrough), usage, cost_reporting, and
     error_mapping.
   - update `gollm-test.json.sample`.
4. Add tests:
   - OpenRouter request-build golden for headers/extras, fallback models,
     routing/plugins, `session_id`, reasoning, cache_control TTL, response
     format, tools, and fail-open strict schema (incl. `$defs` carve-out).
   - effort mapping table test (FS §9 OpenRouter column incl. `none`).
   - response fixture for fallback `model` echo, reasoning details,
     annotations, native cost + usage-embedded `cost_details`/`is_byok`,
     extras, and usage invariant.
   - reasoning replay round-trip: wire response → parts → rebuilt request
     with a wire-identical `reasoning_details` array.
   - stream fixture with SSE comment keep-alives, multi-chunk
     reasoning_details, tool-call deltas, and final usage, asserted as a true
     Collect(stream)-vs-Chat equivalence against the same logical payload.
   - stream-level `ToolCallDropped` fixture (malformed args + missing name
     through the index state machine).
   - attribution precedence (constructor-only and per-request-wins on both
     Chat and ChatStream), mid-stream error fixture, warm-up empty-choices
     error, moderation-metadata error mapping, scoped context-overflow
     mapping, and `Models()` mapping from OpenRouter `/models` metadata.

## Tests

- `TestOpenRouterBuildRequestGolden`: verifies OpenRouter request JSON,
  typed extras, schema fail-open, cache_control TTL, and session affinity.
- `TestOpenRouterEffortMapping`: verifies the FS §9 effort table, including
  `none` → `reasoning: {enabled: false}`.
- `TestOpenRouterStrictToolSchemaDefsCarveOut`: verifies `$defs` names are
  not treated as strict-mode keywords (and keywords inside `$defs` are).
- `TestOpenRouterMapResponseFixture`: verifies fallback model echo, text,
  reasoning details, annotations, native cost with usage-embedded
  `cost_details`/`is_byok`, extras, and usage normalization.
- `TestOpenRouterReasoningReplayRoundTrip`: verifies wire →
  parts → replay produces a wire-identical `reasoning_details` array.
- `TestOpenRouterStreamKeepAliveCollectEquivalent`: verifies comment
  keep-alive lines are tolerated and Collect(stream) matches Chat for the
  same logical payload (parts, usage, stop reason, typed extras, byte-equal
  reasoning raw).
- `TestOpenRouterStreamToolCallDropped`: verifies malformed streamed tool
  calls surface as deterministic `ToolCallDropped` events.
- `TestOpenRouterAttributionPrecedence`: verifies construction-level
  attribution applies and per-request attribution wins on both paths.
- `TestOpenRouterStreamRetriesBeforeYield`/`TestOpenRouterMidStreamError`:
  verify billing-safe pre-stream retries and HTTP-200 stream error chunks.
- `TestOpenRouterWarmupEmptyChoices`, `TestOpenRouterModerationMetadataError`,
  `TestOpenRouterContextTooLongMapping`, `TestOpenRouterModels`.
- `TestWithStreamEnabledPreservesGiantIntegers` (chatcompletions).
- Existing provider tests plus `go vet` and `go test -race ./...`.
