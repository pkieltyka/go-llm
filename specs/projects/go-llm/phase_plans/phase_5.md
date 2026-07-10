---
status: complete
---

# Phase 5: OpenAI Provider

> Historical, non-normative execution record. `functional_spec.md` and
> `architecture.md` define the current contract.

## Overview

This phase adds the direct OpenAI Responses provider and closes the requested
phase 4 review gaps that phase 5 relies on: dropped-tool retry shaping,
retry-warning transport logging, Anthropic context overflow mapping, and pinned
reasoning replay/stream semantics.

## Steps

1. Patch shared phase 4 behaviors needed before review.
   - Update `RetryDroppedToolCalls` so retry corrections are tool-result-shaped
     when valid tool calls survived, and a failed retry returns the prior
     successful response.
   - Add reusable warn-level HTTP retry logging for 429/503/529 with
     `Retry-After` metadata, and wire it through provider logger options.
   - Map actual Anthropic context-overflow error messages to
     `ErrContextTooLong`.

2. Add Anthropic regression coverage.
   - Pin same-provider reasoning replay raw bytes and foreign-provider
     reasoning drop behavior.
   - Add refusal and effort-level fixtures.
   - Strengthen stream/non-stream collect equivalence around reasoning raw
     payloads.

3. Add `providers/openai`.
   - Constructor/options: static or per-request API key, base URL, HTTP
     client, retries, timeout, logger, debug capture, price table, raw SDK
     `Client()`, and Responses-specific `openai.Options`.
   - Request conversion: `input` items, `instructions`,
     `max_output_tokens`, temperature/top-p, tools, tool choice,
     `text.format`, reasoning effort/summary, prompt-cache key, stateless
     `store:false`, encrypted reasoning include, hosted/pass-through options,
     and same-provider reasoning replay.
   - Response conversion: output item ordering, text/refusal, function calls,
     reasoning summaries/raw encrypted items, status stop reasons, usage, cost,
     models, and normalized errors.
   - Streaming conversion: Responses semantic events to normalized stream
     events with collect-equivalent reasoning raw and tool-call validation.

4. Add OpenAI tests and e2e registration.
   - Offline request, response, stream, replay, error, options, models, and
     debug-capture tests.
   - Live OpenAI e2e behind the existing `live` tag/config harness; visibly
     skip when no credential is configured.

5. Verify and prepare for code review.
   - Run formatting and package tests/checks.
   - Run opt-in live OpenAI tests only if credentials are available.
   - Scan modified files/fixtures for secret-shaped values.

## Tests

- Core retry tests for tool-result-shaped correction and prior-response return
  on retry error.
- Retry logging transport test for warning attributes and retry-after parsing.
- Anthropic request/response/stream goldens for reasoning replay, refusal, and
  all effort levels.
- OpenAI request-build goldens for stateless Responses parameters, strict
  schema fail-open, reasoning, and provider options.
- OpenAI response/stream fixtures for text, reasoning, function calls,
  incomplete/content-filter/max-token statuses, usage, and collect
  equivalence.
- OpenAI error mapping, models listing, debug capture redaction, and live e2e
  skip/run coverage.
