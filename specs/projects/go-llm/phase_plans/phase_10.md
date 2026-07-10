---
status: complete
---

# Phase 10 (v0.3): public `chatcompletions.New` + vLLM provider

> Historical, non-normative execution and live-probe record. Current vLLM
> behavior is specified in the main specs and package documentation.

## Overview

Promote the internal chat-completions engine to the public
`providers/chatcompletions` package with a key-optional generic constructor,
ship the `providers/vllm` preset (flagship consumer) live-verified against a
real vLLM 0.24.0 host, add a data-only `providers/ollama` preset, and wire a
`vllm` suite into the live e2e matrix + recorded-fixture replay harness.

## Steps (as executed)

1. `git mv providers/internal/chatcompletions providers/chatcompletions`;
   exported the raw wire types (`JSONObject`, `RawMessage`, `RawChoice`,
   `RawUsage`, `RawToolCall`, `RawError`, ...) as canonical names with
   internal aliases; the Config-based constructor became `NewWithDialect`
   (advanced, stability-exempt, documented as such along with `Dialect`).
2. Added public `New(baseURL string, opts ...Option)`: key-OPTIONAL
   (`WithAPIKey`/`WithAPIKeyFunc`; no env fallback; keyless requests carry
   no Authorization header on the SDK, direct-SSE, and DoJSON paths),
   `WithName` (default "chatcompletions"), `WithCompat`, `WithCapabilities`
   (generous default baseline), plus the standard wiring options.
3. Grew `Compat` (the declarative growth surface) instead of `Dialect`:
   - `MapEffort` semantics changed to return TOP-LEVEL wire fields; the
     nil default is now `{"reasoning_effort": ...}` (OpenAI CC + vLLM
     spelling); OpenRouter nests `{"reasoning": {...}}` itself.
   - `ReasoningReplayField`: same-provider plain-text reasoning replay
     field ("reasoning" / "reasoning_content"); default drops it.
   - `SniffMidStreamErrors`: choice-less SSE error data events (nested and
     flat legacy shapes) → normalized in-stream errors (goose-crash case).
   - `NormalizeToolUseStop`: end-turn + tool calls present → tool_use
     (FS §5 semantics), gated so OpenRouter behavior stays byte-identical.
   - Engine now reads `reasoning_content` as a fallback spelling for
     `reasoning` on both blocking and streaming paths, and tags text-only
     streamed reasoning blocks with the provider name at stream end
     (Collect-equivalence with the blocking path's ReasoningPart.Provider).
4. `providers/vllm`: host-first `New(baseURL, opts...)`, key-optional;
   era-aware (`WithLegacyEra()` switches reasoning replay to
   `reasoning_content`; `response_format: json_schema` is era-portable);
   Effort → `reasoning_effort` (xhigh→high nearest-level, max passes
   through); typed `Options` (top_k, min_p, repetition_penalty,
   stop_token_ids, ChatTemplateKwargs + EnableThinking sugar, XArgs =
   vllm_xargs); live `Models()` mapping `max_model_len` → ContextWindow
   (LoRA `parent` preserved in Raw); numeric status-shaped in-stream error
   codes classified through the canonical status table; `abort` →
   StopReasonError. NO Responses-API surface, NO `/tokenize` extension
   (deferred — `/tokenize` is the recorded next vLLM increment; flip
   criteria for Responses live in plan Future Work).
5. `providers/ollama`: data-only, community-verified preset —
   localhost:11434/v1 convention, name "ollama", `StreamIncludeUsage`;
   no invented behavior.
6. e2e: loader rule `ProviderConfig.Configured()` (base_url alone
   configures keyless self-hosted entries); `TestLiveVLLM`
   scenario-per-capability suite with the Effort-none defaulting wrapper;
   vllm joined the cross-provider handoff rotation (as source and target);
   fixtures recorded to `internal/e2e/fixtures/vllm/live.json` (redaction
   guard passes) and replayed via `providers/vllm/replay_test.go` with the
   new `ReasoningTextMarkers` profile field (vLLM reasoning has no raw
   payload to assert).
7. Docs: README providers table + "Self-hosted" section (snippets mirrored
   in compile-checked example tests); FS §2/§3, ARCH §1/§3.3, release.md
   floors updated; coverage floors added (`providers/chatcompletions` 77,
   `providers/vllm` 76, `providers/ollama` 100; openrouter ratcheted 73→75,
   openai re-covered above its 82 floor after pre-existing 81.9% drift).

## Live findings (vLLM 0.24.0, Qwen/Qwen3.6-27B-FP8, qwen3 parsers)

- **Constrained decoding conflicts with active thinking**: `tool_choice`
  named or `"required"`, and tools with `strict: true`, return HTTP 500
  (`InternalServerError`) while the model is thinking; all three work with
  thinking disabled (`reasoning_effort: "none"` or
  `chat_template_kwargs.enable_thinking=false`). `response_format:
  json_schema` works fine even WITH thinking (constraint applies after
  reasoning). Documented in the package docs; the e2e Effort-none wrapper
  keeps the tool scenarios on the working path.
- **Forced tool calls finish with `"stop"`**, auto-detected ones with
  `"tool_calls"` — hence `Compat.NormalizeToolUseStop`.
- Blocking responses carry `content: null` alongside populated `reasoning`;
  every response includes `"reasoning": null` when the parser produced
  nothing (fixture replay markers must match `"reasoning":"`).
- Errors are the nested `{"error":{message,type,param,code}}` shape with a
  NUMERIC code mirroring the HTTP status; mid-stream errors arrive as
  choice-less data events after HTTP 200.
- `usage.prompt_tokens_details` is `null` without
  `--enable-prompt-tokens-details`; usage arrives on a trailing
  empty-choices chunk under `stream_options.include_usage`.
- Same-provider reasoning replay (assistant `reasoning` field) is accepted
  (no 400); the chat template drops prior thinking by design.
- `reasoning_effort: "max"` accepted; `parallel_tool_calls` works through
  the qwen3 parser (2 calls returned under both auto and required).
- **Anthropic `/v1/messages` recipe: WORKS.**
  `anthropic.New(WithBaseURL(hostRoot), WithAPIKey("dummy"))` completes
  chats with usage; vLLM emits Anthropic-style `thinking` content blocks
  (with a signature hash) that map to reasoning parts. Documented in the
  README with the live result.

## Tests

- `gofmt -l` clean; `go vet ./...` and `go vet -tags live ./...` clean.
- `go build ./...` + `./examples/...`; `go test -race -count=1 ./...`.
- Fuzz targets (4 × 5s) over the serialize corpus.
- `scripts/check-coverage.sh` with the new floors.
- Full live matrix: `TestLiveAnthropic` + `TestLiveOpenAICodex` +
  `TestLiveOpenRouter` + `TestLiveOpenAI` (skip — no key) +
  `TestLiveVLLM` (13 scenarios incl. `anthropic_messages` bonus and
  `cross_provider_handoff`) against `http://pax.local:8000/v1`.
- `-record` pass for the vllm fixture corpus; `TestRecordedFixturesAreRedacted`
  passes over all four fixture files.
