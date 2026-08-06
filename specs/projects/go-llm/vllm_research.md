---
status: complete
---

# vLLM current-stable API surface

> Point-in-time upstream review, verified 2026-08-06 against vLLM v0.26.0.
> The shipped contract is defined by `providers/vllm` and the main project
> specifications.

go-llm targets the current stable vLLM protocol. It does not expose server-era
switches or alternate wire spellings for retired protocol revisions.

## Supported surface

The provider uses the OpenAI-compatible Chat Completions API at
`POST /v1/chat/completions`. Current vLLM also provides:

- `GET /v1/models`, including served aliases and registered LoRA adapters.
- `POST /tokenize`, `POST /detokenize`, and the flag-gated
  `GET /tokenizer_info` endpoints at the server root.
- OpenAI-compatible Responses, embeddings, transcription, translation, and
  batch APIs.
- An Anthropic-compatible `/v1/messages` surface.

Chat Completions remains a supported current API and exposes the extensions
used by the go-llm preset. A Responses-based vLLM provider is a separate
future feature, not a compatibility migration.

## Chat Completions extensions used by go-llm

The provider maps these current request fields:

- `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`,
  and the vLLM-specific `max`.
- `reasoning` on assistant messages and streamed deltas.
- `top_k`, `min_p`, `repetition_penalty`, and `stop_token_ids`.
- `chat_template_kwargs`, including the typed `enable_thinking` convenience.
- `thinking_token_budget`.
- `structured_outputs`.
- `vllm_xargs` for engine-specific values that are not modeled directly.

The server ignores `user`. Tool choice and tool parsing remain dependent on
the model and server configuration. Automatic tool choice requires
`--enable-auto-tool-choice` and a matching `--tool-call-parser`.

## Structured outputs

Current vLLM uses one top-level `structured_outputs` object. go-llm exposes
the native `choice`, `regex`, `grammar`, and `structural_tag` modes through
`vllm.StructuredOutputs`. Exactly one mode is allowed per request.

JSON schema and JSON mode use the provider-neutral
`llm.Request.ResponseFormat`, serialized as `response_format`. Combining it
with `vllm.StructuredOutputs` is rejected before the request is sent because
both configure the same constraint system.

Structured output applied to reasoning requires the server setting
`--structured-outputs-config.enable_in_reasoning=True`. Without it, callers
should disable thinking for constrained requests.

## Reasoning

Servers started with a reasoning parser return reasoning in
`message.reasoning` and `delta.reasoning`. go-llm normalizes these values to
`llm.ReasoningPart` and `llm.ReasoningDelta`, and replays same-provider
reasoning under `reasoning`.

`reasoning_effort` enables thinking in the server chat template. The typed
`ThinkingTokenBudget` option validates the budget and, when `MaxTokens` is
set, reserves 1,024 tokens for visible output.

## Streaming, errors, and usage

Current vLLM streams standard SSE `data:` events followed by `[DONE]`.
Generation failures after the HTTP 200 response are emitted as a choice-less
nested error object:

```json
{"error":{"message":"Internal server error","type":"InternalServerError","param":null,"code":500}}
```

The provider detects that envelope before ordinary chunk decoding and returns
a normalized provider error while preserving any partial response.

With `stream_options.include_usage`, vLLM emits usage in a trailing chunk with
an empty `choices` array. Prefix-cache reads are reported in
`prompt_tokens_details.cached_tokens` only when the server enables prompt
token details.

## Models and exact tokenization

`GET /v1/models` exposes `max_model_len`, which go-llm maps to
`ModelInfo.ContextWindow`. `ResolveModel` selects among served aliases and
deployment suffixes.

`Provider.Tokenize` reuses the same message, tool, effort, and provider-option
conversion as Chat, then calls the server-root `/tokenize` endpoint. Its
result provides the exact rendered prompt token count and `max_model_len`.

## Sources

- [vLLM v0.26.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.26.0)
- [OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
- [Chat Completions protocol](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/openai/chat_completion/protocol/)
- [Structured outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/)
- [Reasoning outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs/)
