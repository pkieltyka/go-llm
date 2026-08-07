---
status: complete
---

# Phase 10 (v0.3): public `chatcompletions.New` + vLLM provider

> Historical execution record. The main specifications and package
> documentation define current behavior.

## Outcome

This phase promoted the shared Chat Completions engine to the public
`providers/chatcompletions` package, added the key-optional
`providers/vllm` preset, added the data-only `providers/ollama` preset, and
integrated vLLM into the live and recorded-fixture test matrices.

The vLLM provider now targets the current stable v0.26.0 protocol only:

- assistant reasoning is read and replayed through `reasoning`;
- `reasoning_effort` is sent without remapping supported levels such as
  `xhigh`;
- mid-stream failures use the nested `{"error": {...}}` envelope;
- JSON schema uses the unified response-format surface, while vLLM-native
  constraints use `structured_outputs`;
- provider options include vLLM sampling fields, chat-template keyword
  arguments, thinking control, and `vllm_xargs`;
- `Models()` surfaces `max_model_len` as the context window and preserves
  model metadata in `Raw`.

The shared engine also gained keyless operation, configurable names and
capabilities, normalized tool-use stops, provider reasoning replay, and
choice-less streaming-error detection. The public dialect hook remains an
advanced escape hatch; normal users construct a provider through its preset.

## Verification

The phase added unit, race, coverage, fixture-redaction, replay, and live e2e
coverage for chat, streaming, tools, parsing, reasoning, usage, model listing,
error mapping, and cross-provider handoff. Current CI is authoritative for the
supported wire contract.
