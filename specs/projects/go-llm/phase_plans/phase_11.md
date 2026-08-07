---
status: complete
---

# Phase 11 (post-v0.3.1): vLLM tokenizer and structured-output extensions

> Historical execution record. The main specifications and package
> documentation define current behavior.

## Outcome

This phase added typed vLLM extension methods and native structured-output
options on top of the provider introduced in phase 10.

- `Tokenize`, `Detokenize`, and `TokenizerInfo` call the server-root vLLM
  endpoints while reusing the provider's normal message, tool, validation,
  authentication, and error paths.
- `TokenizeResult.ContextUsage()` bridges exact server counts and
  `max_model_len` into the unified context-usage type.
- Tokenization mirrors `reasoning_effort` thinking behavior so its rendered
  prompt matches the corresponding chat request.
- `StructuredOutputs` supports regex, choice, grammar, structural-tag, and
  whitespace-pattern fields using the current `structured_outputs` request
  object.
- Response-format and native structured-output constraints are mutually
  exclusive, exactly one native constraint mode is required, and invalid
  structural-tag JSON fails before a network call.
- When reasoning and constrained output are combined, servers must enable
  `--structured-outputs-config.enable_in_reasoning=True`; callers can also
  disable reasoning for the constrained request.

## Verification

Unit tests cover request bodies, endpoint paths, effort injection, error
mapping, all structured-output modes, and invalid option combinations. Live
e2e scenarios cover token-count parity, structured choice, and structured
regex behavior. Current CI is authoritative for the supported wire contract.
