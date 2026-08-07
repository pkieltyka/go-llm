// Package vllm provides a go-llm provider preset for self-hosted vLLM
// servers (https://docs.vllm.ai), riding the public
// providers/chatcompletions engine with vLLM's dialect quirks pre-configured.
//
// Construction is host-first and the API key is optional (vLLM has no auth
// unless started with --api-key):
//
//	p, err := vllm.New("http://localhost:8000/v1")
//	model, err := p.ResolveModel(ctx, "qwen") // e.g. nvidia/Qwen-3.6-27B-NVFP4
//
// # Supported server
//
// The preset targets the current stable vLLM v0.26.0 protocol. Reasoning
// replays under `reasoning`; native constraints use `structured_outputs`.
//
// # Structured outputs
//
// JSON-schema and JSON-mode output ride the unified llm.Request.ResponseFormat
// (sent as response_format). vLLM's additional native
// constraint modes — regex, choice, EBNF grammar, structural tag — are typed
// on Options.StructuredOutputs and sent as structured_outputs. Exactly one
// mode per request; combining with ResponseFormat is
// rejected at build (see the StructuredOutputs type for the conflict rules
// and the thinking interaction observed live).
//
// # Tokenization extensions
//
// vLLM exposes tokenizer endpoints at the SERVER ROOT (outside /v1;
// probe-verified: POST /tokenize works while POST /v1/tokenize is 404), and
// the provider reaches them as typed extension methods beyond the
// llm.Provider interface: Tokenize (exact chat-template-aware prompt token
// count + max_model_len for a request, reusing the engine's message/tool
// conversion — TokenizeResult.ContextUsage bridges to llm.ContextUsage),
// Detokenize, and TokenizerInfo (raw; endpoint is flag-gated server-side).
//
// # Reasoning
//
// Servers started with --reasoning-parser stream `delta.reasoning` fragments,
// which map to llm.ReasoningDelta / llm.ReasoningPart with plain text (vLLM
// reasoning has no signed or encrypted payloads). Request.Effort maps to
// `reasoning_effort` (vLLM accepts none, minimal, low, medium, high, xhigh,
// and the vLLM-specific max). Thinking-by-default models (Qwen3.6)
// honor llm.EffortNone or Options.EnableThinking=false to answer directly.
// Options.ThinkingTokenBudget can opt into vLLM's deployment-specific
// `thinking_token_budget` extension; when Request.MaxTokens is set, the
// provider clamps it to reserve 1,024 tokens for visible output and rejects
// contradictory thinking-disabled controls.
//
// # Server-flag-dependent features
//
// Tool calling with tool_choice "auto" requires the server flags
// --enable-auto-tool-choice --tool-call-parser <name>; without them such
// requests fail server-side (the client cannot detect this from /v1/models).
// Structured output applied to reasoning requires
// --structured-outputs-config.enable_in_reasoning=True; otherwise disable
// thinking (EffortNone / EnableThinking=false) for constrained requests.
// Usage.CacheReadTokens is populated only when
// the server runs with --enable-prompt-tokens-details.
package vllm
