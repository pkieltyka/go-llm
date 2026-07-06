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
// # Server eras
//
// Deployed vLLM spans years of releases and SILENTLY IGNORES unknown request
// fields, so wrong-era parameters degrade silently rather than erroring. The
// preset defaults to the modern era (v0.12+): reasoning replays under the
// `reasoning` field and structured output uses `response_format: json_schema`
// — the portable spelling accepted by both eras. WithLegacyEra switches
// reasoning replay to the pre-rename `reasoning_content` field for older
// servers (and rejects Options.StructuredOutputs, a v0.12+ param); response
// parsing always tolerates both spellings.
//
// # Structured outputs
//
// JSON-schema and JSON-mode output ride the unified llm.Request.ResponseFormat
// (sent as response_format, portable across eras). vLLM's additional native
// constraint modes — regex, choice, EBNF grammar, structural tag — are typed
// on Options.StructuredOutputs and sent as the v0.12+ structured_outputs
// param. Exactly one mode per request; combining with ResponseFormat is
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
// `reasoning_effort` (vLLM accepts none..high plus the vLLM-specific "max";
// the unified "xhigh" maps to "high"). Thinking-by-default models (Qwen3.6)
// honor llm.EffortNone or Options.EnableThinking=false to answer directly.
//
// # Server-flag-dependent features
//
// Tool calling with tool_choice "auto" requires the server flags
// --enable-auto-tool-choice --tool-call-parser <name>; without them such
// requests fail server-side (the client cannot detect this from /v1/models).
// Forced tool choice (named or "required") and strict tools use constrained
// decoding, which on some deployments (observed on vLLM 0.24.0 with the
// qwen3 reasoning parser) returns HTTP 500 while thinking is enabled —
// disable thinking (EffortNone / EnableThinking=false) for forced tool calls
// on reasoning-parser hosts. Usage.CacheReadTokens is populated only when
// the server runs with --enable-prompt-tokens-details.
package vllm
