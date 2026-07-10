---
status: complete
---

# vLLM Server API Surface — Research Report (as of 2026-07)

> **Research, not a normative provider contract.** This point-in-time upstream
> survey explains design choices and may age as vLLM changes. Current shipped
> behavior is defined by the main specs and `providers/vllm` package docs.

Target: first-class vLLM support in go-llm via the shared chat-completions adapter + per-provider dialect hooks / declarative Compat config.

**Version landscape.** vLLM's release cadence accelerated sharply in 2026. Latest release at time of writing: **v0.24.0 (2026-06-26)** ([releases](https://github.com/vllm-project/vllm/releases)). Key API-surface milestones referenced throughout:

| Version | Change |
|---|---|
| v0.10.0 (Jul 2025) | OpenAI **Responses API** added (PR [#20504](https://github.com/vllm-project/vllm/issues/14721)); `cache_salt` on /v1/completions + /v1/responses |
| v0.11.1 (Nov 2025) | **Anthropic `/v1/messages`** endpoint added (PR #22627, issue [#21313](https://github.com/vllm-project/vllm/issues/21313)) |
| ~v0.11.x–v0.12 | `reasoning_content` → **`reasoning`** rename (RFC [#27755](https://github.com/vllm-project/vllm/issues/27755), PR #27752) |
| v0.12.0 | **`guided_json`/`guided_regex`/`guided_choice`/`guided_grammar`/`guided_whitespace_pattern` and `guided_decoding_backend` REMOVED** — replaced by unified `structured_outputs` param ([structured outputs docs](https://docs.vllm.ai/en/latest/features/structured_outputs/)) |
| ~v0.22 (Jan 2026) | WebSocket **`/v1/realtime`** endpoint + streaming inputs (PRs #28973, #33187; [blog](https://vllm.ai/blog/2026-01-31-streaming-realtime)) |
| 2026 | Entrypoints code restructured: `vllm/entrypoints/openai/{chat_completion,completion,engine,models,responses}/protocol.py`, `vllm/entrypoints/anthropic/`, `vllm/entrypoints/serve/tokenize/` |

Because deployed fleets lag, a Go client will encounter servers from ~v0.8 through v0.24; the dialect should tolerate both pre- and post-v0.12 structured-output styles (see §9).

---

## 1. Endpoints

Source: [online serving docs](https://docs.vllm.ai/en/latest/serving/online_serving/), `vllm/entrypoints/openai/api_server.py`.

### OpenAI-compatible
- `POST /v1/chat/completions` — full-featured; the main surface.
- `POST /v1/chat/completions/batch` — batch variant (2026 addition).
- `POST /v1/completions` — legacy completions incl. `suffix`, `echo`, `logprobs`; vLLM extension: `prompt_embeds` (base64 tensors).
- `GET /v1/models` — lists base model + LoRA adapters (see §7).
- `POST /v1/embeddings` — for embedding models; `encoding_format` float/base64; vLLM extras historically include `truncate_prompt_tokens`, `add_special_tokens`, `priority`, `request_id`.
- `POST /v1/responses`, `GET /v1/responses/{response_id}`, `POST /v1/responses/{response_id}/cancel` — **OpenAI Responses API** (since v0.10.0). Supports: streaming, **background mode** (`background: true` + GET/cancel), `store` / per-response storage, `previous_response_id` multi-turn, **reasoning items** (reasoning-token counts derived from the reasoning parser), function tools + **MCP tool servers**, logprobs. Originally built for gpt-oss/harmony; team is aligning with the "Open Responses" spec (Jan 2026). Gaps: multimodal input to /v1/responses is still patchy (images/video fail — issue [#32685](https://github.com/vllm-project/vllm/issues/32685), Jan 2026); `truncation:"auto"` returns 400 instead of truncating ([#38132](https://github.com/vllm-project/vllm/issues/38132)); no way to distinguish length-stop in some cases ([#24184](https://github.com/vllm-project/vllm/issues/24184)). Full statefulness without stores is still an open RFC ([#24603](https://github.com/vllm-project/vllm/issues/24603)). Verdict: real and useful for text+tools, not yet at chat-completions parity.
- `POST /v1/audio/transcriptions`, `POST /v1/audio/translations` — Whisper-style ASR models.

### Anthropic-compatible (yes, it exists)
- `POST /v1/messages`, `POST /v1/messages/count_tokens` — **Anthropic Messages API**, since **v0.11.1** (PR #22627). Implemented in `vllm/entrypoints/anthropic/serving_messages.py`. Documented as a drop-in backend for **Claude Code** ([integration guide](https://docs.vllm.ai/en/latest/serving/integrations/claude_code/)): set `ANTHROPIC_BASE_URL=http://localhost:8000`, dummy `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`, and map model env vars; server needs `--enable-auto-tool-choice --tool-call-parser <parser>`. Known rough edges: reasoning/thinking blocks in /v1/messages still a feature request ([#29915](https://github.com/vllm-project/vllm/issues/29915)); strict validation broke against newer Claude Code role types ([#44000](https://github.com/vllm-project/vllm/issues/44000)); image inputs buggy on some stacks.

### Tokenization (vLLM-specific, valuable client extension)
Source: `vllm/entrypoints/serve/tokenize/protocol.py`.
- `POST /tokenize` — two request shapes:
  - completion-style: `{model?, prompt, add_special_tokens=true, return_token_strs=false}`
  - chat-style: `{model?, messages, add_generation_prompt=true, continue_final_message=false, add_special_tokens=false, chat_template?, chat_template_kwargs?, tools?, mm_processor_kwargs?, media_io_kwargs?, return_token_strs=false}` — i.e. it applies the chat template server-side, including tools → exact prompt token count.
  - Response: `{count, max_model_len, tokens: [int], token_strs?: [str]}` — note it returns **max_model_len**, handy for client-side context budgeting.
- `POST /detokenize` — `{model?, tokens:[int]}` → `{prompt}`.
- `GET /tokenizer_info` — tokenizer class, chat template, config (`extra="allow"` passthrough of tokenizer config).

### Other endpoint families (server-side / ops, client rarely calls)
- Health/ops: `GET /health`, `GET /version`, `GET /load`, `GET /metrics` (Prometheus), `GET /ping` + `POST /invocations` (SageMaker).
- Pooling-model APIs: `/pooling`, `/classify`, `/score` + `/v1/score`, `/rerank` + `/v1/rerank` + `/v2/rerank` (Jina/Cohere-style), `/v2/embed` (Cohere-compatible).
- LoRA: `POST /v1/load_lora_adapter`, `POST /v1/unload_lora_adapter` (gated by `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`; docs: local dev only).
- Realtime: `WS /v1/realtime` — WebSocket, OpenAI-Realtime-flavored messages (`input_audio_buffer.append/commit`, `session.update`, `response.text.delta`, `transcription.delta/done`). ~v0.22.
- Scale-out/dev (ignore for client): `/inference/v1/generate`, `/abort_requests`, `/v1/{chat/}completions/render|derender`, `/scale_elastic_ep`, profiling, and `VLLM_SERVER_DEV_MODE=1` endpoints (`/reset_prefix_cache`, `/sleep`, `/wake_up`, weight update, `/server_info`, ...).

---

## 2. Chat-completions request extensions (the dialect meat)

Source: `vllm/entrypoints/openai/chat_completion/protocol.py` (main, 2026-07) — all go in the top-level JSON body (OpenAI SDK users put them in `extra_body`; a raw HTTP client just sets them as body fields).

### Sampling extras
| Field | Type / default |
|---|---|
| `top_k` | int? (None) |
| `min_p` | float? (None) |
| `repetition_penalty` | float? (None) |
| `length_penalty` | float = 1.0 (beam) |
| `use_beam_search` | bool = false (still present on chat + completions) |
| `stop_token_ids` | list[int] = [] |
| `include_stop_str_in_output` | bool = false |
| `ignore_eos` | bool = false |
| `min_tokens` | int = 0 |
| `skip_special_tokens` | bool = true |
| `spaces_between_special_tokens` | bool = true |
| `allowed_token_ids` | list[int]? |
| `bad_words` | list[str] = [] |
| `prompt_logprobs` | int? (logprobs for the *prompt*; echoed in response) |
| `truncate_prompt_tokens` | int?; `truncation_side`: "left"\|"right"? |
| `logit_bias` | dict[str,float]? — supported (standard) |
| `seed`, `n`, `frequency_penalty`, `presence_penalty`, `max_tokens`/`max_completion_tokens`, `stop` | standard, supported; `n>1` works |
| `user` | accepted, **ignored** ("NOTE this will be ignored by vLLM") |

### Prompt/template control
- `chat_template` (str, per-request override), `chat_template_kwargs` (dict — e.g. `{"enable_thinking": false}` for Qwen3, `{"thinking": true}` for Granite/DeepSeek-V3.1).
- `add_generation_prompt` (true), `continue_final_message` (false), `add_special_tokens` (false for chat), `echo` (false).
- `documents`: `list[dict[str,str]]` — RAG documents passed to chat templates that support them (Command-R style).
- Server flag `--chat-template-content-format auto|openai|string` affects how content parts are rendered into the template.

### Structured output / reasoning
- `structured_outputs`: StructuredOutputsParams (see §3). Legacy `guided_*` fields **gone from protocol** on main.
- `response_format`: `text` | `json_object` | `json_schema` | **`structural_tag`** (vLLM accepts a structural-tag response format type).
- `reasoning_effort`: `"none"|"minimal"|"low"|"medium"|"high"|"max"` — `"max"` is vLLM-specific (DeepSeek-V4); setting effort auto-injects `enable_thinking: true` into chat_template_kwargs.
- `thinking_token_budget` (with server `--reasoning-config` `reasoning_start_str`/`reasoning_end_str`).
- `include_reasoning`: bool = true (drop reasoning from response when false).

### Output/introspection extras
- `return_tokens_as_token_ids` (logprobs tokens rendered as `token_id:<n>`), `return_token_ids`, `return_token_offsets`, `return_prompt_text`, `return_assistant_tokens_mask`.

### Infra-ish extras
- `priority`: int = 0 — priority scheduling (requires server `--scheduling-policy priority`; lower = sooner).
- `request_id`: str (default random uuid) — client-supplied ID, surfaced in logs/`X-Request-Id` (with `--enable-request-id-headers`).
- `cache_salt`: str — partitions the automatic-prefix-cache (multi-tenant cache isolation).
- `kv_transfer_params`: dict (disagg prefill/KV connector); echoed in the response.
- `mm_processor_kwargs`, `media_io_kwargs`: dicts for multimodal processing.
- `vllm_xargs`: `dict[str, str|int|float|list]` — generic escape hatch forwarded to the engine (good mapping target for go-llm's provider-options passthrough).
- `repetition_detection`: RepetitionDetectionParams (2026 addition; abort pathological loops).

### Response-side extensions (chat)
- `choices[].message.reasoning: str?` (see §4).
- `choices[].stop_reason: int|str|null` — **extra sibling of `finish_reason`**: the actual stop string or token id that ended generation (null = EOS). Docs/code: "not part of the OpenAI spec but included in vLLM for legacy reasons" (deprecation RFC [#6202](https://github.com/vllm-project/vllm/issues/6202) never completed).
- `choices[].token_ids: list[int]?` (when requested), `choices[].routed_experts` (base64 MoE routing, niche).
- Top-level: `prompt_logprobs`, `prompt_token_ids`, `prompt_text`, `kv_transfer_params`.

`/v1/completions` mirrors nearly all of the above minus chat-template fields, plus `prompt_embeds: bytes|list[bytes]` (base64 torch tensors as prompt) — `suffix` field exists, `echo`+`logprobs` supported.

---

## 3. Structured outputs / guided decoding

Source: [structured outputs docs](https://docs.vllm.ai/en/latest/features/structured_outputs/).

- **Current (≥ v0.12.0)**: single param `structured_outputs` with exactly one of:
  - `{"structured_outputs": {"choice": [...]}}"`
  - `{"structured_outputs": {"regex": "..."}}` (Rust-regex syntax for xgrammar/guidance backends)
  - `{"structured_outputs": {"json": {<schema>}}}`
  - `{"structured_outputs": {"grammar": "<EBNF>"}}` (context-free EBNF)
  - `{"structured_outputs": {"structural_tag": ...}}` (schema confined within tags — used for strict tool calls)
  - plus `whitespace_pattern`.
- **Removed in v0.12.0**: `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`, `guided_whitespace_pattern`, `structural_tag` (top-level), `guided_decoding_backend`. Because the request model is `extra="allow"`, a legacy `guided_json` sent to ≥v0.12 is **silently ignored** (debug-level log only) — the model just free-forms. This is the #1 footgun for a client library.
- `response_format`: `json_object`, `json_schema` (`{"type":"json_schema","json_schema":{"name","schema","strict"}}`), and `structural_tag` all route into the same structured-outputs machinery. `response_format json_schema` is the **portable** spelling that works on both old and new vLLM.
- Backends: server-side `--structured-outputs-config.backend` (default `auto` picks per request); xgrammar, guidance, outlines, lm-format-enforcer (regex dialect differs: Python `re` for lm-format-enforcer). Per-request backend selection was removed with `guided_decoding_backend`.
- Reasoning interplay: structured output constraint applies to final content after reasoning; `--structured-outputs-config.enable_in_reasoning=True` extends it into reasoning (Qwen3-coder etc., v0.11.2+).
- First-request FSM compilation adds latency; then cached.

---

## 4. Reasoning outputs

Source: [reasoning outputs docs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/), RFC [#27755](https://github.com/vllm-project/vllm/issues/27755).

- Server flag `--reasoning-parser <name>`. Parsers (current docs): `deepseek_r1` (R1 + QwQ-32B), `deepseek_v3` (DeepSeek-V3.1), `qwen3`, `granite`, `glm45`, `gemma4`, `hunyuan_a13b`, `holo2`, `minimax_m2_append_think`, `cohere_command3`, `ernie45`, `gptoss` (harmony), and more; pluggable.
- **Field name: `message.reasoning` (and `delta.reasoning` in streamed chunks).** Renamed from `reasoning_content` (DeepSeek convention) to `reasoning` (OpenAI/gpt-oss convention) via PR #27752, landed ~v0.11.x/v0.12; `reasoning_content` is deprecated. Input-side: request messages containing `reasoning_content` are normalized to `reasoning` for back-compat. **A first-class client should read both `reasoning` and `reasoning_content`** — deployed pre-rename servers emit only the old name (LiteLLM hit exactly this: [litellm#20246](https://github.com/BerriAI/litellm/issues/20246)).
- Toggling: `chat_template_kwargs: {"enable_thinking": true/false}` (Qwen3/Holo2 default-on; Granite/Gemma-4/DeepSeek-V3.1 default-off with `thinking`/`enable_thinking`); `reasoning_effort` auto-enables thinking; `thinking_token_budget` caps it. Known bug: with `enable_thinking=false` some parsers misroute content into `reasoning` ([#40466](https://github.com/vllm-project/vllm/issues/40466)).
- Tool calling parses tools from `content` only, never from `reasoning`; reasoning + tool calls can coexist (model dependent — R1 can't do tools, Command-A-Reasoning can).
- Reasoning content is available on `/v1/chat/completions`, `/v1/responses` (as reasoning items), and (partially) `/v1/messages`.
- Usage does **not** include a reasoning-token count on chat completions (UsageInfo has no completion_tokens_details); Responses API reports reasoning token counts derived from the parser.

---

## 5. Tool calling

Source: [tool calling docs](https://docs.vllm.ai/en/latest/features/tool_calling/).

- Flags: `--enable-auto-tool-choice` + `--tool-call-parser <name>` (+ often a specific `--chat-template`).
- Parsers (current list): `hermes`, `mistral`, `llama3_json`, `llama4_pythonic`, `granite`, `granite4`, `granite-20b-fc`, `internlm`, `jamba`, `xlam`, `deepseek_v3`, `deepseek_v31`, `pythonic`, `openai` (gpt-oss), `kimi_k2`, `hunyuan_a13b`, `cohere_command3`, `longcat`, `glm45`, `glm47`, `functiongemma`, `qwen3_xml`, `qwen` (coder), `olmo3`, `gigachat3`, `apertus`, ... (grows every release; also `--tool-parser-plugin` for custom).
- `tool_choice`:
  - `"auto"` — needs the flags above; text-extraction by parser. If a tool sets `strict: true` and `VLLM_ENFORCE_STRICT_TOOL_CALLING` (default true), vLLM applies **structural-tag constrained decoding** for schema-exact arguments.
  - `"required"` — schema-constrained decoding via structured-outputs backend (guaranteed ≥1 valid call).
  - named `{"type":"function","function":{"name":...}}` — forced via structured outputs (works **without** `--enable-auto-tool-choice`).
  - `"none"` — no calls; `--exclude-tools-when-tool-choice-none` omits tool defs from the prompt.
- `parallel_tool_calls: bool = true` field accepted; actual parallel-call support is model/parser dependent (Llama 3.1: no; Llama 3.2/4, Hermes, Granite 3.1+, xLAM, pythonic: yes; Mistral 7B unreliable).
- Streaming: parsers implement incremental `extract_tool_calls_streaming()` → OpenAI-shaped `delta.tool_calls` (index/id/name/arguments fragments). Fidelity varies by parser; gpt-oss streaming tool calls had bugs ([#27641](https://github.com/vllm-project/vllm/issues/27641)).
- Quirks: Mistral requires 9-digit tool-call IDs (vLLM ships fixed templates); llama3_json sometimes emits stringified arrays; pythonic parser can't mix text + calls; a bug rejected requests with 17+ tools ([#27921](https://github.com/vllm-project/vllm/issues/27921)); empty `tool_calls: []` arrays seen after tool results ([#44104](https://github.com/vllm-project/vllm/issues/44104)).
- `finish_reason` is `"tool_calls"` when calls are parsed (see §6 caveats).

---

## 6. Streaming, usage, errors, finish reasons

Sources: `engine/protocol.py`, [goose#8021](https://github.com/block/goose/issues/8021), PR [#12897](https://github.com/vllm-project/vllm/pull/12897).

- **Usage**: `UsageInfo{prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details?}`; `PromptTokenUsageInfo{cached_tokens, multimodal_tokens}`. `prompt_tokens_details.cached_tokens` (prefix-cache hits) requires server flag **`--enable-prompt-tokens-details`** (had a v1-engine bug, [#16162](https://github.com/vllm-project/vllm/issues/16162), since fixed). No `completion_tokens_details`/reasoning tokens on chat completions.
- **stream_options**: `include_usage` (usage in a final chunk with empty `choices`, before `[DONE]`) and vLLM-specific **`continuous_usage_stats`** (usage on every chunk). Server can force usage via `--enable-force-include-usage`.
- SSE framing is standard `data: {...}\n\n` … `data: [DONE]`. No keep-alive comment lines documented.
- **Errors (non-stream)**: current main = OpenAI-shaped nested `{"error": {"message","type","param","code"}}` (`ErrorResponse{error: ErrorInfo}`). History: originally flat `{"object":"error","message",...,"code"}`; PR #12897 (~v0.7.3, Feb 2025) added a duplicated `error` field for compat; later versions fully nested. **Old deployments still emit flat errors** — parse both.
- **Mid-stream errors**: emitted as an SSE `data:` event whose payload is the error JSON (older: `{"object":"error","message":...,"code":N}`; newer: nested) with **no `choices` key**, after HTTP 200 was already sent — a chunk decoder must sniff for `error`/`object=="error"` before decoding as a chunk (this crashed goose's OpenAI provider).
- HTTP codes: 400 for validation/context-length-exceeded (message contains max context info; vLLM rejects rather than clamps — [#42474](https://github.com/vllm-project/vllm/issues/42474)); 401 bad API key; 404 unknown model/response id; 500 `GenerationError` (internal finish reason); FastAPI validation errors are converted to 400-style vLLM errors.
- **finish_reason**: `stop`, `length`, `tool_calls` (+ `null` in in-progress stream chunks; an intermittent bug returned final `null` — [#27572](https://github.com/vllm-project/vllm/issues/27572)). Engine-level `abort` exists but surfaces as an error/500 rather than a documented finish_reason. Plus the non-standard `stop_reason` sibling field (§2).

---

## 7. Models, LoRA, auth, deployment

Sources: [LoRA docs](https://docs.vllm.ai/en/latest/features/lora/), [serve CLI](https://docs.vllm.ai/en/latest/cli/serve/).

- `GET /v1/models` returns the served model(s) **and every registered LoRA adapter** as separate entries: `{id: <lora name>, root: <path>, parent: <base model id>}`; base models have `parent: null`. Client selects a LoRA by putting its name in `model`. Static: `--lora-modules name=path` or JSON `{"name","path","base_model_name"}`; dynamic load/unload endpoints (`{"lora_name","lora_path"}`) behind `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`.
- `--served-model-name` accepts **multiple aliases**; server answers to any of them; responses' `model` field shows the **first** name. Default model name = the HF path passed to `vllm serve`. (Requests with an unknown model → 404-style error.)
- **Auth**: none by default. `--api-key` (now accepts **multiple keys**) or `VLLM_API_KEY` → middleware requiring `Authorization: Bearer <key>` (per CLI docs: "require one of these keys ... in the header"). No key rotation/scopes. /health-type endpoints are handled by middleware config, not documented as exempt — don't assume metrics/health skip auth when a key is set.
- Base URL convention: `http://host:8000/v1` for OpenAI APIs (Anthropic clients use root, `http://host:8000`, since paths include /v1/messages). TLS via `--ssl-keyfile/--ssl-certfile/--ssl-ca-certs`; reverse-proxy via `--root-path`; CORS via `--allowed-origins` etc.; `--enable-request-id-headers` adds `X-Request-Id` response header (pairs with request `request_id` field); `--api-server-count` scales frontend processes.

---

## 8. Multimodal (client-visible)

Source: [multimodal inputs docs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/).

- Standard `image_url` content parts (http(s) URLs + `data:image/...;base64,`), optional per-part `uuid` for media caching.
- **vLLM-specific content-part types**: `video_url`, `audio_url`, `input_audio` (OpenAI-style `{data, format}`), `image_embeds` (base64 tensors; needs `--enable-mm-embeds`).
- Server limits are client-visible as 400 errors: `--limit-mm-per-prompt '{"image": N, ...}'` (exceeding → request rejected), `--allowed-media-domains`, `--allowed-local-media-path` (file:// URLs), `--media-io-kwargs`. Request-level `mm_processor_kwargs`/`media_io_kwargs` override processing.
- /v1/responses multimodal still incomplete (§1).

---

## 9. Divergences & quirks a client must handle (Compat checklist)

1. **Unknown fields are silently ignored** (`OpenAIBaseModel: model_config = ConfigDict(extra="allow")`, extra fields logged at debug). Consequence: sending the *wrong-era* structured-output param never errors — it's just ignored. go-llm should pick the spelling per configured vLLM version, or default to `response_format json_schema` (portable), or verify via probing.
2. **guided_\* → structured_outputs breaking change at v0.12.0** (removed, not aliased). Dialect needs a switch (e.g. `StructuredOutputsStyle: "guided" | "structured_outputs"`).
3. **`reasoning` vs `reasoning_content`**: read both on responses/deltas; emit assistant-turn reasoning back using `reasoning` (server normalizes `reasoning_content` too).
4. **Flat vs nested error JSON** and **mid-stream SSE error events without `choices`** (§6) — decoder must sniff.
5. **`stop_reason`** extra field next to finish_reason; harmless but present.
6. Extra response fields at multiple levels (`prompt_token_ids`, `token_ids`, `kv_transfer_params`, `routed_experts`) — decoders must not be strict.
7. **Usage nuances**: `cached_tokens` only with `--enable-prompt-tokens-details`; `continuous_usage_stats` is nonstandard; `multimodal_tokens` is a vLLM-only detail field.
8. **Tool-calling requires server-side flags**; without `--enable-auto-tool-choice`, `tool_choice:"auto"` requests error (only named/none work). Client can't detect capability from /v1/models — surface a clear error mapping.
9. `user` ignored; `logit_bias` supported; `n>1` supported; beam search via `use_beam_search`.
10. Old flat `{"object":"error"}` + transitional dual-shape errors from pre-2025 servers.
11. finish_reason `null` bug in some streams; treat missing finish_reason on last chunk defensively.
12. Anthropic /v1/messages validation is stricter than Anthropic's own API in places (#44000).

---

## 10. Implications for go-llm's vLLM preset (dialect/Compat must express)

- **Endpoints**: chat-completions (primary), completions, embeddings, models (with LoRA `parent`), `/tokenize`+`/detokenize`+`/tokenizer_info` (extension capability: exact token counting + max_model_len discovery), Responses API (optional secondary transport), Anthropic /v1/messages (could even reuse go-llm's Anthropic provider pointed at vLLM).
- **Capability flags**: structured outputs (json schema/regex/choice/grammar/structural_tag), reasoning field parsing, tool calling (server-flag dependent), logprobs + prompt_logprobs, multimodal content types incl. `video_url`/`audio_url`, priority, cache_salt, n>1, usage-in-stream.
- **Config knobs**: structured-outputs spelling (pre/post v0.12), reasoning field name tolerance, api-key optionality, served-model aliasing, `vllm_xargs` passthrough for anything not modeled.

## Sources

- https://docs.vllm.ai/en/latest/serving/online_serving/ (endpoint inventory, extra params)
- https://docs.vllm.ai/en/latest/features/structured_outputs/ (v0.12.0 migration)
- https://docs.vllm.ai/en/latest/features/tool_calling/ (parsers, tool_choice, strict mode)
- https://docs.vllm.ai/en/latest/features/reasoning_outputs/ (parsers, `reasoning` field)
- https://docs.vllm.ai/en/latest/features/multimodal_inputs/
- https://docs.vllm.ai/en/latest/features/lora/
- https://docs.vllm.ai/en/latest/serving/integrations/claude_code/
- https://docs.vllm.ai/en/latest/cli/serve/
- https://github.com/vllm-project/vllm — `vllm/entrypoints/openai/chat_completion/protocol.py`, `openai/completion/protocol.py`, `openai/engine/protocol.py`, `openai/api_server.py`, `serve/tokenize/protocol.py`, `entrypoints/anthropic/`
- https://github.com/vllm-project/vllm/releases (v0.24.0, 2026-06-26)
- Issues/PRs: #27755/#27752 (reasoning rename), #21313 + PR #22627 (v1/messages, v0.11.1), #14721/PR #20504 (Responses API, v0.10.0), PR #12897 + #12886 (error shape), #16162 (prompt_tokens_details), #6202 (stop_reason RFC), #24603/#24184/#38132/#32685 (Responses API gaps), #27921/#44104/#27641 (tool bugs), #40466 (enable_thinking bug), #42474 (context-length 400), block/goose#8021 (mid-stream error shape), BerriAI/litellm#20246 (reasoning field)
- https://vllm.ai/blog/2026-01-31-streaming-realtime (WS /v1/realtime)
