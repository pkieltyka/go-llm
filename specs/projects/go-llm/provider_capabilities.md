---
status: complete
---

# Provider Capability Matrix: overlap vs provider-unique

Reference appendix to the functional spec/architecture. Answers: *how much
of each provider's chat surface normalizes into go-llm's unified interface,
and how much is reachable only via the escape hatches
(`ProviderOptions` / extension parts / raw `Client()`)?*

Sources: Anthropic Messages API reference (July 2026), OpenAI Chat
Completions + Responses docs research (July 2026), OpenRouter docs research
(July 2026), Z.AI docs research (July 2026). Surfaces assessed:
**Anthropic Messages API**, **OpenAI Responses API** (the chosen surface for
the OpenAI provider — see Decision note), **OpenRouter chat completions**,
**ZAI chat completions**.

**Methodology.** Chat-relevant features only (adjacent endpoints like
batches/files/audio-gen are out of scope for v1 and excluded). Each feature
is classified: **N** = normalized (expressible via the unified
`Request`/parts/`Usage`/errors surface), **P** = partially normalized
(unified with documented caveats), **U** = provider-unique (escape hatch
only). Feature counting is inherently judgment-laden — percentages are
directional, not precise.

## Summary

| Provider | Normalized (N+P) | Unique (U) | Notes |
|---|---|---|---|
| Anthropic | ~50% (20 of ~40) | ~50% | Uniqueness = platform machinery: server tools, MCP, skills, compaction, task budgets, fast mode |
| OpenAI (Responses) | ~59% (19 of ~32) | ~41% | Uniqueness = hosted tools + conversation state + background mode; Responses dropped most CC legacy knobs |
| OpenRouter | ~51% (19 of ~37) | ~49% | Uniqueness = routing/marketplace: fallbacks, provider prefs, plugins, presets |
| ZAI | ~60% (15 of ~25) | ~40% | Uniqueness = bundled server tools (web_search/retrieval) + extra modalities |

**Cross-provider disjointness:** the four "unique" sets barely overlap with
*each other* (≈95% pairwise disjoint) — Anthropic's uniqueness is platform
tooling, OpenAI's is state/hosted-tools, OpenRouter's is routing, ZAI's is
bundled search + modalities. The one shape all four share in incompatible
forms: **hosted/server-side web search** (Anthropic `web_search` tool,
OpenAI Responses `web_search` tool, OpenRouter web plugin/`:online`, ZAI
`web_search` tool type) — the most promising future unification candidate
(v2), deliberately not unified in v1.

**Usage-weighted reality:** feature-count percentages understate practical
coverage. The normalized surface (chat, streaming, tools, structured
output, effort, images, usage/cost, errors) is what the overwhelming
majority of real calls touch — usage-weighted, the unified interface covers
roughly **90%+** of typical requests; the escape hatches exist for the long
tail.

## Anthropic (Messages API)

**Normalized (N):** messages/roles · system prompt · streaming ·
max_tokens · stop_sequences · temperature/top_p (older models; removed on
4.7+, pass-through) · tools · parallel tool calls · strict tools ·
tool_choice (all modes) · structured output (`output_config.format`) ·
adaptive thinking + effort → `Effort` · thinking blocks → `ReasoningPart`
(signature round-trip via `Raw`) · image input · PDF input → `FilePart` ·
usage incl. cache tokens · stop reasons (incl. `refusal`,
`context_overflow`, `paused`) · errors · models listing.

**Partially normalized (P):** prompt caching `cache_control` → `CacheHint`
(TTL granularity 5m/1h beyond the hint via options).

**Unique (U, escape hatch):** server tools — web search, web fetch, code
execution, computer use, memory, text editor + bash, tool search /
defer_loading, programmatic tool calling · MCP connector · skills/container
· context editing · compaction · task budgets · server-side fallbacks ·
`stop_details` refusal categories · fast mode (`speed`) · `inference_geo` ·
mid-conversation system messages · citations · token-counting endpoint ·
service tiers · beta-header mechanism.

## OpenAI (Responses API)

**Normalized (N):** input items/roles (≈ messages) · `instructions` →
`System` · streaming · `max_output_tokens` → `MaxTokens` ·
temperature/top_p (non-reasoning models) · tools (flattened function
shape) · parallel tool calls · strict (default-on) · tool_choice ·
structured output (`text.format` json_schema) · `reasoning.effort` →
`Effort` (full `none…xhigh` range) · reasoning summaries + encrypted items
→ `ReasoningPart` (`Raw` carries the reasoning item incl.
`encrypted_content` for stateless multi-turn continuity) · image input ·
file/PDF input · usage (incl. `cached_tokens`, `reasoning_tokens`) ·
status/`incomplete_details` → stop reasons · errors · models listing.

**Partially normalized (P):** prompt caching (automatic;
`prompt_cache_key` → `SessionID`).

**Unique (U, escape hatch):** hosted tools — web_search, file_search,
code_interpreter, computer_use, image_generation, remote MCP, shell,
skills, apply_patch, tool_search · conversation state (`store`,
`previous_response_id`, Conversations API) · background mode · `include[]`
mechanism · prompt templates (`prompt: {id, version}`) ·
`context_management` compaction · reasoning summary detail levels ·
`verbosity` · `metadata` · `service_tier` · `safety_identifier` ·
`prompt_cache_retention`.

**Dropped by Responses** (existed on Chat Completions; *not* carried into
go-llm's OpenAI surface): `stop` sequences (→ go-llm gates
`StopSequences` on the `stop-sequences` capability), `n`, `seed`,
`logit_bias`, `logprobs`, presence/frequency penalties, audio modalities,
`prediction` — reachable only by pointing the raw `Client()` at Chat
Completions.

## OpenRouter (chat completions + extensions)

**Normalized (N):** messages/system/streaming/max_tokens/temp/top_p/stop ·
tools + parallel + tool_choice · structured output (`response_format`
json_schema, upstream-dependent) · `reasoning.effort` → `Effort` ·
`reasoning`/`reasoning_details` output → `ReasoningPart` (echo-back via
`Raw`) · image input · usage + cached tokens · **`usage.cost` → `CostUSD`**
(the only provider with native cost) · `session_id` → `SessionID` ·
`cache_control` passthrough → `CacheHint` · normalized + `native` finish
reasons · errors (incl. moderation metadata via `ProviderError`) · models
listing (richest metadata).

**Unique (U, escape hatch):** `models` fallback array · `provider` routing
preferences (order/only/ignore/allow_fallbacks/require_parameters/sort/
max_price/quantizations/zdr/throughput/latency prefs) · model suffixes
(`:free`/`:nitro`/`:floor`/`:online`/`:extended`/`:thinking`/`:exacto`) ·
plugins (web search engines, context-compression, file-parser,
response-healing) · presets (`@preset/slug`) · `prediction` · extra
sampling (`top_k`, `min_p`, `top_a`, `repetition_penalty`) · attribution
headers · reasoning `exclude`/`max_tokens` variants · `verbosity` ·
response extras (`provider`, `cost_details`, `is_byok`, annotations) ·
generation/key/credits endpoints · debug echo.

## ZAI (chat completions)

**Normalized (N):** messages/system/streaming/max_tokens ·
temperature/top_p (clamped ranges [0,1]/[0.01,1], pass-through) · stop
(≤4) · tools (function) · tool-argument streaming (`tool_stream` —
absorbed: auto-set by the adapter) · json-mode (`json_object`) ·
`thinking` + `reasoning_effort` → `Effort` · `reasoning_content` →
`ReasoningPart` · image input · usage + `cached_tokens` · stop reasons
(incl. `sensitive`, `network_error`, `model_context_window_exceeded`) ·
errors (numeric business codes mapped) · models (curated static list).

**Partially normalized (P):** `user_id` → `SessionID` (attribution-grade
affinity only) · `tool_choice` (**`auto` only** — `required`/named →
`ErrUnsupported`).

**Unique (U, escape hatch):** `thinking.clear_thinking` · `do_sample` ·
`request_id` · `web_search` tool type (engines, recency/domain filters,
result sequencing) · `retrieval` tool type (knowledge bases) · `video_url`
input part · `file_url` input part · `web_search[]` response block ·
coding-plan base URL · Anthropic-compatible endpoint · per-model sampling
defaults.

## Decision note: OpenAI provider uses the Responses API

Verified July 2026: OpenAI's docs state *"While Chat Completions remains
supported, Responses is recommended for all new projects"*; GPT-5.5 "works
best in the Responses API". Chat Completions is not deprecated and no
sunset is announced, but on it: reasoning is **discarded between turns**
(Responses' reasoning persistence is worth ~3% SWE-bench on agentic tool
loops and 40%→80% cache utilization), no reasoning content is ever
returned, and hosted tools/background mode don't exist. `openai-go` v3 has
mature first-class Responses support in the same module that powers the
CC-shaped adapter.

**Resulting architecture:** OpenAI = direct wrap of `openai-go` Responses
(stateless: `store: false` + `include: ["reasoning.encrypted_content"]`,
reasoning items round-tripped via `ReasoningPart.Raw`). OpenRouter + ZAI
stay on the shared CC-shaped `openaicompat` adapter — chat completions is
OpenRouter's canonical surface (its `/responses` endpoint is beta,
stateless-only) and ZAI's only OpenAI-style surface (no Responses shape
exists there).
