package vllm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	sdk "github.com/openai/openai-go/v3"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// dialect carries the vLLM-specific behavior for the shared chat-completions
// engine. legacyEra switches the reasoning replay field for pre-v0.12
// servers.
type dialect struct {
	legacyEra bool
}

func (dialect) Name() string           { return providerName }
func (dialect) DefaultBaseURL() string { return "" }
func (dialect) APIKeyEnv() string      { return "" }

func (dialect) Capabilities() []llm.Capability {
	return append([]llm.Capability(nil), capabilities...)
}

func (d dialect) Compat() chatcompletions.Compat {
	// Modern era (v0.12+) replays reasoning under `reasoning`; the server
	// normalizes `reasoning_content` too, but pre-rename servers only accept
	// the old spelling.
	replayField := "reasoning"
	if d.legacyEra {
		replayField = "reasoning_content"
	}
	return chatcompletions.Compat{
		// vLLM reports usage on a trailing empty-choices chunk when
		// stream_options.include_usage is set.
		StreamIncludeUsage: true,
		MapEffort:          mapEffort,
		// vLLM emits mid-stream failures as choice-less SSE error events
		// after HTTP 200 (nested on current servers, flat {"object":"error"}
		// historically) — the goose-crash case; sniff both shapes.
		SniffMidStreamErrors: true,
		// Forced (named) tool calls end with finish_reason "stop" on vLLM
		// (OpenAI CC semantics); normalize to tool_use when calls are present.
		NormalizeToolUseStop: true,
		ReasoningReplayField: replayField,
	}
}

// mapEffort implements FS §9's vLLM column: the wire field is
// `reasoning_effort` with enum none|minimal|low|medium|high|max ("max" is
// vLLM-specific). The unified "xhigh" has no vLLM level and maps to the
// nearest, "high". Non-none efforts auto-inject enable_thinking server-side.
func mapEffort(effort llm.Effort) map[string]any {
	switch effort {
	case "":
		return nil
	case llm.EffortXHigh:
		return map[string]any{"reasoning_effort": "high"}
	default:
		return map[string]any{"reasoning_effort": string(effort)}
	}
}

func (d dialect) ApplyRequest(req *llm.Request, _ *sdk.ChatCompletionNewParams, extras chatcompletions.JSONObject) error {
	options, ok, err := providerutil.OptionsOf[Options](req)
	if err != nil {
		return err
	}
	if !ok {
		return nil
	}
	if options.StructuredOutputs != nil {
		value, err := d.structuredOutputsValue(req, *options.StructuredOutputs)
		if err != nil {
			return err
		}
		extras["structured_outputs"] = value
	}
	if options.TopK != nil {
		extras["top_k"] = *options.TopK
	}
	if options.MinP != nil {
		extras["min_p"] = *options.MinP
	}
	if options.RepetitionPenalty != nil {
		extras["repetition_penalty"] = *options.RepetitionPenalty
	}
	if len(options.StopTokenIDs) > 0 {
		extras["stop_token_ids"] = append([]int(nil), options.StopTokenIDs...)
	}
	if kwargs := options.chatTemplateKwargs(); len(kwargs) > 0 {
		extras["chat_template_kwargs"] = kwargs
	}
	if len(options.XArgs) > 0 {
		xargs := make(map[string]any, len(options.XArgs))
		for key, value := range options.XArgs {
			xargs[key] = value
		}
		extras["vllm_xargs"] = xargs
	}
	return nil
}

// structuredOutputsValue validates and builds the structured_outputs wire
// object. Conflict rules are documented on the StructuredOutputs type: the
// unified ResponseFormat wins the constraint-system slot (two constraint
// systems on one request is ambiguous → fail loud), the param requires a
// modern-era server, and exactly one mode field may be set.
func (d dialect) structuredOutputsValue(req *llm.Request, so StructuredOutputs) (map[string]any, error) {
	if req.ResponseFormat != nil {
		return nil, fmt.Errorf("%w: vllm: Request.ResponseFormat and Options.StructuredOutputs are both set — use one constraint system (JSON schema belongs on ResponseFormat)", llm.ErrBadRequest)
	}
	if d.legacyEra {
		return nil, fmt.Errorf("%w: vllm: StructuredOutputs needs a v0.12+ server (provider has WithLegacyEra): structured_outputs does not exist pre-v0.12, and the legacy guided_* spelling is not emitted because modern servers silently ignore it", llm.ErrUnsupported)
	}
	value := map[string]any{}
	if so.Regex != "" {
		value["regex"] = so.Regex
	}
	if len(so.Choice) > 0 {
		value["choice"] = append([]string(nil), so.Choice...)
	}
	if so.Grammar != "" {
		value["grammar"] = so.Grammar
	}
	if len(so.StructuralTag) > 0 {
		if !json.Valid(so.StructuralTag) {
			return nil, fmt.Errorf("%w: vllm: StructuredOutputs.StructuralTag must be valid JSON", llm.ErrBadRequest)
		}
		value["structural_tag"] = json.RawMessage(so.StructuralTag)
	}
	if len(value) != 1 {
		return nil, fmt.Errorf("%w: vllm: StructuredOutputs requires exactly one of Regex, Choice, Grammar, StructuralTag (%d set)", llm.ErrBadRequest, len(value))
	}
	if so.WhitespacePattern != "" {
		value["whitespace_pattern"] = so.WhitespacePattern
	}
	return value, nil
}

// chatTemplateKwargs merges the typed EnableThinking toggle into the generic
// chat_template_kwargs map; the typed field wins on conflict.
func (o Options) chatTemplateKwargs() map[string]any {
	if len(o.ChatTemplateKwargs) == 0 && o.EnableThinking == nil {
		return nil
	}
	kwargs := make(map[string]any, len(o.ChatTemplateKwargs)+1)
	for key, value := range o.ChatTemplateKwargs {
		kwargs[key] = value
	}
	if o.EnableThinking != nil {
		kwargs["enable_thinking"] = *o.EnableThinking
	}
	return kwargs
}

// MapStopReason layers vLLM's engine-level "abort" over the standard
// chat-completions vocabulary.
func (dialect) MapStopReason(raw string) llm.StopReason {
	if raw == "abort" {
		return llm.StopReasonError
	}
	return chatcompletions.DefaultStopReason(raw)
}

// MapErrorStatus defers to the shared classifier, which handles vLLM's
// numeric in-stream error codes: mid-stream error events carry no HTTP
// status, but their `code` mirrors one (e.g. {"object":"error","code":503}
// after HTTP 200), and the unified classifier maps integral status-like
// codes through the canonical status table.
func (dialect) MapErrorStatus(status int, code, message string) error {
	return chatcompletions.DefaultErrorKind(status, code, message)
}

// ExtractParts defers to the engine's default chat-completions mapping: vLLM
// reasoning is plain text on `reasoning` (or legacy `reasoning_content`,
// which the default mapping also reads), and content/tool-call shapes are
// standard.
func (dialect) ExtractParts(_ chatcompletions.JSONObject, _ chatcompletions.RawMessage) ([]llm.Part, []llm.DroppedToolCall, error) {
	return nil, nil, nil
}

// ExtractExtras exposes the raw response root so vLLM-specific fields
// (stop_reason, prompt_token_ids, kv_transfer_params, system_fingerprint)
// stay reachable through Response.Raw.
func (dialect) ExtractExtras(raw chatcompletions.JSONObject, _ chatcompletions.RawChoice) any {
	if len(raw) == 0 {
		return nil
	}
	return raw
}

// MapUsage is the standard chat-completions normalization.
// prompt_tokens_details.cached_tokens (prefix-cache hits) is populated only
// when the server runs with --enable-prompt-tokens-details; vLLM reports no
// reasoning-token counts on chat completions and no native cost.
func (dialect) MapUsage(model string, raw chatcompletions.RawUsage, table llm.PriceTable) llm.Usage {
	return chatcompletions.DefaultUsage(providerName, model, raw, table)
}

// Models lists the served models (and any registered LoRA adapters) from
// GET /v1/models, surfacing vLLM's max_model_len as ContextWindow. LoRA rows
// carry their base model in the raw payload's `parent` field.
func (dialect) Models(ctx context.Context, p *chatcompletions.Provider) ([]llm.ModelInfo, error) {
	var payload struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := p.DoJSON(ctx, http.MethodGet, "/models", nil, &payload); err != nil {
		return nil, err
	}
	models := make([]llm.ModelInfo, 0, len(payload.Data))
	for _, rawRow := range payload.Data {
		var row struct {
			ID          string `json:"id"`
			MaxModelLen int    `json:"max_model_len"`
		}
		if err := json.Unmarshal(rawRow, &row); err != nil {
			return nil, providerutil.NormalizeRemoteError(providerName, err)
		}
		models = append(models, llm.ModelInfo{
			ID:            row.ID,
			ContextWindow: row.MaxModelLen,
			Raw:           append(json.RawMessage(nil), rawRow...),
		})
	}
	return models, nil
}
