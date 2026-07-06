package openrouter

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"

	sdk "github.com/openai/openai-go/v3"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// dialect carries the OpenRouter-specific behavior for the shared
// chat-completions adapter plus the construction-time attribution headers it
// surfaces as Compat defaults.
type dialect struct {
	headers http.Header
}

func (dialect) Name() string                   { return providerName }
func (dialect) DefaultBaseURL() string         { return defaultOpenRouterBaseURL }
func (dialect) APIKeyEnv() string              { return apiKeyEnv }
func (dialect) Capabilities() []llm.Capability { return append([]llm.Capability(nil), capabilities...) }

func (d dialect) Compat() chatcompletions.Compat {
	return chatcompletions.Compat{
		// OpenRouter reports usage (and native cost) on the final stream
		// chunk; include_usage makes that explicit (matches pi).
		StreamIncludeUsage: true,
		MapEffort:          mapEffort,
		DefaultHeaders:     d.headers,
	}
}

// mapEffort implements FS §9's OpenRouter column: OpenRouter accepts the full
// unified minimal..max enum as reasoning.effort verbatim (nearest-level
// mapping is the identity), and EffortNone disables reasoning explicitly via
// reasoning.enabled=false. Compat.MapEffort returns top-level wire fields, so
// the value nests under the "reasoning" key here.
func mapEffort(effort llm.Effort) map[string]any {
	switch effort {
	case "":
		return nil
	case llm.EffortNone:
		return map[string]any{"reasoning": map[string]any{"enabled": false}}
	default:
		return map[string]any{"reasoning": map[string]any{"effort": string(effort)}}
	}
}

func (dialect) ApplyRequest(req *llm.Request, _ *sdk.ChatCompletionNewParams, extras chatcompletions.JSONObject) error {
	if req.SessionID != "" {
		extras["session_id"] = req.SessionID
	}
	options, err := requestOptions(req)
	if err != nil {
		return err
	}
	if options != nil {
		if len(options.Models) > 0 {
			extras["models"] = options.Models
		}
		if options.Provider != nil {
			extras["provider"] = options.Provider
		}
		if len(options.Plugins) > 0 {
			extras["plugins"] = options.Plugins
		}
		if options.Prediction != nil {
			extras["prediction"] = options.Prediction
		}
		if len(options.Reasoning) > 0 {
			reasoning := map[string]any{}
			for key, value := range options.Reasoning {
				reasoning[key] = value
			}
			// The unified Effort mapping (pre-populated by the adapter)
			// wins over option extras on conflict (FS §14).
			if unified, ok := extras["reasoning"].(map[string]any); ok {
				for key, value := range unified {
					reasoning[key] = value
				}
			}
			extras["reasoning"] = reasoning
		}
		if options.TopK != nil {
			extras["top_k"] = *options.TopK
		}
		if options.MinP != nil {
			extras["min_p"] = *options.MinP
		}
		if options.TopA != nil {
			extras["top_a"] = *options.TopA
		}
		if options.RepetitionPenalty != nil {
			extras["repetition_penalty"] = *options.RepetitionPenalty
		}
	}
	return nil
}

func (dialect) RequestHeaders(req *llm.Request) http.Header {
	// Errors are surfaced by ApplyRequest, which runs earlier in the same
	// request build; a spoofed options type never reaches this point.
	options, _ := requestOptions(req)
	if options == nil {
		return nil
	}
	headers := http.Header{}
	if options.HTTPReferer != "" {
		headers.Set("HTTP-Referer", options.HTTPReferer)
	}
	if options.XTitle != "" {
		headers.Set("X-Title", options.XTitle)
	}
	return headers
}

func requestOptions(req *llm.Request) (*Options, error) {
	options, ok, err := providerutil.OptionsOf[Options](req)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, nil
	}
	return &options, nil
}

func (dialect) MapStopReason(raw string) llm.StopReason {
	switch raw {
	case "stop":
		return llm.StopReasonEndTurn
	case "length":
		return llm.StopReasonMaxTokens
	case "tool_calls", "function_call":
		return llm.StopReasonToolUse
	case "content_filter":
		return llm.StopReasonContentFilter
	case "error":
		return llm.StopReasonError
	case "":
		return ""
	default:
		return llm.StopReasonOther
	}
}

// MapErrorStatus layers OpenRouter's numeric stream error codes over the
// unified classifier: mid-stream errors carry no HTTP status, only a numeric
// code string, so "402" must classify as ErrInsufficientCredits itself.
// Everything else (including moderation vocabulary and the canonical status
// table) is the shared chatcompletions default.
func (dialect) MapErrorStatus(status int, code, message string) error {
	if code == "402" {
		return llm.ErrInsufficientCredits
	}
	return chatcompletions.DefaultErrorKind(status, code, message)
}

// ExtractParts defers to the adapter's default chat-completions mapping —
// OpenRouter's reasoning/reasoning_details, content, refusal, and tool-call
// shapes are exactly the standard ones, and the default mapping tags
// ReasoningPart.Provider with Dialect.Name() for replay.
func (dialect) ExtractParts(_ chatcompletions.JSONObject, _ chatcompletions.RawMessage) ([]llm.Part, []llm.DroppedToolCall, error) {
	return nil, nil, nil
}

func (dialect) ExtractExtras(raw chatcompletions.JSONObject, choice chatcompletions.RawChoice) any {
	extras := &ResponseExtras{Raw: raw}
	if value, ok := raw["provider"].(string); ok {
		extras.Provider = value
	}
	if value, ok := choice.Raw["native_finish_reason"].(string); ok {
		extras.NativeFinishReason = value
	}
	if value, ok := raw["native_finish_reason"].(string); ok && extras.NativeFinishReason == "" {
		extras.NativeFinishReason = value
	}
	// Usage accounting extras live INSIDE the usage object on the wire
	// (usage.{cost, cost_details, is_byok} per the OpenRouter docs), not at
	// the response root.
	if usage, ok := raw["usage"].(map[string]any); ok {
		if value, ok := usage["is_byok"].(bool); ok {
			extras.IsBYOK = &value
		}
		extras.CostDetails = marshalRaw(usage["cost_details"])
	}
	extras.Annotations = choice.Message.Annotations
	extras.ReasoningDetails = choice.Message.ReasoningDetails
	return extras
}

func (dialect) MapUsage(model string, raw chatcompletions.RawUsage, table llm.PriceTable) llm.Usage {
	cacheRead := raw.PromptTokensDetails.CachedTokens
	inputTokens := raw.PromptTokens
	if cacheRead > 0 && inputTokens >= cacheRead {
		inputTokens -= cacheRead
	}
	out := llm.Usage{
		InputTokens:     inputTokens,
		CacheReadTokens: cacheRead,
		OutputTokens:    raw.CompletionTokens,
		ReasoningTokens: raw.CompletionTokensDetails.ReasoningTokens,
		TotalTokens:     raw.TotalTokens,
		Raw:             raw.Raw,
	}
	if out.TotalTokens == 0 {
		out.TotalTokens = out.InputTokens + out.CacheReadTokens + out.OutputTokens
	}
	if raw.Cost != nil {
		// OpenRouter reports billing-grade cost natively (usage.cost).
		cost := *raw.Cost
		out.CostUSD = &cost
		out.CostSource = llm.CostSourceNative
		return out
	}
	if table != nil {
		return llm.EstimateCostWithTable(table, providerName, model, out)
	}
	return llm.EstimateCostForModel(providerName, model, out)
}

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
			ID            string `json:"id"`
			Name          string `json:"name"`
			ContextLength int    `json:"context_length"`
			TopProvider   struct {
				MaxCompletionTokens int `json:"max_completion_tokens"`
			} `json:"top_provider"`
			Pricing struct {
				Prompt     string `json:"prompt"`
				Completion string `json:"completion"`
			} `json:"pricing"`
			CanonicalSlug string `json:"canonical_slug"`
		}
		if err := json.Unmarshal(rawRow, &row); err != nil {
			return nil, err
		}
		info := llm.ModelInfo{
			ID:              row.ID,
			DisplayName:     row.Name,
			ContextWindow:   row.ContextLength,
			MaxOutputTokens: row.TopProvider.MaxCompletionTokens,
			CanonicalID:     row.CanonicalSlug,
			Raw:             append(json.RawMessage(nil), rawRow...),
		}
		if pricing := parsePricing(row.Pricing.Prompt, row.Pricing.Completion); pricing != nil {
			info.Pricing = pricing
		}
		models = append(models, info)
	}
	return models, nil
}

func parsePricing(prompt, completion string) *llm.ModelPricing {
	in, okIn := parseFloat(prompt)
	out, okOut := parseFloat(completion)
	if !okIn && !okOut {
		return nil
	}
	return &llm.ModelPricing{InputPerMTok: in * 1_000_000, OutputPerMTok: out * 1_000_000}
}

func parseFloat(s string) (float64, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	var out float64
	if err := json.Unmarshal([]byte(s), &out); err == nil {
		return out, true
	}
	return 0, false
}

func marshalRaw(value any) json.RawMessage {
	if value == nil {
		return nil
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	return raw
}
