package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"regexp"
	"strconv"
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
		// chunk; include_usage makes that explicit.
		StreamIncludeUsage:       true,
		ToolMessageContentBlocks: true,
		MapEffort:                mapEffort,
		DefaultHeaders:           d.headers,
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
	if cost, ok := reportedCost(raw); ok {
		out.CostUSD = &cost
		out.CostSource = llm.CostSourceNative
		return out
	}
	if table != nil {
		return llm.EstimateCostWithTable(table, providerName, model, out)
	}
	return llm.EstimateCostForModel(providerName, model, out)
}

// reportedCost is the billed cost from OpenRouter's usage accounting.
// usage.cost is what OpenRouter charged. With a BYOK key that is only
// OpenRouter's fee, and the inference itself is billed upstream as
// cost_details.upstream_inference_cost (or, when that is null or absent, its
// prompt and completion parts), so the two are summed. A missing is_byok
// means a plain key; malformed or incomplete accounting reports nothing,
// leaving the price-table estimate (as pi does).
func reportedCost(raw chatcompletions.RawUsage) (float64, bool) {
	if raw.Cost == nil || !validCost(*raw.Cost) {
		return 0, false
	}
	flag, present := raw.Raw["is_byok"]
	byok, ok := flag.(bool)
	if present && !ok {
		return 0, false
	}
	if !byok {
		return *raw.Cost, true
	}
	details, _ := raw.Raw["cost_details"].(map[string]any)
	if details == nil {
		return 0, false
	}
	var upstream float64
	switch value := details["upstream_inference_cost"].(type) {
	case float64:
		upstream = value
	case nil:
		prompt, promptOK := details["upstream_inference_prompt_cost"].(float64)
		completion, completionOK := details["upstream_inference_completions_cost"].(float64)
		if !promptOK || !completionOK || !validCost(prompt) || !validCost(completion) {
			return 0, false
		}
		upstream = prompt + completion
	default:
		return 0, false
	}
	if !validCost(upstream) {
		return 0, false
	}
	return *raw.Cost + upstream, true
}

// validCost reports whether a reported cost is a finite, non-negative amount.
func validCost(value float64) bool {
	return value >= 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

// Models lists the chat catalog and the classifier (decisions) catalog,
// which OpenRouter's default listing leaves out (FS §13C). A failed
// decisions listing (a proxy or mirror that rejects the filter) leaves the
// chat catalog alone; only cancellation fails the call.
func (dialect) Models(ctx context.Context, p *chatcompletions.Provider) ([]llm.ModelInfo, error) {
	chat, err := listModels(ctx, p, "/models")
	if err != nil {
		return nil, err
	}
	classifiers, err := listModels(ctx, p, "/models?output_modalities=decisions")
	if err != nil {
		if ctx.Err() != nil {
			return nil, err
		}
		return chat, nil
	}
	seen := make(map[string]bool, len(chat))
	for _, info := range chat {
		seen[info.ID] = true
	}
	for _, info := range classifiers {
		if !seen[info.ID] {
			chat = append(chat, info)
			seen[info.ID] = true
		}
	}
	return chat, nil
}

func listModels(ctx context.Context, p *chatcompletions.Provider, path string) ([]llm.ModelInfo, error) {
	var payload struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := p.DoJSON(ctx, http.MethodGet, path, nil, &payload); err != nil {
		return nil, err
	}
	models := make([]llm.ModelInfo, 0, len(payload.Data))
	for _, rawRow := range payload.Data {
		var row struct {
			ID                  string   `json:"id"`
			Name                string   `json:"name"`
			ContextLength       int      `json:"context_length"`
			SupportedParameters []string `json:"supported_parameters"`
			Architecture        struct {
				InputModalities  []string `json:"input_modalities"`
				OutputModalities []string `json:"output_modalities"`
			} `json:"architecture"`
			TopProvider struct {
				MaxCompletionTokens int `json:"max_completion_tokens"`
			} `json:"top_provider"`
			Pricing       json.RawMessage `json:"pricing"`
			Reasoning     json.RawMessage `json:"reasoning"`
			CanonicalSlug string          `json:"canonical_slug"`
		}
		if err := json.Unmarshal(rawRow, &row); err != nil {
			return nil, providerutil.NormalizeRemoteError(providerName, err)
		}
		info := llm.ModelInfo{
			ID:              row.ID,
			DisplayName:     row.Name,
			ContextWindow:   row.ContextLength,
			MaxOutputTokens: row.TopProvider.MaxCompletionTokens,
			CanonicalID:     row.CanonicalSlug,
			Capabilities:    openRouterModelCapabilities(row.SupportedParameters, row.Architecture.InputModalities),
			Raw:             append(json.RawMessage(nil), rawRow...),
		}
		for _, modality := range row.Architecture.OutputModalities {
			if strings.TrimSpace(modality) == "decisions" {
				info.Capabilities = append(info.Capabilities, llm.CapabilityClassify)
				break
			}
		}
		if pricing := parsePricing(row.Pricing); pricing != nil {
			info.Pricing = pricing
		}
		mapReasoningMetadata(&info, row.Reasoning)
		models = append(models, info)
	}
	return models, nil
}

func openRouterModelCapabilities(parameters, inputModalities []string) []llm.Capability {
	params := make(map[string]bool, len(parameters))
	for _, parameter := range parameters {
		if parameter = strings.TrimSpace(parameter); parameter != "" {
			params[parameter] = true
		}
	}
	modalities := make(map[string]bool, len(inputModalities))
	for _, modality := range inputModalities {
		if modality = strings.TrimSpace(modality); modality != "" {
			modalities[modality] = true
		}
	}

	var capabilities []llm.Capability
	if params["tools"] {
		capabilities = append(capabilities, llm.CapabilityTools)
	}
	if params["tool_choice"] {
		capabilities = append(capabilities, llm.CapabilityToolChoiceRequired)
	}
	if params["parallel_tool_calls"] {
		capabilities = append(capabilities, llm.CapabilityParallelTools)
	}
	if params["structured_outputs"] {
		capabilities = append(capabilities, llm.CapabilityJSONSchema)
	}
	if params["reasoning"] || params["include_reasoning"] || params["reasoning_effort"] {
		capabilities = append(capabilities, llm.CapabilityReasoning)
	}
	if modalities["image"] {
		capabilities = append(capabilities, llm.CapabilityImageInput)
	}
	if params["stop"] {
		capabilities = append(capabilities, llm.CapabilityStopSequences)
	}
	return capabilities
}

func parsePricing(raw json.RawMessage) *llm.ModelPricing {
	var components map[string]json.RawMessage
	if len(raw) == 0 || json.Unmarshal(raw, &components) != nil || components == nil {
		return nil
	}
	in, inStatus := parsePrice(components["prompt"])
	out, outStatus := parsePrice(components["completion"])
	read, readStatus := parsePrice(components["input_cache_read"])
	write, writeStatus := parsePrice(components["input_cache_write"])
	if inStatus != priceValid && outStatus != priceValid && readStatus != priceValid && writeStatus != priceValid {
		return nil
	}
	return &llm.ModelPricing{
		InputPerMTok:      in,
		OutputPerMTok:     out,
		CacheReadPerMTok:  read,
		CacheWritePerMTok: write,
		Availability: &llm.ModelPricingAvailability{
			InputPerMTok:      inStatus == priceValid,
			OutputPerMTok:     outStatus == priceValid,
			CacheReadPerMTok:  readStatus == priceValid,
			CacheWritePerMTok: writeStatus == priceValid,
		},
	}
}

type priceStatus uint8

const (
	priceInvalid priceStatus = iota
	priceValid
	priceNegative
)

var decimalPricePattern = regexp.MustCompile(`^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$`)

func parsePrice(raw json.RawMessage) (float64, priceStatus) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return 0, priceInvalid
	}
	var valueToken any
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(&valueToken); err != nil {
		return 0, priceInvalid
	}
	var s string
	switch value := valueToken.(type) {
	case string:
		s = strings.TrimSpace(value)
	case json.Number:
		s = string(value)
	default:
		return 0, priceInvalid
	}
	if s == "" {
		return 0, priceInvalid
	}
	if !decimalPricePattern.MatchString(s) {
		return 0, priceInvalid
	}
	value, err := strconv.ParseFloat(s, 64)
	if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, priceInvalid
	}
	if value < 0 {
		return 0, priceNegative
	}
	perMillion := value * 1_000_000
	if math.IsNaN(perMillion) || math.IsInf(perMillion, 0) || perMillion < 0 {
		return 0, priceInvalid
	}
	return perMillion, priceValid
}

var knownEfforts = [...]llm.Effort{
	llm.EffortNone,
	llm.EffortMinimal,
	llm.EffortLow,
	llm.EffortMedium,
	llm.EffortHigh,
	llm.EffortXHigh,
	llm.EffortMax,
}

func mapReasoningMetadata(info *llm.ModelInfo, raw json.RawMessage) {
	if len(raw) == 0 {
		return
	}
	var reasoning map[string]json.RawMessage
	if err := json.Unmarshal(raw, &reasoning); err != nil || reasoning == nil {
		return
	}

	mandatory := false
	if mandatoryRaw, ok := reasoning["mandatory"]; ok {
		_ = json.Unmarshal(mandatoryRaw, &mandatory)
	}
	info.ReasoningRequired = mandatory

	efforts, ladderState := parseSupportedEfforts(reasoning, mandatory)
	if ladderState != effortLadderUnknown {
		info.SupportedEfforts = efforts
	}

	defaultRaw, ok := reasoning["default_effort"]
	if !ok {
		return
	}
	var defaultValue string
	if err := json.Unmarshal(defaultRaw, &defaultValue); err != nil {
		return
	}
	defaultEffort, known := knownEffort(defaultValue)
	if !known || (mandatory && defaultEffort == llm.EffortNone) {
		return
	}
	switch ladderState {
	case effortLadderKnown:
		if len(efforts) > 0 && effortIncluded(efforts, defaultEffort) {
			info.DefaultEffort = defaultEffort
		}
	case effortLadderUnknown:
		if defaultEffort != llm.EffortNone {
			info.DefaultEffort = defaultEffort
		}
	}
}

type effortLadderState uint8

const (
	effortLadderUnknown effortLadderState = iota
	effortLadderKnown
)

func parseSupportedEfforts(reasoning map[string]json.RawMessage, mandatory bool) ([]llm.Effort, effortLadderState) {
	raw, ok := reasoning["supported_efforts"]
	if !ok {
		return nil, effortLadderUnknown
	}
	if strings.TrimSpace(string(raw)) == "null" {
		return filterMandatoryEffort(append([]llm.Effort(nil), knownEfforts[:]...), mandatory), effortLadderKnown
	}
	var values []string
	if err := json.Unmarshal(raw, &values); err != nil || values == nil {
		return nil, effortLadderUnknown
	}

	seen := make(map[llm.Effort]bool, len(values))
	for _, value := range values {
		if effort, known := knownEffort(value); known {
			seen[effort] = true
		}
	}
	efforts := make([]llm.Effort, 0, len(seen))
	for _, effort := range knownEfforts {
		if seen[effort] && (!mandatory || effort != llm.EffortNone) {
			efforts = append(efforts, effort)
		}
	}
	// Preserve the distinction between an explicit empty/unknown-only ladder
	// and an omitted or malformed ladder.
	if efforts == nil {
		efforts = []llm.Effort{}
	}
	return efforts, effortLadderKnown
}

func filterMandatoryEffort(efforts []llm.Effort, mandatory bool) []llm.Effort {
	if !mandatory {
		return efforts
	}
	return efforts[1:]
}

func knownEffort(value string) (llm.Effort, bool) {
	effort := llm.Effort(value)
	for _, known := range knownEfforts {
		if effort == known {
			return effort, true
		}
	}
	return "", false
}

func effortIncluded(efforts []llm.Effort, want llm.Effort) bool {
	for _, effort := range efforts {
		if effort == want {
			return true
		}
	}
	return false
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
