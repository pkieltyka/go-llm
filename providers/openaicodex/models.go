package openaicodex

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/provideroauth"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

const (
	modelsSuccessTTL = 30 * time.Minute
	modelsFailureTTL = 5 * time.Minute
	modelsBodyLimit  = 8 << 20
)

type modelsFlight struct {
	done   chan struct{}
	models []llm.ModelInfo
	err    error
}

type codexCatalogDocument struct {
	Models []json.RawMessage `json:"models"`
	Data   []json.RawMessage `json:"data"`
}

type codexCatalogRow struct {
	ID                        string          `json:"id"`
	Slug                      string          `json:"slug"`
	DisplayName               string          `json:"display_name"`
	Name                      string          `json:"name"`
	Description               string          `json:"description"`
	Visibility                string          `json:"visibility"`
	ContextWindow             int             `json:"context_window"`
	ContextLength             int             `json:"context_length"`
	MaxContextLength          int             `json:"max_context_length"`
	MaxContextWindow          int             `json:"max_context_window"`
	MaxOutputTokens           int             `json:"max_output_tokens"`
	SupportedReasoningLevels  json.RawMessage `json:"supported_reasoning_levels"`
	ReasoningEfforts          []string        `json:"reasoning_efforts"`
	SupportedParameters       []string        `json:"supported_parameters"`
	InputModalities           []string        `json:"input_modalities"`
	Modalities                []string        `json:"modalities"`
	SupportsTools             bool            `json:"supports_tools"`
	SupportsToolCalls         bool            `json:"supports_tool_calls"`
	ToolCall                  bool            `json:"tool_call"`
	SupportsReasoning         bool            `json:"supports_reasoning"`
	Reasoning                 bool            `json:"reasoning"`
	SupportsImages            bool            `json:"supports_images"`
	SupportsPromptCaching     bool            `json:"supports_prompt_caching"`
	SupportsStopSequences     bool            `json:"supports_stop_sequences"`
	SupportsStructuredOutputs bool            `json:"supports_structured_outputs"`
	SupportsParallelToolCalls bool            `json:"supports_parallel_tool_calls"`
	Architecture              struct {
		InputModalities []string `json:"input_modalities"`
	} `json:"architecture"`
}

// Models explicitly discovers the models visible to this Codex subscription.
// Successful catalogs are cached for 30 minutes. Temporary discovery failures
// use a stale successful catalog, or the curated static fallback, and suppress
// another refresh for five minutes. Authentication and context errors remain
// visible to the caller.
func (p *Provider) Models(ctx context.Context) ([]llm.ModelInfo, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if p == nil || p.transport.source == nil || strings.TrimSpace(p.transport.modelsEndpoint) == "" {
		return cloneCodexModels(curatedCodexModels(p)), nil
	}
	ctx, cancel := p.contextWithTimeout(ctx)
	defer cancel()

	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		now := p.modelsClock()()
		p.modelsMu.Lock()
		if len(p.modelsCache) > 0 && now.Before(p.modelsExpiresAt) {
			models := cloneCodexModels(p.modelsCache)
			p.modelsMu.Unlock()
			return models, nil
		}
		if now.Before(p.modelsRetryAfter) {
			models := p.modelsFallbackLocked()
			p.modelsMu.Unlock()
			return cloneCodexModels(models), nil
		}
		if flight := p.modelsFlight; flight != nil {
			p.modelsMu.Unlock()
			select {
			case <-flight.done:
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				return cloneCodexModels(flight.models), flight.err
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		flight := &modelsFlight{done: make(chan struct{})}
		p.modelsFlight = flight
		p.modelsMu.Unlock()

		models, err := p.fetchCodexModels(ctx)
		p.finishModelsFlight(flight, models, err)
		return cloneCodexModels(flight.models), flight.err
	}
}

func (p *Provider) modelsClock() func() time.Time {
	if p.modelsNow != nil {
		return p.modelsNow
	}
	return time.Now
}

func (p *Provider) modelsFallbackLocked() []llm.ModelInfo {
	if len(p.modelsCache) > 0 {
		return p.modelsCache
	}
	return curatedCodexModels(p)
}

func (p *Provider) finishModelsFlight(flight *modelsFlight, models []llm.ModelInfo, err error) {
	now := p.modelsClock()()
	p.modelsMu.Lock()
	logFallback := false

	switch {
	case err == nil:
		p.modelsCache = cloneCodexModels(models)
		p.modelsExpiresAt = now.Add(modelsSuccessTTL)
		p.modelsRetryAfter = time.Time{}
		flight.models = p.modelsCache
	case isModelsVisibleError(err):
		flight.err = err
	default:
		p.modelsRetryAfter = now.Add(modelsFailureTTL)
		flight.models = p.modelsFallbackLocked()
		logFallback = p.logger != nil
	}

	if p.modelsFlight == flight {
		p.modelsFlight = nil
	}
	close(flight.done)
	p.modelsMu.Unlock()
	if logFallback {
		p.logger.Warn("OpenAI Codex model discovery failed; using fallback",
			slog.String("error", llm.SafeError(err)))
	}
}

func isModelsVisibleError(err error) bool {
	return errors.Is(err, context.Canceled) ||
		errors.Is(err, context.DeadlineExceeded) ||
		errors.Is(err, llm.ErrAuth) ||
		errors.Is(err, llm.ErrPermission) ||
		provideroauth.IsPersistenceError(err)
}

func (p *Provider) fetchCodexModels(ctx context.Context) ([]llm.ModelInfo, error) {
	resp, err := p.transport.getModels(ctx)
	if err != nil {
		return nil, p.adapter().MapError(err)
	}
	if resp == nil {
		return nil, &llm.ProviderError{Provider: providerName, Message: "nil HTTP response", Kind: llm.ErrServer}
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, p.adapter().MapHTTPResponseError(resp)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, modelsBodyLimit+1))
	if err != nil {
		return nil, providerutil.NormalizeRemoteError(providerName, err)
	}
	if len(body) > modelsBodyLimit {
		return nil, &llm.ProviderError{Provider: providerName, Message: "models response exceeds limit", Kind: llm.ErrServer}
	}
	models, err := parseCodexCatalog(body, p)
	if err != nil {
		return nil, providerutil.NormalizeRemoteError(providerName, err)
	}
	return models, nil
}

func parseCodexCatalog(body []byte, p *Provider) ([]llm.ModelInfo, error) {
	var document codexCatalogDocument
	if err := json.Unmarshal(body, &document); err != nil {
		return nil, fmt.Errorf("decode OpenAI Codex models response: %w", err)
	}
	rows := append(append([]json.RawMessage(nil), document.Models...), document.Data...)
	models := make([]llm.ModelInfo, 0, len(rows))
	seen := make(map[string]struct{}, len(rows))
	for _, raw := range rows {
		var row codexCatalogRow
		if err := json.Unmarshal(raw, &row); err != nil {
			continue
		}
		id := firstNonEmpty(row.ID, row.Slug)
		visibility := strings.TrimSpace(row.Visibility)
		if id == "" || id == "codex-auto-review" || (visibility != "" && visibility != "list") {
			continue
		}
		if _, duplicate := seen[id]; duplicate {
			continue
		}
		seen[id] = struct{}{}
		model := llm.ModelInfo{
			ID:               id,
			DisplayName:      firstNonEmpty(row.DisplayName, row.Name, row.Description),
			ContextWindow:    firstPositive(row.ContextWindow, row.ContextLength, row.MaxContextLength, row.MaxContextWindow),
			MaxOutputTokens:  positive(row.MaxOutputTokens),
			SupportedEfforts: codexReasoningEfforts(row),
			Capabilities:     codexModelCapabilities(row),
			Raw:              append(json.RawMessage(nil), raw...),
		}
		models = append(models, enrichCodexModel(model, p))
	}
	if len(models) == 0 {
		return nil, fmt.Errorf("OpenAI Codex models response contains no usable visible models")
	}
	return models, nil
}

func codexReasoningEfforts(row codexCatalogRow) []llm.Effort {
	values := codexReasoningLevelValues(row)
	seen := map[llm.Effort]struct{}{}
	efforts := make([]llm.Effort, 0, len(values))
	for _, value := range values {
		effort := llm.Effort(value)
		switch effort {
		case llm.EffortNone, llm.EffortMinimal, llm.EffortLow, llm.EffortMedium, llm.EffortHigh, llm.EffortXHigh, llm.EffortMax:
		default:
			continue
		}
		if _, duplicate := seen[effort]; duplicate {
			continue
		}
		seen[effort] = struct{}{}
		efforts = append(efforts, effort)
	}
	return efforts
}

func codexReasoningLevelValues(row codexCatalogRow) []string {
	values := make([]string, 0, len(row.ReasoningEfforts))
	add := func(value string) {
		if value = strings.TrimSpace(value); value != "" {
			values = append(values, value)
		}
	}
	for _, value := range row.ReasoningEfforts {
		add(value)
	}
	if len(row.SupportedReasoningLevels) > 0 && string(row.SupportedReasoningLevels) != "null" {
		var entries []json.RawMessage
		if json.Unmarshal(row.SupportedReasoningLevels, &entries) == nil {
			for _, entry := range entries {
				var value string
				if json.Unmarshal(entry, &value) != nil {
					var object struct {
						Effort string `json:"effort"`
						Level  string `json:"level"`
						Value  string `json:"value"`
					}
					if json.Unmarshal(entry, &object) == nil {
						value = firstNonEmpty(object.Effort, object.Level, object.Value)
					}
				}
				add(value)
			}
		}
	}
	return values
}

func codexModelCapabilities(row codexCatalogRow) []llm.Capability {
	parameters := stringSet(row.SupportedParameters)
	modalities := stringSet(append(append(append([]string(nil), row.InputModalities...), row.Modalities...), row.Architecture.InputModalities...))
	tools := row.SupportsTools || row.SupportsToolCalls || row.ToolCall || row.SupportsParallelToolCalls || parameters["tools"] || parameters["tool_calls"]
	toolChoice := parameters["tool_choice"]
	structured := row.SupportsStructuredOutputs || parameters["structured_outputs"]
	reasoning := row.SupportsReasoning || row.Reasoning || hasCodexReasoningLevels(row) || parameters["reasoning"] || parameters["reasoning_effort"] || parameters["include_reasoning"]
	images := row.SupportsImages || modalities["image"]
	stop := row.SupportsStopSequences || parameters["stop"] || parameters["stop_sequences"]
	cache := row.SupportsPromptCaching || parameters["prompt_caching"] || parameters["prompt_cache_key"]

	var result []llm.Capability
	if tools {
		result = append(result, llm.CapabilityTools)
	}
	if toolChoice {
		result = append(result, llm.CapabilityToolChoiceRequired)
	}
	if row.SupportsParallelToolCalls {
		result = append(result, llm.CapabilityParallelTools)
	}
	if structured {
		result = append(result, llm.CapabilityJSONSchema)
	}
	if reasoning {
		result = append(result, llm.CapabilityReasoning)
	}
	if images {
		result = append(result, llm.CapabilityImageInput)
	}
	if stop {
		result = append(result, llm.CapabilityStopSequences)
	}
	if cache {
		result = append(result, llm.CapabilityPromptCaching)
	}
	return result
}

func hasCodexReasoningLevels(row codexCatalogRow) bool {
	return len(codexReasoningLevelValues(row)) > 0
}

func curatedCodexModels(p *Provider) []llm.ModelInfo {
	models := cloneCodexModels(staticModels)
	for i := range models {
		models[i] = enrichCodexModel(models[i], p)
	}
	return models
}

func enrichCodexModel(model llm.ModelInfo, p *Provider) llm.ModelInfo {
	var fallback llm.ModelInfo
	for _, candidate := range staticModels {
		if candidate.ID == model.ID {
			fallback = candidate
			break
		}
	}
	if embedded, ok := llm.LookupModelInfo(providerName, model.ID); ok {
		if fallback.DisplayName == "" {
			fallback.DisplayName = embedded.DisplayName
		}
		if fallback.ContextWindow == 0 {
			fallback.ContextWindow = embedded.ContextWindow
		}
		if fallback.MaxOutputTokens == 0 {
			fallback.MaxOutputTokens = embedded.MaxOutputTokens
		}
		if len(fallback.SupportedEfforts) == 0 {
			fallback.SupportedEfforts = embedded.SupportedEfforts
		}
		fallback.Pricing = embedded.Pricing
	}
	if model.DisplayName == "" {
		model.DisplayName = fallback.DisplayName
	}
	if model.ContextWindow == 0 {
		model.ContextWindow = fallback.ContextWindow
	}
	if model.MaxOutputTokens == 0 {
		model.MaxOutputTokens = fallback.MaxOutputTokens
	}
	if len(model.SupportedEfforts) == 0 {
		model.SupportedEfforts = append([]llm.Effort(nil), fallback.SupportedEfforts...)
	}
	if p != nil && p.priceTable != nil {
		model.Pricing = priceForModel(p.priceTable, model.ID)
	} else {
		model.Pricing = cloneModelPricing(fallback.Pricing)
	}
	return model
}

func cloneCodexModels(models []llm.ModelInfo) []llm.ModelInfo {
	if models == nil {
		return nil
	}
	cloned := make([]llm.ModelInfo, len(models))
	for i, model := range models {
		cloned[i] = model
		cloned[i].Pricing = cloneModelPricing(model.Pricing)
		cloned[i].SupportedEfforts = append([]llm.Effort(nil), model.SupportedEfforts...)
		cloned[i].Capabilities = append([]llm.Capability(nil), model.Capabilities...)
		switch raw := model.Raw.(type) {
		case json.RawMessage:
			cloned[i].Raw = append(json.RawMessage(nil), raw...)
		case []byte:
			cloned[i].Raw = append([]byte(nil), raw...)
		}
	}
	return cloned
}

func cloneModelPricing(pricing *llm.ModelPricing) *llm.ModelPricing {
	if pricing == nil {
		return nil
	}
	cloned := *pricing
	cloned.Tiers = append([]llm.ModelPricingTier(nil), pricing.Tiers...)
	return &cloned
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			return value
		}
	}
	return ""
}

func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func positive(value int) int {
	if value > 0 {
		return value
	}
	return 0
}

func stringSet(values []string) map[string]bool {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			set[value] = true
		}
	}
	return set
}
