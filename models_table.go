package llm

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

//go:embed models.json
var embeddedModelsJSON []byte

var defaultModelTable = struct {
	once sync.Once
	err  error
	data parsedModelTable
}{}

type parsedModelTable struct {
	GeneratedAt string
	ByKey       map[string]ModelInfo
	Rows        []modelTableRow
}

type modelTableDocument struct {
	GeneratedAt string          `json:"generated_at"`
	Models      []modelTableRow `json:"models"`
}

type modelTableRow struct {
	Provider         string             `json:"provider"`
	ID               string             `json:"id"`
	CanonicalID      string             `json:"canonical_id,omitempty"`
	DisplayName      string             `json:"display_name,omitempty"`
	ContextWindow    *int               `json:"context_window,omitempty"`
	MaxOutputTokens  *int               `json:"max_output_tokens,omitempty"`
	Pricing          *modelTablePricing `json:"pricing,omitempty"`
	SupportedEfforts []string           `json:"supported_efforts,omitempty"`
}

type modelTablePricing struct {
	InputPerMTok      *float64         `json:"input_per_mtok,omitempty"`
	OutputPerMTok     *float64         `json:"output_per_mtok,omitempty"`
	CacheReadPerMTok  *float64         `json:"cache_read_per_mtok,omitempty"`
	CacheWritePerMTok *float64         `json:"cache_write_per_mtok,omitempty"`
	Tiers             []modelTableTier `json:"tiers,omitempty"`
}

type modelTableTier struct {
	InputTokensAbove  *int64   `json:"input_tokens_above"`
	InputPerMTok      *float64 `json:"input_per_mtok"`
	OutputPerMTok     *float64 `json:"output_per_mtok"`
	CacheReadPerMTok  *float64 `json:"cache_read_per_mtok"`
	CacheWritePerMTok *float64 `json:"cache_write_per_mtok"`
}

// PriceTableDate returns the generated_at stamp from the embedded model table.
// It returns an empty string when the embedded table cannot be parsed.
func PriceTableDate() string {
	table, err := loadDefaultModelTable()
	if err != nil {
		return ""
	}
	return table.GeneratedAt
}

// LookupModelInfo returns embedded model metadata without making a network call.
//
// Lookup first tries provider/model exact match, then an equivalent dot/dash
// spelling, then the longest provider-local prefix match for dated model
// variants, and finally canonical-ID fallback for missing metadata.
func LookupModelInfo(provider, modelID string) (ModelInfo, bool) {
	table, err := loadDefaultModelTable()
	if err != nil {
		return ModelInfo{}, false
	}
	info, ok := table.lookup(provider, modelID)
	return info, ok
}

func loadDefaultModelTable() (parsedModelTable, error) {
	defaultModelTable.once.Do(func() {
		defaultModelTable.data, defaultModelTable.err = parseModelTable(embeddedModelsJSON)
	})
	return defaultModelTable.data, defaultModelTable.err
}

func parseModelTable(raw []byte) (parsedModelTable, error) {
	var doc modelTableDocument
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&doc); err != nil {
		return parsedModelTable{}, err
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return parsedModelTable{}, fmt.Errorf("model table contains multiple JSON documents")
		}
		return parsedModelTable{}, fmt.Errorf("model table has trailing data: %w", err)
	}
	if _, err := time.Parse(time.RFC3339, doc.GeneratedAt); doc.GeneratedAt == "" || err != nil {
		return parsedModelTable{}, fmt.Errorf("model table generated_at must be valid RFC3339: %q", doc.GeneratedAt)
	}
	if len(doc.Models) == 0 {
		return parsedModelTable{}, fmt.Errorf("model table has no models")
	}

	table := parsedModelTable{
		GeneratedAt: doc.GeneratedAt,
		ByKey:       make(map[string]ModelInfo, len(doc.Models)),
		Rows:        append([]modelTableRow(nil), doc.Models...),
	}
	previousKey := ""
	for index, row := range doc.Models {
		if row.Provider == "" || row.ID == "" {
			return parsedModelTable{}, fmt.Errorf("model table row %d has an empty provider or id", index)
		}
		key := modelKey(row.Provider, row.ID)
		if index > 0 && key <= previousKey {
			if key == previousKey {
				return parsedModelTable{}, fmt.Errorf("model table has duplicate key %q", key)
			}
			return parsedModelTable{}, fmt.Errorf("model table row %q is out of canonical order after %q", key, previousKey)
		}
		if row.ContextWindow != nil && *row.ContextWindow <= 0 {
			return parsedModelTable{}, fmt.Errorf("model table %s context_window must be positive", key)
		}
		if row.MaxOutputTokens != nil && *row.MaxOutputTokens <= 0 {
			return parsedModelTable{}, fmt.Errorf("model table %s max_output_tokens must be positive", key)
		}
		if err := validateModelTablePricing(key, row.Pricing); err != nil {
			return parsedModelTable{}, err
		}
		if err := validateModelTableEfforts(key, row.SupportedEfforts); err != nil {
			return parsedModelTable{}, err
		}
		table.ByKey[key] = row.modelInfo()
		previousKey = key
	}
	return table, nil
}

func validateModelTablePricing(key string, pricing *modelTablePricing) error {
	if pricing == nil {
		return nil
	}
	for name, rate := range map[string]*float64{
		"input_per_mtok":       pricing.InputPerMTok,
		"output_per_mtok":      pricing.OutputPerMTok,
		"cache_read_per_mtok":  pricing.CacheReadPerMTok,
		"cache_write_per_mtok": pricing.CacheWritePerMTok,
	} {
		if rate != nil && !validPrice(*rate) {
			return fmt.Errorf("model table %s pricing.%s must be finite and non-negative", key, name)
		}
	}
	previousThreshold := int64(0)
	for index, tier := range pricing.Tiers {
		if tier.InputTokensAbove == nil || *tier.InputTokensAbove <= previousThreshold {
			return fmt.Errorf("model table %s pricing tier %d threshold must be positive and ascending", key, index)
		}
		for name, rate := range map[string]*float64{
			"input_per_mtok":       tier.InputPerMTok,
			"output_per_mtok":      tier.OutputPerMTok,
			"cache_read_per_mtok":  tier.CacheReadPerMTok,
			"cache_write_per_mtok": tier.CacheWritePerMTok,
		} {
			if rate == nil || !validPrice(*rate) {
				return fmt.Errorf("model table %s pricing tier %d %s must be present, finite, and non-negative", key, index, name)
			}
		}
		previousThreshold = *tier.InputTokensAbove
	}
	return nil
}

func validateModelTableEfforts(key string, efforts []string) error {
	if efforts == nil {
		return nil
	}
	if len(efforts) == 0 {
		return fmt.Errorf("model table %s supported_efforts must not be empty", key)
	}
	ranks := map[string]int{"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5, "max": 6}
	previous := -1
	for _, effort := range efforts {
		rank, ok := ranks[effort]
		if !ok || rank <= previous {
			return fmt.Errorf("model table %s supported_efforts must be known, unique, and ordered", key)
		}
		previous = rank
	}
	return nil
}

func (t parsedModelTable) lookup(provider, modelID string) (ModelInfo, bool) {
	if provider == "" || modelID == "" {
		return ModelInfo{}, false
	}

	if info, ok := t.ByKey[modelKey(provider, modelID)]; ok {
		return t.withCanonicalFallback(info), true
	}
	if info, ok := t.lookupSeparatorEquivalent(provider, modelID); ok {
		info.ID = modelID
		return t.withCanonicalFallback(info), true
	}

	var (
		best    ModelInfo
		bestLen int
		found   bool
	)
	for _, row := range t.Rows {
		if row.Provider != provider || !datedModelVariant(modelID, row.ID) {
			continue
		}
		if len(row.ID) <= bestLen {
			continue
		}
		best = row.modelInfo()
		bestLen = len(row.ID)
		found = true
	}
	if !found {
		return ModelInfo{}, false
	}
	best.ID = modelID
	return t.withCanonicalFallback(best), true
}

func (t parsedModelTable) lookupSeparatorEquivalent(provider, modelID string) (ModelInfo, bool) {
	normalizedModelID := normalizeModelSeparators(modelID)
	for _, row := range t.Rows {
		if row.Provider != provider {
			continue
		}
		info := row.modelInfo()
		if normalizeModelSeparators(info.ID) == normalizedModelID ||
			(info.CanonicalID != "" && normalizeModelSeparators(info.CanonicalID) == normalizedModelID) {
			return info, true
		}
	}
	return ModelInfo{}, false
}

func normalizeModelSeparators(value string) string {
	return strings.ReplaceAll(strings.ToLower(value), ".", "-")
}

func datedModelVariant(modelID, prefix string) bool {
	if !modelIDHasBoundaryPrefix(modelID, prefix) || len(modelID) <= len(prefix) {
		return false
	}
	suffix := modelID[len(prefix)+1:]
	return hasCompactDatePrefix(suffix) || hasDashedDatePrefix(suffix)
}

func hasCompactDatePrefix(value string) bool {
	if len(value) < 8 {
		return false
	}
	for i := range 8 {
		if value[i] < '0' || value[i] > '9' {
			return false
		}
	}
	return true
}

func hasDashedDatePrefix(value string) bool {
	if len(value) < len("2006-01-02") || value[4] != '-' || value[7] != '-' {
		return false
	}
	for _, i := range [...]int{0, 1, 2, 3, 5, 6, 8, 9} {
		if value[i] < '0' || value[i] > '9' {
			return false
		}
	}
	return true
}

func (t parsedModelTable) withCanonicalFallback(info ModelInfo) ModelInfo {
	info = cloneModelInfo(info)
	if info.CanonicalID == "" {
		return info
	}
	canonical, ok := t.lookupCanonical(info.CanonicalID)
	if !ok {
		return info
	}
	if info.DisplayName == "" {
		info.DisplayName = canonical.DisplayName
	}
	if info.ContextWindow == 0 {
		info.ContextWindow = canonical.ContextWindow
	}
	if info.MaxOutputTokens == 0 {
		info.MaxOutputTokens = canonical.MaxOutputTokens
	}
	if info.Pricing == nil && canonical.Pricing != nil {
		info.Pricing = cloneModelPricing(canonical.Pricing)
	}
	if len(info.SupportedEfforts) == 0 && len(canonical.SupportedEfforts) > 0 {
		info.SupportedEfforts = append([]Effort(nil), canonical.SupportedEfforts...)
	}
	return info
}

func (t parsedModelTable) lookupCanonical(canonicalID string) (ModelInfo, bool) {
	if strings.Contains(canonicalID, "/") {
		if info, ok := t.ByKey[canonicalID]; ok {
			return cloneModelInfo(info), true
		}
	}
	for _, row := range t.Rows {
		info := row.modelInfo()
		if info.CanonicalID == canonicalID || info.ID == canonicalID {
			return cloneModelInfo(info), true
		}
	}
	return ModelInfo{}, false
}

func (row modelTableRow) modelInfo() ModelInfo {
	canonicalID := row.CanonicalID
	if canonicalID == modelKey(row.Provider, row.ID) {
		canonicalID = ""
	}
	info := ModelInfo{
		ID:              row.ID,
		CanonicalID:     canonicalID,
		DisplayName:     row.DisplayName,
		ContextWindow:   valueOrZero(row.ContextWindow),
		MaxOutputTokens: valueOrZero(row.MaxOutputTokens),
	}
	if row.Pricing != nil {
		info.Pricing = row.Pricing.modelPricing()
	}
	if len(row.SupportedEfforts) > 0 {
		info.SupportedEfforts = make([]Effort, len(row.SupportedEfforts))
		for i, effort := range row.SupportedEfforts {
			info.SupportedEfforts[i] = Effort(effort)
		}
	}
	return info
}

func cloneModelInfo(info ModelInfo) ModelInfo {
	if info.Pricing != nil {
		info.Pricing = cloneModelPricing(info.Pricing)
	}
	if len(info.SupportedEfforts) > 0 {
		info.SupportedEfforts = append([]Effort(nil), info.SupportedEfforts...)
	}
	if len(info.Capabilities) > 0 {
		info.Capabilities = append([]Capability(nil), info.Capabilities...)
	}
	return info
}

func (pricing *modelTablePricing) modelPricing() *ModelPricing {
	if pricing == nil {
		return nil
	}
	out := &ModelPricing{
		InputPerMTok:      valueOrZero(pricing.InputPerMTok),
		OutputPerMTok:     valueOrZero(pricing.OutputPerMTok),
		CacheReadPerMTok:  valueOrZero(pricing.CacheReadPerMTok),
		CacheWritePerMTok: valueOrZero(pricing.CacheWritePerMTok),
	}
	if len(pricing.Tiers) > 0 {
		out.Tiers = make([]ModelPricingTier, len(pricing.Tiers))
		for i, tier := range pricing.Tiers {
			out.Tiers[i] = ModelPricingTier{
				InputTokensAbove:  valueOrZero(tier.InputTokensAbove),
				InputPerMTok:      valueOrZero(tier.InputPerMTok),
				OutputPerMTok:     valueOrZero(tier.OutputPerMTok),
				CacheReadPerMTok:  valueOrZero(tier.CacheReadPerMTok),
				CacheWritePerMTok: valueOrZero(tier.CacheWritePerMTok),
			}
		}
	}
	return out
}

func cloneModelPricing(pricing *ModelPricing) *ModelPricing {
	if pricing == nil {
		return nil
	}
	cloned := *pricing
	if len(pricing.Tiers) > 0 {
		cloned.Tiers = append([]ModelPricingTier(nil), pricing.Tiers...)
	}
	return &cloned
}

func valueOrZero[T int | int64 | float64](value *T) T {
	if value == nil {
		return 0
	}
	return *value
}

func modelKey(provider, modelID string) string {
	return provider + "/" + modelID
}
