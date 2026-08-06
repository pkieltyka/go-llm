package llm

import (
	_ "embed"
	"encoding/json"
	"strings"
	"sync"
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
	Provider         string        `json:"provider"`
	ID               string        `json:"id"`
	CanonicalID      string        `json:"canonical_id,omitempty"`
	DisplayName      string        `json:"display_name,omitempty"`
	ContextWindow    int           `json:"context_window,omitempty"`
	MaxOutputTokens  int           `json:"max_output_tokens,omitempty"`
	Pricing          *ModelPricing `json:"pricing,omitempty"`
	SupportedEfforts []string      `json:"supported_efforts,omitempty"`
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
	if err := json.Unmarshal(raw, &doc); err != nil {
		return parsedModelTable{}, err
	}

	table := parsedModelTable{
		GeneratedAt: doc.GeneratedAt,
		ByKey:       make(map[string]ModelInfo, len(doc.Models)),
		Rows:        append([]modelTableRow(nil), doc.Models...),
	}
	for _, row := range doc.Models {
		if row.Provider == "" || row.ID == "" {
			continue
		}
		table.ByKey[modelKey(row.Provider, row.ID)] = row.modelInfo()
	}
	return table, nil
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
		pricing := *canonical.Pricing
		info.Pricing = &pricing
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
		ContextWindow:   row.ContextWindow,
		MaxOutputTokens: row.MaxOutputTokens,
	}
	if row.Pricing != nil {
		pricing := *row.Pricing
		info.Pricing = &pricing
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
		pricing := *info.Pricing
		info.Pricing = &pricing
	}
	if len(info.SupportedEfforts) > 0 {
		info.SupportedEfforts = append([]Effort(nil), info.SupportedEfforts...)
	}
	return info
}

func modelKey(provider, modelID string) string {
	return provider + "/" + modelID
}
