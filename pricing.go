package llm

import "strings"

// PriceTable stores per-million-token prices keyed by "provider/model-id".
type PriceTable map[string]ModelPricing

// Lookup returns pricing by exact key, then by longest key prefix.
func (t PriceTable) Lookup(provider, modelID string) (ModelPricing, bool) {
	if len(t) == 0 || provider == "" || modelID == "" {
		return ModelPricing{}, false
	}
	key := modelKey(provider, modelID)
	if pricing, ok := t[key]; ok {
		return pricing, true
	}

	var (
		best    ModelPricing
		bestLen int
		found   bool
	)
	prefix := provider + "/"
	for candidate, pricing := range t {
		if !strings.HasPrefix(candidate, prefix) {
			continue
		}
		candidateID := strings.TrimPrefix(candidate, prefix)
		if !modelIDHasBoundaryPrefix(modelID, candidateID) {
			continue
		}
		if len(candidate) <= bestLen {
			continue
		}
		best = pricing
		bestLen = len(candidate)
		found = true
	}
	return best, found
}

func modelIDHasBoundaryPrefix(modelID, prefix string) bool {
	if prefix == "" || !strings.HasPrefix(modelID, prefix) {
		return false
	}
	if len(modelID) == len(prefix) {
		return true
	}
	switch modelID[len(prefix)] {
	case '-', '.', '_', ':', '/':
		return true
	default:
		return false
	}
}

// EstimateCost returns a copy of usage with CostUSD populated from pricing
// and CostSource set to CostSourceEstimated. Provider-reported costs are
// left untouched.
func EstimateCost(usage Usage, pricing ModelPricing) Usage {
	if usage.CostUSD != nil {
		return usage
	}
	cost := (float64(usage.InputTokens)*pricing.InputPerMTok +
		float64(usage.OutputTokens)*pricing.OutputPerMTok +
		float64(usage.CacheReadTokens)*pricing.CacheReadPerMTok +
		float64(usage.CacheWriteTokens)*pricing.CacheWritePerMTok) / 1_000_000
	usage.CostUSD = &cost
	usage.CostSource = CostSourceEstimated
	return usage
}

// EstimateCostForModel estimates response cost from the embedded model table.
func EstimateCostForModel(provider, modelID string, usage Usage) Usage {
	if usage.CostUSD != nil {
		return usage
	}
	info, ok := LookupModelInfo(provider, modelID)
	if !ok || info.Pricing == nil {
		return usage
	}
	return EstimateCost(usage, *info.Pricing)
}

// EstimateCostWithTable estimates response cost from a caller-provided table.
func EstimateCostWithTable(table PriceTable, provider, modelID string, usage Usage) Usage {
	if usage.CostUSD != nil {
		return usage
	}
	pricing, ok := table.Lookup(provider, modelID)
	if !ok {
		return usage
	}
	return EstimateCost(usage, pricing)
}
