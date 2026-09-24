package llm

import (
	"math"
	"strings"
)

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
// and CostSource set to CostSourceEstimated. It leaves cost unknown when a
// nonzero usage component has no available rate. Provider-reported costs are
// left untouched.
func EstimateCost(usage Usage, pricing ModelPricing) Usage {
	if usage.CostUSD != nil {
		return usage
	}
	pricing = pricingForUsage(usage, pricing)
	if !pricingCoversUsage(usage, pricing) {
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

func pricingForUsage(usage Usage, pricing ModelPricing) ModelPricing {
	occupancy := usage.InputTokens + usage.CacheReadTokens + usage.CacheWriteTokens
	bestThreshold := int64(0)
	for _, tier := range pricing.Tiers {
		if occupancy <= tier.InputTokensAbove || tier.InputTokensAbove <= bestThreshold || !validPricingTier(tier) {
			continue
		}
		bestThreshold = tier.InputTokensAbove
		availability := tierAvailability(tier)
		pricing.InputPerMTok = availableRate(tier.InputPerMTok, availability.InputPerMTok)
		pricing.OutputPerMTok = availableRate(tier.OutputPerMTok, availability.OutputPerMTok)
		pricing.CacheReadPerMTok = availableRate(tier.CacheReadPerMTok, availability.CacheReadPerMTok)
		pricing.CacheWritePerMTok = availableRate(tier.CacheWritePerMTok, availability.CacheWritePerMTok)
		pricing.Availability = &availability
	}
	return pricing
}

// availableRate zeroes an unavailable rate so an unvalidated value never
// reaches the cost sum, even when multiplied by zero tokens.
func availableRate(rate float64, available bool) float64 {
	if !available {
		return 0
	}
	return rate
}

func tierAvailability(tier ModelPricingTier) ModelPricingAvailability {
	if tier.Availability == nil {
		return ModelPricingAvailability{
			InputPerMTok: true, OutputPerMTok: true,
			CacheReadPerMTok: true, CacheWritePerMTok: true,
		}
	}
	return *tier.Availability
}

func pricingCoversUsage(usage Usage, pricing ModelPricing) bool {
	if pricing.Availability == nil {
		return true
	}
	return (usage.InputTokens == 0 || pricing.HasInputPrice()) &&
		(usage.OutputTokens == 0 || pricing.HasOutputPrice()) &&
		(usage.CacheReadTokens == 0 || pricing.HasCacheReadPrice()) &&
		(usage.CacheWriteTokens == 0 || pricing.HasCacheWritePrice())
}

// validPricingTier requires a positive threshold and valid known rates.
// Unavailable rates are not validated because they are never used.
func validPricingTier(tier ModelPricingTier) bool {
	available := tierAvailability(tier)
	return tier.InputTokensAbove > 0 &&
		(!available.InputPerMTok || validPrice(tier.InputPerMTok)) &&
		(!available.OutputPerMTok || validPrice(tier.OutputPerMTok)) &&
		(!available.CacheReadPerMTok || validPrice(tier.CacheReadPerMTok)) &&
		(!available.CacheWritePerMTok || validPrice(tier.CacheWritePerMTok))
}

func validPrice(price float64) bool {
	return price >= 0 && !math.IsNaN(price) && !math.IsInf(price, 0)
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
