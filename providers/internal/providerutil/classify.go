package providerutil

import (
	"slices"

	llm "github.com/pkieltyka/go-llm"
)

// EstimateClassifierCost fills in a classify call's cost estimate (FS
// §13C) unless the service reported one. A caller's price table is used as
// given. Otherwise only an embedded model-table row that lists
// CapabilityClassify prices the call: a chat row's price is never assumed
// to hold for classification, so a model without classifier evidence keeps
// an unknown cost.
func EstimateClassifierCost(table llm.PriceTable, provider, model string, usage llm.Usage) llm.Usage {
	if usage.CostUSD != nil {
		return usage
	}
	if table != nil {
		return llm.EstimateCostWithTable(table, provider, model, usage)
	}
	info, ok := llm.LookupModelInfo(provider, model)
	if !ok || info.Pricing == nil || !slices.Contains(info.Capabilities, llm.CapabilityClassify) {
		return usage
	}
	return llm.EstimateCost(usage, *info.Pricing)
}
