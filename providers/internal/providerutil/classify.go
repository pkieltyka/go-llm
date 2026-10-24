package providerutil

import (
	"errors"
	"slices"
	"strings"

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

// ResponseError reports a classify response that could not be used
// (unreadable, malformed, or holding invalid answers) as a ProviderError of
// kind ErrServer. Its Message keeps err's details, which may quote the
// response, so callers see them while llm.SafeError keeps them out of
// operational logs. A leading ErrServer text in err is dropped, since Kind
// already says it. Kind joins err, so its cause stays reachable with
// errors.Is and errors.As.
func ResponseError(provider string, err error) error {
	message := strings.TrimPrefix(err.Error(), llm.ErrServer.Error()+": ")
	return &llm.ProviderError{Provider: provider, Message: message, Kind: errors.Join(llm.ErrServer, err)}
}
