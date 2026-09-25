package llm

// SupportedEffortsForModel returns the reasoning Effort levels the embedded
// model table (sourced from models.dev and OpenRouter) lists for a model,
// ordered weakest → strongest, including canonical-ID fallback for aggregator
// aliases. Models without upstream effort metadata return nil, and an
// explicit empty upstream ladder stays a non-nil empty slice: go-llm does not
// infer ladders from model names. The metadata is ADVISORY — request
// forwarding and server-side validation are unchanged (some gateways ignore
// an unsupported effort, others reject it; both remain the server's call),
// and the returned slice is always a clone.
func SupportedEffortsForModel(provider, modelID string) []Effort {
	table, err := loadDefaultModelTable()
	if err != nil {
		return nil
	}
	return table.supportedEfforts(provider, modelID)
}

func (t parsedModelTable) supportedEfforts(provider, modelID string) []Effort {
	info, ok := t.lookup(provider, modelID)
	if !ok {
		return nil
	}
	return info.SupportedEfforts
}
