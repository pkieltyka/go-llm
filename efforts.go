package llm

// SupportedEffortsForModel returns the reasoning Effort levels the embedded
// model table (sourced from models.dev and OpenRouter) lists for a model,
// ordered weakest → strongest, including canonical-ID fallback for aggregator
// aliases. Models without upstream effort metadata return nil: go-llm does
// not infer ladders from model names. The metadata is ADVISORY — request
// forwarding and server-side validation are unchanged (some gateways ignore
// an unsupported effort, others reject it; both remain the server's call),
// and the returned slice is always a clone.
func SupportedEffortsForModel(provider, modelID string) []Effort {
	if info, ok := LookupModelInfo(provider, modelID); ok && len(info.SupportedEfforts) > 0 {
		return info.SupportedEfforts
	}
	return nil
}
