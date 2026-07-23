package llm

import "strings"

// SupportedEffortsForModel returns the reasoning Effort levels a model
// supports, ordered weakest → strongest: curated embedded-table metadata
// first (including canonical-ID fallback for aggregator aliases), then
// name-family inference for models absent from the table. Unrecognized
// models return nil. The metadata is ADVISORY — request forwarding and
// server-side validation are unchanged (some gateways ignore an unsupported
// effort, others reject it; both remain the server's call), and the
// returned slice is always a clone.
func SupportedEffortsForModel(provider, modelID string) []Effort {
	if info, ok := LookupModelInfo(provider, modelID); ok && len(info.SupportedEfforts) > 0 {
		return info.SupportedEfforts
	}
	family := modelFamilyName(modelID)
	if efforts := inferredEffortsForFamily(family); len(efforts) > 0 {
		return append([]Effort(nil), efforts...)
	}
	return nil
}

// modelFamilyName normalizes a model id for effort-family inference ONLY:
// a "provider:" prefix and any leading vendor "/" segments are stripped
// ("openai/o3-mini" → "o3-mini"), and the result is lowercased. This
// normalization is deliberately NOT shared with pricing lookups — the price
// table is keyed "provider/model-id" with longest-prefix matching and
// preserves aggregator namespaces.
func modelFamilyName(modelID string) string {
	name := modelID
	if i := strings.IndexByte(name, ':'); i >= 0 {
		name = name[i+1:]
	}
	if i := strings.LastIndexByte(name, '/'); i >= 0 {
		name = name[i+1:]
	}
	return strings.ToLower(strings.TrimSpace(name))
}

// inferredEffortsForFamily maps model-name families to effort levels for
// models absent from the embedded table. Curated table metadata always wins
// over inference; keep this list to well-known reasoning families.
func inferredEffortsForFamily(family string) []Effort {
	switch {
	case strings.Contains(family, "chat"):
		// "-chat-latest" variants are non-reasoning despite the gpt-5 name.
		return nil
	case strings.Contains(family, "codex"):
		// Codex reasoning models take low/medium/high; checked before the
		// broader gpt-5 family so "gpt-5.x-codex" ids land here.
		return []Effort{EffortLow, EffortMedium, EffortHigh}
	case strings.HasPrefix(family, "gpt-5"):
		return []Effort{EffortMinimal, EffortLow, EffortMedium, EffortHigh}
	case family == "o3" || strings.HasPrefix(family, "o3-"),
		family == "o4" || strings.HasPrefix(family, "o4-"):
		return []Effort{EffortLow, EffortMedium, EffortHigh}
	default:
		return nil
	}
}
