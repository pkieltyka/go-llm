package openai

import llm "github.com/pkieltyka/go-llm"

// apiModelCapabilities contains only exact aliases/snapshots whose official
// OpenAI API model pages advertise image input. Reviewed 2026-10-27; see
// specs/amendments/2026-10-27-cost-completeness-openai-image.md for sources.
// Never prefix-match: a fine-tune, future snapshot, or similarly named model
// requires its own positive evidence. ChatGPT rows use their live modalities.
func apiModelCapabilities(id string) []llm.Capability {
	switch id {
	case "gpt-4.1", "gpt-4.1-2025-04-14",
		"gpt-4.1-mini", "gpt-4.1-mini-2025-04-14",
		"gpt-4.1-nano", "gpt-4.1-nano-2025-04-14",
		"gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20",
		"gpt-4o-mini", "gpt-4o-mini-2024-07-18",
		"gpt-5", "gpt-5-2025-08-07",
		"gpt-5-mini", "gpt-5-mini-2025-08-07",
		"gpt-5-nano", "gpt-5-nano-2025-08-07",
		"gpt-5.1", "gpt-5.1-2025-11-13":
		return []llm.Capability{llm.CapabilityImageInput}
	default:
		return nil
	}
}
