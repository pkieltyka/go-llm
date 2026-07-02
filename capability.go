package llm

// Capability is a provider feature flag.
type Capability string

const (
	CapabilityStreaming          Capability = "streaming"
	CapabilityTools              Capability = "tools"
	CapabilityToolChoiceRequired Capability = "tool-choice-required"
	CapabilityToolStreaming      Capability = "tool-streaming"
	CapabilityParallelTools      Capability = "parallel-tools"
	CapabilityStrictTools        Capability = "strict-tools"
	CapabilityJSONSchema         Capability = "json-schema"
	CapabilityJSONMode           Capability = "json-mode"
	CapabilityReasoning          Capability = "reasoning"
	CapabilityImageInput         Capability = "image-input"
	CapabilityPDFInput           Capability = "pdf-input"
	CapabilityStopSequences      Capability = "stop-sequences"
	CapabilityPromptCaching      Capability = "prompt-caching"
	CapabilitySessionAffinity    Capability = "session-affinity"
	CapabilityCostReporting      Capability = "cost-reporting"
	CapabilityModelsListing      Capability = "models-listing"
)

var standardCapabilities = map[Capability]struct{}{
	CapabilityStreaming:          {},
	CapabilityTools:              {},
	CapabilityToolChoiceRequired: {},
	CapabilityToolStreaming:      {},
	CapabilityParallelTools:      {},
	CapabilityStrictTools:        {},
	CapabilityJSONSchema:         {},
	CapabilityJSONMode:           {},
	CapabilityReasoning:          {},
	CapabilityImageInput:         {},
	CapabilityPDFInput:           {},
	CapabilityStopSequences:      {},
	CapabilityPromptCaching:      {},
	CapabilitySessionAffinity:    {},
	CapabilityCostReporting:      {},
	CapabilityModelsListing:      {},
}

// CustomCapabilities returns provider-specific capabilities in declaration order.
func CustomCapabilities(p Provider) []Capability {
	if p == nil {
		return nil
	}
	var custom []Capability
	for _, cap := range p.Capabilities() {
		if _, ok := standardCapabilities[cap]; !ok {
			custom = append(custom, cap)
		}
	}
	return custom
}
