package llm

// Capability is a provider feature flag.
type Capability string

// Standard capabilities a provider may declare (FS §12). Request validation
// rejects features the provider does not declare before any network call;
// providers may additionally declare custom "provider/feature" capabilities
// (see CustomCapabilities).
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
	// CapabilityPositionalInstructions and CapabilityPositionalTools mean
	// the provider sends RoleSystem messages and ToolsDeltaPart changes in
	// place for models whose ModelBehavior allows it. Both are advisory:
	// every provider accepts positional messages and folds what a model
	// cannot take in place (FS §7A).
	CapabilityPositionalInstructions Capability = "positional-instructions"
	CapabilityPositionalTools        Capability = "positional-tools"
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

	CapabilityPositionalInstructions: {},
	CapabilityPositionalTools:        {},
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
