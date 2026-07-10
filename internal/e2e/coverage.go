package e2e

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

type ScenarioRun func(context.Context, *testing.T, llm.Provider, string)

type capabilityCoverage struct {
	Capability llm.Capability
	Scenarios  []string
}

// The registry is deliberately exhaustive. A newly advertised capability
// must gain a live scenario or an explicit provider exemption before the
// offline completeness test will pass.
var liveCapabilityCoverage = []capabilityCoverage{
	{llm.CapabilityStreaming, []string{"stream"}},
	{llm.CapabilityModelsListing, []string{"models"}},
	{llm.CapabilityTools, []string{"tools", "cross_provider_handoff"}},
	{llm.CapabilityToolChoiceRequired, []string{"tools"}},
	{llm.CapabilityToolStreaming, []string{"tools_stream"}},
	{llm.CapabilityParallelTools, []string{"parallel_tools"}},
	{llm.CapabilityStrictTools, []string{"tools"}},
	{llm.CapabilityJSONSchema, []string{"parse"}},
	{llm.CapabilityJSONMode, []string{"json_mode"}},
	{llm.CapabilityReasoning, []string{"reasoning", "reasoning_replay"}},
	{llm.CapabilityImageInput, []string{"multimodal"}},
	{llm.CapabilityPDFInput, nil},
	{llm.CapabilityStopSequences, []string{"stop_sequences"}},
	{llm.CapabilityPromptCaching, []string{"prompt_cache"}},
	{llm.CapabilitySessionAffinity, []string{"session_affinity"}},
	{llm.CapabilityCostReporting, []string{"cost_reporting"}},
	{llm.Capability("openrouter/routing"), nil},
	{llm.Capability("openrouter/plugins"), nil},
}

type liveProviderProfile struct {
	Exemptions map[llm.Capability]string
	Extensions map[string]string
}

const pdfLiveExemption = "no stable, provider-neutral synthetic PDF fixture; PDF wire mapping is covered offline and must be enabled here when a durable live fixture is added"

var liveProviderProfiles = map[string]liveProviderProfile{
	"anthropic": {
		Exemptions: map[llm.Capability]string{llm.CapabilityPDFInput: pdfLiveExemption},
	},
	"openai": {
		Exemptions: map[llm.Capability]string{llm.CapabilityPDFInput: pdfLiveExemption},
	},
	"openai-codex": {
		Exemptions: map[llm.Capability]string{llm.CapabilityPDFInput: pdfLiveExemption},
	},
	"openrouter": {
		Exemptions: map[llm.Capability]string{
			llm.Capability("openrouter/routing"): "routing selection is account and upstream availability dependent; request mapping is covered offline",
			llm.Capability("openrouter/plugins"): "plugins invoke external services with nondeterministic availability; request mapping is covered offline",
		},
	},
	"vllm": {
		Extensions: map[string]string{
			"structured_choice":  "vLLM native structured_outputs choice extension",
			"structured_regex":   "vLLM native structured_outputs regex extension",
			"tokenize":           "vLLM tokenize/detokenize extension endpoints",
			"anthropic_messages": "vLLM Anthropic-compatible Messages endpoint preset",
		},
	},
}

var liveScenarioOrder = []string{
	"chat",
	"stream",
	"models",
	"tools",
	"tools_stream",
	"parallel_tools",
	"parse",
	"json_mode",
	"reasoning",
	"reasoning_replay",
	"multimodal",
	"stop_sequences",
	"prompt_cache",
	"session_affinity",
	"usage",
	"cost_reporting",
	"error_mapping",
	"cross_provider_handoff",
	"structured_choice",
	"structured_regex",
	"tokenize",
	"anthropic_messages",
}

type LiveCapabilityExemption struct {
	Provider   string
	Capability llm.Capability
	Reason     string
}

func capabilityScenarioNames(providerID string, capabilities []llm.Capability) ([]string, []LiveCapabilityExemption, error) {
	profile, ok := liveProviderProfiles[providerID]
	if !ok {
		return nil, nil, fmt.Errorf("unknown live provider profile %q", providerID)
	}
	rules := make(map[llm.Capability][]string, len(liveCapabilityCoverage))
	for _, coverage := range liveCapabilityCoverage {
		if coverage.Capability == "" {
			return nil, nil, fmt.Errorf("live capability registry contains an empty capability")
		}
		if _, exists := rules[coverage.Capability]; exists {
			return nil, nil, fmt.Errorf("live capability registry duplicates %q", coverage.Capability)
		}
		rules[coverage.Capability] = coverage.Scenarios
	}

	selected := map[string]bool{"chat": true, "usage": true, "error_mapping": true}
	var exemptions []LiveCapabilityExemption
	seenCapabilities := map[llm.Capability]bool{}
	for _, capability := range capabilities {
		if seenCapabilities[capability] {
			continue
		}
		seenCapabilities[capability] = true
		scenarios, exists := rules[capability]
		if !exists {
			return nil, nil, fmt.Errorf("provider %s advertises unregistered capability %q", providerID, capability)
		}
		if reason := strings.TrimSpace(profile.Exemptions[capability]); reason != "" {
			exemptions = append(exemptions, LiveCapabilityExemption{Provider: providerID, Capability: capability, Reason: reason})
			continue
		}
		if len(scenarios) == 0 {
			return nil, nil, fmt.Errorf("provider %s capability %q has neither a scenario nor an exemption", providerID, capability)
		}
		for _, scenario := range scenarios {
			selected[scenario] = true
		}
	}
	for scenario, reason := range profile.Extensions {
		if strings.TrimSpace(reason) == "" {
			return nil, nil, fmt.Errorf("provider %s extension scenario %q lacks a justification", providerID, scenario)
		}
		selected[scenario] = true
	}

	names := make([]string, 0, len(selected))
	for _, name := range liveScenarioOrder {
		if selected[name] {
			names = append(names, name)
			delete(selected, name)
		}
	}
	if len(selected) > 0 {
		unknown := make([]string, 0, len(selected))
		for name := range selected {
			unknown = append(unknown, name)
		}
		sort.Strings(unknown)
		return nil, nil, fmt.Errorf("live scenarios missing from stable order: %s", strings.Join(unknown, ", "))
	}
	return names, exemptions, nil
}

func liveScenarios(providerID string, provider llm.Provider, runners map[string]ScenarioRun) ([]Scenario, []LiveCapabilityExemption, error) {
	if provider == nil {
		return nil, nil, fmt.Errorf("live provider %s is nil", providerID)
	}
	if provider.Name() != providerID {
		return nil, nil, fmt.Errorf("live provider ID = %q, want %q", provider.Name(), providerID)
	}
	names, exemptions, err := capabilityScenarioNames(providerID, provider.Capabilities())
	if err != nil {
		return nil, nil, err
	}
	scenarios := make([]Scenario, 0, len(names))
	for _, name := range names {
		run := runners[name]
		if run == nil {
			return nil, nil, fmt.Errorf("provider %s has no runner for required live scenario %q", providerID, name)
		}
		scenarios = append(scenarios, Scenario{Name: name, Run: run})
	}
	return scenarios, exemptions, nil
}
