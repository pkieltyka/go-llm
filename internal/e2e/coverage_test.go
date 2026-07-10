package e2e

import (
	"context"
	"iter"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestLiveCapabilityExemptionsAreExplicitAndNarrow(t *testing.T) {
	for _, providerID := range []string{"anthropic", "openai", "openai-codex"} {
		_, exemptions, err := capabilityScenarioNames(providerID, []llm.Capability{llm.CapabilityPDFInput})
		if err != nil {
			t.Fatalf("%s PDF coverage: %v", providerID, err)
		}
		if len(exemptions) != 1 || exemptions[0].Capability != llm.CapabilityPDFInput || strings.TrimSpace(exemptions[0].Reason) == "" {
			t.Fatalf("%s exemptions = %+v", providerID, exemptions)
		}
	}
	if _, _, err := capabilityScenarioNames("vllm", []llm.Capability{llm.CapabilityPDFInput}); err == nil {
		t.Fatal("unexempted PDF capability should fail")
	}
	if _, _, err := capabilityScenarioNames("anthropic", []llm.Capability{"future-capability"}); err == nil {
		t.Fatal("unregistered capability should fail")
	}
}

func TestLiveProviderIDAndRequiredRunnerValidation(t *testing.T) {
	provider := &coverageProvider{name: "wrong", capabilities: []llm.Capability{llm.CapabilityStreaming}}
	runners := map[string]ScenarioRun{
		"chat":          func(context.Context, *testing.T, llm.Provider, string) {},
		"stream":        func(context.Context, *testing.T, llm.Provider, string) {},
		"usage":         func(context.Context, *testing.T, llm.Provider, string) {},
		"error_mapping": func(context.Context, *testing.T, llm.Provider, string) {},
	}
	if _, _, err := liveScenarios("openai", provider, runners); err == nil || !strings.Contains(err.Error(), "provider ID") {
		t.Fatalf("provider ID error = %v", err)
	}

	provider.name = "openai"
	if _, _, err := liveScenarios("openai", provider, map[string]ScenarioRun{}); err == nil || !strings.Contains(err.Error(), "no runner") {
		t.Fatalf("missing runner error = %v", err)
	}
}

func TestVLLMExtensionScenariosAreJustified(t *testing.T) {
	names, _, err := capabilityScenarioNames("vllm", nil)
	if err != nil {
		t.Fatalf("capabilityScenarioNames returned error: %v", err)
	}
	want := map[string]bool{
		"structured_choice":  false,
		"structured_regex":   false,
		"tokenize":           false,
		"anthropic_messages": false,
	}
	for _, name := range names {
		if _, ok := want[name]; ok {
			want[name] = true
		}
	}
	for name, found := range want {
		if !found {
			t.Fatalf("vllm extension scenario %q missing from %v", name, names)
		}
	}
}

type coverageProvider struct {
	name         string
	capabilities []llm.Capability
}

func (p *coverageProvider) Name() string { return p.name }

func (p *coverageProvider) Capabilities() []llm.Capability {
	return append([]llm.Capability(nil), p.capabilities...)
}

func (p *coverageProvider) Models(context.Context) ([]llm.ModelInfo, error) { return nil, nil }

func (p *coverageProvider) Chat(context.Context, *llm.Request) (*llm.Response, error) {
	return nil, nil
}

func (p *coverageProvider) ChatStream(context.Context, *llm.Request) iter.Seq2[llm.Event, error] {
	return nil
}
