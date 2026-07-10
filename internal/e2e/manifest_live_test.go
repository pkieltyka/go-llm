//go:build live

package e2e

import (
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/anthropic"
	"github.com/pkieltyka/go-llm/providers/openai"
	"github.com/pkieltyka/go-llm/providers/openaicodex"
	"github.com/pkieltyka/go-llm/providers/openrouter"
	"github.com/pkieltyka/go-llm/providers/vllm"
)

// TestLiveRunnerManifests is credential-free and performs no provider calls.
// It validates the exact runner maps used by the live suites against each
// provider's actual advertised capabilities.
func TestLiveRunnerManifests(t *testing.T) {
	anthropicProvider, err := anthropic.New(anthropic.WithAPIKey("manifest-test"))
	if err != nil {
		t.Fatalf("construct anthropic: %v", err)
	}
	openAIProvider, err := openai.New(openai.WithAPIKey("manifest-test"))
	if err != nil {
		t.Fatalf("construct openai: %v", err)
	}
	codexProvider, err := openaicodex.New(openaicodex.WithOAuth(llm.AuthCredential{Type: "oauth", Access: "manifest-test"}, nil))
	if err != nil {
		t.Fatalf("construct openai-codex: %v", err)
	}
	openRouterProvider, err := openrouter.New(openrouter.WithAPIKey("manifest-test"))
	if err != nil {
		t.Fatalf("construct openrouter: %v", err)
	}
	vllmProvider, err := vllm.New("http://localhost:8000/v1")
	if err != nil {
		t.Fatalf("construct vllm: %v", err)
	}

	tests := []struct {
		providerID string
		provider   llm.Provider
		runners    map[string]ScenarioRun
	}{
		{"anthropic", anthropicProvider, anthropicLiveScenarioRunners("reasoning-model", "")},
		{"openai", openAIProvider, openAILiveScenarioRunners("")},
		{"openai-codex", codexProvider, openAICodexLiveScenarioRunners()},
		{"openrouter", openRouterProvider, openRouterLiveScenarioRunners("reasoning-model", "cache-model", "parallel-model", "tools-model", "")},
		{"vllm", vllmProvider, vllmLiveScenarioRunners(vllmProvider, "http://localhost:8000/v1")},
	}
	knownScenarios := make(map[string]bool, len(liveScenarioOrder))
	for _, name := range liveScenarioOrder {
		knownScenarios[name] = true
	}
	for _, test := range tests {
		t.Run(test.providerID, func(t *testing.T) {
			for name, runner := range test.runners {
				if !knownScenarios[name] {
					t.Fatalf("real manifest contains unknown scenario %q", name)
				}
				if runner == nil {
					t.Fatalf("real manifest runner %q is nil", name)
				}
			}
			scenarios, exemptions, err := liveScenarios(test.providerID, test.provider, test.runners)
			if err != nil {
				t.Fatalf("validate real manifest: %v", err)
			}
			if len(scenarios) == 0 {
				t.Fatal("real manifest selected no scenarios")
			}
			for _, exemption := range exemptions {
				if exemption.Provider != test.providerID || exemption.Capability == "" || strings.TrimSpace(exemption.Reason) == "" {
					t.Fatalf("invalid exemption: %+v", exemption)
				}
			}
		})
	}
}
