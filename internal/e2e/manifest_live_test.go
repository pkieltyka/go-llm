//go:build live

package e2e

import (
	"context"
	"errors"
	"iter"
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

func TestAnthropicLiveReasoningModelSelection(t *testing.T) {
	if got := anthropicLiveReasoningModel(ProviderConfig{Model: anthropicCheapModel}); got != anthropicReasoningModel {
		t.Fatalf("standard endpoint reasoning model = %q, want %q", got, anthropicReasoningModel)
	}
	if got := anthropicLiveReasoningModel(ProviderConfig{Model: "private-reasoning", BaseURL: "https://example.invalid"}); got != "private-reasoning" {
		t.Fatalf("custom endpoint reasoning model = %q, want private-reasoning", got)
	}
}

func TestVLLMModelRejectsImagesClassification(t *testing.T) {
	unsupported := errors.Join(llm.ErrBadRequest, errors.New("At most 0 image(s) may be provided (parameter=image)"))
	if !vllmModelRejectsImages(unsupported) {
		t.Fatal("explicit zero-image rejection was not classified")
	}
	if vllmModelRejectsImages(errors.Join(llm.ErrBadRequest, errors.New("invalid image"))) {
		t.Fatal("generic bad image was classified as zero image capacity")
	}
	if vllmModelRejectsImages(errors.New("At most 0 image(s) may be provided (parameter=image)")) {
		t.Fatal("non-bad-request error was classified as zero image capacity")
	}
}

func TestOpenRouterStopSequenceRunnerPinsCompatibleModel(t *testing.T) {
	provider := &stopSequenceProvider{}
	runners := openRouterLiveScenarioRunners("reasoning-model", "cache-model", "parallel-model", "tools-model", "")
	runners["stop_sequences"](context.Background(), t, provider, "cheap-model")

	if provider.request == nil {
		t.Fatal("stop-sequence runner did not make a request")
	}
	if provider.request.Model != "tools-model" {
		t.Fatalf("stop-sequence model = %q, want tools-model", provider.request.Model)
	}
	if len(provider.request.StopSequences) != 1 || provider.request.StopSequences[0] != "STOP" {
		t.Fatalf("stop sequences = %q, want [STOP]", provider.request.StopSequences)
	}
}

type stopSequenceProvider struct {
	request *llm.Request
}

func (p *stopSequenceProvider) Name() string { return "openrouter" }

func (p *stopSequenceProvider) Capabilities() []llm.Capability { return nil }

func (p *stopSequenceProvider) Models(context.Context) ([]llm.ModelInfo, error) { return nil, nil }

func (p *stopSequenceProvider) Chat(_ context.Context, req *llm.Request) (*llm.Response, error) {
	p.request = req
	return &llm.Response{
		Parts:      []llm.Part{llm.TextPart{Text: "alpha"}},
		StopReason: llm.StopReasonEndTurn,
	}, nil
}

func (p *stopSequenceProvider) ChatStream(context.Context, *llm.Request) iter.Seq2[llm.Event, error] {
	return nil
}
