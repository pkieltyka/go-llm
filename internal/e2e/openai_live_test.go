//go:build live

package e2e

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	openaiProvider "github.com/pkieltyka/go-llm/providers/openai"
)

const (
	openAICheapModel      = "gpt-4.1-mini"
	openAIReasoningModel  = "gpt-5.1-mini"
	openAIReasoningPrompt = "A store has three boxes. One is labeled apples, one oranges, and one mixed, " +
		"but every label is wrong. You may inspect one fruit from one box. Determine the correct labels, then answer with only: solved."
)

func TestLiveOpenAI(t *testing.T) {
	root, err := RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("LoadConfig returned error: %v", err)
	}
	providerCfg := cfg.Provider("openai", "OPENAI_API_KEY")
	if providerCfg.Auth.Key == "" {
		t.Skip("OpenAI API key missing from gollm-test.json and OPENAI_API_KEY")
	}
	model := providerCfg.Model
	if model == "" {
		model = openAICheapModel
	}

	var captures []llm.WireCapture
	opts := []openaiProvider.Option{
		openaiProvider.WithAPIKey(providerCfg.Auth.Key),
		openaiProvider.WithMaxRetries(0),
		openaiProvider.WithWireCapture(func(c llm.WireCapture) {
			captures = append(captures, c)
		}),
	}
	if providerCfg.BaseURL != "" {
		opts = append(opts, openaiProvider.WithBaseURL(providerCfg.BaseURL))
	}
	p, err := openaiProvider.New(opts...)
	if err != nil {
		t.Fatalf("openai.New returned error: %v", err)
	}
	if *record {
		t.Cleanup(func() {
			path := filepath.Join(root, "internal", "e2e", "fixtures", "openai", "live.json")
			if err := WriteFixture(path, captures, providerCfg.Auth.Key, os.Getenv("OPENAI_API_KEY")); err != nil {
				t.Fatalf("WriteFixture returned error: %v", err)
			}
		})
	}

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	RunScenarios(ctx, t, p, model, []Scenario{
		{Name: "chat", Run: liveChatScenario},
		{Name: "stream", Capability: llm.CapabilityStreaming, Run: liveStreamScenario},
		{Name: "models", Capability: llm.CapabilityModelsListing, Run: liveModelsScenario},
		{Name: "tools", Capability: llm.CapabilityTools, Run: liveToolsScenario},
		{Name: "tools_stream", Capability: llm.CapabilityToolStreaming, Run: liveToolsStreamScenario},
		{Name: "parallel_tools", Capability: llm.CapabilityParallelTools, Run: liveParallelToolsScenario},
		{Name: "parse", Capability: llm.CapabilityJSONSchema, Run: liveOpenAIParseScenario},
		{Name: "reasoning", Capability: llm.CapabilityReasoning, Run: liveOpenAIReasoningScenario},
		{Name: "reasoning_replay", Capability: llm.CapabilityReasoning, Run: liveOpenAIReasoningReplayScenario},
		{Name: "multimodal", Capability: llm.CapabilityImageInput, Run: liveMultimodalScenario},
		{Name: "usage", Run: liveUsageScenario},
		{Name: "error_mapping", Run: func(ctx context.Context, t *testing.T, p llm.Provider, model string) {
			liveOpenAIErrorMappingScenario(ctx, t, p, model, providerCfg.BaseURL)
		}},
		{Name: "cross_provider_handoff", Capability: llm.CapabilityTools, Run: liveCrossProviderHandoffScenario},
	})
}

func liveOpenAIParseScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	temperature := 0.0
	got, resp, err := llm.Parse[liveContact](ctx, p, &llm.Request{
		Model:       model,
		MaxTokens:   64,
		Temperature: &temperature,
		Messages:    []llm.Message{llm.UserText("Extract this contact as JSON: Ada Lovelace.")},
	})
	if err != nil {
		t.Fatalf("Parse returned error: %v", err)
	}
	if resp == nil || !strings.Contains(strings.ToLower(got.Name), "ada") {
		t.Fatalf("Parse result = %+v response=%+v", got, resp)
	}
}

func liveOpenAIReasoningScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	resp := liveOpenAIReasoningResponse(ctx, t, p, model)
	if resp.Reasoning() == "" && !hasOpenAIReasoningRaw(resp) {
		t.Fatalf("reasoning response missing reasoning parts: %+v", resp)
	}

	none, err := p.Chat(ctx, &llm.Request{
		Model:     liveOpenAIReasoningModel(model),
		MaxTokens: 32,
		Effort:    llm.EffortNone,
		Messages:  []llm.Message{llm.UserText("Answer exactly: no reasoning")},
	})
	if err != nil {
		t.Fatalf("EffortNone Chat returned error: %v", err)
	}
	if none.Reasoning() != "" || hasOpenAIReasoningRaw(none) {
		t.Fatalf("EffortNone returned reasoning: %+v", none.Parts)
	}
}

func liveOpenAIReasoningReplayScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	first := liveOpenAIReasoningResponse(ctx, t, p, model)
	if !hasOpenAIReasoningRaw(first) {
		t.Fatalf("reasoning replay needs OpenAI raw reasoning: %+v", first.Parts)
	}
	replayModel := first.Model
	if replayModel == "" {
		replayModel = liveOpenAIReasoningModel(model)
	}
	resp, err := p.Chat(ctx, &llm.Request{
		Model:     replayModel,
		MaxTokens: 64,
		Messages: []llm.Message{
			llm.UserText(openAIReasoningPrompt),
			{
				Role:     llm.RoleAssistant,
				Parts:    first.Parts,
				Provider: first.Provider,
				Model:    first.Model,
			},
			llm.UserText("Now answer exactly: replay ok"),
		},
	})
	if err != nil {
		t.Fatalf("reasoning replay Chat returned error: %v", err)
	}
	if strings.TrimSpace(resp.Text()) == "" {
		t.Fatalf("reasoning replay text empty: %+v", resp)
	}
}

func liveOpenAIReasoningResponse(ctx context.Context, t *testing.T, p llm.Provider, model string) *llm.Response {
	t.Helper()
	resp, err := llm.Collect(p.ChatStream(ctx, &llm.Request{
		Model:     liveOpenAIReasoningModel(model),
		MaxTokens: 768,
		Effort:    llm.EffortLow,
		Messages:  []llm.Message{llm.UserText(openAIReasoningPrompt)},
	}))
	if err != nil {
		t.Fatalf("reasoning stream returned error: %v", err)
	}
	return resp
}

func liveOpenAIReasoningModel(model string) string {
	if strings.HasPrefix(model, "gpt-5") || strings.HasPrefix(model, "o") {
		return model
	}
	return openAIReasoningModel
}

func hasOpenAIReasoningRaw(resp *llm.Response) bool {
	if resp == nil {
		return false
	}
	for _, part := range resp.Parts {
		// Parts are value types (adapters never emit pointer parts).
		if reasoning, ok := part.(llm.ReasoningPart); ok && reasoning.Provider == "openai" && len(reasoning.Raw) > 0 && json.Valid(reasoning.Raw) {
			return true
		}
	}
	return false
}

func liveOpenAIErrorMappingScenario(ctx context.Context, t *testing.T, p llm.Provider, model, baseURL string) {
	t.Helper()
	_, err := p.Chat(ctx, &llm.Request{
		Model:     "definitely-not-a-real-openai-model",
		MaxTokens: 8,
		Messages:  []llm.Message{llm.UserText("hello")},
	})
	if !errors.Is(err, llm.ErrNotFound) && !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("bad model error = %v, want ErrNotFound or ErrBadRequest", err)
	}

	opts := []openaiProvider.Option{
		openaiProvider.WithAPIKey("invalid-openai-key"),
		openaiProvider.WithMaxRetries(0),
	}
	if baseURL != "" {
		opts = append(opts, openaiProvider.WithBaseURL(baseURL))
	}
	badProvider, err := openaiProvider.New(opts...)
	if err != nil {
		t.Fatalf("openai.New with invalid key returned error: %v", err)
	}
	_, err = badProvider.Chat(ctx, &llm.Request{
		Model:     model,
		MaxTokens: 8,
		Messages:  []llm.Message{llm.UserText("hello")},
	})
	if !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("invalid key error = %v, want ErrAuth", err)
	}
}
