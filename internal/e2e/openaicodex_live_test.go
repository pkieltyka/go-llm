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
	"github.com/pkieltyka/go-llm/providers/openaicodex"
)

const (
	openAICodexCheapModel      = "gpt-5.4-mini"
	openAICodexReasoningPrompt = "A store has three boxes. One is labeled apples, one oranges, and one mixed, " +
		"but every label is wrong. You may inspect one fruit from one box. Determine the correct labels, then answer with only: solved."
)

func TestLiveOpenAICodex(t *testing.T) {
	root, err := RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("LoadConfig returned error: %v", err)
	}
	providerCfg := cfg.Provider("openai-codex", "")
	if providerCfg.Auth.Type != "oauth" || providerCfg.Auth.Access == "" {
		t.Skip("OpenAI Codex OAuth credential missing from gollm-test.json")
	}
	model := providerCfg.Model
	if model == "" {
		model = openAICodexCheapModel
	}

	captures := &CaptureLog{}
	secrets := NewSecretSet(providerCfg.Auth.Access, providerCfg.Auth.Refresh, providerCfg.Auth.AccountID, os.Getenv("OPENAI_API_KEY"))
	var scenarioReport ScenarioReport
	if *record {
		path := filepath.Join(root, "internal", "e2e", "fixtures", "openai-codex", "live.json")
		ScheduleFixtureRecording(t, path, captures, secrets, &scenarioReport, *recordAllowIncomplete)
	}
	// Persist rotated refresh tokens back to gollm-test.json — dropping them
	// strands the stored credential after the provider refreshes.
	persist := AuthFilePersistence(filepath.Join(root, "gollm-test.json"), "openai-codex", t.Logf, secrets)
	opts := []openaicodex.Option{
		openaicodex.WithOAuth(providerCfg.Auth, persist),
		openaicodex.WithMaxRetries(0),
		openaicodex.WithWireCapture(captures.Capture),
	}
	if providerCfg.BaseURL != "" {
		opts = append(opts, openaicodex.WithBaseURL(providerCfg.BaseURL))
	}
	p, err := openaicodex.New(opts...)
	if err != nil {
		t.Fatalf("openaicodex.New returned error: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	ctx = RecordingContext(ctx, captures, secrets)
	if err := liveOpenAICodexAuthPreflight(ctx, p, model); err != nil {
		if errors.Is(err, llm.ErrAuth) {
			t.Skipf("OpenAI Codex OAuth credential is invalid or expired: %v", err)
		}
		t.Fatalf("OpenAI Codex auth preflight returned error: %v", err)
	}
	scenarioReport = RunScenarios(ctx, t, p, model, []Scenario{
		{Name: "chat", Run: liveChatScenario},
		{Name: "stream", Capability: llm.CapabilityStreaming, Run: liveStreamScenario},
		{Name: "models", Capability: llm.CapabilityModelsListing, Run: liveModelsScenario},
		{Name: "tools", Capability: llm.CapabilityTools, Run: liveOpenAICodexToolsScenario},
		{Name: "tools_stream", Capability: llm.CapabilityToolStreaming, Run: liveToolsStreamScenario},
		{Name: "parallel_tools", Capability: llm.CapabilityParallelTools, Run: liveParallelToolsScenario},
		{Name: "parse", Capability: llm.CapabilityJSONSchema, Run: liveOpenAIParseScenario},
		{Name: "reasoning", Capability: llm.CapabilityReasoning, Run: liveOpenAICodexReasoningScenario},
		{Name: "reasoning_replay", Capability: llm.CapabilityReasoning, Run: liveOpenAICodexReasoningReplayScenario},
		{Name: "multimodal", Capability: llm.CapabilityImageInput, Run: liveMultimodalScenario},
		{Name: "prompt_cache", Capability: llm.CapabilityPromptCaching, Run: liveOpenAICodexPromptCacheScenario},
		{Name: "usage", Run: liveUsageScenario},
		{Name: "error_mapping", Run: liveOpenAICodexErrorMappingScenario},
		{Name: "cross_provider_handoff", Capability: llm.CapabilityTools, Run: liveCrossProviderHandoffScenario},
	})
}

// liveOpenAICodexToolsScenario extends the shared tools round trip with a
// tool-result image (B5): the follow-up tool result carries the red-pixel
// PNG alongside text, proving the Responses function_call_output
// content-array mapping against the live backend.
func liveOpenAICodexToolsScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	req := &llm.Request{
		Model:     model,
		MaxTokens: 64,
		Messages:  []llm.Message{llm.UserText(`Use the screenshot tool now.`)},
		Tools: []llm.Tool{{
			Name:        "screenshot",
			Description: "Take a screenshot of the current screen.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`),
			Strict:      true,
		}},
		ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "screenshot"},
	}
	resp, err := p.Chat(ctx, req)
	if err != nil {
		t.Fatalf("tool Chat returned error: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID == "" || calls[0].Name != "screenshot" {
		t.Fatalf("tool calls = %+v", calls)
	}

	result := llm.ToolResultParts(calls[0].ID,
		llm.Text("Screenshot captured:"),
		llm.ImageData(RedPixelPNG(t), "image/png"),
	)
	result.Name = calls[0].Name
	followUp := append([]llm.Message(nil), req.Messages...)
	followUp = append(followUp,
		llm.Message{Role: llm.RoleAssistant, Parts: resp.Parts, Provider: resp.Provider, Model: resp.Model},
		llm.Message{Role: llm.RoleTool, Parts: []llm.Part{result}},
		llm.UserText("What color is the square in the screenshot? Answer with one word."),
	)
	second, err := p.Chat(ctx, &llm.Request{
		Model:     model,
		MaxTokens: 32,
		Messages:  followUp,
	})
	if err != nil {
		t.Fatalf("tool-result image follow-up returned error: %v", err)
	}
	if !strings.Contains(strings.ToLower(second.Text()), "red") {
		t.Fatalf("tool-result image follow-up = %q, want red", second.Text())
	}
}

// liveOpenAICodexErrorMappingScenario checks bogus-model error mapping using
// the already-authenticated provider — no extra auth traffic (no invalid-key
// sub-case: that would burn a refresh cycle against the OAuth endpoint).
func liveOpenAICodexErrorMappingScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	_, err := p.Chat(ctx, &llm.Request{
		Model:    "definitely-not-a-real-codex-model",
		Messages: []llm.Message{llm.UserText("hello")},
	})
	if !errors.Is(err, llm.ErrNotFound) && !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("bad model error = %v, want ErrNotFound or ErrBadRequest", err)
	}
}

// liveOpenAICodexPromptCacheScenario exercises SessionID → prompt_cache_key.
// Cache hits are observed with a soft assertion: the codex backend does not
// guarantee cache reads on the second call, so a zero count only logs.
func liveOpenAICodexPromptCacheScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	req := &llm.Request{
		Model:     model,
		System:    strings.Repeat("Codex prompt cache live fixture sentence with stable content. ", 400),
		SessionID: "gollm-live-prompt-cache",
		Messages:  []llm.Message{llm.UserText("Answer exactly: cached")},
	}
	first, err := p.Chat(ctx, req)
	if err != nil {
		t.Fatalf("prompt cache first Chat returned error: %v", err)
	}
	second, err := p.Chat(ctx, req)
	if err != nil {
		t.Fatalf("prompt cache second Chat returned error: %v", err)
	}
	if first.Usage.InputTokens == 0 || second.Usage.InputTokens+second.Usage.CacheReadTokens == 0 {
		t.Fatalf("prompt cache usage missing input tokens: first=%+v second=%+v", first.Usage, second.Usage)
	}
	if second.Usage.CacheReadTokens < 0 {
		t.Fatalf("prompt cache read tokens negative: %+v", second.Usage)
	}
	if second.Usage.CacheReadTokens == 0 {
		t.Logf("prompt cache soft-miss: second call read no cached tokens (first=%+v second=%+v)", first.Usage, second.Usage)
	}
}

func liveOpenAICodexAuthPreflight(ctx context.Context, p llm.Provider, model string) error {
	_, err := p.Chat(ctx, &llm.Request{
		Model:    model,
		Messages: []llm.Message{llm.UserText("Reply with exactly: ok")},
	})
	return err
}

func liveOpenAICodexReasoningScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	resp := liveOpenAICodexReasoningResponse(ctx, t, p, model)
	if resp.Reasoning() == "" && !hasOpenAICodexReasoningRaw(resp) {
		t.Fatalf("reasoning response missing reasoning parts: %+v", resp)
	}

	none, err := p.Chat(ctx, &llm.Request{
		Model:     liveOpenAICodexReasoningModel(model),
		MaxTokens: 32,
		Effort:    llm.EffortNone,
		Messages:  []llm.Message{llm.UserText("Answer exactly: no reasoning")},
	})
	if err != nil {
		t.Fatalf("EffortNone Chat returned error: %v", err)
	}
	if none.Reasoning() != "" || hasOpenAICodexReasoningRaw(none) {
		t.Fatalf("EffortNone returned reasoning: %+v", none.Parts)
	}
}

func liveOpenAICodexReasoningReplayScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	first := liveOpenAICodexReasoningResponse(ctx, t, p, model)
	if !hasOpenAICodexReasoningRaw(first) {
		t.Fatalf("reasoning replay needs Codex raw reasoning: %+v", first.Parts)
	}
	// Replay with the requested alias, NOT first.Model: Response.Model keeps
	// the server-reported identity, which can be a dated snapshot (e.g.
	// gpt-5.4-mini-2026-03-17) that the codex backend rejects as a request
	// model for ChatGPT accounts.
	replayModel := liveOpenAICodexReasoningModel(model)
	resp, err := p.Chat(ctx, &llm.Request{
		Model:     replayModel,
		MaxTokens: 64,
		Messages: []llm.Message{
			llm.UserText(openAICodexReasoningPrompt),
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

func liveOpenAICodexReasoningResponse(ctx context.Context, t *testing.T, p llm.Provider, model string) *llm.Response {
	t.Helper()
	resp, err := llm.Collect(p.ChatStream(ctx, &llm.Request{
		Model:     liveOpenAICodexReasoningModel(model),
		MaxTokens: 768,
		Effort:    llm.EffortLow,
		Messages:  []llm.Message{llm.UserText(openAICodexReasoningPrompt)},
	}))
	if err != nil {
		t.Fatalf("reasoning stream returned error: %v", err)
	}
	return resp
}

func liveOpenAICodexReasoningModel(model string) string {
	if strings.HasPrefix(model, "gpt-5") || strings.HasPrefix(model, "o") {
		return model
	}
	return openAICodexCheapModel
}

func hasOpenAICodexReasoningRaw(resp *llm.Response) bool {
	if resp == nil {
		return false
	}
	for _, part := range resp.Parts {
		// Parts are value types (adapters never emit pointer parts).
		if reasoning, ok := part.(llm.ReasoningPart); ok && reasoning.Provider == "openai-codex" && len(reasoning.Raw) > 0 && json.Valid(reasoning.Raw) {
			return true
		}
	}
	return false
}
