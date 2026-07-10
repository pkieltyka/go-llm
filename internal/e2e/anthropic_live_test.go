//go:build live

package e2e

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/anthropic"
)

const (
	anthropicCheapModel      = "claude-haiku-4-5"
	anthropicReasoningModel  = "claude-sonnet-5"
	anthropicReasoningPrompt = "A store has three boxes. One is labeled apples, one oranges, and one mixed, " +
		"but every label is wrong. You may inspect one fruit from one box. Determine the correct labels, then answer with only: solved."
)

var record = flag.Bool("record", false, "record redacted live provider fixtures")
var recordAllowIncomplete = flag.Bool("record-allow-incomplete", false, "acknowledge replacement with an intentionally partial recording")

func TestLiveAnthropic(t *testing.T) {
	root, err := RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("LoadConfig returned error: %v", err)
	}
	providerCfg := cfg.Provider("anthropic", "ANTHROPIC_API_KEY")
	if providerCfg.Auth.Type == "oauth" {
		if providerCfg.Auth.Access == "" {
			t.Skip("Anthropic OAuth credential missing access token in gollm-test.json")
		}
	} else if providerCfg.Auth.Key == "" {
		t.Skip("Anthropic API key missing from gollm-test.json and ANTHROPIC_API_KEY")
	}
	model := providerCfg.Model
	if model == "" {
		model = anthropicCheapModel
	}
	// Honor an explicit model override for the reasoning scenarios too;
	// otherwise use the pinned reasoning-capable model.
	reasoningModel := providerCfg.Model
	if reasoningModel == "" {
		reasoningModel = anthropicReasoningModel
	}

	captures := &CaptureLog{}
	secrets := NewSecretSet(providerCfg.Auth.Key, providerCfg.Auth.Access, providerCfg.Auth.Refresh, providerCfg.Auth.AccountID, os.Getenv("ANTHROPIC_API_KEY"))
	var scenarioReport ScenarioReport
	if *record {
		path := filepath.Join(root, "internal", "e2e", "fixtures", "anthropic", "live.json")
		ScheduleFixtureRecording(t, path, captures, secrets, &scenarioReport, *recordAllowIncomplete)
	}
	opts := []anthropic.Option{
		anthropic.WithMaxRetries(0),
		anthropic.WithWireCapture(captures.Capture),
	}
	if providerCfg.Auth.Type == "oauth" {
		// Persist rotated refresh tokens back to gollm-test.json — dropping
		// them strands the stored credential after the provider refreshes.
		onRefresh := PersistOnRefresh(filepath.Join(root, "gollm-test.json"), "anthropic", t.Logf, secrets)
		opts = append(opts, anthropic.WithOAuth(providerCfg.Auth, onRefresh))
	} else {
		opts = append(opts, anthropic.WithAPIKey(providerCfg.Auth.Key))
	}
	if providerCfg.BaseURL != "" {
		opts = append(opts, anthropic.WithBaseURL(providerCfg.BaseURL))
	}
	p, err := anthropic.New(opts...)
	if err != nil {
		t.Fatalf("anthropic.New returned error: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	ctx = RecordingContext(ctx, captures, secrets)
	scenarioReport = RunScenarios(ctx, t, p, model, []Scenario{
		{Name: "chat", Run: liveChatScenario},
		{Name: "stream", Capability: llm.CapabilityStreaming, Run: liveStreamScenario},
		{Name: "models", Capability: llm.CapabilityModelsListing, Run: liveModelsScenario},
		{Name: "tools", Capability: llm.CapabilityTools, Run: liveToolsScenario},
		{Name: "tools_stream", Capability: llm.CapabilityToolStreaming, Run: liveToolsStreamScenario},
		{Name: "parallel_tools", Capability: llm.CapabilityParallelTools, Run: liveParallelToolsScenario},
		{Name: "parse", Capability: llm.CapabilityJSONSchema, Run: liveParseScenario},
		{Name: "reasoning", Capability: llm.CapabilityReasoning, Run: func(ctx context.Context, t *testing.T, p llm.Provider, _ string) {
			liveReasoningScenario(ctx, t, p, reasoningModel)
		}},
		{Name: "reasoning_replay", Capability: llm.CapabilityReasoning, Run: func(ctx context.Context, t *testing.T, p llm.Provider, _ string) {
			liveReasoningReplayScenario(ctx, t, p, reasoningModel)
		}},
		{Name: "multimodal", Capability: llm.CapabilityImageInput, Run: liveMultimodalScenario},
		{Name: "prompt_cache", Capability: llm.CapabilityPromptCaching, Run: livePromptCacheScenario},
		{Name: "usage", Run: liveUsageScenario},
		{Name: "error_mapping", Run: func(ctx context.Context, t *testing.T, p llm.Provider, model string) {
			liveErrorMappingScenario(ctx, t, p, model, providerCfg.BaseURL)
		}},
		{Name: "cross_provider_handoff", Capability: llm.CapabilityTools, Run: liveCrossProviderHandoffScenario},
	})
}

func liveChatScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	temperature := 0.0
	resp, err := p.Chat(ctx, &llm.Request{
		Model:       model,
		MaxTokens:   8,
		Temperature: &temperature,
		Messages:    []llm.Message{llm.UserText("Output exactly this word and no other text: pong")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if !strings.Contains(strings.ToLower(resp.Text()), "pong") {
		t.Fatalf("Chat text = %q, want pong", resp.Text())
	}
	if resp.Usage.OutputTokens == 0 {
		t.Fatalf("Chat usage missing output tokens: %+v", resp.Usage)
	}
}

func liveStreamScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	temperature := 0.0
	resp, err := llm.Collect(p.ChatStream(ctx, &llm.Request{
		Model:       model,
		MaxTokens:   8,
		Temperature: &temperature,
		Messages:    []llm.Message{llm.UserText("Output exactly this word and no other text: ok")},
	}))
	if err != nil {
		t.Fatalf("Collect(ChatStream) returned error: %v", err)
	}
	if strings.TrimSpace(resp.Text()) == "" {
		t.Fatalf("stream text empty: %+v", resp)
	}
	if resp.Usage.OutputTokens == 0 {
		t.Fatalf("stream usage missing output tokens: %+v", resp.Usage)
	}
}

func liveModelsScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	models, err := p.Models(ctx)
	if err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	if len(models) == 0 {
		t.Fatalf("Models returned empty list")
	}
	for _, info := range models {
		if info.ID == model {
			return
		}
	}
	t.Logf("model %q was not present in listing of %d models; provider override may be an alias", model, len(models))
}

func liveToolsScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	temperature := 0.0
	req := &llm.Request{
		Model:       model,
		MaxTokens:   64,
		Temperature: &temperature,
		Messages:    []llm.Message{llm.UserText(`Use the lookup tool with q exactly "go".`)},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Description: "Look up a short value.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}`),
			Strict:      true,
		}},
		ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "lookup"},
	}
	resp, err := p.Chat(ctx, req)
	if err != nil {
		t.Fatalf("tool Chat returned error: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID == "" || calls[0].Name != "lookup" {
		t.Fatalf("tool calls = %+v", calls)
	}
	var args struct {
		Q string `json:"q"`
	}
	if err := json.Unmarshal(calls[0].Args, &args); err != nil {
		t.Fatalf("tool call args are invalid JSON: %v: %s", err, calls[0].Args)
	}
	if args.Q != "go" {
		t.Fatalf("tool call args q = %q, want go", args.Q)
	}

	followUp := append([]llm.Message(nil), req.Messages...)
	followUp = append(followUp, llm.Message{
		Role:     llm.RoleAssistant,
		Parts:    resp.Parts,
		Provider: resp.Provider,
		Model:    resp.Model,
	})
	followUp = append(followUp, llm.Message{
		Role:  llm.RoleTool,
		Parts: []llm.Part{llm.ToolResultPart{ToolCallID: calls[0].ID, Name: calls[0].Name, Content: []llm.Part{llm.Text(`{"answer":"ok"}`)}}},
	})
	second, err := p.Chat(ctx, &llm.Request{
		Model:       model,
		MaxTokens:   32,
		Temperature: &temperature,
		Messages:    followUp,
	})
	if err != nil {
		t.Fatalf("tool-result follow-up returned error: %v", err)
	}
	if strings.TrimSpace(second.Text()) == "" {
		t.Fatalf("tool-result follow-up text empty: %+v", second)
	}
}

// liveToolsStreamScenario forces one tool call over the streaming path and
// asserts the collected response parses it — covering ToolCallStart/Delta/End
// event mapping and the tool_use stop reason on the stream path (the
// recorded fixture feeds the offline tool-streaming replay tests).
func liveToolsStreamScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	temperature := 0.0
	resp, err := llm.Collect(p.ChatStream(ctx, &llm.Request{
		Model:       model,
		MaxTokens:   64,
		Temperature: &temperature,
		Messages:    []llm.Message{llm.UserText(`Use the lookup tool with q exactly "go".`)},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Description: "Look up a short value.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}`),
			Strict:      true,
		}},
		ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "lookup"},
	}))
	if err != nil {
		t.Fatalf("tool stream returned error: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID == "" || calls[0].Name != "lookup" {
		t.Fatalf("streamed tool calls = %+v", calls)
	}
	var args struct {
		Q string `json:"q"`
	}
	if err := json.Unmarshal(calls[0].Args, &args); err != nil || args.Q != "go" {
		t.Fatalf("streamed tool call args = %s (err=%v)", calls[0].Args, err)
	}
	if resp.StopReason != llm.StopReasonToolUse {
		t.Fatalf("streamed tool stop reason = %q, want %q", resp.StopReason, llm.StopReasonToolUse)
	}
}

func liveParallelToolsScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	temperature := 0.0
	resp, err := p.Chat(ctx, &llm.Request{
		Model:       model,
		MaxTokens:   96,
		Temperature: &temperature,
		Messages: []llm.Message{llm.UserText(
			`Use both tools in parallel: lookup_weather with city exactly "Paris", and lookup_time with city exactly "Paris". Do not answer with text.`,
		)},
		Tools: []llm.Tool{
			{
				Name:        "lookup_weather",
				Description: "Look up weather for a city.",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}`),
				Strict:      true,
			},
			{
				Name:        "lookup_time",
				Description: "Look up local time for a city.",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}`),
				Strict:      true,
			},
		},
		ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceRequired},
	})
	if err != nil {
		t.Fatalf("parallel tools Chat returned error: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) < 2 {
		t.Fatalf("parallel tool calls len = %d, want at least 2: %+v", len(calls), calls)
	}
	seen := map[string]bool{}
	for _, call := range calls {
		seen[call.Name] = true
		if call.ID == "" {
			t.Fatalf("parallel tool call missing ID: %+v", call)
		}
	}
	if !seen["lookup_weather"] || !seen["lookup_time"] {
		t.Fatalf("parallel tool calls missing expected names: %+v", calls)
	}
}

type liveContact struct {
	Name string `json:"name"`
}

func liveParseScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
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

func liveReasoningScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	resp := liveReasoningResponse(ctx, t, p, model)
	if resp.Reasoning() == "" && !hasAnthropicReasoningRaw(resp) {
		t.Fatalf("reasoning response missing reasoning parts: text_len=%d stop_reason=%s", len(resp.Text()), resp.StopReason)
	}

	none, err := p.Chat(ctx, &llm.Request{
		Model:     model,
		MaxTokens: 16,
		Effort:    llm.EffortNone,
		Messages:  []llm.Message{llm.UserText("Answer exactly: no thinking")},
	})
	if err != nil {
		t.Fatalf("EffortNone Chat returned error: %v", err)
	}
	if none.Reasoning() != "" || hasAnthropicReasoningRaw(none) {
		t.Fatalf("EffortNone returned reasoning: %+v", none.Parts)
	}
}

func liveReasoningReplayScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	first := liveReasoningResponse(ctx, t, p, model)
	if !hasAnthropicReasoningRaw(first) {
		t.Fatalf("reasoning replay needs Anthropic raw reasoning: parts=%d stop_reason=%s", len(first.Parts), first.StopReason)
	}
	replayModel := first.Model
	if replayModel == "" {
		replayModel = model
	}
	resp, err := p.Chat(ctx, &llm.Request{
		Model:     replayModel,
		MaxTokens: 128,
		Messages: []llm.Message{
			llm.UserText(anthropicReasoningPrompt),
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
		t.Fatalf("reasoning replay text empty: stop_reason=%s output_tokens=%d reasoning_tokens=%d", resp.StopReason, resp.Usage.OutputTokens, resp.Usage.ReasoningTokens)
	}
}

func liveReasoningResponse(ctx context.Context, t *testing.T, p llm.Provider, model string) *llm.Response {
	t.Helper()
	temperature := 1.0
	resp, err := llm.Collect(p.ChatStream(ctx, &llm.Request{
		Model:       model,
		MaxTokens:   768,
		Temperature: &temperature,
		Effort:      llm.EffortHigh,
		Messages:    []llm.Message{llm.UserText(anthropicReasoningPrompt)},
	}))
	if err != nil {
		t.Fatalf("reasoning stream returned error: %v", err)
	}
	return resp
}

func hasAnthropicReasoningRaw(resp *llm.Response) bool {
	if resp == nil {
		return false
	}
	for _, part := range resp.Parts {
		// Parts are value types (adapters never emit pointer parts).
		if reasoning, ok := part.(llm.ReasoningPart); ok && reasoning.Provider == "anthropic" && len(reasoning.Raw) > 0 && json.Valid(reasoning.Raw) {
			return true
		}
	}
	return false
}

func liveMultimodalScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	image := RedPixelPNG(t)
	temperature := 0.0
	resp, err := p.Chat(ctx, &llm.Request{
		Model:       model,
		MaxTokens:   16,
		Temperature: &temperature,
		Messages: []llm.Message{llm.UserParts(
			llm.ImageData(image, "image/png"),
			llm.Text("What color is this square? Answer with one word."),
		)},
	})
	if err != nil {
		t.Fatalf("multimodal Chat returned error: %v", err)
	}
	if !strings.Contains(strings.ToLower(resp.Text()), "red") {
		t.Fatalf("multimodal text = %q, want red", resp.Text())
	}
}

func livePromptCacheScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	temperature := 0.0
	req := &llm.Request{
		Model:       model,
		System:      strings.Repeat("Anthropic prompt cache live fixture sentence with stable content. ", 600),
		SystemCache: &llm.CacheHint{TTL: time.Hour},
		MaxTokens:   8,
		Temperature: &temperature,
		Messages:    []llm.Message{llm.UserText("Answer exactly: cached")},
	}
	first, err := p.Chat(ctx, req)
	if err != nil {
		t.Fatalf("prompt cache first Chat returned error: %v", err)
	}
	var second *llm.Response
	for attempt := range 3 {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				t.Fatalf("prompt cache retry context done: %v", ctx.Err())
			case <-time.After(time.Duration(attempt) * time.Second):
			}
		}
		second, err = p.Chat(ctx, req)
		if err != nil {
			t.Fatalf("prompt cache second Chat returned error: %v", err)
		}
		if second.Usage.CacheReadTokens > 0 {
			return
		}
	}
	t.Fatalf("prompt cache second call never read cache: first=%+v second=%+v", first.Usage, second.Usage)
}

func liveUsageScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	temperature := 0.0
	resp, err := p.Chat(ctx, &llm.Request{
		Model:       model,
		MaxTokens:   8,
		Temperature: &temperature,
		Messages:    []llm.Message{llm.UserText("Output exactly: usage")},
	})
	if err != nil {
		t.Fatalf("usage Chat returned error: %v", err)
	}
	if resp.Usage.InputTokens == 0 || resp.Usage.OutputTokens == 0 || resp.Usage.TotalTokens == 0 {
		t.Fatalf("usage missing tokens: %+v", resp.Usage)
	}
}

func liveErrorMappingScenario(ctx context.Context, t *testing.T, p llm.Provider, model, baseURL string) {
	t.Helper()
	_, err := p.Chat(ctx, &llm.Request{
		Model:     "definitely-not-a-real-anthropic-model",
		MaxTokens: 8,
		Messages:  []llm.Message{llm.UserText("hello")},
	})
	if !errors.Is(err, llm.ErrNotFound) && !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("bad model error = %v, want ErrNotFound or ErrBadRequest", err)
	}

	opts := []anthropic.Option{
		anthropic.WithAPIKey("invalid-anthropic-key"),
		anthropic.WithMaxRetries(0),
	}
	if baseURL != "" {
		opts = append(opts, anthropic.WithBaseURL(baseURL))
	}
	badProvider, err := anthropic.New(opts...)
	if err != nil {
		t.Fatalf("anthropic.New with invalid key returned error: %v", err)
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
