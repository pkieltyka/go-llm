//go:build live

package e2e

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	openrouterProvider "github.com/pkieltyka/go-llm/providers/openrouter"
)

const (
	openRouterCheapModel = "qwen/qwen3.6-27b"
	// Reasoning scenarios run on the pinned default model (hybrid-reasoning
	// Qwen 3.6 exposes reasoning/reasoning_details through OpenRouter).
	openRouterReasoningModel = "qwen/qwen3.6-27b"
	// Prompt caching is exercised via OpenRouter's Anthropic cache_control
	// passthrough, which needs an Anthropic-family model.
	openRouterCacheModel = "anthropic/claude-haiku-4.5"
	// Parallel tool fan-out is model behavior: the pinned Qwen model answers
	// the parallel-tools prompt with a single sequential call even at
	// temperature 0, so that scenario runs on an Anthropic-family model that
	// reliably emits parallel calls.
	openRouterParallelToolsModel = "anthropic/claude-haiku-4.5"
	// Tool-calling reliability through OpenRouter is routing-dependent for
	// qwen3.6-27b: some upstream hosts intermittently return no tool_calls at
	// all (observed 2026-07-05: FAIL/FAIL/PASS across identical runs). The
	// tool-dependent scenarios pin an Anthropic-family model; qwen tool paths
	// stay covered by the vLLM suite, which runs the same family natively.
	openRouterToolsModel = "anthropic/claude-haiku-4.5"
)

func TestLiveOpenRouter(t *testing.T) {
	root, err := RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("LoadConfig returned error: %v", err)
	}
	providerCfg := cfg.Provider("openrouter", "OPENROUTER_API_KEY")
	if providerCfg.Auth.Key == "" {
		t.Skip("OpenRouter API key missing from gollm-test.json and OPENROUTER_API_KEY")
	}
	model := providerCfg.Model
	if model == "" {
		model = openRouterCheapModel
	}
	// Honor an explicit model override for the specialized scenarios too;
	// otherwise use the pinned per-capability models.
	reasoningModel := providerCfg.Model
	if reasoningModel == "" {
		reasoningModel = openRouterReasoningModel
	}
	cacheModel := providerCfg.Model
	if cacheModel == "" {
		cacheModel = openRouterCacheModel
	}
	parallelToolsModel := providerCfg.Model
	if parallelToolsModel == "" {
		parallelToolsModel = openRouterParallelToolsModel
	}
	toolsModel := providerCfg.Model
	if toolsModel == "" {
		toolsModel = openRouterToolsModel
	}

	var captures []llm.WireCapture
	opts := []openrouterProvider.Option{
		openrouterProvider.WithAPIKey(providerCfg.Auth.Key),
		openrouterProvider.WithMaxRetries(0),
		openrouterProvider.WithAttribution("https://github.com/pkieltyka/go-llm", "go-llm live tests"),
		openrouterProvider.WithWireCapture(func(c llm.WireCapture) {
			captures = append(captures, c)
		}),
	}
	if providerCfg.BaseURL != "" {
		opts = append(opts, openrouterProvider.WithBaseURL(providerCfg.BaseURL))
	}
	p, err := openrouterProvider.New(opts...)
	if err != nil {
		t.Fatalf("openrouter.New returned error: %v", err)
	}
	if *record {
		t.Cleanup(func() {
			path := filepath.Join(root, "internal", "e2e", "fixtures", "openrouter", "live.json")
			if err := WriteFixture(path, captures, providerCfg.Auth.Key, os.Getenv("OPENROUTER_API_KEY")); err != nil {
				t.Fatalf("WriteFixture returned error: %v", err)
			}
		})
	}

	// The pinned qwen3.6 model is hybrid-reasoning and THINKS BY DEFAULT,
	// starving the generic scenarios' small MaxTokens budgets before any
	// text is emitted. Default unset Effort to none — exactly how callers
	// run hybrid models for terse outputs — using go-llm's own middleware;
	// the reasoning scenarios set Effort explicitly and pass through
	// untouched.
	scenarioProvider := llm.Wrap(p, defaultEffortNoneMiddleware())

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	RunScenarios(ctx, t, scenarioProvider, model, []Scenario{
		{Name: "chat", Run: liveChatScenario},
		{Name: "stream", Capability: llm.CapabilityStreaming, Run: liveStreamScenario},
		{Name: "models", Capability: llm.CapabilityModelsListing, Run: liveModelsScenario},
		{Name: "tools", Capability: llm.CapabilityTools, Run: func(ctx context.Context, t *testing.T, p llm.Provider, _ string) {
			liveToolsScenario(ctx, t, p, toolsModel)
		}},
		{Name: "tools_stream", Capability: llm.CapabilityToolStreaming, Run: func(ctx context.Context, t *testing.T, p llm.Provider, _ string) {
			liveToolsStreamScenario(ctx, t, p, toolsModel)
		}},
		{Name: "parallel_tools", Capability: llm.CapabilityParallelTools, Run: func(ctx context.Context, t *testing.T, p llm.Provider, _ string) {
			liveParallelToolsScenario(ctx, t, p, parallelToolsModel)
		}},
		{Name: "parse", Capability: llm.CapabilityJSONSchema, Run: liveParseScenario},
		{Name: "reasoning", Capability: llm.CapabilityReasoning, Run: func(ctx context.Context, t *testing.T, p llm.Provider, _ string) {
			liveOpenRouterReasoningScenario(ctx, t, p, reasoningModel)
		}},
		{Name: "reasoning_replay", Capability: llm.CapabilityReasoning, Run: func(ctx context.Context, t *testing.T, p llm.Provider, _ string) {
			liveOpenRouterReasoningReplayScenario(ctx, t, p, reasoningModel)
		}},
		{Name: "multimodal", Capability: llm.CapabilityImageInput, Run: liveMultimodalScenario},
		{Name: "prompt_cache", Capability: llm.CapabilityPromptCaching, Run: func(ctx context.Context, t *testing.T, p llm.Provider, _ string) {
			livePromptCacheScenario(ctx, t, p, cacheModel)
		}},
		{Name: "usage", Run: liveUsageScenario},
		{Name: "cost_reporting", Capability: llm.CapabilityCostReporting, Run: liveOpenRouterCostScenario},
		{Name: "error_mapping", Run: func(ctx context.Context, t *testing.T, p llm.Provider, model string) {
			liveOpenRouterErrorMappingScenario(ctx, t, p, model, providerCfg.BaseURL)
		}},
		{Name: "cross_provider_handoff", Capability: llm.CapabilityTools, Run: func(ctx context.Context, t *testing.T, p llm.Provider, _ string) {
			// The handoff source leg forces a tool call — same routing
			// reliability pin as the tools scenarios above.
			liveCrossProviderHandoffScenario(ctx, t, p, toolsModel)
		}},
	})
}

// defaultEffortNoneMiddleware defaults unset Effort to EffortNone so
// hybrid-reasoning models answer tersely in the generic scenarios; requests
// that set Effort explicitly are untouched.
func defaultEffortNoneMiddleware() llm.Middleware {
	return llm.Middleware{
		Chat: func(next llm.ChatFunc) llm.ChatFunc {
			return func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				return next(ctx, withDefaultEffortNone(req))
			}
		},
		Stream: func(next llm.StreamFunc) llm.StreamFunc {
			return func(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
				return next(ctx, withDefaultEffortNone(req))
			}
		},
	}
}

func withDefaultEffortNone(req *llm.Request) *llm.Request {
	if req == nil || req.Effort != "" {
		return req
	}
	copied := *req
	copied.Effort = llm.EffortNone
	return &copied
}

func liveOpenRouterCostScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	temperature := 0.0
	resp, err := p.Chat(ctx, &llm.Request{
		Model:       model,
		MaxTokens:   8,
		Temperature: &temperature,
		Messages:    []llm.Message{llm.UserText("Output exactly: cost")},
	})
	if err != nil {
		t.Fatalf("cost Chat returned error: %v", err)
	}
	if resp.Usage.CostUSD == nil || *resp.Usage.CostUSD <= 0 {
		t.Fatalf("native cost missing: %+v", resp.Usage)
	}
	extras, ok := openrouterProvider.Extras(resp)
	if !ok || extras.Provider == "" {
		t.Fatalf("typed extras missing: ok=%v extras=%+v", ok, extras)
	}
	isBYOK := "unreported"
	if extras.IsBYOK != nil {
		isBYOK = strconv.FormatBool(*extras.IsBYOK)
	}
	t.Logf("cost=%v provider=%s cost_details=%s is_byok=%s", *resp.Usage.CostUSD, extras.Provider, extras.CostDetails, isBYOK)
}

func liveOpenRouterReasoningScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	resp := liveOpenRouterReasoningResponse(ctx, t, p, model)
	if resp.Reasoning() == "" && !hasOpenRouterReasoningRaw(resp) {
		t.Fatalf("reasoning response missing reasoning parts: text_len=%d stop_reason=%s", len(resp.Text()), resp.StopReason)
	}

	none, err := p.Chat(ctx, &llm.Request{
		Model:     model,
		MaxTokens: 64,
		Effort:    llm.EffortNone,
		Messages:  []llm.Message{llm.UserText("Answer exactly: no thinking")},
	})
	if err != nil {
		t.Fatalf("EffortNone Chat returned error: %v", err)
	}
	if none.Reasoning() != "" || hasOpenRouterReasoningRaw(none) {
		t.Fatalf("EffortNone returned reasoning: %+v", none.Parts)
	}
}

func liveOpenRouterReasoningReplayScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	first := liveOpenRouterReasoningResponse(ctx, t, p, model)
	if !hasOpenRouterReasoningRaw(first) {
		t.Fatalf("reasoning replay needs OpenRouter raw reasoning_details: parts=%d stop_reason=%s", len(first.Parts), first.StopReason)
	}
	replayModel := first.Model
	if replayModel == "" {
		replayModel = model
	}
	resp, err := p.Chat(ctx, &llm.Request{
		Model:     replayModel,
		MaxTokens: 2048,
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

func liveOpenRouterReasoningResponse(ctx context.Context, t *testing.T, p llm.Provider, model string) *llm.Response {
	t.Helper()
	resp, err := llm.Collect(p.ChatStream(ctx, &llm.Request{
		Model:     model,
		MaxTokens: 2048,
		Effort:    llm.EffortHigh,
		Messages:  []llm.Message{llm.UserText(anthropicReasoningPrompt)},
	}))
	if err != nil {
		t.Fatalf("reasoning stream returned error: %v", err)
	}
	return resp
}

func hasOpenRouterReasoningRaw(resp *llm.Response) bool {
	if resp == nil {
		return false
	}
	for _, part := range resp.Parts {
		// Parts are value types (adapters never emit pointer parts).
		if reasoning, ok := part.(llm.ReasoningPart); ok && reasoning.Provider == "openrouter" && len(reasoning.Raw) > 0 && json.Valid(reasoning.Raw) {
			return true
		}
	}
	return false
}

func liveOpenRouterErrorMappingScenario(ctx context.Context, t *testing.T, p llm.Provider, model, baseURL string) {
	t.Helper()
	_, err := p.Chat(ctx, &llm.Request{
		Model:     "definitely/not-a-real-openrouter-model",
		MaxTokens: 8,
		Messages:  []llm.Message{llm.UserText("hello")},
	})
	if !errors.Is(err, llm.ErrNotFound) && !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("bad model error = %v, want ErrNotFound or ErrBadRequest", err)
	}

	opts := []openrouterProvider.Option{
		openrouterProvider.WithAPIKey("invalid-openrouter-key"),
		openrouterProvider.WithMaxRetries(0),
	}
	if baseURL != "" {
		opts = append(opts, openrouterProvider.WithBaseURL(baseURL))
	}
	badProvider, err := openrouterProvider.New(opts...)
	if err != nil {
		t.Fatalf("openrouter.New with invalid key returned error: %v", err)
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
