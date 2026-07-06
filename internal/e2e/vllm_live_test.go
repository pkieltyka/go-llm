//go:build live

package e2e

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	anthropicProvider "github.com/pkieltyka/go-llm/providers/anthropic"
	vllmProvider "github.com/pkieltyka/go-llm/providers/vllm"
)

// TestLiveVLLM exercises the vllm preset against a self-hosted server
// (gollm-test.json entry "vllm": keyless, base_url required — the loader's
// base_url-presence rule marks it configured). The suite assumes a server
// with --enable-auto-tool-choice and a reasoning parser, serving a
// thinking-by-default model (Qwen3.6): unset Effort defaults to none via
// go-llm middleware so terse scenarios stay within their token budgets, and
// — a live finding on vLLM 0.24.0 — constrained tool choice (named/required/
// strict) 500s while thinking is enabled, so the same default keeps the tool
// scenarios on the working path. Reasoning scenarios set Effort explicitly.
func TestLiveVLLM(t *testing.T) {
	root, err := RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("LoadConfig returned error: %v", err)
	}
	providerCfg := cfg.Provider("vllm", "")
	// vLLM is host-first: a base URL is the one thing this suite requires
	// (Configured() would also accept a key-only entry, which is useless here).
	if providerCfg.BaseURL == "" {
		t.Skip("vLLM base_url missing from gollm-test.json (keyless self-hosted entry)")
	}

	var captures []llm.WireCapture
	opts := []vllmProvider.Option{
		vllmProvider.WithMaxRetries(0),
		vllmProvider.WithWireCapture(func(c llm.WireCapture) {
			captures = append(captures, c)
		}),
	}
	if providerCfg.Auth.Key != "" {
		opts = append(opts, vllmProvider.WithAPIKey(providerCfg.Auth.Key))
	}
	p, err := vllmProvider.New(providerCfg.BaseURL, opts...)
	if err != nil {
		t.Fatalf("vllm.New returned error: %v", err)
	}
	if *record {
		t.Cleanup(func() {
			path := filepath.Join(root, "internal", "e2e", "fixtures", "vllm", "live.json")
			if err := WriteFixture(path, captures, providerCfg.Auth.Key); err != nil {
				t.Fatalf("WriteFixture returned error: %v", err)
			}
		})
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	modelPreference := providerCfg.Model
	if modelPreference == "" {
		modelPreference = "qwen"
	}
	resolvedModel, err := p.ResolveModel(ctx, modelPreference)
	if err != nil {
		t.Fatalf("vLLM model discovery failed for preference %q: %v", modelPreference, err)
	}
	model := resolvedModel.ID
	if model != providerCfg.Model {
		t.Logf("resolved vLLM model preference %q to served model %q", modelPreference, model)
	}

	// Same rationale as the OpenRouter suite: hybrid Qwen thinks by default,
	// so unset Effort defaults to none through go-llm's own middleware.
	scenarioProvider := llm.Wrap(p, defaultEffortNoneMiddleware())

	RunScenarios(ctx, t, scenarioProvider, model, []Scenario{
		{Name: "chat", Run: liveChatScenario},
		{Name: "stream", Capability: llm.CapabilityStreaming, Run: liveStreamScenario},
		{Name: "models", Capability: llm.CapabilityModelsListing, Run: liveVLLMModelsScenario},
		{Name: "tools", Capability: llm.CapabilityTools, Run: liveToolsScenario},
		{Name: "tools_stream", Capability: llm.CapabilityToolStreaming, Run: liveVLLMToolsStreamScenario},
		{Name: "parallel_tools", Capability: llm.CapabilityParallelTools, Run: liveParallelToolsScenario},
		{Name: "parse", Capability: llm.CapabilityJSONSchema, Run: liveParseScenario},
		{Name: "structured_choice", Run: liveVLLMStructuredChoiceScenario},
		{Name: "structured_regex", Run: liveVLLMStructuredRegexScenario},
		{Name: "tokenize", Run: func(ctx context.Context, t *testing.T, _ llm.Provider, model string) {
			// Extension methods live on the concrete provider, not the
			// wrapped llm.Provider.
			liveVLLMTokenizeScenario(ctx, t, p, model)
		}},
		{Name: "reasoning", Capability: llm.CapabilityReasoning, Run: liveVLLMReasoningScenario},
		{Name: "reasoning_replay", Capability: llm.CapabilityReasoning, Run: liveVLLMReasoningReplayScenario},
		{Name: "usage", Run: liveUsageScenario},
		{Name: "error_mapping", Run: liveVLLMErrorMappingScenario},
		{Name: "cross_provider_handoff", Capability: llm.CapabilityTools, Run: liveCrossProviderHandoffScenario},
		{Name: "anthropic_messages", Run: func(ctx context.Context, t *testing.T, _ llm.Provider, model string) {
			liveVLLMAnthropicMessagesScenario(ctx, t, providerCfg.BaseURL, model)
		}},
	})
}

// liveVLLMToolsStreamScenario streams one tool call through the server-side
// auto tool parser (tool_choice auto + --tool-call-parser). It deliberately
// differs from the shared forced-call scenario: vLLM ends FORCED (named)
// tool calls with finish_reason "stop" — OpenAI Chat Completions semantics —
// so only auto-detected calls carry the "tool_calls" finish reason this
// scenario asserts.
func liveVLLMToolsStreamScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	temperature := 0.0
	resp, err := llm.Collect(p.ChatStream(ctx, &llm.Request{
		Model:       model,
		MaxTokens:   256,
		Temperature: &temperature,
		Messages:    []llm.Message{llm.UserText(`Use the lookup tool with q exactly "go".`)},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Description: "Look up a short value.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}`),
		}},
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

// liveVLLMModelsScenario asserts the served model is listed with vLLM's
// max_model_len surfaced as ContextWindow.
func liveVLLMModelsScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	models, err := p.Models(ctx)
	if err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	if len(models) == 0 {
		t.Fatalf("Models returned empty list")
	}
	for _, info := range models {
		if info.ID != model {
			continue
		}
		if info.ContextWindow <= 0 {
			t.Fatalf("model %q missing max_model_len context window: %+v", model, info)
		}
		return
	}
	t.Fatalf("model %q not present in listing of %d models", model, len(models))
}

// liveVLLMReasoningScenario asserts the reasoning parser's plain-text output
// maps to ReasoningPart (text only, no raw payload — vLLM reasoning carries
// no signatures) and that EffortNone suppresses thinking.
func liveVLLMReasoningScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	resp := liveVLLMReasoningResponse(ctx, t, p, model)
	reasoning := vllmReasoningPart(resp)
	if reasoning == nil || strings.TrimSpace(reasoning.Text) == "" {
		t.Fatalf("reasoning response missing plain-text reasoning: parts=%+v stop_reason=%s", resp.Parts, resp.StopReason)
	}
	if len(reasoning.Raw) != 0 {
		t.Fatalf("vLLM reasoning should have no raw payload: %s", reasoning.Raw)
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
	if none.Reasoning() != "" {
		t.Fatalf("EffortNone returned reasoning: %+v", none.Parts)
	}
}

// liveVLLMReasoningReplayScenario replays a reasoning-bearing turn back to
// the same server. vLLM reasoning is plain text and open-model chat
// templates DROP prior thinking on replay by design, so the assertion is
// that the continuation succeeds (no 400 for the replayed `reasoning`
// field) and produces text — not that the thinking influenced anything.
func liveVLLMReasoningReplayScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	first := liveVLLMReasoningResponse(ctx, t, p, model)
	if part := vllmReasoningPart(first); part == nil {
		t.Fatalf("reasoning replay needs a reasoning part: parts=%d stop_reason=%s", len(first.Parts), first.StopReason)
	}
	replayModel := first.Model
	if replayModel == "" {
		replayModel = model
	}
	resp, err := p.Chat(ctx, &llm.Request{
		Model:     replayModel,
		MaxTokens: 128,
		Effort:    llm.EffortNone,
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
		t.Fatalf("reasoning replay text empty: stop_reason=%s output_tokens=%d", resp.StopReason, resp.Usage.OutputTokens)
	}
}

func liveVLLMReasoningResponse(ctx context.Context, t *testing.T, p llm.Provider, model string) *llm.Response {
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

func vllmReasoningPart(resp *llm.Response) *llm.ReasoningPart {
	if resp == nil {
		return nil
	}
	for _, part := range resp.Parts {
		// Parts are value types (adapters never emit pointer parts).
		if reasoning, ok := part.(llm.ReasoningPart); ok && reasoning.Provider == "vllm" {
			return &reasoning
		}
	}
	return nil
}

// liveVLLMErrorMappingScenario asserts vLLM's nested error shape maps to the
// normalized sentinels. Keyless hosts have no invalid-key case to exercise
// (vLLM ignores Authorization unless started with --api-key).
func liveVLLMErrorMappingScenario(ctx context.Context, t *testing.T, p llm.Provider, _ string) {
	t.Helper()
	_, err := p.Chat(ctx, &llm.Request{
		Model:     "definitely-not-a-served-vllm-model",
		MaxTokens: 8,
		Messages:  []llm.Message{llm.UserText("hello")},
	})
	if !errors.Is(err, llm.ErrNotFound) && !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("bad model error = %v, want ErrNotFound or ErrBadRequest", err)
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Provider != "vllm" {
		t.Fatalf("bad model error missing vllm provider metadata: %v", err)
	}
}

// liveVLLMTokenizeScenario exercises the /tokenize extension family. Wire
// findings baked into the assertions: the endpoints live at the server root
// (/v1/tokenize is 404 — the provider derives the root from the base URL),
// and the count is exact — a follow-up chat with the identical request
// reports prompt_tokens matching Count (Effort is set explicitly so the
// tokenize body and the chat body agree on the thinking toggle).
func liveVLLMTokenizeScenario(ctx context.Context, t *testing.T, p *vllmProvider.Provider, model string) {
	t.Helper()
	req := &llm.Request{
		Model:     model,
		MaxTokens: 64,
		Effort:    llm.EffortNone,
		Messages:  []llm.Message{llm.UserText("Output exactly this word and no other text: pong")},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Description: "Look up a short value.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}`),
		}},
	}
	result, err := p.Tokenize(ctx, req)
	if err != nil {
		t.Fatalf("Tokenize returned error: %v", err)
	}
	if result.Count <= 0 {
		t.Fatalf("tokenize count = %d, want > 0", result.Count)
	}
	if result.MaxModelLen <= 0 {
		t.Fatalf("tokenize max_model_len = %d, want > 0", result.MaxModelLen)
	}
	if len(result.Tokens) != result.Count {
		t.Fatalf("tokenize tokens len = %d, count = %d", len(result.Tokens), result.Count)
	}
	usage := result.ContextUsage()
	if usage.UsedTokens != int64(result.Count) || usage.Window != int64(result.MaxModelLen) || usage.Remaining != usage.Window-usage.UsedTokens {
		t.Fatalf("ContextUsage bridge = %+v", usage)
	}

	resp, err := p.Chat(ctx, req)
	if err != nil {
		t.Fatalf("follow-up Chat returned error: %v", err)
	}
	prompt := resp.Usage.InputTokens + resp.Usage.CacheReadTokens
	diff := prompt - int64(result.Count)
	if diff < 0 {
		diff = -diff
	}
	t.Logf("tokenize count=%d, chat prompt_tokens=%d (diff %d), max_model_len=%d", result.Count, prompt, diff, result.MaxModelLen)
	if diff > 8 {
		t.Fatalf("tokenize count %d vs chat prompt_tokens %d: diff %d exceeds tolerance", result.Count, prompt, diff)
	}

	detok, err := p.Detokenize(ctx, result.Tokens)
	if err != nil {
		t.Fatalf("Detokenize returned error: %v", err)
	}
	if !strings.Contains(detok, "pong") {
		t.Fatalf("detokenized prompt missing user text: %q", detok)
	}

	info, err := p.TokenizerInfo(ctx)
	switch {
	case errors.Is(err, llm.ErrNotFound):
		// Flag-gated endpoint (--enable-tokenizer-info-endpoint); 404
		// observed on the live host while /tokenize and /detokenize work.
		t.Logf("tokenizer_info not enabled on this host: %v", err)
	case err != nil:
		t.Fatalf("TokenizerInfo returned error: %v", err)
	case len(info) == 0:
		t.Fatalf("TokenizerInfo returned empty payload")
	}
}

// liveVLLMStructuredChoiceScenario constrains decoding to a fixed choice set
// via the native structured_outputs param. Effort is pinned to none: the
// live finding on this host (0.23/0.24-family, qwen3 reasoning parser) is
// that constraint modes corrupt output while thinking is active (a choice
// probe returned "greengreen"), while thinking-off returns an exact member.
func liveVLLMStructuredChoiceScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	choices := []string{"red", "green", "blue"}
	resp, err := p.Chat(ctx, &llm.Request{
		Model:     model,
		MaxTokens: 16,
		Effort:    llm.EffortNone,
		Messages:  []llm.Message{llm.UserText("What color is grass? Answer with one word.")},
		ProviderOptions: vllmProvider.Options{
			StructuredOutputs: &vllmProvider.StructuredOutputs{Choice: choices},
		},
	})
	if err != nil {
		t.Fatalf("structured choice Chat returned error: %v", err)
	}
	got := strings.TrimSpace(resp.Text())
	for _, choice := range choices {
		if got == choice {
			t.Logf("structured choice answer = %q", got)
			return
		}
	}
	t.Fatalf("structured choice answer %q not in %v", got, choices)
}

// liveVLLMStructuredRegexScenario constrains decoding to a regex (anchored
// form probe-verified on the xgrammar-family backend). Effort none for the
// same thinking interaction as the choice scenario.
func liveVLLMStructuredRegexScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	resp, err := p.Chat(ctx, &llm.Request{
		Model:     model,
		MaxTokens: 16,
		Effort:    llm.EffortNone,
		Messages:  []llm.Message{llm.UserText("What year did World War II end? Answer with just the 4-digit year.")},
		ProviderOptions: vllmProvider.Options{
			StructuredOutputs: &vllmProvider.StructuredOutputs{Regex: "^[0-9]{4}$"},
		},
	})
	if err != nil {
		t.Fatalf("structured regex Chat returned error: %v", err)
	}
	got := resp.Text()
	if !regexp.MustCompile(`^[0-9]{4}$`).MatchString(got) {
		t.Fatalf("structured regex answer %q does not match ^[0-9]{4}$", got)
	}
	t.Logf("structured regex answer = %q", got)
}

// liveVLLMAnthropicMessagesScenario is the bonus recipe check: vLLM ≥0.11.1
// serves an Anthropic /v1/messages endpoint at the server root, so go-llm's
// anthropic provider can target it via WithBaseURL plus a dummy key. The
// smoke asserts a successful, usage-bearing completion; thinking-by-default
// models may spend the whole budget on a thinking block, so content may be
// reasoning rather than text.
func liveVLLMAnthropicMessagesScenario(ctx context.Context, t *testing.T, baseURL, model string) {
	t.Helper()
	root := strings.TrimSuffix(strings.TrimRight(baseURL, "/"), "/v1")
	p, err := anthropicProvider.New(
		anthropicProvider.WithBaseURL(root),
		anthropicProvider.WithAPIKey("dummy"),
		anthropicProvider.WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("anthropic.New against vLLM returned error: %v", err)
	}
	resp, err := p.Chat(ctx, &llm.Request{
		Model:     model,
		MaxTokens: 512,
		Messages:  []llm.Message{llm.UserText("Output exactly this word and no other text: pong")},
	})
	if err != nil {
		t.Fatalf("anthropic-endpoint Chat returned error: %v", err)
	}
	if strings.TrimSpace(resp.Text()) == "" && strings.TrimSpace(resp.Reasoning()) == "" {
		t.Fatalf("anthropic-endpoint response has no content: %+v", resp.Parts)
	}
	if resp.Usage.OutputTokens == 0 {
		t.Fatalf("anthropic-endpoint usage missing output tokens: %+v", resp.Usage)
	}
	t.Logf("anthropic /v1/messages via vLLM: text=%q reasoning_len=%d stop=%s", resp.Text(), len(resp.Reasoning()), resp.StopReason)
}
