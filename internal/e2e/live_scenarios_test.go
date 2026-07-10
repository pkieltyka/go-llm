//go:build live

package e2e

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func commonLiveScenarioRunners() map[string]ScenarioRun {
	return map[string]ScenarioRun{
		"chat":                   liveChatScenario,
		"stream":                 liveStreamScenario,
		"models":                 liveModelsScenario,
		"tools":                  liveToolsScenario,
		"tools_stream":           liveToolsStreamScenario,
		"parallel_tools":         liveParallelToolsScenario,
		"parse":                  liveParseScenario,
		"json_mode":              liveJSONModeScenario,
		"multimodal":             liveMultimodalScenario,
		"stop_sequences":         liveStopSequencesScenario,
		"session_affinity":       liveSessionAffinityScenario,
		"usage":                  liveUsageScenario,
		"cross_provider_handoff": liveCrossProviderHandoffScenario,
	}
}

func requireLiveProviderConfig(t *testing.T, provider string, cfg ProviderConfig) {
	t.Helper()
	err := ValidateLiveProviderConfig(provider, cfg)
	if errors.Is(err, errLiveCredentialsMissing) {
		t.Skipf("%v", err)
	}
	if err != nil {
		t.Fatalf("invalid %s live configuration: %v", provider, err)
	}
}

func liveJSONModeScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	resp, err := p.Chat(ctx, &llm.Request{
		Model:          model,
		MaxTokens:      32,
		Messages:       []llm.Message{llm.UserText(`Return one JSON object with key "status" and value "ok".`)},
		ResponseFormat: &llm.ResponseFormat{Type: llm.FormatJSONMode},
	})
	if err != nil {
		t.Fatalf("JSON mode Chat returned error: %v", err)
	}
	var value map[string]any
	if err := json.Unmarshal([]byte(resp.Text()), &value); err != nil {
		t.Fatalf("JSON mode response is invalid JSON: %v: %q", err, resp.Text())
	}
	if value["status"] != "ok" {
		t.Fatalf("JSON mode response = %#v, want status=ok", value)
	}
}

func liveStopSequencesScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	temperature := 0.0
	resp, err := p.Chat(ctx, &llm.Request{
		Model:         model,
		MaxTokens:     32,
		Temperature:   &temperature,
		StopSequences: []string{"STOP"},
		Messages:      []llm.Message{llm.UserText("Output exactly: alpha STOP omega")},
	})
	if err != nil {
		t.Fatalf("stop-sequence Chat returned error: %v", err)
	}
	text := strings.TrimSpace(resp.Text())
	if text == "" || strings.Contains(text, "STOP") || strings.Contains(strings.ToLower(text), "omega") {
		t.Fatalf("stop-sequence response = %q", resp.Text())
	}
	if resp.StopReason != llm.StopReasonStopSequence {
		t.Fatalf("stop-sequence reason = %q, want %q", resp.StopReason, llm.StopReasonStopSequence)
	}
}

func liveSessionAffinityScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	for _, prompt := range []string{"Answer exactly: session one", "Answer exactly: session two"} {
		resp, err := p.Chat(ctx, &llm.Request{
			Model:     model,
			MaxTokens: 16,
			SessionID: "go-llm-live-session-affinity",
			Messages:  []llm.Message{llm.UserText(prompt)},
		})
		if err != nil {
			t.Fatalf("session-affinity Chat returned error: %v", err)
		}
		if strings.TrimSpace(resp.Text()) == "" || resp.Usage.OutputTokens == 0 {
			t.Fatalf("session-affinity response missing content or usage: %+v", resp)
		}
	}
}

func liveImplicitPromptCacheScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	t.Helper()
	req := &llm.Request{
		Model:     model,
		MaxTokens: 16,
		SessionID: "go-llm-live-prompt-cache",
		System:    strings.Repeat("Stable prompt cache live sentence. ", 400),
		Messages:  []llm.Message{llm.UserText("Answer exactly: cached")},
	}
	first, second, err := probePromptCache(ctx, p.Name(), defaultPromptCacheProbePolicy(), func(ctx context.Context) (*llm.Response, error) {
		return p.Chat(ctx, req)
	})
	if err != nil {
		t.Fatalf("prompt cache evidence failed: %v (first=%+v last=%+v)", err, responseUsage(first), responseUsage(second))
	}
	if first.Usage.InputTokens == 0 || second.Usage.InputTokens+second.Usage.CacheReadTokens == 0 {
		t.Fatalf("prompt cache usage missing input tokens: first=%+v second=%+v", first.Usage, second.Usage)
	}
}
