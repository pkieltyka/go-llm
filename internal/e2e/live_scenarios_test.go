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

// liveClassifyScenario asks a classifier model (FS §13C) one question of
// each type about an unambiguous state and checks the answers' shape,
// direction and usage.
func liveClassifyScenario(model string) ScenarioRun {
	return func(ctx context.Context, t *testing.T, p llm.Provider, _ string) {
		t.Helper()
		classifier := llm.NewClassifierModel(p, model, llm.WithClassifierQuestions(
			llm.ChoiceQuestion("category", "Classify the support message.",
				llm.Choice{Value: "bug", Description: "A defect report"},
				llm.Choice{Value: "praise", Description: "Positive feedback"}),
			llm.ScoreQuestion("urgency", "How urgent is the message?", "low", "medium", "high"),
			llm.BoolQuestion("satisfied", "Is the user satisfied?", "Satisfied", "Not satisfied"),
		))
		result, err := classifier.Classify(ctx, map[string]string{"message": "The app crashes every time I press save and I lose all my work!"})
		if err != nil {
			t.Fatalf("Classify(%s): %v", model, err)
		}
		category, urgency, satisfied := result.Answers["category"], result.Answers["urgency"], result.Answers["satisfied"]
		if category.Type != llm.QuestionChoice || category.Choice != "bug" || category.Probabilities["bug"] < 0.5 {
			t.Fatalf("category = %+v", category)
		}
		if urgency.Type != llm.QuestionScore || urgency.Score < 1 || urgency.Score > 2 {
			t.Fatalf("urgency = %+v", urgency)
		}
		if satisfied.Type != llm.QuestionBool || satisfied.Probability > 0.5 {
			t.Fatalf("satisfied = %+v", satisfied)
		}
		if result.Usage.InputTokens <= 0 {
			t.Fatalf("usage = %+v", result.Usage)
		}
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
	liveStopSequencesScenarioWithReasons(ctx, t, p, model, llm.StopReasonStopSequence)
}

func liveOpenAICompatibleStopSequencesScenario(ctx context.Context, t *testing.T, p llm.Provider, model string) {
	// Chat Completions uses finish_reason "stop" for both a natural stop and
	// a matched stop sequence, so these adapters cannot distinguish the two.
	liveStopSequencesScenarioWithReasons(ctx, t, p, model, llm.StopReasonStopSequence, llm.StopReasonEndTurn)
}

func liveStopSequencesScenarioWithReasons(ctx context.Context, t *testing.T, p llm.Provider, model string, allowed ...llm.StopReason) {
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
	for _, reason := range allowed {
		if resp.StopReason == reason {
			return
		}
	}
	t.Fatalf("stop-sequence reason = %q, want one of %q", resp.StopReason, allowed)
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

// liveReasoningModelSwitchScenario produces a reasoning-bearing tool call on
// producer, then continues the tool loop on consumer with the full,
// provenance-stamped history. It asserts the continuation succeeds and that
// no part of the producer's raw reasoning payload reached the consumer
// request (FS §18 model-aware replay). Tool choice stays auto because
// Anthropic-family models reject forced tool use with thinking enabled.
func liveReasoningModelSwitchScenario(ctx context.Context, t *testing.T, p llm.Provider, producer, consumer string) {
	t.Helper()
	captures := RecordingCaptures(ctx)
	if captures == nil {
		t.Fatal("reasoning_model_switch needs a RecordingContext capture log")
	}
	tool := llm.Tool{
		Name:        "get_weather",
		Description: "Get the current weather for a city.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}`),
	}
	user := llm.UserText("What's the weather in Paris? Use the get_weather tool, then answer in one short sentence.")
	first, err := p.Chat(ctx, &llm.Request{
		Model: producer, MaxTokens: 2048, Effort: llm.EffortLow,
		Tools: []llm.Tool{tool}, Messages: []llm.Message{user},
	})
	if err != nil {
		t.Fatalf("producer %s Chat returned error: %v", producer, err)
	}
	calls := first.ToolCalls()
	marker := reasoningRawMarker(first)
	if len(calls) == 0 || marker == "" {
		t.Fatalf("producer %s must return a tool call with raw reasoning: tool_calls=%d raw_marker=%t stop_reason=%s", producer, len(calls), marker != "", first.StopReason)
	}
	results := make([]llm.Part, 0, len(calls))
	for _, call := range calls {
		results = append(results, llm.ToolResult(call.ID, `{"city":"Paris","temp_c":18,"sky":"cloudy"}`))
	}

	before := len(captures.Snapshot().Captures)
	resp, err := p.Chat(ctx, &llm.Request{
		Model: consumer, MaxTokens: 2048, Effort: llm.EffortLow,
		Tools: []llm.Tool{tool},
		Messages: []llm.Message{
			user,
			{Role: llm.RoleAssistant, Parts: first.Parts, Provider: first.Provider, Model: first.Model},
			{Role: llm.RoleTool, Parts: results},
		},
	})
	if err != nil {
		t.Fatalf("consumer %s continuation returned error: %v", consumer, err)
	}
	if strings.TrimSpace(resp.Text()) == "" {
		t.Fatalf("consumer %s continuation text empty: stop_reason=%s", consumer, resp.StopReason)
	}
	sent := captures.Snapshot().Captures[before:]
	if len(sent) == 0 {
		t.Fatal("no wire capture recorded for the consumer request")
	}
	escaped, _ := json.Marshal(marker)
	for _, capture := range sent {
		body := string(capture.RequestBody)
		if strings.Contains(body, marker) || strings.Contains(body, strings.Trim(string(escaped), `"`)) {
			t.Fatalf("consumer %s request replayed the producer's raw reasoning payload", consumer)
		}
	}
	t.Logf("model switch %s -> %s: continuation ok, producer raw reasoning absent from %d consumer request(s)", producer, consumer, len(sent))
}

// reasoningRawMarker returns a wire-presence marker from the response's raw
// reasoning payloads: the longest opaque value (signature, data, or encrypted
// content) of at least 16 bytes, falling back to the longest string value.
// Opaque values cannot legitimately appear in a prompt or tool schema. Empty
// when none exists.
func reasoningRawMarker(resp *llm.Response) string {
	var opaque, best string
	var walk func(key string, value any)
	walk = func(key string, value any) {
		switch v := value.(type) {
		case string:
			if len(v) < 16 {
				return
			}
			switch key {
			case "signature", "data", "encrypted_content", "encrypted":
				if len(v) > len(opaque) {
					opaque = v
				}
			}
			if len(v) > len(best) {
				best = v
			}
		case []any:
			for _, item := range v {
				walk(key, item)
			}
		case map[string]any:
			for k, item := range v {
				walk(k, item)
			}
		}
	}
	for _, part := range resp.Parts {
		reasoning, ok := part.(llm.ReasoningPart)
		if !ok || len(reasoning.Raw) == 0 {
			continue
		}
		var decoded any
		if json.Unmarshal(reasoning.Raw, &decoded) == nil {
			walk("", decoded)
		}
	}
	if opaque != "" {
		return opaque
	}
	return best
}

// livePositionalChangesScenario adds a tool and a prompt section with a
// mid-conversation system message, then removes the tool (FS §7A). model
// must take positional changes in place. With expectCacheRead, turn 2 must
// read the prefix turn 1 cached: folding would have rewritten the system
// prompt and missed.
func livePositionalChangesScenario(ctx context.Context, t *testing.T, p llm.Provider, model string, expectCacheRead bool) {
	t.Helper()
	info, ok := llm.LookupModelInfo(p.Name(), model)
	if !ok || !info.Behavior.PositionalSupport().Tools {
		t.Fatalf("%s/%s does not take positional tool changes in place; pin a model whose behavior does", p.Name(), model)
	}
	cityTool := func(name, description string) llm.Tool {
		return llm.Tool{Name: name, Description: description, InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)}
	}
	history := []llm.Message{llm.UserParts(llm.TextPart{Text: "Say OK.", Cache: &llm.CacheHint{}})}
	req := func() *llm.Request {
		return &llm.Request{
			Model:       model,
			System:      strings.Repeat("Positional changes live fixture sentence with stable content. ", 600),
			SystemCache: &llm.CacheHint{},
			Tools:       []llm.Tool{cityTool("get_time", "Get the current time in a city.")},
			MaxTokens:   512,
			Messages:    llm.CloneMessages(history),
		}
	}
	first, err := p.Chat(ctx, req())
	if err != nil {
		t.Fatalf("turn 1 returned error: %v", err)
	}

	history = append(history, assistantMessage(first),
		llm.UserParts(llm.TextPart{Text: "Use the get_weather tool for Paris.", Cache: &llm.CacheHint{}}),
		llm.Message{Role: llm.RoleSystem, Parts: []llm.Part{
			llm.SystemDeltaPart{Section: "weather", Text: "Weather questions must be answered with the get_weather tool."},
			llm.ToolsDeltaPart{Added: []llm.Tool{cityTool("get_weather", "Get the current weather in a city.")}},
		}})
	second, err := p.Chat(ctx, req())
	if err != nil {
		t.Fatalf("turn 2 (tool added in place) returned error: %v", err)
	}
	calls := second.ToolCalls()
	if len(calls) == 0 || calls[0].Name != "get_weather" {
		t.Fatalf("turn 2 tool calls = %+v, want get_weather (added in place)", calls)
	}
	if expectCacheRead && second.Usage.CacheReadTokens == 0 {
		t.Fatalf("turn 2 read no cached prefix (turn 1 usage %+v, turn 2 usage %+v): the tool addition rewrote the prefix", first.Usage, second.Usage)
	}

	history = append(history, assistantMessage(second),
		llm.Message{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResult(calls[0].ID, "18C and sunny")}},
		llm.UserText("Thanks. Now the weather in Rome, if a weather tool is still available; otherwise reply with exactly UNAVAILABLE."),
		llm.Message{Role: llm.RoleSystem, Parts: []llm.Part{llm.ToolsDeltaPart{Removed: []string{"get_weather"}}}})
	third, err := p.Chat(ctx, req())
	if err != nil {
		t.Fatalf("turn 3 (tool removed in place) returned error: %v", err)
	}
	for _, call := range third.ToolCalls() {
		if call.Name == "get_weather" {
			t.Fatalf("turn 3 called get_weather after it was removed: %+v", third.ToolCalls())
		}
	}
}

func assistantMessage(resp *llm.Response) llm.Message {
	return llm.Message{Role: llm.RoleAssistant, Parts: llm.CloneParts(resp.Parts), Provider: resp.Provider, Model: resp.Model}
}
