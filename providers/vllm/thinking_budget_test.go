package vllm

import (
	"encoding/json"
	"errors"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func intPointer(value int) *int { return &value }

func boolPointer(value bool) *bool { return &value }

func buildThinkingBudgetBody(t *testing.T, req *llm.Request) map[string]any {
	t.Helper()
	p, err := New("http://vllm.test/v1")
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	params, err := p.inner.BuildParams(req, false)
	if err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("unmarshal params: %v", err)
	}
	return body
}

func thinkingBudgetRequest(options Options) *llm.Request {
	return &llm.Request{
		Model:           "m",
		Messages:        []llm.Message{llm.UserText("hi")},
		ProviderOptions: options,
	}
}

func TestThinkingTokenBudgetWireBehavior(t *testing.T) {
	tests := []struct {
		name      string
		budget    *int
		maxTokens int
		want      int
		wantField bool
	}{
		{name: "absent"},
		{name: "ordinary", budget: intPointer(2048), maxTokens: 4096, want: 2048, wantField: true},
		{name: "clamped", budget: intPointer(5000), maxTokens: 4096, want: 3072, wantField: true},
		{name: "no ceiling", budget: intPointer(5000), want: 5000, wantField: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := thinkingBudgetRequest(Options{ThinkingTokenBudget: tt.budget})
			req.MaxTokens = tt.maxTokens
			body := buildThinkingBudgetBody(t, req)
			got, ok := body["thinking_token_budget"]
			if ok != tt.wantField {
				t.Fatalf("thinking_token_budget present = %v, want %v; body=%v", ok, tt.wantField, body)
			}
			if ok && got != float64(tt.want) {
				t.Fatalf("thinking_token_budget = %v, want %d", got, tt.want)
			}
		})
	}
}

func TestThinkingTokenBudgetRejectsContradictions(t *testing.T) {
	tests := []struct {
		name      string
		budget    int
		maxTokens int
		effort    llm.Effort
		enable    *bool
		kwargs    map[string]any
	}{
		{name: "zero budget", budget: 0},
		{name: "negative budget", budget: -1},
		{name: "max tokens below reserve", budget: 1, maxTokens: 1000},
		{name: "max tokens at reserve", budget: 1, maxTokens: 1024},
		{name: "effort none", budget: 1, effort: llm.EffortNone},
		{name: "typed thinking disabled", budget: 1, enable: boolPointer(false)},
		{name: "generic thinking disabled", budget: 1, kwargs: map[string]any{"enable_thinking": false}},
		{name: "generic thinking non boolean", budget: 1, kwargs: map[string]any{"enable_thinking": "yes"}},
	}
	p, err := New("http://vllm.test/v1")
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := thinkingBudgetRequest(Options{
				ThinkingTokenBudget: intPointer(tt.budget),
				EnableThinking:      tt.enable,
				ChatTemplateKwargs:  tt.kwargs,
			})
			req.MaxTokens = tt.maxTokens
			req.Effort = tt.effort
			if _, err := p.inner.BuildParams(req, false); !errors.Is(err, llm.ErrBadRequest) {
				t.Fatalf("BuildParams error = %v, want ErrBadRequest", err)
			}
		})
	}
}

func TestThinkingTokenBudgetTypedPrecedenceAndStructuredOutput(t *testing.T) {
	choice := &StructuredOutputs{Choice: []string{"yes", "no"}}
	req := thinkingBudgetRequest(Options{
		ThinkingTokenBudget: intPointer(2048),
		EnableThinking:      boolPointer(true),
		ChatTemplateKwargs:  map[string]any{"enable_thinking": false},
		StructuredOutputs:   choice,
	})
	req.MaxTokens = 4096
	body := buildThinkingBudgetBody(t, req)
	if got := body["thinking_token_budget"]; got != float64(2048) {
		t.Fatalf("thinking_token_budget = %v, want 2048", got)
	}
	kwargs, ok := body["chat_template_kwargs"].(map[string]any)
	if !ok || kwargs["enable_thinking"] != true {
		t.Fatalf("chat_template_kwargs = %#v, want typed enable_thinking=true", body["chat_template_kwargs"])
	}
	structured, ok := body["structured_outputs"].(map[string]any)
	if !ok || structured["choice"] == nil {
		t.Fatalf("structured_outputs = %#v, want choice constraint", body["structured_outputs"])
	}
}
