package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

type recordingToolResultSink struct {
	calls   int
	results []llm.ToolResultPart
}

func (s *recordingToolResultSink) AddToolResults(results ...llm.ToolResultPart) {
	s.calls++
	s.results = append(s.results, results...)
}

func TestDispatchToolCallFailuresBecomeErrorResults(t *testing.T) {
	t.Parallel()

	tool := llm.Tool{
		Name: "weather",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"city": map[string]any{"type": "string"}},
			"required":   []string{"city"},
		},
	}
	tools := map[string]toolHandler{
		"weather": {
			tool: tool,
			execute: func(context.Context, json.RawMessage) (string, error) {
				return "", errors.New("station unavailable")
			},
		},
	}

	tests := []struct {
		name string
		call llm.ToolCallPart
		want string
	}{
		{name: "unknown tool", call: llm.ToolCall("unknown", "search", json.RawMessage(`{}`)), want: `unknown tool "search"`},
		{name: "malformed arguments", call: llm.ToolCall("malformed", "weather", json.RawMessage(`{"city":`)), want: "invalid arguments"},
		{name: "schema violation", call: llm.ToolCall("schema", "weather", json.RawMessage(`{"city":42}`)), want: "invalid arguments"},
		{name: "execution error", call: llm.ToolCall("execution", "weather", json.RawMessage(`{"city":"Toronto"}`)), want: "execution failed: station unavailable"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := dispatchToolCall(context.Background(), tools, test.call)
			if !result.IsError {
				t.Fatalf("IsError = false, want true: %#v", result)
			}
			if result.ToolCallID != test.call.ID || result.Name != test.call.Name {
				t.Fatalf("result identity = %q/%q, want %q/%q", result.ToolCallID, result.Name, test.call.ID, test.call.Name)
			}
			if got := toolResultText(t, result); !strings.Contains(got, test.want) {
				t.Fatalf("result text = %q, want substring %q", got, test.want)
			}
		})
	}
}

func TestDispatchToolCallsGroupsParallelResults(t *testing.T) {
	t.Parallel()

	tool := llm.Tool{
		Name: "echo",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"value": map[string]any{"type": "string"}},
			"required":   []string{"value"},
		},
	}
	tools := map[string]toolHandler{
		"echo": {
			tool: tool,
			execute: func(_ context.Context, args json.RawMessage) (string, error) {
				return string(args), nil
			},
		},
	}
	calls := []llm.ToolCallPart{
		llm.ToolCall("call_1", "echo", json.RawMessage(`{"value":"one"}`)),
		llm.ToolCall("call_2", "echo", json.RawMessage(`{"value":"two"}`)),
	}
	sink := new(recordingToolResultSink)

	dispatchToolCalls(context.Background(), sink, tools, calls)

	if sink.calls != 1 {
		t.Fatalf("AddToolResults calls = %d, want 1", sink.calls)
	}
	if len(sink.results) != len(calls) {
		t.Fatalf("result count = %d, want %d", len(sink.results), len(calls))
	}
	for i, result := range sink.results {
		if result.IsError || result.ToolCallID != calls[i].ID || result.Name != calls[i].Name {
			t.Fatalf("result[%d] = %#v, want successful result for %#v", i, result, calls[i])
		}
	}
}

func toolResultText(t *testing.T, result llm.ToolResultPart) string {
	t.Helper()
	if len(result.Content) != 1 {
		t.Fatalf("result content count = %d, want 1", len(result.Content))
	}
	text, ok := result.Content[0].(llm.TextPart)
	if !ok {
		t.Fatalf("result content type = %T, want llm.TextPart", result.Content[0])
	}
	return text.Text
}
