package llm

import (
	"encoding/json"
	"testing"
)

func TestHistory(t *testing.T) {
	var h History
	h.AddUserText("hello")

	resp := &Response{
		Provider: "openai",
		Model:    "gpt-test",
		Parts: []Part{
			ReasoningPart{Text: "thinking", Raw: []byte(`{"encrypted":"payload"}`), Provider: "openai"},
			Text("I can call a tool."),
			ToolCall("call_1", "lookup", []byte(`{"q":"go"}`)),
		},
	}
	h.AddResponse(resp)
	h.AddToolResults(
		ToolResultPart{ToolCallID: "call_1", Name: "lookup", Content: []Part{Text("result one")}},
		ToolResultPart{ToolCallID: "call_2", Name: "search", Content: []Part{Text("result two")}},
	)

	msgs := h.Messages()
	if len(msgs) != 3 {
		t.Fatalf("history len = %d, want 3", len(msgs))
	}
	if msgs[0].Role != RoleUser || msgs[1].Role != RoleAssistant || msgs[2].Role != RoleTool {
		t.Fatalf("roles = %q, %q, %q", msgs[0].Role, msgs[1].Role, msgs[2].Role)
	}
	if msgs[1].Provider != "openai" || msgs[1].Model != "gpt-test" {
		t.Fatalf("assistant provenance = (%q, %q), want (openai, gpt-test)", msgs[1].Provider, msgs[1].Model)
	}
	if len(msgs[2].Parts) != 2 {
		t.Fatalf("tool result group len = %d, want 2", len(msgs[2].Parts))
	}
	firstResult, ok := msgs[2].Parts[0].(ToolResultPart)
	if !ok {
		t.Fatalf("tool message part type = %T, want ToolResultPart", msgs[2].Parts[0])
	}
	if firstResult.ToolCallID != "call_1" || firstResult.Name != "lookup" {
		t.Fatalf("tool result = %+v, want call_1/lookup", firstResult)
	}

	msgs[0].Parts[0] = Text("mutated")
	call := msgs[1].Parts[2].(ToolCallPart)
	call.Args[0] = '['
	msgs[1].Parts[2] = call

	again := h.Messages()
	if got := again[0].Parts[0].(TextPart).Text; got != "hello" {
		t.Fatalf("stored user text = %q, want hello", got)
	}
	if got := string(again[1].Parts[2].(ToolCallPart).Args); got != `{"q":"go"}` {
		t.Fatalf("stored tool args = %q, want original JSON", got)
	}
}

// A tool a ToolsDeltaPart adds is owned by the history: mutating its schema
// through a snapshot or the original never reaches the stored message.
func TestHistoryOwnsToolsDeltaSchemas(t *testing.T) {
	raw := json.RawMessage(`{"type":"object"}`)
	shaped := map[string]any{"type": "object", "required": []any{"q"}}
	var h History
	h.Add(Message{Role: RoleSystem, Parts: []Part{ToolsDeltaPart{Added: []Tool{
		{Name: "raw", InputSchema: raw},
		{Name: "shaped", InputSchema: shaped},
	}}}})
	raw[0] = '!'
	shaped["type"] = "mutated"
	snapshot := h.Messages()[0].Parts[0].(ToolsDeltaPart)
	snapshot.Added[0].InputSchema.(json.RawMessage)[0] = '!'
	snapshot.Added[1].InputSchema.(map[string]any)["required"].([]any)[0] = "mutated"

	stored := h.Messages()[0].Parts[0].(ToolsDeltaPart)
	if got := string(stored.Added[0].InputSchema.(json.RawMessage)); got != `{"type":"object"}` {
		t.Fatalf("raw schema = %s", got)
	}
	schema := stored.Added[1].InputSchema.(map[string]any)
	if schema["type"] != "object" || schema["required"].([]any)[0] != "q" {
		t.Fatalf("JSON-shaped schema = %v", schema)
	}
}
