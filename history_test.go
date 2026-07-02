package llm

import "testing"

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
		ToolResultWithName("call_1", "lookup", "result one"),
		ToolResultPartsWithName("call_2", "search", Text("result two")),
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

func TestHistoryForeignReasoningAsTextOption(t *testing.T) {
	h := NewHistory(WithForeignReasoningAsText())
	if !h.ForeignReasoningAsText() {
		t.Fatalf("ForeignReasoningAsText() = false, want true")
	}
}
