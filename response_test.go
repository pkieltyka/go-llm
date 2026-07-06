package llm

import "testing"

func TestResponseAccessorsHandlePointerParts(t *testing.T) {
	resp := &Response{
		Parts: []Part{
			&TextPart{Text: "hello"},
			&ReasoningPart{Text: "because"},
			&ToolCallPart{ID: "call_1", Name: "lookup", Args: []byte(`{"q":"go"}`)},
		},
	}

	if got := resp.Text(); got != "hello" {
		t.Fatalf("Text() = %q, want hello", got)
	}
	if got := resp.Reasoning(); got != "because" {
		t.Fatalf("Reasoning() = %q, want because", got)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(calls))
	}
	if calls[0].ID != "call_1" || calls[0].Name != "lookup" || string(calls[0].Args) != `{"q":"go"}` {
		t.Fatalf("tool call = %+v", calls[0])
	}

	calls[0].Args[0] = '['
	if got := string(resp.Parts[2].(*ToolCallPart).Args); got != `{"q":"go"}` {
		t.Fatalf("stored args = %q, want original JSON", got)
	}
}
