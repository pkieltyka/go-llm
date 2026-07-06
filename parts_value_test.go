package llm_test

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestPointerPartsNormalizeOnEntry pins the Part value-type doctrine: parts
// passed as pointers behave exactly like their value forms end-to-end — a
// Chat request build (History clone + provider request path) and the
// canonical serialize round-trip both normalize them to value parts.
func TestPointerPartsNormalizeOnEntry(t *testing.T) {
	cases := []struct {
		name    string
		pointer llm.Part
		value   llm.Part
	}{
		{"text", &llm.TextPart{Text: "hello"}, llm.TextPart{Text: "hello"}},
		{"image", &llm.ImagePart{URL: "https://example.test/red.png"}, llm.ImagePart{URL: "https://example.test/red.png"}},
		{"file", &llm.FilePart{URL: "https://example.test/doc.pdf", MediaType: "application/pdf"}, llm.FilePart{URL: "https://example.test/doc.pdf", MediaType: "application/pdf"}},
		{"tool_call", &llm.ToolCallPart{ID: "call_1", Name: "lookup", Args: json.RawMessage(`{"q":"go"}`)}, llm.ToolCallPart{ID: "call_1", Name: "lookup", Args: json.RawMessage(`{"q":"go"}`)}},
		{
			"tool_result",
			&llm.ToolResultPart{ToolCallID: "call_1", Name: "lookup", Content: []llm.Part{&llm.TextPart{Text: "ok"}}},
			llm.ToolResultPart{ToolCallID: "call_1", Name: "lookup", Content: []llm.Part{llm.TextPart{Text: "ok"}}},
		},
		{"reasoning", &llm.ReasoningPart{Text: "hmm", Raw: json.RawMessage(`{"sig":"x"}`), Provider: "fake"}, llm.ReasoningPart{Text: "hmm", Raw: json.RawMessage(`{"sig":"x"}`), Provider: "fake"}},
		{"unknown", &llm.UnknownPart{Type: "x/y", Data: json.RawMessage(`{"type":"x/y"}`)}, llm.UnknownPart{Type: "x/y", Data: json.RawMessage(`{"type":"x/y"}`)}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			role := llm.RoleUser
			if tc.name == "tool_result" {
				role = llm.RoleTool
			}

			// Serialize round-trip: pointer and value forms marshal to
			// identical bytes, and unmarshal to the value form.
			pointerMsg := llm.Message{Role: role, Parts: []llm.Part{tc.pointer}}
			valueMsg := llm.Message{Role: role, Parts: []llm.Part{tc.value}}
			pointerRaw, err := llm.MarshalMessage(pointerMsg)
			if err != nil {
				t.Fatalf("MarshalMessage(pointer) returned error: %v", err)
			}
			valueRaw, err := llm.MarshalMessage(valueMsg)
			if err != nil {
				t.Fatalf("MarshalMessage(value) returned error: %v", err)
			}
			if !bytes.Equal(pointerRaw, valueRaw) {
				t.Fatalf("pointer marshal = %s, value marshal = %s", pointerRaw, valueRaw)
			}
			decoded, err := llm.UnmarshalMessage(pointerRaw)
			if err != nil {
				t.Fatalf("UnmarshalMessage returned error: %v", err)
			}
			redecoded, err := llm.MarshalMessage(decoded)
			if err != nil {
				t.Fatalf("MarshalMessage(decoded) returned error: %v", err)
			}
			if !bytes.Equal(redecoded, valueRaw) {
				t.Fatalf("round-trip marshal = %s, want %s", redecoded, valueRaw)
			}

			// Chat request build: a pointer part flows through History and
			// the provider Chat path without error, and the recorded request
			// serializes identically to the value form.
			p := llmtest.New(llmtest.WithName("fake"))
			p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text("ok")}})
			h := llm.NewHistory()
			h.Add(pointerMsg)
			if _, err := p.Chat(context.Background(), &llm.Request{
				Model:    "model-a",
				Messages: h.Messages(),
			}); err != nil {
				t.Fatalf("Chat returned error: %v", err)
			}
			requests := p.Requests()
			if len(requests) != 1 || len(requests[0].Messages) != 1 {
				t.Fatalf("recorded requests = %+v", requests)
			}
			sentRaw, err := llm.MarshalMessage(requests[0].Messages[0])
			if err != nil {
				t.Fatalf("MarshalMessage(sent) returned error: %v", err)
			}
			if !bytes.Equal(sentRaw, valueRaw) {
				t.Fatalf("request-built message = %s, want %s", sentRaw, valueRaw)
			}
		})
	}
}

// TestPointerPartsResponseAccessors covers the deref path in Response's
// convenience accessors.
func TestPointerPartsResponseAccessors(t *testing.T) {
	resp := &llm.Response{Parts: []llm.Part{
		&llm.TextPart{Text: "hello"},
		&llm.ReasoningPart{Text: "why"},
		&llm.ToolCallPart{ID: "call_1", Name: "lookup", Args: json.RawMessage(`{}`)},
	}}
	if resp.Text() != "hello" || resp.Reasoning() != "why" {
		t.Fatalf("accessors = (%q, %q)", resp.Text(), resp.Reasoning())
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call_1" {
		t.Fatalf("ToolCalls = %+v", calls)
	}
}
