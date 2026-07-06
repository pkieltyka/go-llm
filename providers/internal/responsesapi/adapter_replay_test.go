package responsesapi_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func marshalWire(t *testing.T, v any) string {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal params returned error: %v", err)
	}
	return string(raw)
}

func TestBuildParamsRichRequest(t *testing.T) {
	adapter := replayAdapter()
	temperature := 0.25
	topP := 0.9
	req := &llm.Request{
		Model:       "replay-model",
		System:      "be terse",
		MaxTokens:   128,
		Temperature: &temperature,
		TopP:        &topP,
		SessionID:   "sess_replay",
		Effort:      llm.EffortHigh,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Parts: []llm.Part{
				llm.Text("hello"),
				llm.ImagePart{URL: "https://example.test/red.png"},
				llm.ImagePart{Data: []byte{1, 2, 3}, MediaType: "image/png"},
				llm.FilePart{Data: []byte{4, 5, 6}, MediaType: "application/pdf"},
				llm.UnknownPart{Type: "x/y"},
			}},
			{Role: llm.RoleAssistant, Parts: []llm.Part{
				llm.Text("calling tool"),
				llm.ToolCallPart{ID: "call_1", Name: "lookup", Args: json.RawMessage(`{"q":"go"}`)},
				llm.ReasoningPart{Provider: replayAdapterName, Raw: json.RawMessage(`{"type":"reasoning","id":"rs_1","summary":[]}`)},
				llm.ReasoningPart{Provider: "someone-else", Raw: json.RawMessage(`{"type":"foreign"}`)},
			}},
			{Role: llm.RoleTool, Parts: []llm.Part{
				llm.ToolResultPart{ToolCallID: "call_1", Name: "lookup", Content: []llm.Part{llm.Text(`{"answer":"ok"}`)}},
			}},
		},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Description: "Look up a short value.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}`),
			Strict:      true,
		}},
		ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "lookup"},
		ResponseFormat: &llm.ResponseFormat{
			Type:   llm.FormatJSONSchema,
			Name:   "contact",
			Schema: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}`),
			Strict: true,
		},
	}
	params, err := adapter.BuildParams(req, true)
	if err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	wire := marshalWire(t, params)
	for _, want := range []string{
		`"model":"replay-model"`,
		`"instructions":"be terse"`,
		`"max_output_tokens":128`,
		`"temperature":0.25`,
		`"top_p":0.9`,
		`"prompt_cache_key":"sess_replay"`,
		`"store":false`,
		`"include":["reasoning.encrypted_content"]`,
		`"reasoning":{"effort":"high","summary":"auto"}`,
		`"type":"input_image"`,
		`"image_url":"data:image/png;base64,AQID"`,
		`"type":"input_file"`,
		`"type":"function_call"`,
		`"call_id":"call_1"`,
		`"type":"function_call_output"`,
		`"type":"reasoning","id":"rs_1"`,
		`"tool_choice":{"name":"lookup","type":"function"}`,
		`"name":"contact"`,
	} {
		if !strings.Contains(wire, want) {
			t.Fatalf("wire params missing %s in:\n%s", want, wire)
		}
	}
	if strings.Contains(wire, "foreign") {
		t.Fatalf("wire params leaked foreign reasoning: %s", wire)
	}
}

func TestBuildParamsEffortAndToolChoiceModes(t *testing.T) {
	adapter := replayAdapter()
	for effort, want := range map[llm.Effort]string{
		llm.EffortNone:    `"effort":"none"`,
		llm.EffortMinimal: `"effort":"minimal"`,
		llm.EffortLow:     `"effort":"low"`,
		llm.EffortMedium:  `"effort":"medium"`,
		llm.EffortXHigh:   `"effort":"xhigh"`,
		llm.EffortMax:     `"effort":"xhigh"`,
	} {
		params, err := adapter.BuildParams(&llm.Request{
			Model:    "replay-model",
			Effort:   effort,
			Messages: []llm.Message{llm.UserText("hi")},
		}, false)
		if err != nil {
			t.Fatalf("BuildParams(effort=%s) returned error: %v", effort, err)
		}
		if wire := marshalWire(t, params); !strings.Contains(wire, want) {
			t.Fatalf("effort %s wire missing %s: %s", effort, want, wire)
		}
	}

	tool := llm.Tool{
		Name:        "lookup",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}`),
	}
	for mode, want := range map[llm.ToolChoiceMode]string{
		llm.ToolChoiceAuto:     `"tool_choice":"auto"`,
		llm.ToolChoiceNone:     `"tool_choice":"none"`,
		llm.ToolChoiceRequired: `"tool_choice":"required"`,
	} {
		params, err := adapter.BuildParams(&llm.Request{
			Model:      "replay-model",
			Messages:   []llm.Message{llm.UserText("hi")},
			Tools:      []llm.Tool{tool},
			ToolChoice: llm.ToolChoice{Mode: mode},
		}, false)
		if err != nil {
			t.Fatalf("BuildParams(tool choice %s) returned error: %v", mode, err)
		}
		if wire := marshalWire(t, params); !strings.Contains(wire, want) {
			t.Fatalf("tool choice %s wire missing %s: %s", mode, want, wire)
		}
	}
}

func TestBuildParamsRejectsMalformedParts(t *testing.T) {
	adapter := replayAdapter()
	cases := []struct {
		name string
		msgs []llm.Message
		want error
	}{
		{"tool message without tool result", []llm.Message{{Role: llm.RoleTool, Parts: []llm.Part{llm.Text("nope")}}}, llm.ErrBadRequest},
		{"tool call missing id", []llm.Message{{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ToolCallPart{Name: "lookup"}}}}, llm.ErrBadRequest},
		{"tool call invalid args", []llm.Message{{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ToolCallPart{ID: "call_1", Name: "lookup", Args: json.RawMessage(`{`)}}}}, llm.ErrBadRequest},
		{"tool result missing call id", []llm.Message{{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResultPart{Content: []llm.Part{llm.Text("ok")}}}}}, llm.ErrBadRequest},
		{"reasoning replay invalid raw", []llm.Message{{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ReasoningPart{Provider: replayAdapterName, Raw: json.RawMessage(`{`)}}}}, llm.ErrBadRequest},
		{"assistant image part", []llm.Message{{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ImagePart{URL: "https://example.test/x.png"}}}}, llm.ErrUnsupported},
		{"image without url or data", []llm.Message{{Role: llm.RoleUser, Parts: []llm.Part{llm.ImagePart{}}}}, llm.ErrBadRequest},
		{"image data without media type", []llm.Message{{Role: llm.RoleUser, Parts: []llm.Part{llm.ImagePart{Data: []byte{1}}}}}, llm.ErrBadRequest},
		{"file with unsupported media type", []llm.Message{{Role: llm.RoleUser, Parts: []llm.Part{llm.FilePart{Data: []byte{1}, MediaType: "text/csv"}}}}, llm.ErrUnsupported},
		{"file without url or data", []llm.Message{{Role: llm.RoleUser, Parts: []llm.Part{llm.FilePart{MediaType: "application/pdf"}}}}, llm.ErrBadRequest},
		{"unknown role", []llm.Message{{Role: "narrator", Parts: []llm.Part{llm.Text("hi")}}}, llm.ErrBadRequest},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := adapter.BuildParams(&llm.Request{Model: "replay-model", Messages: tc.msgs}, false)
			if !errors.Is(err, tc.want) {
				t.Fatalf("BuildParams error = %v, want %v", err, tc.want)
			}
		})
	}
}

// TestBuildInputPointerPartsMatchValueParts pins the value-type doctrine at
// the Responses adapter boundary: pointer parts build byte-identical input
// items to their value forms (providerutil.DerefPart normalizes on entry).
func TestBuildInputPointerPartsMatchValueParts(t *testing.T) {
	adapter := replayAdapter()
	valueMsgs := []llm.Message{
		{Role: llm.RoleUser, Parts: []llm.Part{
			llm.TextPart{Text: "hello"},
			llm.ImagePart{URL: "https://example.test/red.png"},
			llm.FilePart{Data: []byte{4, 5, 6}, MediaType: "application/pdf"},
		}},
		{Role: llm.RoleAssistant, Parts: []llm.Part{
			llm.TextPart{Text: "calling"},
			llm.ToolCallPart{ID: "call_1", Name: "lookup", Args: json.RawMessage(`{"q":"go"}`)},
			llm.ReasoningPart{Provider: replayAdapterName, Raw: json.RawMessage(`{"type":"reasoning","id":"rs_1"}`)},
		}},
		{Role: llm.RoleTool, Parts: []llm.Part{
			llm.ToolResultPart{ToolCallID: "call_1", Content: []llm.Part{llm.TextPart{Text: "ok"}}},
		}},
	}
	pointerMsgs := []llm.Message{
		{Role: llm.RoleUser, Parts: []llm.Part{
			&llm.TextPart{Text: "hello"},
			&llm.ImagePart{URL: "https://example.test/red.png"},
			&llm.FilePart{Data: []byte{4, 5, 6}, MediaType: "application/pdf"},
		}},
		{Role: llm.RoleAssistant, Parts: []llm.Part{
			&llm.TextPart{Text: "calling"},
			&llm.ToolCallPart{ID: "call_1", Name: "lookup", Args: json.RawMessage(`{"q":"go"}`)},
			&llm.ReasoningPart{Provider: replayAdapterName, Raw: json.RawMessage(`{"type":"reasoning","id":"rs_1"}`)},
		}},
		{Role: llm.RoleTool, Parts: []llm.Part{
			&llm.ToolResultPart{ToolCallID: "call_1", Content: []llm.Part{&llm.TextPart{Text: "ok"}}},
		}},
	}

	valueInput, err := adapter.BuildInput(valueMsgs)
	if err != nil {
		t.Fatalf("BuildInput(value) returned error: %v", err)
	}
	pointerInput, err := adapter.BuildInput(pointerMsgs)
	if err != nil {
		t.Fatalf("BuildInput(pointer) returned error: %v", err)
	}
	if got, want := marshalWire(t, pointerInput), marshalWire(t, valueInput); got != want {
		t.Fatalf("pointer input = %s\nwant value input = %s", got, want)
	}
}

// TestToolResultWithImageBuildsContentArray covers B5: images (and PDF
// files) inside a ToolResultPart map to the Responses function_call_output
// content-array form (input_text / input_image / input_file) instead of
// failing with ErrUnsupported — screenshot-returning tools work on OpenAI.
func TestToolResultWithImageBuildsContentArray(t *testing.T) {
	adapter := replayAdapter()
	input, err := adapter.BuildInput([]llm.Message{
		{Role: llm.RoleTool, Parts: []llm.Part{
			llm.ToolResultParts("call_1",
				llm.Text("screenshot attached"),
				llm.ImageData([]byte{1, 2, 3}, "image/png"),
				llm.FileData([]byte{4, 5, 6}, "application/pdf", "doc.pdf"),
			),
		}},
	})
	if err != nil {
		t.Fatalf("BuildInput returned error: %v", err)
	}
	wire := marshalWire(t, input)
	for _, want := range []string{
		`"type":"function_call_output"`,
		`"call_id":"call_1"`,
		`"type":"input_text"`,
		`"text":"screenshot attached"`,
		`"type":"input_image"`,
		`"image_url":"data:image/png;base64,AQID"`,
		`"type":"input_file"`,
		`"filename":"doc.pdf"`,
	} {
		if !strings.Contains(wire, want) {
			t.Fatalf("tool-result content wire missing %s in:\n%s", want, wire)
		}
	}

	// Text-only results keep the plain string output form.
	textOnly, err := adapter.BuildInput([]llm.Message{
		{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResult("call_1", "plain")}},
	})
	if err != nil {
		t.Fatalf("BuildInput(text-only) returned error: %v", err)
	}
	if wire := marshalWire(t, textOnly); !strings.Contains(wire, `"output":"plain"`) {
		t.Fatalf("text-only tool result should use string output: %s", wire)
	}

	// Non-PDF files inside tool results still error loudly.
	if _, err := adapter.BuildInput([]llm.Message{
		{Role: llm.RoleTool, Parts: []llm.Part{
			llm.ToolResultParts("call_1", llm.FileData([]byte{1}, "text/csv", "x.csv")),
		}},
	}); !errors.Is(err, llm.ErrUnsupported) {
		t.Fatalf("non-PDF tool-result file error = %v, want ErrUnsupported", err)
	}
}
