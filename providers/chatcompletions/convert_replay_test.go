package chatcompletions_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	sdk "github.com/openai/openai-go/v3"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
)

type unsupportedParallelExtrasDialect struct{ replayDialect }

func (unsupportedParallelExtrasDialect) Capabilities() []llm.Capability {
	return []llm.Capability{
		llm.CapabilityStreaming,
		llm.CapabilityTools,
		llm.CapabilityToolStreaming,
	}
}

func (unsupportedParallelExtrasDialect) ApplyRequest(_ *llm.Request, _ *sdk.ChatCompletionNewParams, extras chatcompletions.JSONObject) error {
	extras["parallel_tool_calls"] = true
	return nil
}

func newReplayProvider(t *testing.T) *chatcompletions.Provider {
	t.Helper()
	p, err := chatcompletions.NewWithDialect(chatcompletions.Config{
		Dialect: replayDialect{},
		APIKey:  "replay-key",
	})
	if err != nil {
		t.Fatalf("chatcompletions.NewWithDialect returned error: %v", err)
	}
	return p
}

func TestBuildParamsRichRequest(t *testing.T) {
	p := newReplayProvider(t)
	temperature := 0.25
	topP := 0.9
	req := &llm.Request{
		Model:         "replay-model",
		System:        "be terse",
		SystemCache:   &llm.CacheHint{TTL: time.Hour},
		MaxTokens:     128,
		Temperature:   &temperature,
		TopP:          &topP,
		StopSequences: []string{"STOP"},
		Effort:        llm.EffortHigh,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Parts: []llm.Part{
				llm.TextPart{Text: "hello", Cache: &llm.CacheHint{TTL: time.Minute}},
				llm.ImagePart{URL: "https://example.test/red.png"},
				llm.ImagePart{Data: []byte{1, 2, 3}, MediaType: "image/png"},
				llm.UnknownPart{Type: "x/y"},
			}},
			{Role: llm.RoleAssistant, Parts: []llm.Part{
				llm.Text("calling tool"),
				llm.ToolCallPart{ID: "call_1", Name: "lookup", Args: json.RawMessage(`{"q":"go"}`)},
				llm.ReasoningPart{Provider: replayDialectName, Raw: json.RawMessage(`[{"type":"reasoning.text","text":"hmm"}]`)},
				llm.ReasoningPart{Provider: "someone-else", Raw: json.RawMessage(`[{"type":"foreign"}]`)},
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
	}
	params, err := p.BuildParams(req, true)
	if err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params returned error: %v", err)
	}

	// Full semantic wire golden (key order and whitespace insensitive): the
	// UnknownPart is skipped and the foreign-provider reasoning_details are
	// not replayed.
	want := `{
	"max_tokens": 128,
	"messages": [
		{
			"content": [
				{
					"cache_control": {
						"ttl": "1h",
						"type": "ephemeral"
					},
					"text": "be terse",
					"type": "text"
				}
			],
			"role": "system"
		},
		{
			"content": [
				{
					"cache_control": {
						"type": "ephemeral"
					},
					"text": "hello",
					"type": "text"
				},
				{
					"image_url": {
						"url": "https://example.test/red.png"
					},
					"type": "image_url"
				},
				{
					"image_url": {
						"url": "data:image/png;base64,AQID"
					},
					"type": "image_url"
				}
			],
			"role": "user"
		},
		{
			"content": "calling tool",
			"reasoning_details": [
				{
					"text": "hmm",
					"type": "reasoning.text"
				}
			],
			"role": "assistant",
			"tool_calls": [
				{
					"function": {
						"arguments": "{\"q\":\"go\"}",
						"name": "lookup"
					},
					"id": "call_1",
					"type": "function"
				}
			]
		},
		{
			"content": "{\"answer\":\"ok\"}",
			"role": "tool",
			"tool_call_id": "call_1"
		}
	],
	"model": "replay-model",
	"n": 1,
	"parallel_tool_calls": true,
	"reasoning_effort": "high",
	"stop": [
		"STOP"
	],
	"stream_options": {
		"include_usage": true
	},
	"temperature": 0.25,
	"tool_choice": {
		"function": {
			"name": "lookup"
		},
		"type": "function"
	},
	"tools": [
		{
			"function": {
				"description": "Look up a short value.",
				"name": "lookup",
				"parameters": {
					"additionalProperties": false,
					"properties": {
						"q": {
							"type": "string"
						}
					},
					"required": [
						"q"
					],
					"type": "object"
				},
				"strict": true
			},
			"type": "function"
		}
	],
	"top_p": 0.9
}`
	testutil.AssertJSONEqual(t, string(raw), want)
	if strings.Contains(string(raw), "foreign") {
		t.Fatalf("wire params leaked foreign reasoning: %s", raw)
	}
}

func TestBuildParamsToolChoiceModes(t *testing.T) {
	p := newReplayProvider(t)
	for mode, want := range map[llm.ToolChoiceMode]string{
		llm.ToolChoiceAuto:     `"auto"`,
		llm.ToolChoiceNone:     `"none"`,
		llm.ToolChoiceRequired: `"required"`,
	} {
		params, err := p.BuildParams(&llm.Request{
			Model:    "replay-model",
			Messages: []llm.Message{llm.UserText("hi")},
			Tools: []llm.Tool{{
				Name:        "lookup",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}`),
			}},
			ToolChoice: llm.ToolChoice{Mode: mode},
		}, false)
		if err != nil {
			t.Fatalf("BuildParams(%s) returned error: %v", mode, err)
		}
		raw, _ := json.Marshal(params)
		if !strings.Contains(string(raw), `"tool_choice":`+want) {
			t.Fatalf("tool choice %s missing %s: %s", mode, want, raw)
		}
	}
}

func TestBuildParamsOmitsParallelToolCallsWithoutCapability(t *testing.T) {
	p, err := chatcompletions.New("http://localhost.invalid/v1",
		chatcompletions.WithCapabilities(
			llm.CapabilityStreaming,
			llm.CapabilityTools,
			llm.CapabilityToolStreaming,
		),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	params, err := p.BuildParams(&llm.Request{
		Model:    "m",
		Messages: []llm.Message{llm.UserText("hi")},
		Tools: []llm.Tool{{
			Name:        "lookup",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
	}, true)
	if err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params returned error: %v", err)
	}
	if strings.Contains(string(raw), `"parallel_tool_calls"`) {
		t.Fatalf("parallel_tool_calls must be omitted without capability: %s", raw)
	}
}

func TestBuildParamsRejectsDialectParallelToolExtraWithoutCapability(t *testing.T) {
	p, err := chatcompletions.NewWithDialect(chatcompletions.Config{
		Dialect: unsupportedParallelExtrasDialect{},
		APIKey:  "replay-key",
	})
	if err != nil {
		t.Fatalf("NewWithDialect returned error: %v", err)
	}
	params, err := p.BuildParams(&llm.Request{
		Model:    "m",
		Messages: []llm.Message{llm.UserText("hi")},
		Tools: []llm.Tool{{
			Name:        "lookup",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		}},
	}, true)
	if err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params returned error: %v", err)
	}
	if strings.Contains(string(raw), `"parallel_tool_calls"`) {
		t.Fatalf("dialect extra bypassed unsupported capability: %s", raw)
	}
}

func TestBuildParamsResponseFormats(t *testing.T) {
	p := newReplayProvider(t)
	schema := json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}`)

	params, err := p.BuildParams(&llm.Request{
		Model:          "replay-model",
		Messages:       []llm.Message{llm.UserText("hi")},
		ResponseFormat: &llm.ResponseFormat{Type: llm.FormatJSONSchema, Name: "contact", Schema: schema, Strict: true},
	}, false)
	if err != nil {
		t.Fatalf("BuildParams(json_schema) returned error: %v", err)
	}
	raw, _ := json.Marshal(params)
	if !strings.Contains(string(raw), `"type":"json_schema"`) || !strings.Contains(string(raw), `"name":"contact"`) || !strings.Contains(string(raw), `"strict":true`) {
		t.Fatalf("json_schema response format wire = %s", raw)
	}

	params, err = p.BuildParams(&llm.Request{
		Model:          "replay-model",
		Messages:       []llm.Message{llm.UserText("hi")},
		ResponseFormat: &llm.ResponseFormat{Type: llm.FormatJSONMode},
	}, false)
	if err != nil {
		t.Fatalf("BuildParams(json_mode) returned error: %v", err)
	}
	raw, _ = json.Marshal(params)
	if !strings.Contains(string(raw), `"type":"json_object"`) {
		t.Fatalf("json_mode response format wire = %s", raw)
	}

	if _, err := p.BuildParams(&llm.Request{
		Model:          "replay-model",
		Messages:       []llm.Message{llm.UserText("hi")},
		ResponseFormat: &llm.ResponseFormat{Type: "bogus"},
	}, false); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("BuildParams(bogus format) error = %v, want ErrBadRequest", err)
	}
}

func TestBuildParamsRejectsMalformedParts(t *testing.T) {
	p := newReplayProvider(t)
	cases := []struct {
		name string
		msgs []llm.Message
		want error
	}{
		{"tool message without tool result", []llm.Message{{Role: llm.RoleTool, Parts: []llm.Part{llm.Text("nope")}}}, llm.ErrBadRequest},
		{"tool call missing id", []llm.Message{{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ToolCallPart{Name: "lookup"}}}}, llm.ErrBadRequest},
		{"tool call invalid args", []llm.Message{{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ToolCallPart{ID: "call_1", Name: "lookup", Args: json.RawMessage(`{`)}}}}, llm.ErrBadRequest},
		{"reasoning replay invalid raw", []llm.Message{{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ReasoningPart{Provider: replayDialectName, Raw: json.RawMessage(`{`)}}}}, llm.ErrBadRequest},
		{"assistant image part", []llm.Message{{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ImagePart{URL: "https://example.test/x.png"}}}}, llm.ErrUnsupported},
		{"image without url or data", []llm.Message{{Role: llm.RoleUser, Parts: []llm.Part{llm.ImagePart{}}}}, llm.ErrBadRequest},
		{"image data without media type", []llm.Message{{Role: llm.RoleUser, Parts: []llm.Part{llm.ImagePart{Data: []byte{1}}}}}, llm.ErrBadRequest},
		{"unknown role", []llm.Message{{Role: "narrator", Parts: []llm.Part{llm.Text("hi")}}}, llm.ErrBadRequest},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := p.BuildParams(&llm.Request{Model: "replay-model", Messages: tc.msgs}, false)
			if !errors.Is(err, tc.want) {
				t.Fatalf("BuildParams error = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestDefaultErrorKindTable(t *testing.T) {
	cases := []struct {
		status  int
		code    string
		message string
		want    error
	}{
		{401, "", "", llm.ErrAuth},
		{403, "", "", llm.ErrPermission},
		{404, "", "", llm.ErrNotFound},
		{408, "", "", llm.ErrTimeout},
		{429, "", "", llm.ErrRateLimited},
		{503, "", "", llm.ErrOverloaded},
		// 529 → ErrOverloaded is part of the FS §16 canonical status table
		// shared by every engine (matching anthropic and responsesapi).
		{529, "", "", llm.ErrOverloaded},
		{500, "", "", llm.ErrServer},
		{502, "", "", llm.ErrServer},
		{400, "", "", llm.ErrBadRequest},
		{402, "", "", llm.ErrInsufficientCredits},
		{400, "", "maximum context length exceeded", llm.ErrContextTooLong},
		{400, "content_filter", "", llm.ErrContentFiltered},
		{400, "invalid_api_key", "", llm.ErrAuth},
		{200, "", "", llm.ErrServer},
	}
	for _, tc := range cases {
		if got := chatcompletions.DefaultErrorKind(tc.status, tc.code, tc.message); !errors.Is(got, tc.want) {
			t.Fatalf("DefaultErrorKind(%d, %q, %q) = %v, want %v", tc.status, tc.code, tc.message, got, tc.want)
		}
	}
}

// TestBuildParamsPointerPartsMatchValueParts pins the value-type doctrine at
// the adapter boundary: pointer parts build byte-identical wire params to
// their value forms (providerutil.DerefPart normalizes on entry).
func TestBuildParamsPointerPartsMatchValueParts(t *testing.T) {
	p := newReplayProvider(t)
	valueReq := &llm.Request{
		Model: "replay-model",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Parts: []llm.Part{
				llm.TextPart{Text: "hello"},
				llm.ImagePart{URL: "https://example.test/red.png"},
			}},
			{Role: llm.RoleAssistant, Parts: []llm.Part{
				llm.TextPart{Text: "calling"},
				llm.ToolCallPart{ID: "call_1", Name: "lookup", Args: json.RawMessage(`{"q":"go"}`)},
				llm.ReasoningPart{Provider: replayDialectName, Raw: json.RawMessage(`[{"type":"reasoning.text","text":"hmm"}]`)},
			}},
			{Role: llm.RoleTool, Parts: []llm.Part{
				llm.ToolResultPart{ToolCallID: "call_1", Content: []llm.Part{llm.TextPart{Text: "ok"}}},
			}},
		},
		Tools: []llm.Tool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}},
	}
	pointerReq := &llm.Request{
		Model: "replay-model",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Parts: []llm.Part{
				&llm.TextPart{Text: "hello"},
				&llm.ImagePart{URL: "https://example.test/red.png"},
			}},
			{Role: llm.RoleAssistant, Parts: []llm.Part{
				&llm.TextPart{Text: "calling"},
				&llm.ToolCallPart{ID: "call_1", Name: "lookup", Args: json.RawMessage(`{"q":"go"}`)},
				&llm.ReasoningPart{Provider: replayDialectName, Raw: json.RawMessage(`[{"type":"reasoning.text","text":"hmm"}]`)},
			}},
			{Role: llm.RoleTool, Parts: []llm.Part{
				&llm.ToolResultPart{ToolCallID: "call_1", Content: []llm.Part{&llm.TextPart{Text: "ok"}}},
			}},
		},
		Tools: []llm.Tool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}},
	}

	valueParams, err := p.BuildParams(valueReq, false)
	if err != nil {
		t.Fatalf("BuildParams(value) returned error: %v", err)
	}
	pointerParams, err := p.BuildParams(pointerReq, false)
	if err != nil {
		t.Fatalf("BuildParams(pointer) returned error: %v", err)
	}
	valueJSON, err := json.Marshal(valueParams)
	if err != nil {
		t.Fatalf("marshal value params: %v", err)
	}
	pointerJSON, err := json.Marshal(pointerParams)
	if err != nil {
		t.Fatalf("marshal pointer params: %v", err)
	}
	if string(valueJSON) != string(pointerJSON) {
		t.Fatalf("pointer params = %s\nwant value params = %s", pointerJSON, valueJSON)
	}
}

// TestToolResultImageStaysUnsupported pins the deliberate asymmetry with the
// Responses adapter (B5): the chat-completions wire accepts only string
// content for role:"tool" messages, so images in tool results keep failing
// loudly with ErrUnsupported here.
func TestToolResultImageStaysUnsupported(t *testing.T) {
	p := newReplayProvider(t)
	_, err := p.BuildParams(&llm.Request{
		Model: "replay-model",
		Messages: []llm.Message{
			{Role: llm.RoleTool, Parts: []llm.Part{
				llm.ToolResultParts("call_1", llm.ImageData([]byte{1, 2, 3}, "image/png")),
			}},
		},
	}, false)
	if !errors.Is(err, llm.ErrUnsupported) {
		t.Fatalf("tool-result image error = %v, want ErrUnsupported", err)
	}
}
