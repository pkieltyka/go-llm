package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
)

func TestOpenRouterBuildRequestGolden(t *testing.T) {
	temp := 0.2
	topP := 0.8
	topK := 20
	minP := 0.1
	p, err := New(WithAPIKey("test"))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	params, err := p.buildParams(&llm.Request{
		Model:       "openai/gpt-test",
		System:      "You are terse.",
		SystemCache: &llm.CacheHint{TTL: time.Hour},
		MaxTokens:   99,
		Temperature: &temp,
		TopP:        &topP,
		StopSequences: []string{
			"STOP",
		},
		SessionID: "session_1",
		Messages: []llm.Message{
			llm.UserParts(llm.Text("hello"), llm.ImageData([]byte("png"), "image/png")),
			llm.AssistantParts(
				llm.ReasoningPart{Provider: providerName, Raw: json.RawMessage(`[{"type":"reasoning.text","text":"why"}]`)},
				llm.ReasoningPart{Provider: "openai", Raw: json.RawMessage(`{"encrypted_content":"foreign"}`)},
				llm.Text("previous"),
				llm.ToolCall("call_1", "lookup", json.RawMessage(`{"q":"go"}`)),
				llm.ToolCall("call_2", "lookup", json.RawMessage(`{"q":"rust"}`)),
			),
			{Role: llm.RoleTool, Parts: []llm.Part{
				llm.ToolResultPart{ToolCallID: "call_1", Name: "lookup", Content: []llm.Part{llm.Text("result 1")}},
				llm.ToolResultPart{ToolCallID: "call_2", Name: "lookup", Content: []llm.Part{llm.Text("result 2")}},
			}},
		},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Description: "Look up a value.",
			Strict:      true,
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string","format":"uri"}},"required":["q"],"additionalProperties":false}`),
		}},
		ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "lookup"},
		ResponseFormat: &llm.ResponseFormat{
			Type:   llm.FormatJSONSchema,
			Name:   "answer",
			Strict: true,
			Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string","pattern":"^ok$"}},"required":["answer"]}`),
		},
		Effort: llm.EffortHigh,
		ProviderOptions: Options{
			Models:   []string{"anthropic/claude-test", "openai/gpt-test"},
			Provider: map[string]any{"require_parameters": true},
			Plugins:  []any{map[string]any{"id": "web"}},
			TopK:     &topK,
			MinP:     &minP,
		},
	}, true)
	if err != nil {
		t.Fatalf("buildParams returned error: %v", err)
	}
	got := testutil.MustCompactJSON(t, params)
	want := `{
	"messages": [
		{
			"content": [
				{
					"cache_control": {
						"ttl": "1h",
						"type": "ephemeral"
					},
					"text": "You are terse.",
					"type": "text"
				}
			],
			"role": "system"
		},
		{
			"content": [
				{
					"text": "hello",
					"type": "text"
				},
				{
					"image_url": {
						"url": "data:image/png;base64,cG5n"
					},
					"type": "image_url"
				}
			],
			"role": "user"
		},
		{
			"content": "previous",
			"reasoning_details": [
				{
					"text": "why",
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
				},
				{
					"function": {
						"arguments": "{\"q\":\"rust\"}",
						"name": "lookup"
					},
					"id": "call_2",
					"type": "function"
				}
			]
		},
		{
			"content": "result 1",
			"role": "tool",
			"tool_call_id": "call_1"
		},
		{
			"content": "result 2",
			"role": "tool",
			"tool_call_id": "call_2"
		}
	],
	"model": "openai/gpt-test",
	"max_tokens": 99,
	"n": 1,
	"temperature": 0.2,
	"top_p": 0.8,
	"parallel_tool_calls": true,
	"stop": [
		"STOP"
	],
	"stream_options": {
		"include_usage": true
	},
	"response_format": {
		"json_schema": {
			"name": "answer",
			"schema": {
				"properties": {
					"answer": {
						"pattern": "^ok$",
						"type": "string"
					}
				},
				"required": [
					"answer"
				],
				"type": "object"
			},
			"strict": false
		},
		"type": "json_schema"
	},
	"tool_choice": {
		"function": {
			"name": "lookup"
		},
		"type": "function"
	},
	"tools": [
		{
			"function": {
				"description": "Look up a value.",
				"name": "lookup",
				"parameters": {
					"additionalProperties": false,
					"properties": {
						"q": {
							"format": "uri",
							"type": "string"
						}
					},
					"required": [
						"q"
					],
					"type": "object"
				},
				"strict": false
			},
			"type": "function"
		}
	],
	"min_p": 0.1,
	"models": [
		"anthropic/claude-test",
		"openai/gpt-test"
	],
	"plugins": [
		{
			"id": "web"
		}
	],
	"provider": {
		"require_parameters": true
	},
	"reasoning": {
		"effort": "high"
	},
	"session_id": "session_1",
	"top_k": 20
}`
	testutil.AssertJSONEqual(t, got, want)
	if strings.Contains(got, "foreign") {
		t.Fatalf("foreign reasoning was replayed: %s", got)
	}
}

func TestOpenRouterEffortMapping(t *testing.T) {
	// FS §9 OpenRouter column: every unified level passes through verbatim as
	// reasoning.effort; none maps to reasoning.enabled=false; empty sends
	// nothing.
	tests := []struct {
		effort llm.Effort
		want   string // "" = no reasoning field
	}{
		{effort: "", want: ""},
		{effort: llm.EffortNone, want: `{"enabled":false}`},
		{effort: llm.EffortMinimal, want: `{"effort":"minimal"}`},
		{effort: llm.EffortLow, want: `{"effort":"low"}`},
		{effort: llm.EffortMedium, want: `{"effort":"medium"}`},
		{effort: llm.EffortHigh, want: `{"effort":"high"}`},
		{effort: llm.EffortXHigh, want: `{"effort":"xhigh"}`},
		{effort: llm.EffortMax, want: `{"effort":"max"}`},
	}
	p, err := New(WithAPIKey("test"))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	for _, tt := range tests {
		name := string(tt.effort)
		if name == "" {
			name = "default"
		}
		t.Run(name, func(t *testing.T) {
			params, err := p.buildParams(&llm.Request{
				Model:    "openai/gpt-test",
				Effort:   tt.effort,
				Messages: []llm.Message{llm.UserText("hello")},
			}, false)
			if err != nil {
				t.Fatalf("buildParams returned error: %v", err)
			}
			var body struct {
				Reasoning json.RawMessage `json:"reasoning"`
			}
			if err := json.Unmarshal([]byte(testutil.MustCompactJSON(t, params)), &body); err != nil {
				t.Fatalf("params are invalid JSON: %v", err)
			}
			if tt.want == "" {
				if body.Reasoning != nil {
					t.Fatalf("reasoning = %s, want absent", body.Reasoning)
				}
				return
			}
			testutil.AssertJSONEqual(t, string(body.Reasoning), tt.want)
		})
	}
}

func TestOpenRouterStrictToolSchemaDefsCarveOut(t *testing.T) {
	p, err := New(WithAPIKey("test"))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	// A $defs entry NAMED "format" is a definition name, not the "format"
	// keyword: strict stays on.
	params, err := p.buildParams(&llm.Request{
		Model:    "openai/gpt-test",
		Messages: []llm.Message{llm.UserText("hello")},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Strict:      true,
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"$ref":"#/$defs/format"}},"required":["q"],"additionalProperties":false,"$defs":{"format":{"type":"string"}}}`),
		}},
	}, false)
	if err != nil {
		t.Fatalf("buildParams returned error: %v", err)
	}
	if !strings.Contains(testutil.MustCompactJSON(t, params), `"strict":true`) {
		t.Fatalf("strict = false for schema whose $defs entry is named format:\n%s", testutil.MustCompactJSON(t, params))
	}

	// A keyword inside a $defs subschema still disables strict mode.
	params, err = p.buildParams(&llm.Request{
		Model:    "openai/gpt-test",
		Messages: []llm.Message{llm.UserText("hello")},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Strict:      true,
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"$ref":"#/$defs/item"}},"required":["q"],"additionalProperties":false,"$defs":{"item":{"type":"string","pattern":"^a$"}}}`),
		}},
	}, false)
	if err != nil {
		t.Fatalf("buildParams returned error: %v", err)
	}
	if strings.Contains(testutil.MustCompactJSON(t, params), `"strict":true`) {
		t.Fatalf("strict = true for schema whose $defs subschema uses pattern:\n%s", testutil.MustCompactJSON(t, params))
	}
}

// openRouterFixtureResponse is the documented non-streaming wire shape:
// usage accounting extras (cost, cost_details, is_byok) live INSIDE usage.
const openRouterFixtureResponse = `{
	"id":"gen_1","model":"anthropic/claude-fallback","provider":"Anthropic","native_finish_reason":"stop",
	"usage":{"prompt_tokens":10,"prompt_tokens_details":{"cached_tokens":3},"completion_tokens":5,"completion_tokens_details":{"reasoning_tokens":2},"total_tokens":15,"cost":0.00042,"cost_details":{"upstream_inference_cost":0.0004},"is_byok":false},
	"choices":[{"index":0,"finish_reason":"stop","native_finish_reason":"end_turn","message":{
		"role":"assistant","content":"hi","reasoning":"why",
		"reasoning_details":[{"type":"reasoning.text","text":"why"}],
		"annotations":[{"type":"url_citation","url":"https://example.com"}],
		"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"go\"}"}},{"id":"bad","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]
	}}]
}`

func TestOpenRouterMapResponseFixture(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		if got := r.Header.Get("HTTP-Referer"); got != "https://request.example.com" {
			t.Fatalf("referer header = %q", got)
		}
		if got := r.Header.Get("X-Title"); got != "request title" {
			t.Fatalf("title header = %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, openRouterFixtureResponse)
	})

	resp, err := p.Chat(context.Background(), &llm.Request{
		Model:    "openai/gpt-test",
		Messages: []llm.Message{llm.UserText("hello")},
		ProviderOptions: Options{
			HTTPReferer: "https://request.example.com",
			XTitle:      "request title",
		},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Provider != providerName || resp.Model != "anthropic/claude-fallback" || resp.StopReason != llm.StopReasonEndTurn {
		t.Fatalf("identity/stop = %+v", resp)
	}
	if resp.Text() != "hi" || resp.Reasoning() != "why" {
		t.Fatalf("text/reasoning = %q/%q", resp.Text(), resp.Reasoning())
	}
	if resp.Usage.InputTokens != 7 || resp.Usage.CacheReadTokens != 3 || resp.Usage.OutputTokens != 5 || resp.Usage.ReasoningTokens != 2 || resp.Usage.CostUSD == nil || *resp.Usage.CostUSD != 0.00042 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	if resp.Usage.CostSource != llm.CostSourceNative {
		t.Fatalf("cost source = %q, want %q (usage.cost is billing-grade)", resp.Usage.CostSource, llm.CostSourceNative)
	}
	if len(resp.ToolCalls()) != 1 || len(resp.DroppedToolCalls) != 1 {
		t.Fatalf("tool calls/dropped = %+v/%+v", resp.ToolCalls(), resp.DroppedToolCalls)
	}
	extras, ok := Extras(resp)
	if !ok || extras.Provider != "Anthropic" || extras.NativeFinishReason != "end_turn" || !bytes.Contains(extras.Annotations, []byte("url_citation")) || !bytes.Contains(extras.CostDetails, []byte("upstream_inference_cost")) {
		t.Fatalf("extras = %+v ok=%v", extras, ok)
	}
	if extras.IsBYOK == nil || *extras.IsBYOK {
		t.Fatalf("is_byok = %v, want false", extras.IsBYOK)
	}
}

func TestOpenRouterReasoningReplayRoundTrip(t *testing.T) {
	// True round trip: wire response → ExtractParts → buildMessage must
	// replay a wire-identical reasoning_details ARRAY (not a nested [[...]]).
	const details = `[{"type":"reasoning.text","text":"why ","signature":"sig_1"},{"type":"reasoning.encrypted","id":"rd_1","data":"enc"}]`
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"id":"gen_1","model":"openai/gpt-test","provider":"OpenAI",
			"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6,"cost":0.0001},
			"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi","reasoning":"why ","reasoning_details":`+details+`}}]}`)
	})
	resp, err := p.Chat(context.Background(), &llm.Request{Model: "openai/gpt-test", Messages: []llm.Message{llm.UserText("hello")}})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	reasoning := reasoningPartsOf(resp)
	if len(reasoning) != 1 || reasoning[0].Provider != providerName {
		t.Fatalf("reasoning parts = %+v", reasoning)
	}
	testutil.AssertJSONEqual(t, string(reasoning[0].Raw), details)

	params, err := p.buildParams(&llm.Request{
		Model: "openai/gpt-test",
		Messages: []llm.Message{
			llm.UserText("hello"),
			{Role: llm.RoleAssistant, Parts: resp.Parts, Provider: resp.Provider, Model: resp.Model},
			llm.UserText("continue"),
		},
	}, false)
	if err != nil {
		t.Fatalf("replay buildParams returned error: %v", err)
	}
	var body struct {
		Messages []struct {
			Role             string          `json:"role"`
			ReasoningDetails json.RawMessage `json:"reasoning_details"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte(testutil.MustCompactJSON(t, params)), &body); err != nil {
		t.Fatalf("params are invalid JSON: %v", err)
	}
	if len(body.Messages) != 3 || body.Messages[1].Role != "assistant" {
		t.Fatalf("messages = %+v", body.Messages)
	}
	testutil.AssertJSONEqual(t, string(body.Messages[1].ReasoningDetails), details)
}

func TestOpenRouterStreamKeepAliveCollectEquivalent(t *testing.T) {
	// The same logical payload served as a keep-alive-laced stream (with
	// reasoning_details split across chunks) and as a blocking response must
	// produce equivalent Responses (ARCH §2.5 Collect equivalence).
	streamProvider := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		mustWrite(t, w, ": OPENROUTER PROCESSING\n\n")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","provider":"OpenAI","choices":[{"index":0,"delta":{"role":"assistant"}}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"reasoning":"why ","reasoning_details":[{"type":"reasoning.text","text":"why "}]}}]}`+"\n\n")
		mustWrite(t, w, ": OPENROUTER PROCESSING\n\n")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"reasoning":"not","reasoning_details":[{"type":"reasoning.text","text":"not"}]}}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"content":"hi","annotations":[{"type":"url_citation","url":"https://example.com"}]}}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"native_finish_reason":"tool_use","delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"go\"}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6,"cost":0.0001,"cost_details":{"upstream_inference_cost":0.00009},"is_byok":false}}`+"\n\n")
		mustWrite(t, w, "data: [DONE]\n\n")
	})
	chatProvider := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"id":"gen_1","model":"openai/gpt-test","provider":"OpenAI",
			"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6,"cost":0.0001,"cost_details":{"upstream_inference_cost":0.00009},"is_byok":false},
			"choices":[{"index":0,"finish_reason":"tool_calls","native_finish_reason":"tool_use","message":{
				"role":"assistant","content":"hi","reasoning":"why not",
				"reasoning_details":[{"type":"reasoning.text","text":"why "},{"type":"reasoning.text","text":"not"}],
				"annotations":[{"type":"url_citation","url":"https://example.com"}],
				"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"go\"}"}}]}}]}`)
	})
	req := &llm.Request{Model: "openai/gpt-test", Messages: []llm.Message{llm.UserText("hello")}}

	collected, err := llm.Collect(streamProvider.ChatStream(context.Background(), req))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	blocking, err := chatProvider.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}

	if collected.ID != blocking.ID || collected.Provider != blocking.Provider || collected.Model != blocking.Model {
		t.Fatalf("identity mismatch: stream=%+v chat=%+v", collected, blocking)
	}
	if collected.StopReason != blocking.StopReason || collected.StopReasonRaw != blocking.StopReasonRaw {
		t.Fatalf("stop mismatch: stream=%s/%s chat=%s/%s", collected.StopReason, collected.StopReasonRaw, blocking.StopReason, blocking.StopReasonRaw)
	}
	if !reflect.DeepEqual(collected.Parts, blocking.Parts) {
		t.Fatalf("parts mismatch:\nstream: %#v\nchat:   %#v", collected.Parts, blocking.Parts)
	}
	if !reflect.DeepEqual(collected.Usage, blocking.Usage) {
		t.Fatalf("usage mismatch:\nstream: %+v\nchat:   %+v", collected.Usage, blocking.Usage)
	}
	// DeepEqual (not len) so nil-vs-empty asymmetry between the paths fails.
	if !reflect.DeepEqual(collected.DroppedToolCalls, blocking.DroppedToolCalls) {
		t.Fatalf("dropped mismatch: stream=%#v chat=%#v", collected.DroppedToolCalls, blocking.DroppedToolCalls)
	}
	streamExtras, ok := Extras(collected)
	if !ok {
		t.Fatalf("stream extras missing: %+v", collected.Raw)
	}
	chatExtras, ok := Extras(blocking)
	if !ok {
		t.Fatalf("chat extras missing: %+v", blocking.Raw)
	}
	// Raw carries the (path-specific) full payload; every typed extra must
	// agree between the two paths.
	if streamExtras.Provider != chatExtras.Provider || streamExtras.NativeFinishReason != chatExtras.NativeFinishReason {
		t.Fatalf("extras mismatch: stream=%+v chat=%+v", streamExtras, chatExtras)
	}
	testutil.AssertJSONEqual(t, string(streamExtras.CostDetails), string(chatExtras.CostDetails))
	testutil.AssertJSONEqual(t, string(streamExtras.Annotations), string(chatExtras.Annotations))
	testutil.AssertJSONEqual(t, string(streamExtras.ReasoningDetails), string(chatExtras.ReasoningDetails))
	if streamExtras.IsBYOK == nil || chatExtras.IsBYOK == nil || *streamExtras.IsBYOK != *chatExtras.IsBYOK {
		t.Fatalf("is_byok mismatch: stream=%v chat=%v", streamExtras.IsBYOK, chatExtras.IsBYOK)
	}
	// The replay contract: streamed reasoning Raw must be the merged wire
	// array, byte-identical to the blocking path.
	streamReasoning := reasoningPartsOf(collected)
	chatReasoning := reasoningPartsOf(blocking)
	if len(streamReasoning) != 1 || len(chatReasoning) != 1 || !bytes.Equal(streamReasoning[0].Raw, chatReasoning[0].Raw) {
		t.Fatalf("reasoning raw mismatch:\nstream: %s\nchat:   %s", streamReasoning[0].Raw, chatReasoning[0].Raw)
	}
}

func TestOpenRouterStreamToolCallDropped(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"go\"}"}}]}}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"tool_calls":[{"index":2,"id":"call_3","type":"function","function":{"arguments":"{\"x\":1}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"cost":0.00001}}`+"\n\n")
		mustWrite(t, w, "data: [DONE]\n\n")
	})
	var droppedEvents []llm.ToolCallDropped
	for event, err := range p.ChatStream(context.Background(), &llm.Request{Model: "openai/gpt-test", Messages: []llm.Message{llm.UserText("hello")}}) {
		if err != nil {
			t.Fatalf("stream returned error: %v", err)
		}
		if dropped, ok := event.(llm.ToolCallDropped); ok {
			droppedEvents = append(droppedEvents, dropped)
		}
	}
	if len(droppedEvents) != 2 {
		t.Fatalf("dropped events = %+v, want 2", droppedEvents)
	}
	// Pending-call flush is index-ordered: the malformed-args call (block
	// index 3) drops before the missing-name call (block index 4).
	if droppedEvents[0].Reason != "invalid tool arguments JSON" || droppedEvents[1].Reason != "missing tool name" {
		t.Fatalf("dropped reasons = %+v", droppedEvents)
	}
	if droppedEvents[0].Index >= droppedEvents[1].Index {
		t.Fatalf("dropped order not deterministic: %+v", droppedEvents)
	}
}

func TestOpenRouterAttributionPrecedence(t *testing.T) {
	requestOptions := Options{HTTPReferer: "https://request.example.com", XTitle: "request title"}
	tests := []struct {
		name        string
		options     llm.ProviderOptions
		wantReferer string
		wantTitle   string
	}{
		{name: "constructor_only", options: nil, wantReferer: "https://constructor.example.com", wantTitle: "constructor title"},
		{name: "per_request_wins", options: requestOptions, wantReferer: "https://request.example.com", wantTitle: "request title"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var referers, titles []string
			p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
				referers = append(referers, r.Header.Get("HTTP-Referer"))
				titles = append(titles, r.Header.Get("X-Title"))
				if r.Header.Get("Accept") == "text/event-stream" {
					w.Header().Set("Content-Type", "text/event-stream")
					mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\n")
					mustWrite(t, w, "data: [DONE]\n\n")
					return
				}
				w.Header().Set("Content-Type", "application/json")
				mustWrite(t, w, `{"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]}`)
			}, WithAttribution("https://constructor.example.com", "constructor title"))

			req := &llm.Request{Model: "openai/gpt-test", Messages: []llm.Message{llm.UserText("hello")}, ProviderOptions: tt.options}
			if _, err := p.Chat(context.Background(), req); err != nil {
				t.Fatalf("Chat returned error: %v", err)
			}
			if _, err := llm.Collect(p.ChatStream(context.Background(), req)); err != nil {
				t.Fatalf("Collect(ChatStream) returned error: %v", err)
			}
			if len(referers) != 2 {
				t.Fatalf("requests seen = %d, want 2", len(referers))
			}
			for i, path := range []string{"chat", "stream"} {
				if referers[i] != tt.wantReferer || titles[i] != tt.wantTitle {
					t.Fatalf("%s attribution = %q/%q, want %q/%q", path, referers[i], titles[i], tt.wantReferer, tt.wantTitle)
				}
			}
		})
	}
}

func TestOpenRouterStreamRetriesBeforeYield(t *testing.T) {
	attempts := 0
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			mustWrite(t, w, `{"error":{"code":"rate_limit","message":"slow down"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"cost":0.00001}}`+"\n\n")
		mustWrite(t, w, "data: [DONE]\n\n")
	}, WithMaxRetries(1))
	resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{Model: "openai/gpt-test", Messages: []llm.Message{llm.UserText("hello")}}))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if attempts != 2 || resp.Text() != "ok" {
		t.Fatalf("attempts=%d resp=%+v", attempts, resp)
	}
}

func TestOpenRouterMidStreamError(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"content":"partial"}}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"finish_reason":"error","error":{"code":502,"message":"upstream failed","metadata":{"provider_name":"x"}}}]}`+"\n\n")
	})
	resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{Model: "openai/gpt-test", Messages: []llm.Message{llm.UserText("hello")}}))
	if err == nil || resp == nil || resp.Text() != "partial" {
		t.Fatalf("resp=%+v err=%v", resp, err)
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Code != "502" || providerErr.Metadata["provider_name"] != "x" {
		t.Fatalf("provider error = %+v err=%v", providerErr, err)
	}
}

func TestOpenRouterBadRequestContentMessageIsNotContentFiltered(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		mustWrite(t, w, `{"error":{"code":"invalid_request","message":"message content is required"}}`)
	})
	_, err := p.Chat(context.Background(), &llm.Request{Model: "openai/gpt-test", Messages: []llm.Message{llm.UserText("hello")}})
	if !errors.Is(err, llm.ErrBadRequest) || errors.Is(err, llm.ErrContentFiltered) {
		t.Fatalf("err = %v, want ErrBadRequest only", err)
	}
}

func TestOpenRouterContextTooLongMapping(t *testing.T) {
	tests := []struct {
		name string
		body string
		want error
	}{
		{
			name: "context_length_exceeded_code",
			body: `{"error":{"code":"context_length_exceeded","message":"too many tokens"}}`,
			want: llm.ErrContextTooLong,
		},
		{
			name: "maximum_context_length_message",
			body: `{"error":{"code":"invalid_request","message":"This model's maximum context length is 8192 tokens"}}`,
			want: llm.ErrContextTooLong,
		},
		{
			// A message merely mentioning "context" must not be classified
			// as a context overflow.
			name: "bare_context_word",
			body: `{"error":{"code":"invalid_request","message":"invalid context provided for tool"}}`,
			want: llm.ErrBadRequest,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusBadRequest)
				mustWrite(t, w, tt.body)
			})
			_, err := p.Chat(context.Background(), &llm.Request{Model: "openai/gpt-test", Messages: []llm.Message{llm.UserText("hello")}})
			if !errors.Is(err, tt.want) {
				t.Fatalf("err = %v, want %v", err, tt.want)
			}
			if tt.want == llm.ErrBadRequest && errors.Is(err, llm.ErrContextTooLong) {
				t.Fatalf("err = %v unexpectedly matches ErrContextTooLong", err)
			}
		})
	}
}

func TestOpenRouterWarmupEmptyChoices(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"id":"gen_1","model":"openai/gpt-test","choices":[]}`)
	})
	_, err := p.Chat(context.Background(), &llm.Request{Model: "openai/gpt-test", Messages: []llm.Message{llm.UserText("hello")}})
	if !errors.Is(err, llm.ErrServer) {
		t.Fatalf("err = %v, want ErrServer", err)
	}
}

func TestOpenRouterModerationMetadataError(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		mustWrite(t, w, `{"error":{"code":403,"message":"moderated","metadata":{"reasons":["violence"]}}}`)
	})
	_, err := p.Chat(context.Background(), &llm.Request{Model: "openai/gpt-test", Messages: []llm.Message{llm.UserText("hello")}})
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || !errors.Is(err, llm.ErrContentFiltered) {
		t.Fatalf("err = %v providerErr=%+v", err, providerErr)
	}
	if providerErr.Metadata["reasons"] == nil {
		t.Fatalf("metadata = %+v", providerErr.Metadata)
	}
}

func TestOpenRouterModels(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		mustWrite(t, w, `{"data":[{"id":"openai/gpt-test","name":"GPT Test","context_length":128000,"top_provider":{"max_completion_tokens":4096},"pricing":{"prompt":"0.000001","completion":"0.000002"},"canonical_slug":"openai/gpt-test","supported_parameters":["tools"],"modalities":["text","image"]}]}`)
	})
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	if len(models) != 1 || models[0].ID != "openai/gpt-test" || models[0].ContextWindow != 128000 || models[0].MaxOutputTokens != 4096 || models[0].Pricing == nil || models[0].Pricing.InputPerMTok != 1 {
		t.Fatalf("models = %+v", models)
	}
	raw, ok := models[0].Raw.(json.RawMessage)
	if !ok || !bytes.Contains(raw, []byte(`"supported_parameters"`)) || !bytes.Contains(raw, []byte(`"modalities"`)) {
		t.Fatalf("raw model payload = %T %s", models[0].Raw, raw)
	}
}

func reasoningPartsOf(resp *llm.Response) []llm.ReasoningPart {
	var out []llm.ReasoningPart
	for _, part := range resp.Parts {
		switch value := part.(type) {
		case llm.ReasoningPart:
			out = append(out, value)
		case *llm.ReasoningPart:
			if value != nil {
				out = append(out, *value)
			}
		}
	}
	return out
}

func newTestProvider(t *testing.T, handler http.HandlerFunc, opts ...Option) *Provider {
	t.Helper()
	client := &http.Client{Transport: testutil.RoundTripFunc(func(req *http.Request) (*http.Response, error) {
		rec := &responseRecorder{header: http.Header{}, status: http.StatusOK}
		handler.ServeHTTP(rec, req)
		return rec.response(req), nil
	})}
	all := []Option{WithAPIKey("test"), WithBaseURL("https://openrouter.test"), WithHTTPClient(client), WithMaxRetries(0)}
	all = append(all, opts...)
	p, err := New(all...)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	return p
}

func mustWrite(t *testing.T, w io.Writer, s string) {
	t.Helper()
	if _, err := io.WriteString(w, s); err != nil {
		t.Fatalf("WriteString returned error: %v", err)
	}
}

type responseRecorder struct {
	header http.Header
	body   bytes.Buffer
	status int
}

func (r *responseRecorder) Header() http.Header { return r.header }

func (r *responseRecorder) WriteHeader(status int) { r.status = status }

func (r *responseRecorder) Write(data []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}
	return r.body.Write(data)
}

func (r *responseRecorder) response(req *http.Request) *http.Response {
	status := r.status
	if status == 0 {
		status = http.StatusOK
	}
	return &http.Response{
		StatusCode: status,
		Status:     http.StatusText(status),
		Header:     r.header,
		Body:       io.NopCloser(bytes.NewReader(r.body.Bytes())),
		Request:    req,
	}
}
