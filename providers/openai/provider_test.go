package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	sdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
)

func TestOpenAIBuildRequestGolden(t *testing.T) {
	temp := 0.2
	topP := 0.8
	rawReasoning := json.RawMessage(`{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"because"}],"encrypted_content":"enc","status":"completed"}`)
	p := &Provider{}
	params, err := p.adapter().BuildParams(&llm.Request{
		Model:     "gpt-test",
		System:    "You are terse.",
		MaxTokens: 99,
		Messages: []llm.Message{
			llm.UserParts(llm.Text("hello"), llm.ImageData([]byte("png"), "image/png")),
			llm.AssistantParts(
				llm.ReasoningPart{Provider: providerName, Raw: rawReasoning},
				llm.ReasoningPart{Provider: "anthropic", Raw: json.RawMessage(`{"type":"thinking","signature":"foreign"}`)},
				llm.Text("previous"),
				llm.ToolCall("call_1", "lookup", json.RawMessage(`{"q":"go"}`)),
			),
			{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResultPart{ToolCallID: "call_1", Name: "lookup", Content: []llm.Part{llm.Text("result")}}}},
		},
		Temperature: &temp,
		TopP:        &topP,
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
		Effort:    llm.EffortHigh,
		SessionID: "session_1",
	}, false)
	if err != nil {
		t.Fatalf("buildParams returned error: %v", err)
	}

	got := testutil.MustCompactJSON(t, params)
	want := `{
	"instructions": "You are terse.",
	"max_output_tokens": 99,
	"store": false,
	"temperature": 0.2,
	"top_p": 0.8,
	"prompt_cache_key": "session_1",
	"include": [
		"reasoning.encrypted_content"
	],
	"input": [
		{
			"content": [
				{
					"text": "hello",
					"type": "input_text"
				},
				{
					"detail": "auto",
					"image_url": "data:image/png;base64,cG5n",
					"type": "input_image"
				}
			],
			"role": "user"
		},
		{
			"id": "rs_1",
			"type": "reasoning",
			"summary": [
				{
					"type": "summary_text",
					"text": "because"
				}
			],
			"encrypted_content": "enc",
			"status": "completed"
		},
		{
			"content": [
				{
					"text": "previous",
					"type": "output_text"
				}
			],
			"status": "completed",
			"role": "assistant",
			"type": "message"
		},
		{
			"arguments": "{\"q\":\"go\"}",
			"call_id": "call_1",
			"name": "lookup",
			"type": "function_call"
		},
		{
			"call_id": "call_1",
			"output": "result",
			"type": "function_call_output"
		}
	],
	"model": "gpt-test",
	"reasoning": {
		"effort": "high",
		"summary": "auto"
	},
	"text": {
		"format": {
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
			"strict": false,
			"type": "json_schema"
		}
	},
	"tool_choice": {
		"name": "lookup",
		"type": "function"
	},
	"tools": [
		{
			"strict": false,
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
			"name": "lookup",
			"description": "Look up a value.",
			"type": "function"
		}
	]
}`
	testutil.AssertJSONEqual(t, got, want)
	if strings.Contains(got, "foreign") {
		t.Fatalf("foreign reasoning was replayed: %s", got)
	}
}

func TestOpenAIProviderOptionsGolden(t *testing.T) {
	store := true
	background := true
	p := &Provider{}
	params, err := p.adapter().BuildParams(&llm.Request{
		Model:    "gpt-test",
		Messages: []llm.Message{llm.UserText("hello")},
		ProviderOptions: Options{
			Store:                &store,
			PreviousResponseID:   "resp_prev",
			Include:              []responses.ResponseIncludable{responses.ResponseIncludableMessageOutputTextLogprobs},
			Background:           &background,
			Verbosity:            responses.ResponseTextConfigVerbosityLow,
			Metadata:             shared.Metadata{"purpose": "test"},
			ServiceTier:          responses.ResponseNewParamsServiceTierDefault,
			SafetyIdentifier:     "user_hash",
			PromptCacheRetention: responses.ResponseNewParamsPromptCacheRetention24h,
		},
	}, false)
	if err != nil {
		t.Fatalf("buildParams returned error: %v", err)
	}
	got := testutil.MustCompactJSON(t, params)
	want := `{"background":true,"previous_response_id":"resp_prev","store":true,"safety_identifier":"user_hash","include":["message.output_text.logprobs"],"metadata":{"purpose":"test"},"prompt_cache_retention":"24h","service_tier":"default","input":[{"content":[{"text":"hello","type":"input_text"}],"role":"user"}],"model":"gpt-test","text":{"verbosity":"low"}}`
	testutil.AssertJSONEqual(t, got, want)
}

func TestOpenAIEncryptedReasoningIncludeRequiresStatelessRequest(t *testing.T) {
	store := true
	tests := []struct {
		name    string
		options Options
		want    bool
	}{
		{name: "default stateless", want: true},
		{name: "store true", options: Options{Store: &store}, want: false},
		{name: "previous response", options: Options{PreviousResponseID: "resp_prev"}, want: false},
		{name: "conversation id", options: Options{ConversationID: "conv_1"}, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &llm.Request{
				Model:    "gpt-test",
				Messages: []llm.Message{llm.UserText("hello")},
			}
			if tt.options.Store != nil || tt.options.PreviousResponseID != "" || tt.options.ConversationID != "" {
				req.ProviderOptions = tt.options
			}
			params, err := (&Provider{}).adapter().BuildParams(req, false)
			if err != nil {
				t.Fatalf("buildParams returned error: %v", err)
			}
			got := false
			for _, include := range params.Include {
				if include == responses.ResponseIncludableReasoningEncryptedContent {
					got = true
				}
			}
			if got != tt.want {
				t.Fatalf("encrypted reasoning include = %v, want %v; includes=%+v", got, tt.want, params.Include)
			}
		})
	}
}

func TestOpenAIReasoningReplayGolden(t *testing.T) {
	raw := json.RawMessage(`{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"because"}],"encrypted_content":"enc","status":"completed"}`)
	foreign := json.RawMessage(`{"type":"thinking","signature":"foreign"}`)
	input, err := (&Provider{}).adapter().BuildInput([]llm.Message{llm.AssistantParts(
		llm.Text("before"),
		llm.ReasoningPart{Provider: providerName, Raw: raw},
		llm.ReasoningPart{Provider: "anthropic", Raw: foreign},
		llm.ReasoningPart{Provider: providerName, Text: "summary only"},
	)})
	if err != nil {
		t.Fatalf("buildInput returned error: %v", err)
	}
	if len(input) != 2 {
		t.Fatalf("input len = %d, want text + same-provider reasoning", len(input))
	}
	got, err := json.Marshal(input[1])
	if err != nil {
		t.Fatalf("Marshal replay item returned error: %v", err)
	}
	if !bytes.Equal(got, raw) {
		t.Fatalf("replay raw changed\ngot:  %s\nwant: %s", got, raw)
	}
	all, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("Marshal input returned error: %v", err)
	}
	if bytes.Contains(all, foreign) || bytes.Contains(all, []byte("summary only")) {
		t.Fatalf("foreign or summary-only reasoning was replayed: %s", all)
	}
}

func TestOpenAIMapResponseFixtures(t *testing.T) {
	resp := mustOpenAIResponse(t, `{
		"id":"resp_1","model":"gpt-test","status":"completed",
		"output":[
			{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"why"}],"encrypted_content":"enc","status":"completed"},
			{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hi","annotations":[]}]},
			{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"},
			{"id":"fc_bad","type":"function_call","call_id":"call_bad","name":"lookup","arguments":"{\"q\":","status":"completed"}
		],
		"usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":3},"output_tokens":5,"output_tokens_details":{"reasoning_tokens":2},"total_tokens":15}
	}`)
	mapped, err := (&Provider{}).adapter().MapResponse(resp)
	if err != nil {
		t.Fatalf("mapResponse returned error: %v", err)
	}
	if mapped.Provider != providerName || mapped.Model != "gpt-test" || mapped.StopReason != llm.StopReasonToolUse {
		t.Fatalf("response identity/stop = %+v", mapped)
	}
	if mapped.Text() != "hi" || mapped.Reasoning() != "why" {
		t.Fatalf("text/reasoning = %q/%q", mapped.Text(), mapped.Reasoning())
	}
	reasoning := mapped.Parts[0].(llm.ReasoningPart)
	if reasoning.Provider != providerName || !bytes.Contains(reasoning.Raw, []byte(`"encrypted_content":"enc"`)) {
		t.Fatalf("reasoning = %+v", reasoning)
	}
	calls := mapped.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call_1" || calls[0].Name != "lookup" || string(calls[0].Args) != `{"q":"go"}` {
		t.Fatalf("tool calls = %+v", calls)
	}
	if len(mapped.DroppedToolCalls) != 1 || mapped.DroppedToolCalls[0].Reason != "invalid tool arguments JSON" {
		t.Fatalf("dropped calls = %+v", mapped.DroppedToolCalls)
	}
	if mapped.Usage.InputTokens != 7 || mapped.Usage.CacheReadTokens != 3 || mapped.Usage.OutputTokens != 5 || mapped.Usage.ReasoningTokens != 2 || mapped.Usage.TotalTokens != 15 {
		t.Fatalf("usage = %+v", mapped.Usage)
	}
}

func TestOpenAIStopReasonFixtures(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want llm.StopReason
	}{
		{name: "completed", raw: `{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok","annotations":[]}]}]}`, want: llm.StopReasonEndTurn},
		{name: "max output", raw: `{"id":"resp_1","model":"gpt-test","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[]}`, want: llm.StopReasonMaxTokens},
		{name: "content filter", raw: `{"id":"resp_1","model":"gpt-test","status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[]}`, want: llm.StopReasonContentFilter},
		{name: "failed", raw: `{"id":"resp_1","model":"gpt-test","status":"failed","output":[],"error":{"code":"server_error","message":"failed"}}`, want: llm.StopReasonError},
		{name: "refusal", raw: `{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"refusal","refusal":"no"}]}]}`, want: llm.StopReasonRefusal},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := (&Provider{}).adapter().MapResponse(mustOpenAIResponse(t, tt.raw))
			if err != nil {
				t.Fatalf("mapResponse returned error: %v", err)
			}
			if resp.StopReason != tt.want {
				t.Fatalf("stop reason = %q, want %q", resp.StopReason, tt.want)
			}
		})
	}
}

func TestOpenAIStopReasonDroppedOnlyFunctionCall(t *testing.T) {
	resp, err := (&Provider{}).adapter().MapResponse(mustOpenAIResponse(t, `{
		"id":"resp_1","model":"gpt-test","status":"completed",
		"output":[{"id":"fc_bad","type":"function_call","call_id":"call_bad","name":"lookup","arguments":"{\"q\":","status":"completed"}]
	}`))
	if err != nil {
		t.Fatalf("mapResponse returned error: %v", err)
	}
	if resp.StopReason != llm.StopReasonToolUse {
		t.Fatalf("stop reason = %q, want %q", resp.StopReason, llm.StopReasonToolUse)
	}
	if len(resp.ToolCalls()) != 0 || len(resp.DroppedToolCalls) != 1 {
		t.Fatalf("tool calls/dropped = %+v/%+v", resp.ToolCalls(), resp.DroppedToolCalls)
	}
}

func TestOpenAIStreamFixturesCollectEquivalent(t *testing.T) {
	finalResponseRaw := `{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"why "},{"type":"summary_text","text":"therefore"}],"encrypted_content":"enc","status":"completed"},{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hi","annotations":[]}]},{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}],"usage":{"input_tokens":4,"input_tokens_details":{"cached_tokens":1},"output_tokens":2,"output_tokens_details":{"reasoning_tokens":1},"total_tokens":6}}`
	rawEvents := []string{
		`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`,
		`{"type":"response.reasoning_summary_text.delta","sequence_number":1,"item_id":"rs_1","output_index":0,"summary_index":0,"delta":"why "}`,
		`{"type":"response.reasoning_summary_text.delta","sequence_number":2,"item_id":"rs_1","output_index":0,"summary_index":1,"delta":"therefore"}`,
		`{"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"why "},{"type":"summary_text","text":"therefore"}],"encrypted_content":"enc","status":"completed"}}`,
		`{"type":"response.output_text.delta","sequence_number":4,"item_id":"msg_1","output_index":1,"content_index":0,"delta":"hi","logprobs":[]}`,
		`{"type":"response.output_item.added","sequence_number":5,"output_index":2,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"","status":"in_progress"}}`,
		`{"type":"response.function_call_arguments.delta","sequence_number":6,"item_id":"fc_1","output_index":2,"delta":"{\"q\":"}`,
		`{"type":"response.function_call_arguments.delta","sequence_number":7,"item_id":"fc_1","output_index":2,"delta":"\"go\"}"}`,
		`{"type":"response.output_item.done","sequence_number":8,"output_index":2,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}}`,
		`{"type":"response.completed","sequence_number":9,"response":` + finalResponseRaw + `}`,
	}
	resp := collectOpenAIStream(t, rawEvents)
	nonStreaming, err := (&Provider{}).adapter().MapResponse(mustOpenAIResponse(t, finalResponseRaw))
	if err != nil {
		t.Fatalf("non-streaming mapResponse returned error: %v", err)
	}
	if resp.Text() != nonStreaming.Text() || resp.Reasoning() != nonStreaming.Reasoning() || resp.StopReason != nonStreaming.StopReason {
		t.Fatalf("stream/non-stream mismatch: stream=%+v non-stream=%+v", resp, nonStreaming)
	}
	if len(resp.ToolCalls()) != 1 || !reflect.DeepEqual(resp.ToolCalls(), nonStreaming.ToolCalls()) {
		t.Fatalf("stream/non-stream tool calls = %+v/%+v", resp.ToolCalls(), nonStreaming.ToolCalls())
	}
	streamReasoning := reasoningParts(resp.Parts)
	nonStreamReasoning := reasoningParts(nonStreaming.Parts)
	if !reflect.DeepEqual(streamReasoning, nonStreamReasoning) {
		t.Fatalf("stream/non-stream reasoning parts = %+v/%+v", streamReasoning, nonStreamReasoning)
	}
	streamRaw := resp.Parts[0].(llm.ReasoningPart).Raw
	nonStreamRaw := nonStreaming.Parts[0].(llm.ReasoningPart).Raw
	if !bytes.Equal(streamRaw, nonStreamRaw) {
		t.Fatalf("reasoning raw mismatch\nstream:     %s\nnon-stream: %s", streamRaw, nonStreamRaw)
	}
	if resp.Usage.InputTokens != 3 || resp.Usage.CacheReadTokens != 1 || resp.Usage.OutputTokens != 2 || resp.Usage.ReasoningTokens != 1 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestOpenAIStreamDroppedToolCallCollectEquivalent(t *testing.T) {
	// One good call (output item 1) and one started-then-dropped call with
	// invalid JSON args (output item 2). Stream and non-stream must agree on
	// parts AND on DroppedToolCall indices (output-item positions).
	finalResponseRaw := `{"id":"resp_1","model":"gpt-test","status":"completed","output":[` +
		`{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hi","annotations":[]}]},` +
		`{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"},` +
		`{"id":"fc_bad","type":"function_call","call_id":"call_bad","name":"lookup","arguments":"{\"q\":","status":"completed"}` +
		`],"usage":{"input_tokens":4,"input_tokens_details":{"cached_tokens":0},"output_tokens":2,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":6}}`
	rawEvents := []string{
		`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`,
		`{"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"hi","logprobs":[]}`,
		`{"type":"response.output_item.added","sequence_number":2,"output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"","status":"in_progress"}}`,
		`{"type":"response.function_call_arguments.delta","sequence_number":3,"item_id":"fc_1","output_index":1,"delta":"{\"q\":\"go\"}"}`,
		`{"type":"response.output_item.done","sequence_number":4,"output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}}`,
		`{"type":"response.output_item.added","sequence_number":5,"output_index":2,"item":{"id":"fc_bad","type":"function_call","call_id":"call_bad","name":"lookup","arguments":"","status":"in_progress"}}`,
		`{"type":"response.function_call_arguments.delta","sequence_number":6,"item_id":"fc_bad","output_index":2,"delta":"{\"q\":"}`,
		`{"type":"response.output_item.done","sequence_number":7,"output_index":2,"item":{"id":"fc_bad","type":"function_call","call_id":"call_bad","name":"lookup","arguments":"{\"q\":","status":"completed"}}`,
		`{"type":"response.completed","sequence_number":8,"response":` + finalResponseRaw + `}`,
	}
	resp := collectOpenAIStream(t, rawEvents)
	nonStreaming, err := (&Provider{}).adapter().MapResponse(mustOpenAIResponse(t, finalResponseRaw))
	if err != nil {
		t.Fatalf("non-streaming mapResponse returned error: %v", err)
	}
	if !reflect.DeepEqual(resp.DroppedToolCalls, nonStreaming.DroppedToolCalls) {
		t.Fatalf("dropped calls stream/non-stream = %+v/%+v", resp.DroppedToolCalls, nonStreaming.DroppedToolCalls)
	}
	if len(resp.DroppedToolCalls) != 1 || resp.DroppedToolCalls[0].Index != 2 || resp.DroppedToolCalls[0].Reason != "invalid tool arguments JSON" {
		t.Fatalf("dropped calls = %+v, want index 2 (output-item position)", resp.DroppedToolCalls)
	}
	if !reflect.DeepEqual(resp.Parts, nonStreaming.Parts) {
		t.Fatalf("parts stream/non-stream mismatch:\nstream:     %+v\nnon-stream: %+v", resp.Parts, nonStreaming.Parts)
	}
	if resp.Text() != "hi" || len(resp.ToolCalls()) != 1 || resp.ToolCalls()[0].ID != "call_1" {
		t.Fatalf("collected response = text %q calls %+v", resp.Text(), resp.ToolCalls())
	}
}

func TestOpenAIStrictToolAllowsPropertyNamedFormat(t *testing.T) {
	params, err := (&Provider{}).adapter().BuildParams(&llm.Request{
		Model:    "gpt-test",
		Messages: []llm.Message{llm.UserText("hello")},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Strict:      true,
			InputSchema: json.RawMessage(`{"type":"object","properties":{"format":{"type":"string"},"q":{"type":"string"}},"required":["format","q"],"additionalProperties":false}`),
		}},
	}, false)
	if err != nil {
		t.Fatalf("buildParams returned error: %v", err)
	}
	if len(params.Tools) != 1 || params.Tools[0].OfFunction == nil {
		t.Fatalf("tools = %+v", params.Tools)
	}
	// A property NAMED "format" is not the "format" keyword: strict stays on.
	if !params.Tools[0].OfFunction.Strict.Value {
		t.Fatalf("strict = false for schema whose property is named format")
	}

	// The keyword one level deeper (inside the property's schema) still
	// disables strict mode.
	params, err = (&Provider{}).adapter().BuildParams(&llm.Request{
		Model:    "gpt-test",
		Messages: []llm.Message{llm.UserText("hello")},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Strict:      true,
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string","format":"uri"}},"required":["q"],"additionalProperties":false}`),
		}},
	}, false)
	if err != nil {
		t.Fatalf("buildParams returned error: %v", err)
	}
	if params.Tools[0].OfFunction.Strict.Value {
		t.Fatalf("strict = true for schema using the format keyword")
	}
}

func TestOpenAIStrictToolAllowsDefsNamedLikeKeywords(t *testing.T) {
	// Keys directly under $defs/definitions are definition NAMES, not schema
	// keywords — a shared definition named "format" must not disable strict.
	params, err := (&Provider{}).adapter().BuildParams(&llm.Request{
		Model:    "gpt-test",
		Messages: []llm.Message{llm.UserText("hello")},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Strict:      true,
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"$ref":"#/$defs/format"}},"required":["q"],"additionalProperties":false,"$defs":{"format":{"type":"string"}},"definitions":{"pattern":{"type":"integer"}}}`),
		}},
	}, false)
	if err != nil {
		t.Fatalf("buildParams returned error: %v", err)
	}
	if !params.Tools[0].OfFunction.Strict.Value {
		t.Fatalf("strict = false for schema whose $defs/definitions entries are named like keywords")
	}

	// A keyword inside a $defs subschema still disables strict mode.
	params, err = (&Provider{}).adapter().BuildParams(&llm.Request{
		Model:    "gpt-test",
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
	if params.Tools[0].OfFunction.Strict.Value {
		t.Fatalf("strict = true for schema whose $defs subschema uses pattern")
	}
}

func TestOpenAIReasoningTextComposition(t *testing.T) {
	// A reasoning item carrying BOTH summary and reasoning_text content must
	// produce the summary as Text on both paths; reasoning_text is used only
	// when no summary text arrived.
	dualItem := `{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"summary wins"}],"content":[{"type":"reasoning_text","text":"raw thoughts"}],"encrypted_content":"enc","status":"completed"}`
	textOnlyItem := `{"id":"rs_2","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"raw only"}],"encrypted_content":"enc2","status":"completed"}`
	finalResponseRaw := `{"id":"resp_1","model":"gpt-test","status":"completed","output":[` + dualItem + `,` + textOnlyItem + `]}`

	nonStreaming, err := (&Provider{}).adapter().MapResponse(mustOpenAIResponse(t, finalResponseRaw))
	if err != nil {
		t.Fatalf("mapResponse returned error: %v", err)
	}
	nonStreamReasoning := reasoningParts(nonStreaming.Parts)
	if len(nonStreamReasoning) != 2 || nonStreamReasoning[0].Text != "summary wins" || nonStreamReasoning[1].Text != "raw only" {
		t.Fatalf("non-stream reasoning = %+v", nonStreamReasoning)
	}

	rawEvents := []string{
		`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`,
		`{"type":"response.reasoning_summary_text.delta","sequence_number":1,"item_id":"rs_1","output_index":0,"summary_index":0,"delta":"summary wins"}`,
		`{"type":"response.reasoning_text.delta","sequence_number":2,"item_id":"rs_1","output_index":0,"content_index":0,"delta":"raw thoughts"}`,
		`{"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":` + dualItem + `}`,
		`{"type":"response.reasoning_text.delta","sequence_number":4,"item_id":"rs_2","output_index":1,"content_index":0,"delta":"raw "}`,
		`{"type":"response.reasoning_text.delta","sequence_number":5,"item_id":"rs_2","output_index":1,"content_index":0,"delta":"only"}`,
		`{"type":"response.output_item.done","sequence_number":6,"output_index":1,"item":` + textOnlyItem + `}`,
		`{"type":"response.completed","sequence_number":7,"response":` + finalResponseRaw + `}`,
	}
	resp := collectOpenAIStream(t, rawEvents)
	streamReasoning := reasoningParts(resp.Parts)
	if len(streamReasoning) != 2 {
		t.Fatalf("stream reasoning parts = %+v", streamReasoning)
	}
	if streamReasoning[0].Text != nonStreamReasoning[0].Text || streamReasoning[1].Text != nonStreamReasoning[1].Text {
		t.Fatalf("stream/non-stream reasoning text mismatch: %+v vs %+v", streamReasoning, nonStreamReasoning)
	}
}

func TestOpenAIStreamTerminalAndErrorEvents(t *testing.T) {
	t.Run("incomplete", func(t *testing.T) {
		finalResponseRaw := `{"id":"resp_1","model":"gpt-test","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"id":"msg_1","type":"message","role":"assistant","status":"incomplete","content":[{"type":"output_text","text":"partial","annotations":[]}]}],"usage":{"input_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":3}}`
		resp := collectOpenAIStream(t, []string{
			`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`,
			`{"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"partial","logprobs":[]}`,
			`{"type":"response.incomplete","sequence_number":2,"response":` + finalResponseRaw + `}`,
		})
		if resp.StopReason != llm.StopReasonMaxTokens || resp.StopReasonRaw != "incomplete:max_output_tokens" {
			t.Fatalf("stop reason = %q/%q", resp.StopReason, resp.StopReasonRaw)
		}
		if resp.Text() != "partial" {
			t.Fatalf("text = %q", resp.Text())
		}
	})

	t.Run("failed", func(t *testing.T) {
		state := (&Provider{}).adapter().NewStreamState()
		var created responses.ResponseStreamEventUnion
		if err := json.Unmarshal([]byte(`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`), &created); err != nil {
			t.Fatalf("unmarshal created event: %v", err)
		}
		if _, err := state.MapEvent(created); err != nil {
			t.Fatalf("mapEvent(created) returned error: %v", err)
		}
		var failed responses.ResponseStreamEventUnion
		raw := `{"type":"response.failed","sequence_number":1,"response":{"id":"resp_1","model":"gpt-test","status":"failed","output":[],"error":{"code":"server_error","message":"boom"}}}`
		if err := json.Unmarshal([]byte(raw), &failed); err != nil {
			t.Fatalf("unmarshal failed event: %v", err)
		}
		_, err := state.MapEvent(failed)
		if !errors.Is(err, llm.ErrServer) {
			t.Fatalf("response.failed error = %v, want ErrServer", err)
		}
		var providerErr *llm.ProviderError
		if !errors.As(err, &providerErr) || providerErr.Code != "server_error" {
			t.Fatalf("provider error = %+v", providerErr)
		}
	})

	t.Run("error event", func(t *testing.T) {
		state := (&Provider{}).adapter().NewStreamState()
		var errEvent responses.ResponseStreamEventUnion
		raw := `{"type":"error","code":"rate_limit_exceeded","message":"slow down","param":"","sequence_number":0}`
		if err := json.Unmarshal([]byte(raw), &errEvent); err != nil {
			t.Fatalf("unmarshal error event: %v", err)
		}
		_, err := state.MapEvent(errEvent)
		if !errors.Is(err, llm.ErrRateLimited) {
			t.Fatalf("error event = %v, want ErrRateLimited", err)
		}
	})
}

func TestOpenAIExtrasAnnotationsBothPaths(t *testing.T) {
	finalResponseRaw := `{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"cited","annotations":[{"type":"url_citation","url":"https://example.test/doc","title":"Doc","start_index":0,"end_index":5}]}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}`

	nonStreaming, err := (&Provider{}).adapter().MapResponse(mustOpenAIResponse(t, finalResponseRaw))
	if err != nil {
		t.Fatalf("mapResponse returned error: %v", err)
	}
	extras, ok := Extras(nonStreaming)
	if !ok || len(extras.Annotations) != 1 || extras.Annotations[0].URL != "https://example.test/doc" {
		t.Fatalf("non-stream extras = %+v ok=%v", extras, ok)
	}

	resp := collectOpenAIStream(t, []string{
		`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`,
		`{"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"cited","logprobs":[]}`,
		`{"type":"response.completed","sequence_number":2,"response":` + finalResponseRaw + `}`,
	})
	extras, ok = Extras(resp)
	if !ok || len(extras.Annotations) != 1 || extras.Annotations[0].URL != "https://example.test/doc" {
		t.Fatalf("stream extras = %+v ok=%v", extras, ok)
	}

	if extras, ok := Extras(&llm.Response{Provider: "someone-else"}); ok || extras != nil {
		t.Fatalf("foreign provider extras = %+v ok=%v", extras, ok)
	}
}

func TestOpenAIStreamBuffersToolArgsUntilName(t *testing.T) {
	tests := []struct {
		name   string
		events []string
		wantID string
	}{
		{
			name:   "output item supplies name",
			wantID: "call_1",
			events: []string{
				`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`,
				`{"type":"response.function_call_arguments.delta","sequence_number":1,"item_id":"fc_1","output_index":0,"delta":"{\"q\":"}`,
				`{"type":"response.function_call_arguments.delta","sequence_number":2,"item_id":"fc_1","output_index":0,"delta":"\"go\"}"}`,
				`{"type":"response.output_item.added","sequence_number":3,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"","status":"in_progress"}}`,
				`{"type":"response.output_item.done","sequence_number":4,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}}`,
			},
		},
		{
			name:   "arguments done supplies name",
			wantID: "call_1",
			events: []string{
				`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`,
				`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"","arguments":"","status":"in_progress"}}`,
				`{"type":"response.function_call_arguments.delta","sequence_number":2,"item_id":"fc_1","output_index":0,"delta":"{\"q\":"}`,
				`{"type":"response.function_call_arguments.delta","sequence_number":3,"item_id":"fc_1","output_index":0,"delta":"\"go\"}"}`,
				`{"type":"response.function_call_arguments.done","sequence_number":4,"item_id":"fc_1","output_index":0,"name":"lookup","arguments":"{\"q\":\"go\"}"}`,
				`{"type":"response.output_item.done","sequence_number":5,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}}`,
			},
		},
		{
			name:   "output item done supplies call id",
			wantID: "call_late",
			events: []string{
				`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`,
				`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"","name":"lookup","arguments":"","status":"in_progress"}}`,
				`{"type":"response.function_call_arguments.delta","sequence_number":2,"item_id":"fc_1","output_index":0,"delta":"{\"q\":"}`,
				`{"type":"response.function_call_arguments.delta","sequence_number":3,"item_id":"fc_1","output_index":0,"delta":"\"go\"}"}`,
				`{"type":"response.output_item.done","sequence_number":4,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_late","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}}`,
			},
		},
		{
			name:   "output item done supplies first function metadata",
			wantID: "call_done",
			events: []string{
				`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`,
				`{"type":"response.output_item.done","sequence_number":1,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_done","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}}`,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := collectOpenAIStream(t, tt.events)
			calls := resp.ToolCalls()
			if len(calls) != 1 {
				t.Fatalf("tool calls = %+v, want one", calls)
			}
			if calls[0].ID != tt.wantID || calls[0].Name != "lookup" || string(calls[0].Args) != `{"q":"go"}` {
				t.Fatalf("tool call = %+v", calls[0])
			}
		})
	}
}

func TestOpenAIStreamRefusalEvents(t *testing.T) {
	finalResponseRaw := `{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"refusal","refusal":"no"}]}]}`
	tests := []struct {
		name   string
		events []string
	}{
		{
			name: "delta then done",
			events: []string{
				`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`,
				`{"type":"response.refusal.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"no"}`,
				`{"type":"response.refusal.done","sequence_number":2,"item_id":"msg_1","output_index":0,"content_index":0,"refusal":"no"}`,
				`{"type":"response.completed","sequence_number":3,"response":` + finalResponseRaw + `}`,
			},
		},
		{
			name: "done only",
			events: []string{
				`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`,
				`{"type":"response.refusal.done","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"refusal":"no"}`,
				`{"type":"response.completed","sequence_number":2,"response":` + finalResponseRaw + `}`,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := collectOpenAIStream(t, tt.events)
			if resp.Text() != "no" || resp.StopReason != llm.StopReasonRefusal {
				t.Fatalf("response text/stop = %q/%q", resp.Text(), resp.StopReason)
			}
		})
	}
}

func TestOpenAIErrorMapping(t *testing.T) {
	tests := []struct {
		name   string
		status int
		code   string
		typ    string
		msg    string
		want   error
	}{
		{name: "auth", status: 401, code: "invalid_api_key", typ: "invalid_request_error", msg: "bad key", want: llm.ErrAuth},
		{name: "rate", status: 429, code: "rate_limit_exceeded", typ: "rate_limit_error", msg: "slow down", want: llm.ErrRateLimited},
		{name: "quota", status: 429, code: "insufficient_quota", typ: "insufficient_quota", msg: "quota", want: llm.ErrInsufficientCredits},
		{name: "context", status: 400, code: "context_length_exceeded", typ: "invalid_request_error", msg: "maximum context length", want: llm.ErrContextTooLong},
		{name: "overload", status: 503, code: "server_error", typ: "server_error", msg: "try again", want: llm.ErrOverloaded},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var apiErr sdk.Error
			raw := `{"code":` + strconvQuote(tt.code) + `,"message":` + strconvQuote(tt.msg) + `,"param":"","type":` + strconvQuote(tt.typ) + `}`
			if err := json.Unmarshal([]byte(raw), &apiErr); err != nil {
				t.Fatalf("unmarshal api error: %v", err)
			}
			apiErr.StatusCode = tt.status
			apiErr.Response = &http.Response{Header: http.Header{"Retry-After": []string{"2"}}}
			err := mapError(&apiErr)
			if !errors.Is(err, tt.want) {
				t.Fatalf("error = %v, want %v", err, tt.want)
			}
			var providerErr *llm.ProviderError
			if !errors.As(err, &providerErr) {
				t.Fatalf("error does not unwrap to ProviderError: %v", err)
			}
			if providerErr.RetryAfter != 2*time.Second || providerErr.Code != tt.code {
				t.Fatalf("provider error = %+v", providerErr)
			}
		})
	}
}

func TestOpenAIOptionsAndDebugCapture(t *testing.T) {
	var sawAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"pong","annotations":[]}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}`)
	}))
	defer server.Close()

	var captures []llm.WireCapture
	p, err := New(
		WithAPIKeyFunc(func(context.Context) (string, error) { return "dynamic-secret", nil }),
		WithBaseURL(server.URL),
		WithHTTPClient(server.Client()),
		WithMaxRetries(0),
		WithWireCapture(func(c llm.WireCapture) { captures = append(captures, c) }),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if p.Client() == nil {
		t.Fatalf("Client returned nil")
	}
	resp, err := p.Chat(context.Background(), &llm.Request{
		Model:    "gpt-test",
		Messages: []llm.Message{llm.UserText("ping")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Text() != "pong" || sawAuth != "Bearer dynamic-secret" {
		t.Fatalf("response/auth = %q/%q", resp.Text(), sawAuth)
	}
	if len(captures) != 1 {
		t.Fatalf("captures len = %d, want 1", len(captures))
	}
	if got := captures[0].RequestHeaders.Get("Authorization"); got != "[REDACTED]" {
		t.Fatalf("captured Authorization = %q, want redacted", got)
	}
	if bytes.Contains(captures[0].RequestBody, []byte("dynamic-secret")) || bytes.Contains(captures[0].ResponseBody, []byte("dynamic-secret")) {
		t.Fatalf("capture leaked API key: %+v", captures[0])
	}
}

func TestOpenAINewNeutralizesAmbientSDKEnv(t *testing.T) {
	t.Setenv(apiKeyEnv, "env-secret")
	t.Setenv("OPENAI_BASE_URL", "https://env.example.test/v1/")
	t.Setenv("OPENAI_ORG_ID", "env-org")
	t.Setenv("OPENAI_PROJECT_ID", "env-project")
	t.Setenv("OPENAI_ADMIN_KEY", "env-admin")
	t.Setenv(customHeadersEnv, "X-Ambient-Safe: retained\nAuthorization: Bearer custom-secret")

	var sawURLs []string
	var sawHeaders []http.Header
	client := &http.Client{Transport: testutil.RoundTripFunc(func(req *http.Request) (*http.Response, error) {
		sawURLs = append(sawURLs, req.URL.String())
		sawHeaders = append(sawHeaders, req.Header.Clone())
		var body []byte
		if req.Body != nil {
			body, _ = io.ReadAll(req.Body)
		}
		contentType := "application/json"
		responseBody := ""
		switch {
		case strings.HasSuffix(req.URL.Path, "/models"):
			responseBody = `{"object":"list","data":[{"id":"gpt-test","created":0,"object":"model","owned_by":"openai"}]}`
		case bytes.Contains(body, []byte(`"stream":true`)):
			contentType = "text/event-stream"
			responseBody = "event: response.output_text.delta\n" +
				`data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"pong","logprobs":[]}` + "\n\n" +
				"event: response.completed\n" +
				`data: {"type":"response.completed","sequence_number":2,"response":{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"pong","annotations":[]}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}}` + "\n\n"
		default:
			responseBody = `{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"pong","annotations":[]}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}`
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{contentType}},
			Body:       io.NopCloser(strings.NewReader(responseBody)),
			Request:    req,
		}, nil
	})}

	p, err := New(WithAPIKey("explicit-secret"), WithHTTPClient(client), WithMaxRetries(0))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	resp, err := p.Chat(context.Background(), &llm.Request{
		Model:    "gpt-test",
		Messages: []llm.Message{llm.UserText("ping")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Text() != "pong" {
		t.Fatalf("response text = %q", resp.Text())
	}
	streamed, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{
		Model:    "gpt-test",
		Messages: []llm.Message{llm.UserText("ping")},
	}))
	if err != nil {
		t.Fatalf("ChatStream returned error: %v", err)
	}
	if streamed.Text() != "pong" {
		t.Fatalf("stream response text = %q", streamed.Text())
	}
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	if len(models) != 1 || models[0].ID != "gpt-test" {
		t.Fatalf("models = %+v", models)
	}
	if len(sawHeaders) != 3 {
		t.Fatalf("requests = %d, want 3", len(sawHeaders))
	}
	for i, header := range sawHeaders {
		if !strings.HasPrefix(sawURLs[i], defaultOpenAIBaseURL) {
			t.Fatalf("request %d URL = %q, want production base URL", i, sawURLs[i])
		}
		if got := header.Get("Authorization"); got != "Bearer explicit-secret" {
			t.Fatalf("request %d Authorization = %q", i, got)
		}
		for _, key := range []string{organizationHeader, projectHeader} {
			if got := header.Get(key); got != "" {
				t.Fatalf("request %d %s header leaked from SDK env: %q", i, key, got)
			}
		}
		if got := header.Get("X-Ambient-Safe"); got != "retained" {
			t.Fatalf("request %d X-Ambient-Safe = %q", i, got)
		}
	}
}

func TestOpenAIExplicitOrganizationProjectHeaders(t *testing.T) {
	var sawHeaders http.Header
	client := &http.Client{Transport: testutil.RoundTripFunc(func(req *http.Request) (*http.Response, error) {
		sawHeaders = req.Header.Clone()
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body: io.NopCloser(strings.NewReader(
				`{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"pong","annotations":[]}]}]}`,
			)),
			Request: req,
		}, nil
	})}

	p, err := New(
		WithAPIKey("explicit-secret"),
		WithOrganization("org-explicit"),
		WithProject("proj-explicit"),
		WithHTTPClient(client),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	_, err = p.Chat(context.Background(), &llm.Request{
		Model:    "gpt-test",
		Messages: []llm.Message{llm.UserText("ping")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if sawHeaders.Get(organizationHeader) != "org-explicit" || sawHeaders.Get(projectHeader) != "proj-explicit" {
		t.Fatalf("organization/project headers = %q/%q", sawHeaders.Get(organizationHeader), sawHeaders.Get(projectHeader))
	}
}

func TestOpenAIModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"gpt-test","created":0,"object":"model","owned_by":"openai"}]}`)
	}))
	defer server.Close()

	p, err := New(WithAPIKey("test-key"), WithBaseURL(server.URL), WithHTTPClient(server.Client()), WithMaxRetries(0))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	if len(models) != 1 || models[0].ID != "gpt-test" {
		t.Fatalf("models = %+v", models)
	}
}

func TestOpenAINewRequiresAPIKey(t *testing.T) {
	t.Setenv(apiKeyEnv, "")
	_, err := New()
	if !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("New error = %v, want ErrAuth", err)
	}
}

func TestOpenAIStreamProviderContractEOF(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		wantPartial bool
	}{
		{name: "empty EOF"},
		{
			name:        "truncated EOF",
			body:        "data: {\"type\":\"response.output_text.delta\",\"output_index\":4,\"content_index\":0,\"delta\":\"partial\"}\n\n",
			wantPartial: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newOpenAIStreamFixtureProvider(t, func(req *http.Request) (*http.Response, error) {
				return openAIStreamResponse(req, tt.body), nil
			})
			resp, err := llm.Collect(p.ChatStream(context.Background(), streamFixtureRequest()))
			assertOpenAIProviderServerError(t, err)
			if tt.wantPartial {
				if resp == nil || resp.Text() != "partial" || resp.Model != "requested-model" {
					t.Fatalf("partial response = %#v, want text partial and request model", resp)
				}
			} else if resp != nil {
				t.Fatalf("empty stream response = %#v, want nil", resp)
			}
		})
	}
}

func TestOpenAIStreamProviderBuffersPreStartAndFallsBackToRequestModel(t *testing.T) {
	body := "data: {\"type\":\"response.output_text.delta\",\"output_index\":4,\"content_index\":0,\"delta\":\"hello\"}\n\n" +
		"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n"
	p := newOpenAIStreamFixtureProvider(t, func(req *http.Request) (*http.Response, error) {
		return openAIStreamResponse(req, body), nil
	})

	var events []llm.Event
	for event, err := range p.ChatStream(context.Background(), streamFixtureRequest()) {
		if err != nil {
			t.Fatalf("ChatStream returned error: %v", err)
		}
		events = append(events, event)
	}
	if len(events) != 3 {
		t.Fatalf("events = %#v, want start, text, end", events)
	}
	start, ok := events[0].(llm.MessageStart)
	if !ok || start.Model != "requested-model" {
		t.Fatalf("first event = %#v, want MessageStart with request model", events[0])
	}
	if delta, ok := events[1].(llm.TextDelta); !ok || delta.Index != 4 || delta.Text != "hello" {
		t.Fatalf("second event = %#v, want stable provider TextDelta index 4", events[1])
	}
	if _, ok := events[2].(llm.MessageEnd); !ok {
		t.Fatalf("last event = %T, want MessageEnd", events[2])
	}
}

func TestOpenAIStreamProviderFlushesPreStartContentBeforeResponseFailed(t *testing.T) {
	body := "data: {\"type\":\"response.output_text.delta\",\"output_index\":4,\"content_index\":0,\"delta\":\"partial\"}\n\n" +
		"data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_failed\",\"model\":\"failure-model\",\"status\":\"failed\",\"output\":[],\"error\":{\"code\":\"server_error\",\"message\":\"remote failure\"}}}\n\n"
	p := newOpenAIStreamFixtureProvider(t, func(req *http.Request) (*http.Response, error) {
		return openAIStreamResponse(req, body), nil
	})

	resp, err := llm.Collect(p.ChatStream(context.Background(), streamFixtureRequest()))
	assertOpenAIProviderServerError(t, err)
	if resp == nil || resp.ID != "" || resp.Model != "requested-model" || resp.Text() != "partial" {
		t.Fatalf("partial failed response = %#v, want streamed content and original fallback identity", resp)
	}
}

func TestOpenAIStreamProviderPreservesReasoningBeforeErrors(t *testing.T) {
	tests := []struct {
		name           string
		body           string
		wantModel      string
		transportError bool
	}{
		{
			name: "semantic error",
			body: "data: {\"type\":\"response.reasoning_text.delta\",\"output_index\":0,\"delta\":\"private thought\"}\n\n" +
				"data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_failed\",\"model\":\"failure-model\",\"status\":\"failed\",\"output\":[],\"error\":{\"code\":\"server_error\",\"message\":\"remote failure\"}}}\n\n",
			wantModel: "requested-model",
		},
		{
			name: "transport error",
			body: "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"response-model\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
				"data: {\"type\":\"response.reasoning_text.delta\",\"output_index\":0,\"delta\":\"private thought\"}\n\n",
			wantModel:      "response-model",
			transportError: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newOpenAIStreamFixtureProvider(t, func(req *http.Request) (*http.Response, error) {
				resp := openAIStreamResponse(req, tt.body)
				if tt.transportError {
					resp.Body = &errorAfterReadCloser{
						Reader: strings.NewReader(tt.body),
						err:    errors.New("remote body read failed"),
					}
				}
				return resp, nil
			})
			resp, err := llm.Collect(p.ChatStream(context.Background(), streamFixtureRequest()))
			assertOpenAIProviderServerError(t, err)
			if resp == nil || resp.Model != tt.wantModel || resp.Reasoning() != "private thought" {
				t.Fatalf("partial reasoning response = %#v, want model %q and preserved reasoning", resp, tt.wantModel)
			}
		})
	}
}

func TestOpenAIStreamProviderPreservesInterleavedContentBeforeTransportError(t *testing.T) {
	body := "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"response-model\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
		"data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_partial\",\"type\":\"function_call\",\"call_id\":\"call_partial\",\"name\":\"lookup\",\"arguments\":\"\",\"status\":\"in_progress\"}}\n\n" +
		"data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"q\\\":\"}\n\n" +
		"data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"content_index\":0,\"delta\":\"deferred text\"}\n\n" +
		"data: {\"type\":\"response.reasoning_text.delta\",\"output_index\":2,\"delta\":\"deferred reasoning\"}\n\n"
	p := newOpenAIStreamFixtureProvider(t, func(req *http.Request) (*http.Response, error) {
		resp := openAIStreamResponse(req, body)
		resp.Body = &errorAfterReadCloser{
			Reader: strings.NewReader(body),
			err:    errors.New("remote body read failed"),
		}
		return resp, nil
	})

	resp, err := llm.Collect(p.ChatStream(context.Background(), streamFixtureRequest()))
	assertOpenAIProviderServerError(t, err)
	if resp == nil || resp.Text() != "deferred text" || resp.Reasoning() != "deferred reasoning" {
		t.Fatalf("partial response = %#v, want all deferred wire content", resp)
	}
	if calls := resp.ToolCalls(); len(calls) != 1 || string(calls[0].Args) != `{"q":` {
		t.Fatalf("partial tool calls = %#v, want visible incomplete arguments", calls)
	}
}

func TestOpenAIStreamProviderErrorOnlyFirstEventReturnsNilResponse(t *testing.T) {
	p := newOpenAIStreamFixtureProvider(t, func(req *http.Request) (*http.Response, error) {
		return openAIStreamResponse(req, "data: {\"type\":\"error\",\"code\":\"server_error\",\"message\":\"failed\"}\n\n"), nil
	})
	resp, err := llm.Collect(p.ChatStream(context.Background(), streamFixtureRequest()))
	assertOpenAIProviderServerError(t, err)
	if resp != nil {
		t.Fatalf("response = %#v, want nil before identity or content", resp)
	}
}

func TestOpenAIStreamProviderEarlyBreakDoesNotSynthesizeError(t *testing.T) {
	body := "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-test\",\"status\":\"in_progress\",\"output\":[]}}\n\n" +
		"data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"unread\"}\n\n"
	p := newOpenAIStreamFixtureProvider(t, func(req *http.Request) (*http.Response, error) {
		return openAIStreamResponse(req, body), nil
	})

	count := 0
	for event, err := range p.ChatStream(context.Background(), streamFixtureRequest()) {
		if err != nil {
			t.Fatalf("early-break event returned error: %v", err)
		}
		if _, ok := event.(llm.MessageStart); !ok {
			t.Fatalf("first event = %T, want MessageStart", event)
		}
		count++
		break
	}
	if count != 1 {
		t.Fatalf("events before break = %d, want 1", count)
	}
}

func TestOpenAIStreamProviderNormalizesDecodeAndTransportErrors(t *testing.T) {
	t.Run("decode", func(t *testing.T) {
		p := newOpenAIStreamFixtureProvider(t, func(req *http.Request) (*http.Response, error) {
			return openAIStreamResponse(req, "data: {not-json}\n\n"), nil
		})
		_, err := llm.Collect(p.ChatStream(context.Background(), streamFixtureRequest()))
		assertOpenAIProviderServerError(t, err)
	})

	t.Run("transport", func(t *testing.T) {
		p := newOpenAIStreamFixtureProvider(t, func(*http.Request) (*http.Response, error) {
			return nil, errors.New("remote transport failed")
		})
		_, err := llm.Collect(p.ChatStream(context.Background(), streamFixtureRequest()))
		assertOpenAIProviderServerError(t, err)
	})
}

func newOpenAIStreamFixtureProvider(t *testing.T, roundTrip testutil.RoundTripFunc) *Provider {
	t.Helper()
	p, err := New(
		WithAPIKey("test-key"),
		WithHTTPClient(&http.Client{Transport: roundTrip}),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	return p
}

func openAIStreamResponse(req *http.Request, body string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    req,
	}
}

type errorAfterReadCloser struct {
	io.Reader
	err      error
	returned bool
}

func (r *errorAfterReadCloser) Read(p []byte) (int, error) {
	n, err := r.Reader.Read(p)
	if err == io.EOF && !r.returned {
		r.returned = true
		return 0, r.err
	}
	return n, err
}

func (*errorAfterReadCloser) Close() error { return nil }

func streamFixtureRequest() *llm.Request {
	return &llm.Request{
		Model:    "requested-model",
		Messages: []llm.Message{llm.UserText("hello")},
	}
}

func assertOpenAIProviderServerError(t *testing.T, err error) {
	t.Helper()
	if !errors.Is(err, llm.ErrServer) {
		t.Fatalf("error = %v, want ErrServer", err)
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Provider != providerName {
		t.Fatalf("error = %#v, want %s ProviderError", err, providerName)
	}
}

func mustOpenAIResponse(t *testing.T, raw string) *responses.Response {
	t.Helper()
	var resp responses.Response
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("unmarshal response: %v\n%s", err, raw)
	}
	return &resp
}

func collectOpenAIStream(t *testing.T, rawEvents []string) *llm.Response {
	t.Helper()
	state := (&Provider{}).adapter().NewStreamState()
	return testutil.CollectRawEvents(t, rawEvents, state.MapEvent)
}

func reasoningParts(parts []llm.Part) []llm.ReasoningPart {
	var out []llm.ReasoningPart
	for _, part := range parts {
		if reasoning, ok := part.(llm.ReasoningPart); ok {
			out = append(out, reasoning)
		}
	}
	return out
}

func strconvQuote(s string) string {
	raw, _ := json.Marshal(s)
	return string(raw)
}
