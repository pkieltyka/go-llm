package openaicodex

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
	"github.com/pkieltyka/go-llm/llmtest"
)

const (
	codexActivationLiteModel   = "gpt-5.6-sol"
	codexActivationLegacyModel = "gpt-5.5"
)

// TestOpenAICodexConformance machine-checks the llm.Provider contract
// against a fixture server speaking the codex SSE wire shape (the codex
// backend streams both blocking and streaming calls).
func TestOpenAICodexConformance(t *testing.T) {
	llmtest.RunConformance(t, func(t *testing.T) llm.Provider {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch llmtest.ConformanceScenarioFromRequest(r) {
			case llmtest.ConformanceEmpty:
				w.Header().Set("Content-Type", "text/event-stream")
				return
			case llmtest.ConformanceCancel:
				writeCodexSSEStart(w)
				w.(http.Flusher).Flush()
				<-r.Context().Done()
				return
			case llmtest.ConformanceTruncated:
				writeCodexSSEStart(w)
				return
			case llmtest.ConformanceTools:
				writeCodexSSETools(w)
				return
			}
			writeCodexSSESuccess(w)
		}))
		t.Cleanup(server.Close)

		p, err := New(
			WithOAuth(llm.AuthCredential{Type: "oauth", Access: fakeCodexJWT(t, "acct-conformance"), Refresh: "refresh"}, func(ctx context.Context, _ llm.AuthCredential) error {
				return ctx.Err()
			}),
			WithBaseURL(server.URL),
			withOAuthTokenURL(server.URL+"/oauth/token"),
			WithHTTPClient(server.Client()),
			WithMaxRetries(0),
		)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		return p
	})
}

// TestOpenAICodexCapabilityConformance pins native activation for both the
// gpt-5.6 Responses Lite contract and the legacy Codex Responses contract.
// Every case carries a real explicit model; the runner never selects one.
func TestOpenAICodexCapabilityConformance(t *testing.T) {
	toolRequest := func(model string, choice llm.ToolChoiceMode) func() *llm.Request {
		return func() *llm.Request {
			request := testutil.ToolActivationRequest(model, choice)
			request.System = "activation system"
			return request
		}
	}
	profile := llmtest.CapabilityProfile{Cases: []llmtest.CapabilityCase{
		{Name: "tools-lite", Capability: llm.CapabilityTools, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: toolRequest(codexActivationLiteModel, llm.ToolChoiceAuto), Assert: assertCodexActivationResponse(codexActivationLiteModel, testutil.AssertActivationToolCall)},
		{Name: "tools-legacy", Capability: llm.CapabilityTools, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: toolRequest(codexActivationLegacyModel, llm.ToolChoiceAuto), Assert: assertCodexActivationResponse(codexActivationLegacyModel, testutil.AssertActivationToolCall)},
		{Name: "required-tool-choice-lite", Capability: llm.CapabilityToolChoiceRequired, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: toolRequest(codexActivationLiteModel, llm.ToolChoiceRequired), Assert: assertCodexActivationResponse(codexActivationLiteModel, testutil.AssertActivationToolCall)},
		{Name: "required-tool-choice-legacy", Capability: llm.CapabilityToolChoiceRequired, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: toolRequest(codexActivationLegacyModel, llm.ToolChoiceRequired), Assert: assertCodexActivationResponse(codexActivationLegacyModel, testutil.AssertActivationToolCall)},
		{Name: "json-schema-lite", Capability: llm.CapabilityJSONSchema, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request { return testutil.JSONSchemaActivationRequest(codexActivationLiteModel) }, Assert: assertCodexActivationResponse(codexActivationLiteModel, testutil.AssertActivationText)},
		{Name: "image-input-lite", Capability: llm.CapabilityImageInput, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request { return testutil.ImageActivationRequest(codexActivationLiteModel) }, Assert: assertCodexActivationResponse(codexActivationLiteModel, testutil.AssertActivationText)},
		{Name: "prompt-cache-key-lite", Capability: llm.CapabilityPromptCaching, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request {
			return &llm.Request{Model: codexActivationLiteModel, Messages: []llm.Message{llm.UserText("cache")}, SessionID: "codex.activation/1"}
		}, Assert: assertCodexActivationResponse(codexActivationLiteModel, testutil.AssertActivationText)},
		{Name: "reasoning-lite", Capability: llm.CapabilityReasoning, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request { return testutil.ReasoningActivationRequest(codexActivationLiteModel) }, Assert: assertCodexActivationResponse(codexActivationLiteModel, testutil.AssertActivationReasoning)},
		{Name: "reasoning-legacy", Capability: llm.CapabilityReasoning, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request { return testutil.ReasoningActivationRequest(codexActivationLegacyModel) }, Assert: assertCodexActivationResponse(codexActivationLegacyModel, testutil.AssertActivationReasoning)},
	}}

	llmtest.RunCapabilityConformance(t, func(t *testing.T, invocation llmtest.CapabilityInvocation) llm.Provider {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			expectedModel, _, expectationErr := codexActivationExpectation(invocation)
			if expectationErr != nil {
				t.Errorf("activation expectation: %v", expectationErr)
				http.Error(w, "invalid fixture invocation", http.StatusInternalServerError)
				return
			}
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode activation request: %v", err)
				http.Error(w, "invalid JSON", http.StatusBadRequest)
				return
			}
			if err := assertCodexActivationRequest(r, body, invocation); err != nil {
				t.Errorf("%s native request: %v; body=%#v", invocation.CaseName, err, body)
			}
			writeCodexActivationResponse(w, invocation, expectedModel)
		}))
		t.Cleanup(server.Close)
		provider, err := New(
			WithOAuth(llm.AuthCredential{Type: "oauth", Access: fakeCodexJWT(t, "acct-capability"), Refresh: "refresh"}, func(ctx context.Context, _ llm.AuthCredential) error { return ctx.Err() }),
			WithBaseURL(server.URL),
			withOAuthTokenURL(server.URL+"/oauth/token"),
			WithHTTPClient(server.Client()),
			WithMaxRetries(0),
		)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		return provider
	}, profile)
}

func assertCodexActivationRequest(r *http.Request, body map[string]any, invocation llmtest.CapabilityInvocation) error {
	expectedModel, lite, err := codexActivationExpectation(invocation)
	if err != nil {
		return err
	}
	if model, _ := body["model"].(string); model != expectedModel {
		return fmt.Errorf("model = %q, want %q", model, expectedModel)
	}
	wantLiteHeader := ""
	if lite {
		wantLiteHeader = "true"
	}
	if got := r.Header.Get(codexResponsesLiteHeader); got != wantLiteHeader {
		return fmt.Errorf("Lite header = %q, want %q", got, wantLiteHeader)
	}
	if body["stream"] != true {
		return fmt.Errorf("stream = %#v, want true", body["stream"])
	}
	reasoning, _ := body["reasoning"].(map[string]any)
	if lite {
		if body["parallel_tool_calls"] != false || reasoning["context"] != "all_turns" {
			return fmt.Errorf("Lite branch parallel_tool_calls=%#v reasoning=%#v", body["parallel_tool_calls"], reasoning)
		}
	} else if _, ok := body["parallel_tool_calls"]; ok || reasoning["context"] != nil {
		return fmt.Errorf("legacy branch gained Lite fields: parallel=%#v reasoning=%#v", body["parallel_tool_calls"], reasoning)
	}

	switch invocation.Capability {
	case llm.CapabilityTools, llm.CapabilityToolChoiceRequired:
		var tools []any
		if lite {
			if _, exists := body["tools"]; exists || body["instructions"] != nil {
				return fmt.Errorf("Lite tools/instructions remained top-level")
			}
			input, _ := body["input"].([]any)
			if len(input) < 3 {
				return fmt.Errorf("Lite input prefix = %#v", body["input"])
			}
			additional, _ := input[0].(map[string]any)
			developer, _ := input[1].(map[string]any)
			tools, _ = additional["tools"].([]any)
			developerContent, _ := developer["content"].([]any)
			if additional["type"] != "additional_tools" || additional["role"] != "developer" || developer["type"] != "message" || developer["role"] != "developer" || len(developerContent) != 1 {
				return fmt.Errorf("Lite developer prefix = %#v %#v", additional, developer)
			}
			instruction, _ := developerContent[0].(map[string]any)
			if instruction["type"] != "input_text" || instruction["text"] != "activation system" {
				return fmt.Errorf("Lite developer instruction = %#v", instruction)
			}
		} else {
			tools, _ = body["tools"].([]any)
			if body["instructions"] != "activation system" {
				return fmt.Errorf("legacy instructions = %#v", body["instructions"])
			}
		}
		if len(tools) != 1 {
			return fmt.Errorf("native tools = %#v", tools)
		}
		tool, _ := tools[0].(map[string]any)
		if tool["type"] != "function" || tool["name"] != testutil.ActivationToolName || tool["strict"] != true {
			return fmt.Errorf("function tool = %#v", tool)
		}
		if err := testutil.AssertActivationToolSchema(tool["parameters"]); err != nil {
			return err
		}
		if invocation.Capability == llm.CapabilityToolChoiceRequired && body["tool_choice"] != "required" {
			return fmt.Errorf("tool_choice = %#v, want required", body["tool_choice"])
		}
	case llm.CapabilityJSONSchema:
		text, _ := body["text"].(map[string]any)
		format, _ := text["format"].(map[string]any)
		if format["type"] != "json_schema" || format["name"] != "activation_result" || format["strict"] != true {
			return fmt.Errorf("text.format = %#v", format)
		}
		if err := testutil.AssertActivationResponseSchema(format["schema"]); err != nil {
			return err
		}
	case llm.CapabilityImageInput:
		input, _ := body["input"].([]any)
		content, _ := input[0].(map[string]any)["content"].([]any)
		image, _ := content[1].(map[string]any)
		if image["type"] != "input_image" || image["image_url"] != "data:image/png;base64,AQID" {
			return fmt.Errorf("input_image = %#v", image)
		}
	case llm.CapabilityPromptCaching:
		if body["prompt_cache_key"] != "codex.activation/1" {
			return fmt.Errorf("prompt_cache_key = %#v", body["prompt_cache_key"])
		}
	case llm.CapabilityReasoning:
		if reasoning["effort"] != "high" {
			return fmt.Errorf("reasoning.effort = %#v", reasoning["effort"])
		}
	}
	return nil
}

func codexActivationExpectation(invocation llmtest.CapabilityInvocation) (model string, lite bool, err error) {
	switch {
	case strings.HasSuffix(invocation.CaseName, "-lite"):
		return codexActivationLiteModel, true, nil
	case strings.HasSuffix(invocation.CaseName, "-legacy"):
		return codexActivationLegacyModel, false, nil
	default:
		return "", false, fmt.Errorf("case %q does not identify a Codex request branch", invocation.CaseName)
	}
}

func assertCodexActivationResponse(model string, assert func(*llm.Response) error) func(*llm.Response) error {
	return func(response *llm.Response) error {
		if response.Model != model {
			return fmt.Errorf("response model = %q, want %q", response.Model, model)
		}
		return assert(response)
	}
}

func writeCodexActivationResponse(w http.ResponseWriter, invocation llmtest.CapabilityInvocation, model string) {
	w.Header().Set("Content-Type", "text/event-stream")
	_, _ = io.WriteString(w, `data: {"type":"response.created","response":{"id":"resp_activation","model":"`+model+`","status":"in_progress","output":[]}}`+"\n\n")
	output := []any{map[string]any{
		"id": "msg_activation", "type": "message", "role": "assistant", "status": "completed",
		"content": []any{map[string]any{"type": "output_text", "text": "activated", "annotations": []any{}}},
	}}
	usage := map[string]any{
		"input_tokens": 1, "input_tokens_details": map[string]any{"cached_tokens": 0},
		"output_tokens": 2, "output_tokens_details": map[string]any{"reasoning_tokens": 0}, "total_tokens": 3,
	}
	switch invocation.Capability {
	case llm.CapabilityTools, llm.CapabilityToolChoiceRequired:
		output = []any{map[string]any{"id": "fc_activation", "type": "function_call", "call_id": "call_activation", "name": testutil.ActivationToolName, "arguments": `{"value":"activated"}`, "status": "completed"}}
	case llm.CapabilityReasoning:
		output = append([]any{map[string]any{"id": "rs_activation", "type": "reasoning", "summary": []any{map[string]any{"type": "summary_text", "text": "because"}}, "status": "completed"}}, output...)
		usage["output_tokens_details"] = map[string]any{"reasoning_tokens": 1}
	}
	payload, _ := json.Marshal(map[string]any{
		"type":     "response.completed",
		"response": map[string]any{"id": "resp_activation", "model": model, "status": "completed", "output": output, "usage": usage},
	})
	_, _ = io.WriteString(w, "data: "+string(payload)+"\n\n")
}

func writeCodexSSEStart(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	_, _ = io.WriteString(w, `data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.4-mini","status":"in_progress","output":[]}}`+"\n\n")
}

func writeCodexSSETools(w http.ResponseWriter) {
	writeCodexSSEStart(w)
	_, _ = io.WriteString(w, `data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_conformance","name":"conformance_echo","arguments":"","status":"in_progress"}}`+"\n\n")
	_, _ = io.WriteString(w, `data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"value\":\"pong\"}"}`+"\n\n")
	_, _ = io.WriteString(w, `data: {"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_conformance","name":"conformance_echo","arguments":"{\"value\":\"pong\"}","status":"completed"}}`+"\n\n")
	_, _ = io.WriteString(w, `data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.4-mini","status":"completed","output":[{"id":"fc_1","type":"function_call","call_id":"call_conformance","name":"conformance_echo","arguments":"{\"value\":\"pong\"}","status":"completed"}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}}`+"\n\n")
}
