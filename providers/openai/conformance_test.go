package openai

import (
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

// TestOpenAIConformance machine-checks the llm.Provider contract
// (single-use streams, cancellation, concurrency, panic-freedom, Collect
// partial shape) against a fixture server speaking the Responses wire shape.
func TestOpenAIConformance(t *testing.T) {
	llmtest.RunConformance(t, func(t *testing.T) llm.Provider {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			scenario := llmtest.ConformanceScenarioFromRequest(r)
			body, _ := io.ReadAll(r.Body)
			if strings.Contains(string(body), `"stream":true`) {
				w.Header().Set("Content-Type", "text/event-stream")
				if scenario == llmtest.ConformanceEmpty {
					return
				}
				_, _ = io.WriteString(w, `event: response.created`+"\n")
				_, _ = io.WriteString(w, `data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`+"\n\n")
				switch scenario {
				case llmtest.ConformanceCancel:
					w.(http.Flusher).Flush()
					<-r.Context().Done()
					return
				case llmtest.ConformanceTruncated:
					return
				case llmtest.ConformanceTools:
					_, _ = io.WriteString(w, `event: response.output_item.added`+"\n")
					_, _ = io.WriteString(w, `data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_conformance","name":"conformance_echo","arguments":"","status":"in_progress"}}`+"\n\n")
					_, _ = io.WriteString(w, `event: response.function_call_arguments.delta`+"\n")
					_, _ = io.WriteString(w, `data: {"type":"response.function_call_arguments.delta","sequence_number":2,"item_id":"fc_1","output_index":0,"delta":"{\"value\":\"pong\"}"}`+"\n\n")
					_, _ = io.WriteString(w, `event: response.output_item.done`+"\n")
					_, _ = io.WriteString(w, `data: {"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_conformance","name":"conformance_echo","arguments":"{\"value\":\"pong\"}","status":"completed"}}`+"\n\n")
					_, _ = io.WriteString(w, `event: response.completed`+"\n")
					_, _ = io.WriteString(w, `data: {"type":"response.completed","sequence_number":4,"response":{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"fc_1","type":"function_call","call_id":"call_conformance","name":"conformance_echo","arguments":"{\"value\":\"pong\"}","status":"completed"}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}}`+"\n\n")
					return
				}
				_, _ = io.WriteString(w, `event: response.output_text.delta`+"\n")
				_, _ = io.WriteString(w, `data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"pong","logprobs":[]}`+"\n\n")
				_, _ = io.WriteString(w, `event: response.completed`+"\n")
				_, _ = io.WriteString(w, `data: {"type":"response.completed","sequence_number":2,"response":{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"pong","annotations":[]}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}}`+"\n\n")
				return
			}
			if scenario == llmtest.ConformanceCancel {
				<-r.Context().Done()
				return
			}
			w.Header().Set("Content-Type", "application/json")
			if scenario == llmtest.ConformanceTools {
				_, _ = io.WriteString(w, `{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"fc_1","type":"function_call","call_id":"call_conformance","name":"conformance_echo","arguments":"{\"value\":\"pong\"}","status":"completed"}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}`)
				return
			}
			_, _ = io.WriteString(w, `{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"pong","annotations":[]}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}`)
		}))
		t.Cleanup(server.Close)

		p, err := New(
			WithAPIKey("test-key"),
			WithBaseURL(server.URL),
			WithHTTPClient(server.Client()),
			WithMaxRetries(0),
		)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		return p
	})
}

// TestOpenAICapabilityConformance proves reviewed Responses API fields and
// their normalized results without credentials. Reasoning-summary selection
// remains covered by its dedicated focused tests.
func TestOpenAICapabilityConformance(t *testing.T) {
	const model = "gpt-5.4-mini"
	profile := llmtest.CapabilityProfile{Cases: []llmtest.CapabilityCase{
		{Name: "tools", Capability: llm.CapabilityTools, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request { return testutil.ToolActivationRequest(model, llm.ToolChoiceAuto) }, Assert: testutil.AssertActivationToolCall},
		{Name: "required-tool-choice", Capability: llm.CapabilityToolChoiceRequired, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request { return testutil.ToolActivationRequest(model, llm.ToolChoiceRequired) }, Assert: testutil.AssertActivationToolCall},
		{Name: "json-schema", Capability: llm.CapabilityJSONSchema, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request { return testutil.JSONSchemaActivationRequest(model) }, Assert: testutil.AssertActivationText},
		{Name: "image-input", Capability: llm.CapabilityImageInput, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request { return testutil.ImageActivationRequest(model) }, Assert: testutil.AssertActivationText},
		{Name: "prompt-cache-key", Capability: llm.CapabilityPromptCaching, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request {
			return &llm.Request{Model: model, Messages: []llm.Message{llm.UserText("cache")}, SessionID: "session.activation/1"}
		}, Assert: testutil.AssertActivationText},
		{Name: "reasoning", Capability: llm.CapabilityReasoning, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request { return testutil.ReasoningActivationRequest(model) }, Assert: testutil.AssertActivationReasoning},
	}}

	llmtest.RunCapabilityConformance(t, func(t *testing.T, invocation llmtest.CapabilityInvocation) llm.Provider {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode activation request: %v", err)
				http.Error(w, "invalid JSON", http.StatusBadRequest)
				return
			}
			if err := assertOpenAIActivationRequest(body, invocation); err != nil {
				t.Errorf("%s native request: %v; body=%#v", invocation.CaseName, err, body)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, openAIActivationResponse(invocation, model))
		}))
		t.Cleanup(server.Close)
		provider, err := New(WithAPIKey("test-key"), WithBaseURL(server.URL), WithHTTPClient(server.Client()), WithMaxRetries(0))
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		return provider
	}, profile)
}

func assertOpenAIActivationRequest(body map[string]any, invocation llmtest.CapabilityInvocation) error {
	if body["model"] != "gpt-5.4-mini" {
		return fmt.Errorf("model = %#v", body["model"])
	}
	switch invocation.Capability {
	case llm.CapabilityTools, llm.CapabilityToolChoiceRequired:
		tools, _ := body["tools"].([]any)
		if len(tools) != 1 {
			return fmt.Errorf("tools = %#v", body["tools"])
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
		if len(input) != 1 {
			return fmt.Errorf("input = %#v", body["input"])
		}
		content, _ := input[0].(map[string]any)["content"].([]any)
		if len(content) != 2 {
			return fmt.Errorf("image content = %#v", input[0])
		}
		image, _ := content[1].(map[string]any)
		if image["type"] != "input_image" || image["image_url"] != "data:image/png;base64,AQID" {
			return fmt.Errorf("input_image = %#v", image)
		}
	case llm.CapabilityPromptCaching:
		if body["prompt_cache_key"] != "session.activation/1" {
			return fmt.Errorf("prompt_cache_key = %#v", body["prompt_cache_key"])
		}
	case llm.CapabilityReasoning:
		reasoning, _ := body["reasoning"].(map[string]any)
		if reasoning["effort"] != "high" {
			return fmt.Errorf("reasoning.effort = %#v", reasoning["effort"])
		}
	}
	return nil
}

func openAIActivationResponse(invocation llmtest.CapabilityInvocation, model string) string {
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
	payload := map[string]any{"id": "resp_activation", "model": model, "status": "completed", "output": output, "usage": usage}
	encoded, _ := json.Marshal(payload)
	return string(encoded)
}

// TestProviderIdentitySurface pins the trivial identity accessors.
func TestProviderIdentitySurface(t *testing.T) {
	p, err := New(WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if p.Name() != "openai" {
		t.Fatalf("Name = %q, want openai", p.Name())
	}
	caps := p.Capabilities()
	if len(caps) == 0 || caps[0] != llm.CapabilityStreaming {
		t.Fatalf("capabilities = %+v", caps)
	}
}
