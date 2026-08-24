package anthropic

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestAnthropicConformance machine-checks the llm.Provider contract
// (single-use streams, cancellation, concurrency, panic-freedom, Collect
// partial shape) against a fixture server speaking the Messages wire shape.
func TestAnthropicConformance(t *testing.T) {
	llmtest.RunConformance(t, func(t *testing.T) llm.Provider {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			scenario := llmtest.ConformanceScenarioFromRequest(r)
			body, _ := io.ReadAll(r.Body)
			if strings.Contains(string(body), `"stream":true`) {
				w.Header().Set("Content-Type", "text/event-stream")
				start := strings.Join([]string{
					`event: message_start`,
					`data: {"type":"message_start","message":{"id":"msg_1","model":"claude-test","usage":{"input_tokens":1,"output_tokens":0}}}`,
					``,
				}, "\n") + "\n"
				switch scenario {
				case llmtest.ConformanceEmpty:
					return
				case llmtest.ConformanceCancel:
					_, _ = io.WriteString(w, start)
					w.(http.Flusher).Flush()
					<-r.Context().Done()
					return
				case llmtest.ConformanceTruncated:
					_, _ = io.WriteString(w, start)
					return
				case llmtest.ConformanceTools:
					_, _ = io.WriteString(w, start+strings.Join([]string{
						`event: content_block_start`,
						`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_conformance","name":"conformance_echo","input":{}}}`,
						``,
						`event: content_block_delta`,
						`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"value\":\"pong\"}"}}`,
						``,
						`event: content_block_stop`,
						`data: {"type":"content_block_stop","index":0}`,
						``,
						`event: message_delta`,
						`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":1}}`,
						``,
						`event: message_stop`,
						`data: {"type":"message_stop"}`,
						``,
						``,
					}, "\n"))
					return
				}
				_, _ = io.WriteString(w, start+strings.Join([]string{
					`event: content_block_start`,
					`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"po"}}`,
					``,
					`event: content_block_delta`,
					`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ng"}}`,
					``,
					`event: content_block_stop`,
					`data: {"type":"content_block_stop","index":0}`,
					``,
					`event: message_delta`,
					`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
					``,
					`event: message_stop`,
					`data: {"type":"message_stop"}`,
					``,
					``,
				}, "\n"))
				return
			}
			if scenario == llmtest.ConformanceCancel {
				<-r.Context().Done()
				return
			}
			w.Header().Set("Content-Type", "application/json")
			if scenario == llmtest.ConformanceTools {
				_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"tool_use","id":"call_conformance","name":"conformance_echo","input":{"value":"pong"}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`)
				return
			}
			_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"pong"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
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

// TestAnthropicCapabilityConformance proves the reviewed Messages request
// fields, including explicit cache blocks and normalized cache/thinking usage.
func TestAnthropicCapabilityConformance(t *testing.T) {
	const model = "claude-sonnet-4-6"
	profile := llmtest.CapabilityProfile{Cases: []llmtest.CapabilityCase{
		{Name: "tools", Capability: llm.CapabilityTools, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request { return testutil.ToolActivationRequest(model, llm.ToolChoiceAuto) }, Assert: testutil.AssertActivationToolCall},
		{Name: "required-tool-choice", Capability: llm.CapabilityToolChoiceRequired, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request { return testutil.ToolActivationRequest(model, llm.ToolChoiceRequired) }, Assert: testutil.AssertActivationToolCall},
		{Name: "json-schema", Capability: llm.CapabilityJSONSchema, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request { return testutil.JSONSchemaActivationRequest(model) }, Assert: testutil.AssertActivationText},
		{Name: "image-input", Capability: llm.CapabilityImageInput, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request { return testutil.ImageActivationRequest(model) }, Assert: testutil.AssertActivationText},
		{Name: "prompt-cache-block", Capability: llm.CapabilityPromptCaching, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat, llmtest.ConformanceStream}, Request: func() *llm.Request {
			return &llm.Request{Model: model, Messages: []llm.Message{llm.UserParts(llm.TextPart{Text: "cache", Cache: &llm.CacheHint{TTL: time.Hour}})}}
		}, Assert: assertAnthropicCacheResponse},
		{Name: "stop-sequences", Capability: llm.CapabilityStopSequences, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat}, Request: func() *llm.Request { return testutil.StopActivationRequest(model) }, Assert: testutil.AssertActivationText},
		{Name: "reasoning", Capability: llm.CapabilityReasoning, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat, llmtest.ConformanceStream}, Request: func() *llm.Request { return testutil.ReasoningActivationRequest(model) }, Assert: testutil.AssertActivationReasoning},
	}}

	llmtest.RunCapabilityConformance(t, func(t *testing.T, invocation llmtest.CapabilityInvocation) llm.Provider {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode activation request: %v", err)
				http.Error(w, "invalid JSON", http.StatusBadRequest)
				return
			}
			if err := assertAnthropicActivationRequest(body, invocation); err != nil {
				t.Errorf("%s native request: %v; body=%#v", invocation.CaseName, err, body)
			}
			if invocation.Path == llmtest.ConformanceStream {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, anthropicActivationStream(invocation, model))
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, anthropicActivationResponse(invocation, model))
		}))
		t.Cleanup(server.Close)
		provider, err := New(WithAPIKey("test-key"), WithBaseURL(server.URL), WithHTTPClient(server.Client()), WithMaxRetries(0))
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		return provider
	}, profile)
}

func assertAnthropicActivationRequest(body map[string]any, invocation llmtest.CapabilityInvocation) error {
	if body["model"] != "claude-sonnet-4-6" {
		return fmt.Errorf("model = %#v", body["model"])
	}
	switch invocation.Capability {
	case llm.CapabilityTools, llm.CapabilityToolChoiceRequired:
		tools, _ := body["tools"].([]any)
		if len(tools) != 1 {
			return fmt.Errorf("tools = %#v", body["tools"])
		}
		tool, _ := tools[0].(map[string]any)
		if tool["type"] != "custom" || tool["name"] != testutil.ActivationToolName || tool["strict"] != true {
			return fmt.Errorf("native tool = %#v", tool)
		}
		if err := testutil.AssertActivationToolSchema(tool["input_schema"]); err != nil {
			return err
		}
		if invocation.Capability == llm.CapabilityToolChoiceRequired {
			choice, _ := body["tool_choice"].(map[string]any)
			if choice["type"] != "any" {
				return fmt.Errorf("tool_choice = %#v, want any", choice)
			}
		}
	case llm.CapabilityJSONSchema:
		output, _ := body["output_config"].(map[string]any)
		format, _ := output["format"].(map[string]any)
		if format["type"] != "json_schema" {
			return fmt.Errorf("output_config.format = %#v", format)
		}
		if err := testutil.AssertActivationResponseSchema(format["schema"]); err != nil {
			return err
		}
	case llm.CapabilityImageInput:
		messages, _ := body["messages"].([]any)
		if len(messages) != 1 {
			return fmt.Errorf("messages = %#v", body["messages"])
		}
		content, _ := messages[0].(map[string]any)["content"].([]any)
		if len(content) != 2 {
			return fmt.Errorf("image content = %#v", messages[0])
		}
		image, _ := content[1].(map[string]any)
		source, _ := image["source"].(map[string]any)
		if image["type"] != "image" || source["type"] != "base64" || source["media_type"] != "image/png" || source["data"] != "AQID" {
			return fmt.Errorf("image block = %#v", image)
		}
	case llm.CapabilityPromptCaching:
		messages, _ := body["messages"].([]any)
		content, _ := messages[0].(map[string]any)["content"].([]any)
		block, _ := content[0].(map[string]any)
		cache, _ := block["cache_control"].(map[string]any)
		if cache["type"] != "ephemeral" || cache["ttl"] != "1h" {
			return fmt.Errorf("cache_control = %#v", cache)
		}
	case llm.CapabilityStopSequences:
		stop, _ := body["stop_sequences"].([]any)
		if len(stop) != 2 || stop[0] != "END" || stop[1] != "HALT" {
			return fmt.Errorf("stop_sequences = %#v", body["stop_sequences"])
		}
	case llm.CapabilityReasoning:
		output, _ := body["output_config"].(map[string]any)
		thinking, _ := body["thinking"].(map[string]any)
		if output["effort"] != "high" || thinking["type"] != "adaptive" || thinking["display"] != "summarized" {
			return fmt.Errorf("reasoning output_config=%#v thinking=%#v", output, thinking)
		}
	}
	return nil
}

func anthropicActivationResponse(invocation llmtest.CapabilityInvocation, model string) string {
	content := []any{map[string]any{"type": "text", "text": "activated"}}
	stopReason := "end_turn"
	usage := map[string]any{"input_tokens": 1, "output_tokens": 2}
	switch invocation.Capability {
	case llm.CapabilityTools, llm.CapabilityToolChoiceRequired:
		content = []any{map[string]any{"type": "tool_use", "id": "call_activation", "name": testutil.ActivationToolName, "input": map[string]any{"value": "activated"}}}
		stopReason = "tool_use"
	case llm.CapabilityPromptCaching:
		usage = map[string]any{"input_tokens": 1, "cache_read_input_tokens": 2, "cache_creation_input_tokens": 3, "output_tokens": 1}
	case llm.CapabilityReasoning:
		content = append([]any{map[string]any{"type": "thinking", "thinking": "because", "signature": "fixture"}}, content...)
		usage["output_tokens_details"] = map[string]any{"thinking_tokens": 1}
	}
	payload := map[string]any{
		"id": "msg_activation", "type": "message", "role": "assistant", "model": model,
		"content": content, "stop_reason": stopReason, "usage": usage,
	}
	encoded, _ := json.Marshal(payload)
	return string(encoded)
}

func anthropicActivationStream(invocation llmtest.CapabilityInvocation, model string) string {
	inputUsage := map[string]any{"input_tokens": 1, "output_tokens": 0}
	if invocation.Capability == llm.CapabilityPromptCaching {
		inputUsage["cache_read_input_tokens"] = 2
		inputUsage["cache_creation_input_tokens"] = 3
	}
	message := map[string]any{"id": "msg_activation", "type": "message", "role": "assistant", "model": model, "usage": inputUsage}
	start, _ := json.Marshal(map[string]any{"type": "message_start", "message": message})
	events := []string{"event: message_start\ndata: " + string(start) + "\n\n"}
	index := 0
	if invocation.Capability == llm.CapabilityReasoning {
		events = append(events,
			`event: content_block_start`+"\n"+`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"because","signature":"fixture"}}`+"\n\n",
			`event: content_block_stop`+"\n"+`data: {"type":"content_block_stop","index":0}`+"\n\n",
		)
		index = 1
	}
	events = append(events,
		fmt.Sprintf("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":%d,\"content_block\":{\"type\":\"text\",\"text\":\"activated\"}}\n\n", index),
		fmt.Sprintf("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":%d}\n\n", index),
	)
	outputUsage := map[string]any{"output_tokens": 1}
	if invocation.Capability == llm.CapabilityReasoning {
		outputUsage["output_tokens"] = 2
		outputUsage["output_tokens_details"] = map[string]any{"thinking_tokens": 1}
	}
	delta, _ := json.Marshal(map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": "end_turn"}, "usage": outputUsage})
	events = append(events, "event: message_delta\ndata: "+string(delta)+"\n\n", "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	return strings.Join(events, "")
}

func assertAnthropicCacheResponse(response *llm.Response) error {
	if err := testutil.AssertActivationText(response); err != nil {
		return err
	}
	if response.Usage.InputTokens != 1 || response.Usage.CacheReadTokens != 2 || response.Usage.CacheWriteTokens != 3 || response.Usage.TotalTokens != 7 {
		return fmt.Errorf("cache usage = %+v", response.Usage)
	}
	return nil
}
