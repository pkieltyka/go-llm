package testutil

import (
	"encoding/json"
	"fmt"
	"reflect"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

const ActivationToolName = "activation_echo"

// CompatibleCapabilityProfile is the shared activation profile for the
// public generic chat-completions engine and data-only compatible presets.
// Provider fixtures remain responsible for asserting their exact wire shape.
func CompatibleCapabilityProfile(model string) llmtest.CapabilityProfile {
	return llmtest.CapabilityProfile{Cases: []llmtest.CapabilityCase{
		{
			Name:       "tools",
			Capability: llm.CapabilityTools,
			Paths:      []llmtest.ConformancePath{llmtest.ConformanceChat},
			Request:    func() *llm.Request { return ToolActivationRequest(model, llm.ToolChoiceAuto) },
			Assert:     AssertActivationToolCall,
		},
		{
			Name:       "required-tool-choice",
			Capability: llm.CapabilityToolChoiceRequired,
			Paths:      []llmtest.ConformancePath{llmtest.ConformanceChat},
			Request:    func() *llm.Request { return ToolActivationRequest(model, llm.ToolChoiceRequired) },
			Assert:     AssertActivationToolCall,
		},
		{
			Name:       "json-schema",
			Capability: llm.CapabilityJSONSchema,
			Paths:      []llmtest.ConformancePath{llmtest.ConformanceChat},
			Request:    func() *llm.Request { return JSONSchemaActivationRequest(model) },
			Assert:     AssertActivationText,
		},
		{
			Name:       "image-input",
			Capability: llm.CapabilityImageInput,
			Paths:      []llmtest.ConformancePath{llmtest.ConformanceChat},
			Request:    func() *llm.Request { return ImageActivationRequest(model) },
			Assert:     AssertActivationText,
		},
		{
			Name:       "stop-sequences",
			Capability: llm.CapabilityStopSequences,
			Paths:      []llmtest.ConformancePath{llmtest.ConformanceChat},
			Request:    func() *llm.Request { return StopActivationRequest(model) },
			Assert:     AssertActivationText,
		},
		{
			Name:       "reasoning",
			Capability: llm.CapabilityReasoning,
			Paths:      []llmtest.ConformancePath{llmtest.ConformanceChat},
			Request:    func() *llm.Request { return ReasoningActivationRequest(model) },
			Assert:     AssertActivationReasoning,
		},
	}}
}

func ToolActivationRequest(model string, choice llm.ToolChoiceMode) *llm.Request {
	return &llm.Request{
		Model:    model,
		Messages: []llm.Message{llm.UserText("call the tool")},
		Tools: []llm.Tool{{
			Name:        ActivationToolName,
			Description: "Return activation status",
			InputSchema: map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"value": map[string]any{"type": "string"}},
				"required":             []string{"value"},
				"additionalProperties": false,
			},
			Strict: true,
		}},
		ToolChoice: llm.ToolChoice{Mode: choice},
	}
}

func JSONSchemaActivationRequest(model string) *llm.Request {
	return &llm.Request{
		Model:    model,
		Messages: []llm.Message{llm.UserText("return structured data")},
		ResponseFormat: &llm.ResponseFormat{
			Type: llm.FormatJSONSchema,
			Name: "activation_result",
			Schema: map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"ok": map[string]any{"type": "boolean"}},
				"required":             []string{"ok"},
				"additionalProperties": false,
			},
			Strict: true,
		},
	}
}

func ImageActivationRequest(model string) *llm.Request {
	return &llm.Request{
		Model: model,
		Messages: []llm.Message{llm.UserParts(
			llm.Text("inspect image"),
			llm.ImageData([]byte{0x01, 0x02, 0x03}, "image/png"),
		)},
	}
}

func StopActivationRequest(model string) *llm.Request {
	return &llm.Request{Model: model, Messages: []llm.Message{llm.UserText("stop")}, StopSequences: []string{"END", "HALT"}}
}

func ReasoningActivationRequest(model string) *llm.Request {
	return &llm.Request{Model: model, Messages: []llm.Message{llm.UserText("reason")}, Effort: llm.EffortHigh}
}

func AssertActivationText(response *llm.Response) error {
	if response.Text() != "activated" {
		return fmt.Errorf("text = %q, want activated", response.Text())
	}
	if response.StopReason != llm.StopReasonEndTurn {
		return fmt.Errorf("stop reason = %q, want %q", response.StopReason, llm.StopReasonEndTurn)
	}
	return nil
}

func AssertActivationToolCall(response *llm.Response) error {
	calls := response.ToolCalls()
	if len(calls) != 1 {
		return fmt.Errorf("tool calls = %d, want 1", len(calls))
	}
	if calls[0].Name != ActivationToolName || string(calls[0].Args) != `{"value":"activated"}` {
		return fmt.Errorf("tool call = name %q args %s", calls[0].Name, calls[0].Args)
	}
	if response.StopReason != llm.StopReasonToolUse {
		return fmt.Errorf("stop reason = %q, want %q", response.StopReason, llm.StopReasonToolUse)
	}
	return nil
}

func AssertActivationReasoning(response *llm.Response) error {
	if response.Reasoning() != "because" || response.Text() != "activated" {
		return fmt.Errorf("reasoning = %q text = %q", response.Reasoning(), response.Text())
	}
	if response.Usage.ReasoningTokens != 1 {
		return fmt.Errorf("reasoning tokens = %d, want 1", response.Usage.ReasoningTokens)
	}
	return nil
}

// AssertActivationToolSchema structurally pins the complete native schema used
// by activation tool fixtures, including strict-schema closure semantics.
func AssertActivationToolSchema(schema any) error {
	expected := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"value": map[string]any{"type": "string"},
		},
		"required":             []any{"value"},
		"additionalProperties": false,
	}
	if !reflect.DeepEqual(schema, expected) {
		return fmt.Errorf("tool schema = %#v, want %#v", schema, expected)
	}
	return nil
}

// AssertActivationResponseSchema structurally pins the complete schema used
// by structured-output activation fixtures.
func AssertActivationResponseSchema(schema any) error {
	expected := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"ok": map[string]any{"type": "boolean"},
		},
		"required":             []any{"ok"},
		"additionalProperties": false,
	}
	if !reflect.DeepEqual(schema, expected) {
		return fmt.Errorf("response schema = %#v, want %#v", schema, expected)
	}
	return nil
}

// AssertCompatibleActivationRequest checks the reviewed standard compatible
// fields. Callers assert dialect-specific reasoning/cache fields themselves.
func AssertCompatibleActivationRequest(body map[string]any, invocation llmtest.CapabilityInvocation, expectedModel string) error {
	if got, _ := body["model"].(string); got != expectedModel {
		return fmt.Errorf("model = %q, want %q", got, expectedModel)
	}
	switch invocation.Capability {
	case llm.CapabilityTools, llm.CapabilityToolChoiceRequired:
		tools, ok := body["tools"].([]any)
		if !ok || len(tools) != 1 {
			return fmt.Errorf("tools = %#v, want one native tool", body["tools"])
		}
		function, ok := tools[0].(map[string]any)["function"].(map[string]any)
		if !ok || function["name"] != ActivationToolName || function["strict"] != true {
			return fmt.Errorf("native function tool = %#v", tools[0])
		}
		if err := AssertActivationToolSchema(function["parameters"]); err != nil {
			return err
		}
		if invocation.Capability == llm.CapabilityToolChoiceRequired && body["tool_choice"] != "required" {
			return fmt.Errorf("tool_choice = %#v, want required", body["tool_choice"])
		}
	case llm.CapabilityJSONSchema:
		format, _ := body["response_format"].(map[string]any)
		schema, _ := format["json_schema"].(map[string]any)
		if format["type"] != "json_schema" || schema["name"] != "activation_result" || schema["strict"] != true {
			return fmt.Errorf("response_format = %#v", format)
		}
		if err := AssertActivationResponseSchema(schema["schema"]); err != nil {
			return err
		}
	case llm.CapabilityImageInput:
		messages, _ := body["messages"].([]any)
		if len(messages) != 1 {
			return fmt.Errorf("messages = %#v", body["messages"])
		}
		content, _ := messages[0].(map[string]any)["content"].([]any)
		if len(content) != 2 {
			return fmt.Errorf("image message content = %#v", messages[0])
		}
		image, _ := content[1].(map[string]any)
		imageURL, _ := image["image_url"].(map[string]any)
		if image["type"] != "image_url" || imageURL["url"] != "data:image/png;base64,AQID" {
			return fmt.Errorf("image block = %#v", image)
		}
	case llm.CapabilityStopSequences:
		if !reflect.DeepEqual(body["stop"], []any{"END", "HALT"}) {
			return fmt.Errorf("stop = %#v, want [END HALT]", body["stop"])
		}
	}
	return nil
}

func CompatibleActivationResponse(invocation llmtest.CapabilityInvocation, model string) string {
	message := map[string]any{"role": "assistant", "content": "activated"}
	finishReason := "stop"
	usage := map[string]any{"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
	switch invocation.Capability {
	case llm.CapabilityTools, llm.CapabilityToolChoiceRequired:
		message["content"] = nil
		message["tool_calls"] = []any{map[string]any{
			"id": "call_activation", "type": "function",
			"function": map[string]any{"name": ActivationToolName, "arguments": `{"value":"activated"}`},
		}}
		finishReason = "tool_calls"
	case llm.CapabilityReasoning:
		message["reasoning"] = "because"
		usage["completion_tokens_details"] = map[string]any{"reasoning_tokens": 1}
	}
	payload := map[string]any{
		"id": "activation_response", "model": model,
		"choices": []any{map[string]any{"index": 0, "finish_reason": finishReason, "message": message}},
		"usage":   usage,
	}
	encoded, _ := json.Marshal(payload)
	return string(encoded)
}
