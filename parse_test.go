package llm_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

func TestParseModes(t *testing.T) {
	t.Run("native schema", func(t *testing.T) {
		p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityJSONSchema))
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(`{"name":"Ada","age":37}`)}})
		got, _, err := llm.Parse[parsePerson](context.Background(), p, parseRequest())
		if err != nil {
			t.Fatalf("Parse returned error: %v", err)
		}
		if got.Name != "Ada" || got.Age != 37 {
			t.Fatalf("parsed = %+v", got)
		}
		req := p.Requests()[0]
		if req.ResponseFormat == nil || req.ResponseFormat.Type != llm.FormatJSONSchema {
			t.Fatalf("response format = %+v", req.ResponseFormat)
		}
	})

	t.Run("native mode replaces json mode with derived schema", func(t *testing.T) {
		p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityJSONSchema))
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(`{"name":"Ada"}`)}})
		req := parseRequest()
		req.ResponseFormat = &llm.ResponseFormat{Type: llm.FormatJSONMode, Name: "person_result"}

		if _, _, err := llm.Parse[parsePerson](context.Background(), p, req, llm.WithParseMode(llm.ModeNative)); err != nil {
			t.Fatalf("Parse returned error: %v", err)
		}
		recorded := p.Requests()[0]
		if recorded.ResponseFormat == nil || recorded.ResponseFormat.Type != llm.FormatJSONSchema || recorded.ResponseFormat.Name != "person_result" || recorded.ResponseFormat.Schema == nil || !recorded.ResponseFormat.Strict {
			t.Fatalf("native response format = %+v", recorded.ResponseFormat)
		}
		if req.ResponseFormat.Type != llm.FormatJSONMode || req.ResponseFormat.Schema != nil {
			t.Fatalf("caller request was mutated: %+v", req.ResponseFormat)
		}
	})

	t.Run("forced tool", func(t *testing.T) {
		p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityTools, llm.CapabilityToolChoiceRequired, llm.CapabilityStrictTools))
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.ToolCall("call_1", "parse_result", []byte(`{"name":"Ada"}`))}})
		got, _, err := llm.Parse[parsePerson](context.Background(), p, parseRequest())
		if err != nil {
			t.Fatalf("Parse returned error: %v", err)
		}
		if got.Name != "Ada" {
			t.Fatalf("parsed = %+v", got)
		}
		req := p.Requests()[0]
		if req.ToolChoice.Mode != llm.ToolChoiceTool || len(req.Tools) != 1 || !req.Tools[0].Strict {
			t.Fatalf("tool parse request = %+v", req)
		}
	})

	t.Run("forced tool replaces caller tools and matches by collision-free name", func(t *testing.T) {
		p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityTools, llm.CapabilityToolChoiceRequired))
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{
			llm.ToolCall("call_unrelated", "lookup", []byte(`{"q":"Ada"}`)),
			llm.ToolCall("call_parse", "parse_result_3", []byte(`{"name":"Ada"}`)),
		}})
		req := parseRequest()
		req.Tools = []llm.Tool{{Name: "parse_result"}, {Name: "parse_result_2"}, {Name: "lookup"}}

		got, _, err := llm.Parse[parsePerson](context.Background(), p, req, llm.WithParseMode(llm.ModeTool))
		if err != nil {
			t.Fatalf("Parse returned error: %v", err)
		}
		if got.Name != "Ada" {
			t.Fatalf("parsed = %+v", got)
		}
		recorded := p.Requests()[0]
		if len(recorded.Tools) != 1 || recorded.Tools[0].Name != "parse_result_3" {
			t.Fatalf("tools = %+v, want one collision-free parse tool", recorded.Tools)
		}
		if recorded.ToolChoice.Mode != llm.ToolChoiceTool || recorded.ToolChoice.Name != "parse_result_3" {
			t.Fatalf("tool choice = %+v", recorded.ToolChoice)
		}
		if len(req.Tools) != 3 {
			t.Fatalf("caller tools were mutated: %+v", req.Tools)
		}
	})

	t.Run("forced tool requires exactly one matching call", func(t *testing.T) {
		tests := []struct {
			name  string
			parts []llm.Part
		}{
			{name: "zero", parts: []llm.Part{llm.ToolCall("call_1", "other", []byte(`{"name":"Ada"}`))}},
			{name: "multiple", parts: []llm.Part{
				llm.ToolCall("call_1", "parse_result", []byte(`{"name":"Ada"}`)),
				llm.ToolCall("call_2", "parse_result", []byte(`{"name":"Grace"}`)),
			}},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityTools, llm.CapabilityToolChoiceRequired))
				p.EnqueueResponse(&llm.Response{Parts: tt.parts})
				_, _, err := llm.Parse[parsePerson](context.Background(), p, parseRequest(), llm.WithParseMode(llm.ModeTool))
				if !errors.Is(err, llm.ErrBadRequest) {
					t.Fatalf("error = %v, want ErrBadRequest", err)
				}
			})
		}
	})

	t.Run("forced tool retry uses tool result", func(t *testing.T) {
		p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityTools, llm.CapabilityToolChoiceRequired))
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.ToolCall("call_1", "parse_result", []byte(`{}`))}})
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.ToolCall("call_2", "parse_result", []byte(`{"name":"Ada"}`))}})
		got, _, err := llm.Parse[parsePerson](
			context.Background(),
			p,
			parseRequest(),
			llm.WithParseRetries(1),
		)
		if err != nil {
			t.Fatalf("Parse returned error: %v", err)
		}
		if got.Name != "Ada" {
			t.Fatalf("parsed = %+v", got)
		}
		requests := p.Requests()
		if len(requests) != 2 {
			t.Fatalf("requests len = %d, want 2", len(requests))
		}
		retryMessages := requests[1].Messages
		if len(retryMessages) != 3 {
			t.Fatalf("retry messages = %+v, want user, assistant, tool", retryMessages)
		}
		if retryMessages[1].Role != llm.RoleAssistant || retryMessages[2].Role != llm.RoleTool {
			t.Fatalf("retry roles = %q, %q; want assistant, tool", retryMessages[1].Role, retryMessages[2].Role)
		}
		result, ok := retryMessages[2].Parts[0].(llm.ToolResultPart)
		if !ok {
			t.Fatalf("retry correction part = %T, want ToolResultPart", retryMessages[2].Parts[0])
		}
		if result.ToolCallID != "call_1" || !result.IsError || !strings.Contains(msgText(llm.Message{Parts: result.Content}), "$.name is required") {
			t.Fatalf("retry tool result = %+v", result)
		}
	})

	t.Run("forced tool clears existing response format", func(t *testing.T) {
		p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityTools, llm.CapabilityToolChoiceRequired))
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.ToolCall("call_1", "parse_result", []byte(`{"name":"Ada"}`))}})
		req := parseRequest()
		req.ResponseFormat = &llm.ResponseFormat{
			Type:   llm.FormatJSONSchema,
			Name:   "custom_person",
			Schema: []byte(`{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}`),
			Strict: true,
		}

		got, _, err := llm.Parse[parseCustomSchemaPerson](context.Background(), p, req)
		if err != nil {
			t.Fatalf("Parse returned error: %v", err)
		}
		if got.Name != "Ada" {
			t.Fatalf("parsed = %+v", got)
		}
		requests := p.Requests()
		if requests[0].ResponseFormat != nil {
			t.Fatalf("forced-tool request retained response format: %+v", requests[0].ResponseFormat)
		}
		if len(requests[0].Tools) != 1 || requests[0].Tools[0].InputSchema == nil {
			t.Fatalf("forced-tool request missing synthetic tool schema: %+v", requests[0].Tools)
		}
	})

	t.Run("json mode retry and validator", func(t *testing.T) {
		p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityJSONMode))
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(`{"name":""}`)}})
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(`{"name":"Grace"}`)}})
		got, _, err := llm.Parse[parsePerson](
			context.Background(),
			p,
			parseRequest(),
			llm.WithParseRetries(1),
			llm.WithParseValidator(func(p parsePerson) error {
				if p.Name == "" {
					return errors.New("name required")
				}
				return nil
			}),
		)
		if err != nil {
			t.Fatalf("Parse returned error: %v", err)
		}
		if got.Name != "Grace" {
			t.Fatalf("parsed = %+v", got)
		}
		requests := p.Requests()
		if len(requests) != 2 {
			t.Fatalf("requests len = %d", len(requests))
		}
		if requests[0].ResponseFormat == nil || requests[0].ResponseFormat.Type != llm.FormatJSONMode {
			t.Fatalf("json mode response format = %+v", requests[0].ResponseFormat)
		}
		if len(requests[1].Messages) < 3 || !strings.Contains(msgText(requests[1].Messages[len(requests[1].Messages)-1]), "name required") {
			t.Fatalf("retry messages = %+v", requests[1].Messages)
		}
	})

	t.Run("json mode guidance is system prompt", func(t *testing.T) {
		p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityJSONMode))
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(`{"name":"Ada"}`)}})
		req := parseRequest()
		req.System = "base system"

		if _, _, err := llm.Parse[parsePerson](context.Background(), p, req); err != nil {
			t.Fatalf("Parse returned error: %v", err)
		}
		recorded := p.Requests()[0]
		if !strings.Contains(recorded.System, "base system") || !strings.Contains(recorded.System, "Return only JSON matching this JSON Schema") {
			t.Fatalf("system prompt missing JSON guidance: %q", recorded.System)
		}
		if got := msgText(recorded.Messages[0]); got != "extract" {
			t.Fatalf("user message = %q, want extract", got)
		}
	})

	t.Run("json mode retries schema-invalid output", func(t *testing.T) {
		p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityJSONMode))
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(`{}`)}})
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(`{"name":"Ada"}`)}})
		got, _, err := llm.Parse[parsePerson](
			context.Background(),
			p,
			parseRequest(),
			llm.WithParseRetries(1),
		)
		if err != nil {
			t.Fatalf("Parse returned error: %v", err)
		}
		if got.Name != "Ada" {
			t.Fatalf("parsed = %+v", got)
		}
		requests := p.Requests()
		if len(requests) != 2 {
			t.Fatalf("requests len = %d", len(requests))
		}
		if got := msgText(requests[1].Messages[len(requests[1].Messages)-1]); !strings.Contains(got, "$.name is required") {
			t.Fatalf("retry correction = %q", got)
		}
	})

	t.Run("retries exhausted", func(t *testing.T) {
		p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityJSONMode))
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(`{}`)}})
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(`{}`)}})

		_, _, err := llm.Parse[parsePerson](
			context.Background(),
			p,
			parseRequest(),
			llm.WithParseRetries(1),
		)
		if !errors.Is(err, llm.ErrBadRequest) {
			t.Fatalf("Parse error = %v, want ErrBadRequest", err)
		}
		if strings.Count(err.Error(), llm.ErrBadRequest.Error()) != 1 || strings.Contains(err.Error(), "parse failed: llm: bad request") {
			t.Fatalf("Parse error double-prefixed bad request: %v", err)
		}
		if len(p.Requests()) != 2 {
			t.Fatalf("requests len = %d, want 2", len(p.Requests()))
		}
	})

	t.Run("existing response format schema bypasses generator", func(t *testing.T) {
		p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityJSONSchema))
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(`{"name":"Ada"}`)}})
		req := parseRequest()
		req.ResponseFormat = &llm.ResponseFormat{
			Type:   llm.FormatJSONSchema,
			Name:   "custom_person",
			Schema: []byte(`{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}`),
			Strict: true,
		}

		got, _, err := llm.Parse[parseCustomSchemaPerson](context.Background(), p, req)
		if err != nil {
			t.Fatalf("Parse returned error: %v", err)
		}
		if got.Name != "Ada" {
			t.Fatalf("parsed = %+v", got)
		}
		requests := p.Requests()
		if requests[0].ResponseFormat == nil || requests[0].ResponseFormat.Name != "custom_person" {
			t.Fatalf("response format = %+v", requests[0].ResponseFormat)
		}
	})

	t.Run("unsupported mode override", func(t *testing.T) {
		p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityJSONMode))
		_, _, err := llm.Parse[parsePerson](context.Background(), p, parseRequest(), llm.WithParseMode(llm.ModeNative))
		if !errors.Is(err, llm.ErrUnsupported) {
			t.Fatalf("error = %v, want ErrUnsupported", err)
		}
	})

	t.Run("validator type mismatch fails fast", func(t *testing.T) {
		p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityJSONMode))
		// Validator typed for a different T than Parse's — must fail with
		// ErrBadRequest before any provider call, with a clear message.
		_, _, err := llm.Parse[parsePerson](
			context.Background(),
			p,
			parseRequest(),
			llm.WithParseValidator(func(int) error { return nil }),
		)
		if !errors.Is(err, llm.ErrBadRequest) {
			t.Fatalf("error = %v, want ErrBadRequest", err)
		}
		if !strings.Contains(err.Error(), "parse validator") || !strings.Contains(err.Error(), "func(int) error") {
			t.Fatalf("validator mismatch message = %v", err)
		}
		if len(p.Requests()) != 0 {
			t.Fatalf("provider was called %d times, want 0", len(p.Requests()))
		}
	})
}

type parsePerson struct {
	Name string `json:"name"`
	Age  int    `json:"age,omitempty"`
}

type parseCustomSchemaPerson struct {
	Name        string   `json:"name"`
	Unsupported chan int `json:"unsupported"`
}

func parseRequest() *llm.Request {
	return &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("extract")}}
}
