package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/pkieltyka/go-llm/internal/schemajson"
)

// ParseMode selects the structured-output strategy for Parse.
type ParseMode string

const (
	// ModeAuto picks the best supported strategy: native json-schema, then
	// forced-tool extraction, then json mode.
	ModeAuto ParseMode = ""
	// ModeNative forces provider-native json-schema structured output.
	ModeNative ParseMode = "native"
	// ModeTool forces extraction through a synthetic forced tool call.
	ModeTool ParseMode = "tool"
	// ModeJSON forces plain JSON mode plus client-side validation.
	ModeJSON ParseMode = "json"
)

// ParseOption configures Parse. Options are non-generic so call sites stay
// clean (llm.WithParseRetries(2)); only WithParseValidator carries a type
// parameter, checked against Parse's T at call time.
type ParseOption func(*parseConfig)

type parseConfig struct {
	mode    ParseMode
	retries int
	// validator holds a func(T) error; Parse[T] type-asserts it and returns
	// ErrBadRequest on a mismatch.
	validator any
}

// WithParseRetries sets the bounded retry count for invalid parsed output.
func WithParseRetries(n int) ParseOption {
	return func(opts *parseConfig) {
		if n > 0 {
			opts.retries = n
		}
	}
}

// WithParseMode forces a Parse strategy instead of auto resolution.
func WithParseMode(m ParseMode) ParseOption {
	return func(opts *parseConfig) { opts.mode = m }
}

// WithParseValidator adds a semantic validator. Validator failures can retry.
// The validator's T must match Parse's T; a mismatch fails the Parse call
// with ErrBadRequest.
func WithParseValidator[T any](fn func(T) error) ParseOption {
	return func(opts *parseConfig) { opts.validator = fn }
}

// Parse derives a schema for T, calls p, and decodes structured output.
func Parse[T any](ctx context.Context, p Provider, req *Request, opts ...ParseOption) (T, *Response, error) {
	var zero T
	if p == nil {
		return zero, nil, fmt.Errorf("%w: nil provider", ErrBadRequest)
	}
	if req == nil {
		return zero, nil, fmt.Errorf("%w: nil request", ErrBadRequest)
	}

	cfg := parseConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	var validator func(T) error
	if cfg.validator != nil {
		fn, ok := cfg.validator.(func(T) error)
		if !ok {
			return zero, nil, fmt.Errorf("%w: parse validator is %T, want func(%T) error", ErrBadRequest, cfg.validator, zero)
		}
		validator = fn
	}

	rawSchema, err := parseSchemaForRequest[T](req)
	if err != nil {
		return zero, nil, err
	}
	mode, err := resolveParseMode(cfg.mode, p.Capabilities())
	if err != nil {
		return zero, nil, err
	}

	working := cloneRequest(req)
	for attempt := 0; ; attempt++ {
		attemptReq := cloneRequest(working)
		applyParseMode(attemptReq, mode, rawSchema, hasCapability(p.Capabilities(), CapabilityStrictTools))

		resp, err := p.Chat(ctx, attemptReq)
		if err != nil {
			return zero, resp, err
		}

		value, err := decodeParseResponse[T](mode, rawSchema, resp)
		if err == nil && validator != nil {
			if validateErr := validator(value); validateErr != nil {
				err = validateErr
			}
		}
		if err == nil {
			return value, resp, nil
		}
		if attempt >= cfg.retries {
			return zero, resp, fmt.Errorf("%w: parse failed: %s", ErrBadRequest, badRequestDetail(err))
		}
		appendParseRetryTurn(working, mode, resp, err)
	}
}

func resolveParseMode(mode ParseMode, caps []Capability) (ParseMode, error) {
	if mode != ModeAuto {
		if parseModeSupported(mode, caps) {
			return mode, nil
		}
		return "", fmt.Errorf("%w: parse mode %q", ErrUnsupported, mode)
	}
	if hasCapability(caps, CapabilityJSONSchema) {
		return ModeNative, nil
	}
	if hasCapability(caps, CapabilityTools) && hasCapability(caps, CapabilityToolChoiceRequired) {
		return ModeTool, nil
	}
	if hasCapability(caps, CapabilityJSONMode) {
		return ModeJSON, nil
	}
	return "", fmt.Errorf("%w: parse requires json-schema, forced tools, or json-mode", ErrUnsupported)
}

func parseModeSupported(mode ParseMode, caps []Capability) bool {
	switch mode {
	case ModeNative:
		return hasCapability(caps, CapabilityJSONSchema)
	case ModeTool:
		return hasCapability(caps, CapabilityTools) && hasCapability(caps, CapabilityToolChoiceRequired)
	case ModeJSON:
		return hasCapability(caps, CapabilityJSONMode)
	default:
		return false
	}
}

func parseSchemaForRequest[T any](req *Request) (any, error) {
	if req != nil && req.ResponseFormat != nil && req.ResponseFormat.Schema != nil {
		return cloneSchemaValue(req.ResponseFormat.Schema), nil
	}
	return schemajson.For[T]()
}

func applyParseMode(req *Request, mode ParseMode, rawSchema any, strictTools bool) {
	switch mode {
	case ModeNative:
		if req.ResponseFormat == nil {
			req.ResponseFormat = &ResponseFormat{
				Type:   FormatJSONSchema,
				Name:   "parse_result",
				Schema: cloneSchemaValue(rawSchema),
				Strict: true,
			}
		}
	case ModeTool:
		req.ResponseFormat = nil
		req.Tools = append(req.Tools, Tool{
			Name:        "parse_result",
			Description: "Return the structured result.",
			InputSchema: cloneSchemaValue(rawSchema),
			Strict:      strictTools,
		})
		req.ToolChoice = ToolChoice{Mode: ToolChoiceTool, Name: "parse_result"}
	case ModeJSON:
		req.ResponseFormat = &ResponseFormat{Type: FormatJSONMode}
		guidance := "Return only JSON matching this JSON Schema:\n" + schemaString(rawSchema)
		if req.System == "" {
			req.System = guidance
		} else {
			req.System += "\n\n" + guidance
		}
	}
}

func decodeParseResponse[T any](mode ParseMode, rawSchema any, resp *Response) (T, error) {
	var zero T
	if resp == nil {
		return zero, fmt.Errorf("%w: nil response", ErrBadRequest)
	}
	var raw json.RawMessage
	switch mode {
	case ModeTool:
		calls := resp.ToolCalls()
		if len(calls) == 0 {
			return zero, fmt.Errorf("%w: parse tool was not called", ErrBadRequest)
		}
		raw = calls[0].Args
	default:
		raw = json.RawMessage(strings.TrimSpace(resp.Text()))
	}
	if len(raw) == 0 {
		return zero, fmt.Errorf("%w: empty parse output", ErrBadRequest)
	}
	if err := validateParseRaw(rawSchema, raw); err != nil {
		return zero, err
	}
	var out T
	if err := json.Unmarshal(raw, &out); err != nil {
		return zero, err
	}
	return out, nil
}

func validateParseRaw(rawSchema any, raw json.RawMessage) error {
	err := schemajson.ValidateArgs("parse_result", rawSchema, raw)
	if errors.Is(err, schemajson.ErrBadRequest) {
		return fmt.Errorf("%w: %s", ErrBadRequest, schemajson.BadRequestDetail(err))
	}
	return err
}

func appendParseRetryTurn(req *Request, mode ParseMode, resp *Response, parseErr error) {
	if resp != nil {
		req.Messages = append(req.Messages, Message{
			Role:     RoleAssistant,
			Parts:    cloneParts(resp.Parts),
			Provider: resp.Provider,
			Model:    resp.Model,
		})
		if mode == ModeTool {
			if results := parseRetryToolResults(resp, parseErr); len(results) > 0 {
				req.Messages = append(req.Messages, Message{Role: RoleTool, Parts: results})
				return
			}
		}
	}
	req.Messages = append(req.Messages, UserText(parseRetryCorrectionText(parseErr)))
}

func parseRetryToolResults(resp *Response, parseErr error) []Part {
	calls := resp.ToolCalls()
	if len(calls) == 0 {
		return nil
	}
	parts := make([]Part, len(calls))
	for i, call := range calls {
		parts[i] = ToolResultPart{
			ToolCallID: call.ID,
			Name:       call.Name,
			Content:    []Part{Text(parseRetryCorrectionText(parseErr))},
			IsError:    true,
		}
	}
	return parts
}

func parseRetryCorrectionText(parseErr error) string {
	return "The structured output was invalid: " + parseErr.Error() + ". Return corrected output that matches the requested schema."
}

func badRequestDetail(err error) string {
	if err == nil {
		return ""
	}
	message := err.Error()
	if !errors.Is(err, ErrBadRequest) {
		return message
	}
	prefix := ErrBadRequest.Error()
	if message == prefix {
		return prefix
	}
	return strings.TrimPrefix(message, prefix+": ")
}

// cloneRequest deep-clones a request so per-attempt mutations (Parse mode
// application, retry correction turns) never leak into the caller's Request.
// Shared by Parse and RetryDroppedToolCalls.
func cloneRequest(req *Request) *Request {
	if req == nil {
		return nil
	}
	copied := *req
	copied.Messages = cloneMessages(req.Messages)
	copied.SystemCache = cloneCacheHint(req.SystemCache)
	copied.StopSequences = append([]string(nil), req.StopSequences...)
	copied.Tools = cloneTools(req.Tools)
	if req.ResponseFormat != nil {
		format := *req.ResponseFormat
		format.Schema = cloneSchemaValue(format.Schema)
		copied.ResponseFormat = &format
	}
	return &copied
}

func cloneTools(tools []Tool) []Tool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]Tool, len(tools))
	for i, tool := range tools {
		out[i] = tool
		out[i].InputSchema = cloneSchemaValue(tool.InputSchema)
	}
	return out
}

func cloneSchemaValue(value any) any {
	switch v := value.(type) {
	case json.RawMessage:
		return append(json.RawMessage(nil), v...)
	case []byte:
		return append([]byte(nil), v...)
	default:
		return value
	}
}

func schemaString(value any) string {
	switch v := value.(type) {
	case json.RawMessage:
		return string(v)
	case []byte:
		return string(v)
	default:
		raw, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprint(v)
		}
		return string(raw)
	}
}

func hasCapability(caps []Capability, want Capability) bool {
	for _, cap := range caps {
		if cap == want {
			return true
		}
	}
	return false
}
