package llm

import (
	"fmt"
	"reflect"
)

// ValidateRequest checks required request fields and capability-gated options.
func ValidateRequest(caps []Capability, req *Request) error {
	if req == nil {
		return fmt.Errorf("%w: nil request", ErrBadRequest)
	}
	if req.Model == "" {
		return fmt.Errorf("%w: model is required", ErrBadRequest)
	}
	if len(req.Messages) == 0 {
		return fmt.Errorf("%w: messages are required", ErrBadRequest)
	}
	if req.MaxTokens < 0 {
		return fmt.Errorf("%w: max tokens cannot be negative", ErrBadRequest)
	}

	capSet := makeCapabilitySet(caps)
	if err := validateTools(capSet, req); err != nil {
		return err
	}
	if err := validateResponseFormat(capSet, req.ResponseFormat); err != nil {
		return err
	}
	if err := validateEffort(capSet, req.Effort); err != nil {
		return err
	}
	if len(req.StopSequences) > 0 && !capSet.has(CapabilityStopSequences) {
		return unsupported(CapabilityStopSequences)
	}
	if err := validateMessageParts(capSet, req.Messages); err != nil {
		return err
	}
	return nil
}

// ValidateStreamRequest checks a request for use with ChatStream.
//
// Tools remain gated by CapabilityTools alone: CapabilityToolStreaming
// describes incremental tool-argument deltas (FS §6), not whether tools may
// be used with ChatStream, so it is deliberately not checked here. Providers
// without it deliver tool-call arguments non-incrementally.
func ValidateStreamRequest(caps []Capability, req *Request) error {
	if err := ValidateRequest(caps, req); err != nil {
		return err
	}
	if !makeCapabilitySet(caps).has(CapabilityStreaming) {
		return unsupported(CapabilityStreaming)
	}
	return nil
}

// ValidateProviderOptions checks that request provider options match a provider name.
func ValidateProviderOptions(provider string, req *Request) error {
	if req == nil || req.ProviderOptions == nil {
		return nil
	}
	if isNilProviderOptions(req.ProviderOptions) {
		return fmt.Errorf("%w: nil provider options", ErrBadRequest)
	}
	optionsProvider := req.ProviderOptions.ForProvider()
	if optionsProvider != provider {
		return fmt.Errorf("%w: provider options for %q used with %q", ErrBadRequest, optionsProvider, provider)
	}
	return nil
}

func isNilProviderOptions(options ProviderOptions) bool {
	value := reflect.ValueOf(options)
	for value.IsValid() && value.Kind() == reflect.Interface {
		if value.IsNil() {
			return true
		}
		value = value.Elem()
	}
	if !value.IsValid() {
		return true
	}
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

type capabilitySet map[Capability]struct{}

func (s capabilitySet) has(cap Capability) bool {
	_, ok := s[cap]
	return ok
}

func makeCapabilitySet(caps []Capability) capabilitySet {
	set := make(capabilitySet, len(caps))
	for _, cap := range caps {
		set[cap] = struct{}{}
	}
	return set
}

func validateTools(caps capabilitySet, req *Request) error {
	toolNames := make(map[string]struct{}, len(req.Tools))
	for _, tool := range req.Tools {
		if tool.Name == "" {
			return fmt.Errorf("%w: tool name is required", ErrBadRequest)
		}
		if _, exists := toolNames[tool.Name]; exists {
			return fmt.Errorf("%w: duplicate tool name %q", ErrBadRequest, tool.Name)
		}
		toolNames[tool.Name] = struct{}{}
	}
	if req.ToolChoice.Mode == ToolChoiceTool {
		if req.ToolChoice.Name == "" {
			return fmt.Errorf("%w: tool choice name is required", ErrBadRequest)
		}
		if _, exists := toolNames[req.ToolChoice.Name]; !exists {
			return fmt.Errorf("%w: tool choice names undeclared tool %q", ErrBadRequest, req.ToolChoice.Name)
		}
	}
	if len(req.Tools) == 0 && req.ToolChoice.Mode != "" && req.ToolChoice.Mode != ToolChoiceAuto && req.ToolChoice.Mode != ToolChoiceNone {
		return fmt.Errorf("%w: tool choice requires tools", ErrBadRequest)
	}

	if len(req.Tools) > 0 && !caps.has(CapabilityTools) {
		return unsupported(CapabilityTools)
	}
	for _, tool := range req.Tools {
		if tool.Strict && !caps.has(CapabilityStrictTools) {
			return unsupported(CapabilityStrictTools)
		}
	}

	switch req.ToolChoice.Mode {
	case "", ToolChoiceAuto:
		return nil
	case ToolChoiceNone, ToolChoiceRequired:
		if !caps.has(CapabilityToolChoiceRequired) {
			return unsupported(CapabilityToolChoiceRequired)
		}
	case ToolChoiceTool:
		if !caps.has(CapabilityToolChoiceRequired) {
			return unsupported(CapabilityToolChoiceRequired)
		}
	default:
		return fmt.Errorf("%w: unknown tool choice mode %q", ErrBadRequest, req.ToolChoice.Mode)
	}
	return nil
}

func validateResponseFormat(caps capabilitySet, format *ResponseFormat) error {
	if format == nil {
		return nil
	}
	switch format.Type {
	case FormatJSONSchema:
		if !caps.has(CapabilityJSONSchema) {
			return unsupported(CapabilityJSONSchema)
		}
		if format.Schema == nil {
			return fmt.Errorf("%w: response format schema is required", ErrBadRequest)
		}
	case FormatJSONMode:
		if !caps.has(CapabilityJSONMode) {
			return unsupported(CapabilityJSONMode)
		}
	default:
		return fmt.Errorf("%w: unknown response format %q", ErrBadRequest, format.Type)
	}
	return nil
}

func validateEffort(caps capabilitySet, effort Effort) error {
	if effort == "" {
		return nil
	}
	switch effort {
	case EffortNone, EffortMinimal, EffortLow, EffortMedium, EffortHigh, EffortXHigh, EffortMax:
		if !caps.has(CapabilityReasoning) {
			return unsupported(CapabilityReasoning)
		}
	default:
		return fmt.Errorf("%w: unknown effort %q", ErrBadRequest, effort)
	}
	return nil
}

func validateMessageParts(caps capabilitySet, msgs []Message) error {
	for _, msg := range msgs {
		for _, part := range msg.Parts {
			if err := validatePart(caps, part); err != nil {
				return err
			}
		}
	}
	return nil
}

func validatePart(caps capabilitySet, part Part) error {
	switch p := derefPart(part).(type) {
	case nil:
		return fmt.Errorf("%w: nil part", ErrBadRequest)
	case TextPart, ToolCallPart, ReasoningPart, UnknownPart:
		return nil
	case ImagePart:
		if !caps.has(CapabilityImageInput) {
			return unsupported(CapabilityImageInput)
		}
	case FilePart:
		if !caps.has(CapabilityPDFInput) {
			return unsupported(CapabilityPDFInput)
		}
	case ToolResultPart:
		for _, nested := range p.Content {
			if err := validatePart(caps, nested); err != nil {
				return err
			}
		}
	}
	return nil
}

func unsupported(cap Capability) error {
	return fmt.Errorf("%w: %s", ErrUnsupported, cap)
}
