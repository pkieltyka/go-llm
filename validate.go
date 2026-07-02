package llm

import "fmt"

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
	if req.ProviderOptions.ForProvider() != provider {
		return fmt.Errorf("%w: provider options for %q used with %q", ErrBadRequest, req.ProviderOptions.ForProvider(), provider)
	}
	return nil
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
	if len(req.Tools) > 0 && !caps.has(CapabilityTools) {
		return unsupported(CapabilityTools)
	}
	for _, tool := range req.Tools {
		if tool.Name == "" {
			return fmt.Errorf("%w: tool name is required", ErrBadRequest)
		}
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
		if req.ToolChoice.Name == "" {
			return fmt.Errorf("%w: tool choice name is required", ErrBadRequest)
		}
		if !caps.has(CapabilityToolChoiceRequired) {
			return unsupported(CapabilityToolChoiceRequired)
		}
	default:
		return fmt.Errorf("%w: unknown tool choice mode %q", ErrBadRequest, req.ToolChoice.Mode)
	}
	if len(req.Tools) == 0 && req.ToolChoice.Mode != ToolChoiceNone {
		return fmt.Errorf("%w: tool choice requires tools", ErrBadRequest)
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
	switch p := part.(type) {
	case nil:
		return fmt.Errorf("%w: nil part", ErrBadRequest)
	case TextPart, ToolCallPart, ReasoningPart, UnknownPart:
		return nil
	case *TextPart:
		if p == nil {
			return fmt.Errorf("%w: nil text part", ErrBadRequest)
		}
	case *ToolCallPart:
		if p == nil {
			return fmt.Errorf("%w: nil tool call part", ErrBadRequest)
		}
	case *ReasoningPart:
		if p == nil {
			return fmt.Errorf("%w: nil reasoning part", ErrBadRequest)
		}
	case *UnknownPart:
		if p == nil {
			return fmt.Errorf("%w: nil unknown part", ErrBadRequest)
		}
	case ImagePart:
		if !caps.has(CapabilityImageInput) {
			return unsupported(CapabilityImageInput)
		}
	case *ImagePart:
		if p == nil {
			return fmt.Errorf("%w: nil image part", ErrBadRequest)
		}
		if !caps.has(CapabilityImageInput) {
			return unsupported(CapabilityImageInput)
		}
	case FilePart:
		if !caps.has(CapabilityPDFInput) {
			return unsupported(CapabilityPDFInput)
		}
	case *FilePart:
		if p == nil {
			return fmt.Errorf("%w: nil file part", ErrBadRequest)
		}
		if !caps.has(CapabilityPDFInput) {
			return unsupported(CapabilityPDFInput)
		}
	case ToolResultPart:
		for _, nested := range p.Parts {
			if err := validatePart(caps, nested); err != nil {
				return err
			}
		}
	case *ToolResultPart:
		if p == nil {
			return fmt.Errorf("%w: nil tool result part", ErrBadRequest)
		}
		for _, nested := range p.Parts {
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
