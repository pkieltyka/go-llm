package openrouter

import (
	"encoding/json"

	llm "github.com/pkieltyka/go-llm"
)

// ResponseExtras contains OpenRouter-specific response metadata.
type ResponseExtras struct {
	Provider           string
	NativeFinishReason string
	CostDetails        json.RawMessage
	Annotations        json.RawMessage
	ReasoningDetails   json.RawMessage
	IsBYOK             *bool
	Raw                any
}

// Extras returns OpenRouter-specific response metadata when available.
func Extras(resp *llm.Response) (*ResponseExtras, bool) {
	if resp == nil {
		return nil, false
	}
	extras, ok := resp.Raw.(*ResponseExtras)
	if ok && extras != nil {
		return extras, true
	}
	switch raw := resp.Raw.(type) {
	case []byte:
		return &ResponseExtras{Raw: raw}, true
	case json.RawMessage:
		return &ResponseExtras{Raw: raw}, true
	default:
		return nil, false
	}
}
