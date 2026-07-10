package llm

import (
	"errors"
	"strings"
	"testing"
)

func TestValidateRequestUnsupportedCapabilities(t *testing.T) {
	base := func() *Request {
		return &Request{
			Model:    "model-a",
			Messages: []Message{UserText("hello")},
		}
	}

	tests := []struct {
		name string
		caps []Capability
		req  *Request
		want Capability
	}{
		{
			name: "tools",
			req: func() *Request {
				req := base()
				req.Tools = []Tool{{Name: "lookup"}}
				return req
			}(),
			want: CapabilityTools,
		},
		{
			name: "strict tools",
			caps: []Capability{CapabilityTools},
			req: func() *Request {
				req := base()
				req.Tools = []Tool{{Name: "lookup", Strict: true}}
				return req
			}(),
			want: CapabilityStrictTools,
		},
		{
			name: "tool choice required",
			caps: []Capability{CapabilityTools},
			req: func() *Request {
				req := base()
				req.Tools = []Tool{{Name: "lookup"}}
				req.ToolChoice = ToolChoice{Mode: ToolChoiceRequired}
				return req
			}(),
			want: CapabilityToolChoiceRequired,
		},
		{
			name: "json schema",
			req: func() *Request {
				req := base()
				req.ResponseFormat = &ResponseFormat{Type: FormatJSONSchema, Schema: map[string]any{"type": "object"}}
				return req
			}(),
			want: CapabilityJSONSchema,
		},
		{
			name: "json mode",
			req: func() *Request {
				req := base()
				req.ResponseFormat = &ResponseFormat{Type: FormatJSONMode}
				return req
			}(),
			want: CapabilityJSONMode,
		},
		{
			name: "reasoning",
			req: func() *Request {
				req := base()
				req.Effort = EffortHigh
				return req
			}(),
			want: CapabilityReasoning,
		},
		{
			name: "image input",
			req: func() *Request {
				req := base()
				req.Messages = []Message{UserParts(ImageURL("https://example.test/image.png"))}
				return req
			}(),
			want: CapabilityImageInput,
		},
		{
			name: "pdf input",
			req: func() *Request {
				req := base()
				req.Messages = []Message{UserParts(FileURL("https://example.test/file.pdf", "application/pdf"))}
				return req
			}(),
			want: CapabilityPDFInput,
		},
		{
			name: "stop sequences",
			req: func() *Request {
				req := base()
				req.StopSequences = []string{"END"}
				return req
			}(),
			want: CapabilityStopSequences,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateRequest(tt.caps, tt.req)
			if !errors.Is(err, ErrUnsupported) {
				t.Fatalf("error = %v, want ErrUnsupported", err)
			}
			if !strings.Contains(err.Error(), string(tt.want)) {
				t.Fatalf("error = %q, want capability %q", err, tt.want)
			}
		})
	}
}

func TestValidateRequestAllowsSupportedFeatures(t *testing.T) {
	req := &Request{
		Model: "model-a",
		Messages: []Message{
			UserParts(ImageURL("https://example.test/image.png"), FileURL("https://example.test/file.pdf", "application/pdf")),
		},
		StopSequences: []string{"END"},
		Tools:         []Tool{{Name: "lookup", Strict: true}},
		ToolChoice:    ToolChoice{Mode: ToolChoiceRequired},
		ResponseFormat: &ResponseFormat{
			Type:   FormatJSONSchema,
			Schema: map[string]any{"type": "object"},
		},
		Effort: EffortMedium,
	}
	caps := []Capability{
		CapabilityImageInput,
		CapabilityPDFInput,
		CapabilityStopSequences,
		CapabilityTools,
		CapabilityStrictTools,
		CapabilityToolChoiceRequired,
		CapabilityJSONSchema,
		CapabilityReasoning,
	}

	if err := ValidateRequest(caps, req); err != nil {
		t.Fatalf("ValidateRequest returned error: %v", err)
	}
}

func TestValidateRequestPointerParts(t *testing.T) {
	req := &Request{
		Model:    "model-a",
		Messages: []Message{{Role: RoleUser, Parts: []Part{&ImagePart{URL: "https://example.test/image.png"}}}},
	}

	err := ValidateRequest(nil, req)
	if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("error = %v, want ErrUnsupported", err)
	}
	if !strings.Contains(err.Error(), string(CapabilityImageInput)) {
		t.Fatalf("error = %q, want image capability", err)
	}
}

func TestValidateRequestNilPointerPart(t *testing.T) {
	var part *TextPart
	req := &Request{
		Model:    "model-a",
		Messages: []Message{{Role: RoleUser, Parts: []Part{part}}},
	}

	err := ValidateRequest(nil, req)
	if !errors.Is(err, ErrBadRequest) {
		t.Fatalf("error = %v, want ErrBadRequest", err)
	}
}

func TestValidateStreamRequest(t *testing.T) {
	req := &Request{
		Model:    "model-a",
		Messages: []Message{UserText("hello")},
	}

	err := ValidateStreamRequest(nil, req)
	if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("error = %v, want ErrUnsupported", err)
	}
	if !strings.Contains(err.Error(), string(CapabilityStreaming)) {
		t.Fatalf("error = %q, want streaming capability", err)
	}

	if err := ValidateStreamRequest([]Capability{CapabilityStreaming}, req); err != nil {
		t.Fatalf("ValidateStreamRequest returned error: %v", err)
	}
}

func TestValidateStreamRequestAllowsToolsWithoutToolStreaming(t *testing.T) {
	// tool-streaming means incremental tool-argument deltas, not "tools
	// usable with ChatStream" — tools are gated by CapabilityTools alone.
	req := &Request{
		Model:    "model-a",
		Messages: []Message{UserText("hello")},
		Tools:    []Tool{{Name: "lookup"}},
	}

	if err := ValidateStreamRequest([]Capability{CapabilityStreaming, CapabilityTools}, req); err != nil {
		t.Fatalf("ValidateStreamRequest returned error: %v", err)
	}
}

func TestValidateProviderOptions(t *testing.T) {
	req := &Request{ProviderOptions: testProviderOptions("openai")}

	if err := ValidateProviderOptions("openai", req); err != nil {
		t.Fatalf("ValidateProviderOptions returned error: %v", err)
	}

	err := ValidateProviderOptions("anthropic", req)
	if !errors.Is(err, ErrBadRequest) {
		t.Fatalf("error = %v, want ErrBadRequest", err)
	}
}

func TestValidateProviderOptionsRejectsTypedNilBeforeMethodCall(t *testing.T) {
	var pointer *nilPointerProviderOptions
	var channel nilChanProviderOptions
	var function nilFuncProviderOptions
	var mapping nilMapProviderOptions
	var slice nilSliceProviderOptions

	tests := []struct {
		name    string
		options ProviderOptions
	}{
		{name: "chan", options: channel},
		{name: "func", options: function},
		{name: "map", options: mapping},
		{name: "pointer", options: pointer},
		{name: "slice", options: slice},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateProviderOptions("openai", &Request{ProviderOptions: tt.options})
			if !errors.Is(err, ErrBadRequest) {
				t.Fatalf("error = %v, want ErrBadRequest", err)
			}
		})
	}
}

func TestValidateProviderOptionsCallsForProviderOnce(t *testing.T) {
	options := &countingProviderOptions{provider: "anthropic"}
	err := ValidateProviderOptions("openai", &Request{ProviderOptions: options})
	if !errors.Is(err, ErrBadRequest) {
		t.Fatalf("error = %v, want ErrBadRequest", err)
	}
	if options.calls != 1 {
		t.Fatalf("ForProvider calls = %d, want 1", options.calls)
	}
}

func TestValidateRequestRejectsInvalidNumericAndToolPreflight(t *testing.T) {
	base := func() *Request {
		return &Request{
			Model:    "model-a",
			Messages: []Message{UserText("hello")},
			Tools:    []Tool{{Name: "lookup"}},
		}
	}
	caps := []Capability{CapabilityTools, CapabilityToolChoiceRequired}

	tests := []struct {
		name string
		req  *Request
	}{
		{name: "negative max tokens", req: func() *Request {
			req := base()
			req.MaxTokens = -1
			return req
		}()},
		{name: "duplicate tool names", req: func() *Request {
			req := base()
			req.Tools = append(req.Tools, Tool{Name: "lookup"})
			return req
		}()},
		{name: "forced undeclared tool", req: func() *Request {
			req := base()
			req.ToolChoice = ToolChoice{Mode: ToolChoiceTool, Name: "missing"}
			return req
		}()},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateRequest(caps, tt.req); !errors.Is(err, ErrBadRequest) {
				t.Fatalf("error = %v, want ErrBadRequest", err)
			}
		})
	}
}

type testProviderOptions string

func (o testProviderOptions) ForProvider() string {
	return string(o)
}

type nilPointerProviderOptions struct{}

func (*nilPointerProviderOptions) ForProvider() string { panic("ForProvider called on typed nil") }

type nilChanProviderOptions chan struct{}

func (nilChanProviderOptions) ForProvider() string { panic("ForProvider called on typed nil") }

type nilFuncProviderOptions func()

func (nilFuncProviderOptions) ForProvider() string { panic("ForProvider called on typed nil") }

type nilMapProviderOptions map[string]string

func (nilMapProviderOptions) ForProvider() string { panic("ForProvider called on typed nil") }

type nilSliceProviderOptions []string

func (nilSliceProviderOptions) ForProvider() string { panic("ForProvider called on typed nil") }

type countingProviderOptions struct {
	provider string
	calls    int
}

func (o *countingProviderOptions) ForProvider() string {
	o.calls++
	return o.provider
}
