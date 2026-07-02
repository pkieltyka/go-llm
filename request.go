package llm

// Request is the provider-neutral shape for a chat completion.
type Request struct {
	Model           string
	Messages        []Message
	System          string
	SystemCache     *CacheHint
	MaxTokens       int
	Temperature     *float64
	TopP            *float64
	StopSequences   []string
	Tools           []Tool
	ToolChoice      ToolChoice
	ResponseFormat  *ResponseFormat
	Effort          Effort
	SessionID       string
	ProviderOptions ProviderOptions
}

// Tool describes a caller-managed function/tool the model may invoke.
type Tool struct {
	Name        string
	Description string
	InputSchema any
	Strict      bool
	Annotations ToolAnnotations
}

// ToolAnnotations are caller-consumed hints; providers never receive them.
type ToolAnnotations struct {
	ReadOnly    bool
	Destructive bool
	Idempotent  bool
	OpenWorld   bool
}

// ToolChoice controls whether and how a provider may use tools.
type ToolChoice struct {
	Mode ToolChoiceMode
	Name string
}

// ToolChoiceMode is the provider-neutral tool choice mode.
type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = "auto"
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceTool     ToolChoiceMode = "tool"
)

// ResponseFormat requests structured output.
type ResponseFormat struct {
	Type   ResponseFormatType
	Name   string
	Schema any
	Strict bool
}

// ResponseFormatType identifies the structured-output mechanism.
type ResponseFormatType string

const (
	FormatJSONSchema ResponseFormatType = "json_schema"
	FormatJSONMode   ResponseFormatType = "json_mode"
)

// Effort controls provider reasoning/thinking effort.
type Effort string

const (
	EffortNone    Effort = "none"
	EffortMinimal Effort = "minimal"
	EffortLow     Effort = "low"
	EffortMedium  Effort = "medium"
	EffortHigh    Effort = "high"
	EffortXHigh   Effort = "xhigh"
	EffortMax     Effort = "max"
)

// ProviderOptions is implemented by per-provider extension option structs.
type ProviderOptions interface {
	ForProvider() string
}
