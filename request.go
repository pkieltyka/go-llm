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
	// ToolChoiceAuto lets the model decide whether to call tools (default).
	ToolChoiceAuto ToolChoiceMode = "auto"
	// ToolChoiceNone forbids tool calls for this request.
	ToolChoiceNone ToolChoiceMode = "none"
	// ToolChoiceRequired forces the model to call at least one tool.
	ToolChoiceRequired ToolChoiceMode = "required"
	// ToolChoiceTool forces a call to the tool named in ToolChoice.Name.
	ToolChoiceTool ToolChoiceMode = "tool"
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
	// FormatJSONSchema requests schema-constrained output (strict when the
	// provider and schema support it).
	FormatJSONSchema ResponseFormatType = "json_schema"
	// FormatJSONMode requests syntactically valid JSON without a schema.
	FormatJSONMode ResponseFormatType = "json_mode"
)

// Effort controls provider reasoning/thinking effort.
type Effort string

// Unified reasoning-effort levels. Adapters map each level to the nearest
// native equivalent (thinking budgets, reasoning.effort, ...); EffortNone
// disables reasoning where the provider allows it, and an empty Effort leaves
// the provider default untouched.
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
// The field is deliberately singular — a request targets one provider;
// routing/failover layers that re-dispatch a request across providers must
// swap or strip ProviderOptions per target provider (adapters reject
// options whose ForProvider does not match).
type ProviderOptions interface {
	ForProvider() string
}
