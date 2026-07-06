package llm

import "strings"

// Response is the normalized output from a provider.
type Response struct {
	ID               string
	Provider         string
	Model            string
	Parts            []Part
	StopReason       StopReason
	StopReasonRaw    string
	Usage            Usage
	DroppedToolCalls []DroppedToolCall
	Raw              any
}

// Text returns all top-level text parts concatenated.
func (r *Response) Text() string {
	if r == nil {
		return ""
	}
	var b strings.Builder
	for _, part := range r.Parts {
		if text, ok := derefPart(part).(TextPart); ok {
			b.WriteString(text.Text)
		}
	}
	return b.String()
}

// Reasoning returns all top-level reasoning text concatenated.
func (r *Response) Reasoning() string {
	if r == nil {
		return ""
	}
	var b strings.Builder
	for _, part := range r.Parts {
		if reasoning, ok := derefPart(part).(ReasoningPart); ok {
			b.WriteString(reasoning.Text)
		}
	}
	return b.String()
}

// ToolCalls returns all top-level tool calls in order.
func (r *Response) ToolCalls() []ToolCallPart {
	if r == nil {
		return nil
	}
	var calls []ToolCallPart
	for _, part := range r.Parts {
		if call, ok := derefPart(part).(ToolCallPart); ok {
			calls = append(calls, clonePart(call).(ToolCallPart))
		}
	}
	return calls
}

// StopReason is the provider-neutral reason a response stopped.
type StopReason string

// Normalized stop reasons every adapter maps its native finish/stop values
// onto. The provider's verbatim value is preserved in Response.StopReasonRaw.
const (
	// StopReasonEndTurn is a natural completion.
	StopReasonEndTurn StopReason = "end_turn"
	// StopReasonMaxTokens means the output budget was exhausted.
	StopReasonMaxTokens StopReason = "max_tokens"
	// StopReasonStopSequence means a requested stop sequence matched.
	StopReasonStopSequence StopReason = "stop_sequence"
	// StopReasonToolUse means the model stopped to call tools.
	StopReasonToolUse StopReason = "tool_use"
	// StopReasonContentFilter means provider moderation halted the output.
	StopReasonContentFilter StopReason = "content_filter"
	// StopReasonRefusal means the model refused; content may be empty.
	StopReasonRefusal StopReason = "refusal"
	// StopReasonContextOverflow means the context window was exceeded.
	StopReasonContextOverflow StopReason = "context_overflow"
	// StopReasonPaused means the provider paused the turn (e.g. Anthropic
	// pause_turn) and expects the conversation to be continued.
	StopReasonPaused StopReason = "paused"
	// StopReasonError means the provider reported a terminal error state.
	StopReasonError StopReason = "error"
	// StopReasonOther is any native stop value with no unified equivalent;
	// consult Response.StopReasonRaw.
	StopReasonOther StopReason = "other"
)

// Usage contains normalized token and cost accounting.
//
// Every adapter maps its native usage onto one ADDITIVE shape (FS §11):
// prompt occupancy = InputTokens + CacheReadTokens + CacheWriteTokens, and
// TotalTokens = prompt occupancy + OutputTokens. Fields are zero when the
// provider does not report them.
type Usage struct {
	// InputTokens counts non-cached prompt tokens only — cache reads and
	// writes are EXCLUDED (providers that report cached tokens as a subset
	// of their prompt total are re-normalized by their adapter).
	InputTokens int64
	// OutputTokens counts all generated tokens, including reasoning.
	OutputTokens int64
	// TotalTokens is prompt occupancy plus OutputTokens.
	TotalTokens int64
	// CacheReadTokens counts prompt tokens served from a provider cache;
	// zero when unreported or no cache was hit.
	CacheReadTokens int64
	// CacheWriteTokens counts prompt tokens written to a provider cache;
	// zero when unreported.
	CacheWriteTokens int64
	// ReasoningTokens is an informational SUBSET of OutputTokens, never
	// additive — context and cost math count output exactly once. Exception
	// (FS §11): when a provider's native accounting violates the subset
	// property (observed on OpenRouter for some upstreams), adapters pass
	// the native values through without clamping.
	ReasoningTokens int64
	// CostUSD is the request cost in US dollars; nil when unknown (neither
	// provider-reported nor estimable from a price table).
	CostUSD *float64
	// CostSource says where CostUSD came from: CostSourceNative for
	// provider-reported billing, CostSourceEstimated for price-table
	// estimates. Empty when CostUSD is nil.
	CostSource string
	// Raw is the provider's native usage payload; not serialized.
	Raw any
}

// CostUSD provenance values for Usage.CostSource.
const (
	// CostSourceNative marks a cost reported by the provider itself
	// (billing-grade, e.g. OpenRouter usage.cost).
	CostSourceNative = "native"
	// CostSourceEstimated marks a cost computed from a price table
	// (best-effort estimate, not billing data).
	CostSourceEstimated = "estimated"
)

// DroppedToolCall records a malformed tool call that could not be rescued by
// an adapter.
type DroppedToolCall struct {
	Index  int    `json:"index"`
	Reason string `json:"reason"`
}

// ContextUsage reports how much of a model context window is occupied.
type ContextUsage struct {
	UsedTokens  int64
	Window      int64
	Remaining   int64
	UsedPercent float64
}

// ContextUsage counts prompt occupancy including cached tokens. Providers
// often report cache read/write tokens outside InputTokens, so excluding them
// undercounts the active conversation size.
func (u Usage) ContextUsage(window int64) ContextUsage {
	used := u.InputTokens + u.CacheReadTokens + u.CacheWriteTokens + u.OutputTokens
	out := ContextUsage{
		UsedTokens: used,
		Window:     window,
	}
	if window > 0 {
		out.Remaining = window - used
		out.UsedPercent = float64(used) / float64(window) * 100
	}
	return out
}
