package llm

import "strings"

// Response is the normalized output from a provider.
type Response struct {
	ID            string
	Provider      string
	Model         string
	Parts         []Part
	StopReason    StopReason
	StopReasonRaw string
	Usage         Usage
	Raw           any
}

// Text returns all top-level text parts concatenated.
func (r *Response) Text() string {
	if r == nil {
		return ""
	}
	var b strings.Builder
	for _, part := range r.Parts {
		switch text := part.(type) {
		case TextPart:
			b.WriteString(text.Text)
		case *TextPart:
			if text != nil {
				b.WriteString(text.Text)
			}
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
		switch reasoning := part.(type) {
		case ReasoningPart:
			b.WriteString(reasoning.Text)
		case *ReasoningPart:
			if reasoning != nil {
				b.WriteString(reasoning.Text)
			}
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
		switch call := part.(type) {
		case ToolCallPart:
			calls = append(calls, clonePart(call).(ToolCallPart))
		case *ToolCallPart:
			if call != nil {
				calls = append(calls, *clonePart(call).(*ToolCallPart))
			}
		}
	}
	return calls
}

// StopReason is the provider-neutral reason a response stopped.
type StopReason string

const (
	StopReasonEndTurn         StopReason = "end_turn"
	StopReasonMaxTokens       StopReason = "max_tokens"
	StopReasonStopSequence    StopReason = "stop_sequence"
	StopReasonToolUse         StopReason = "tool_use"
	StopReasonContentFilter   StopReason = "content_filter"
	StopReasonRefusal         StopReason = "refusal"
	StopReasonContextOverflow StopReason = "context_overflow"
	StopReasonPaused          StopReason = "paused"
	StopReasonError           StopReason = "error"
	StopReasonOther           StopReason = "other"
)

// Usage contains normalized token and cost accounting.
type Usage struct {
	InputTokens      int64
	OutputTokens     int64
	TotalTokens      int64
	CacheReadTokens  int64
	CacheWriteTokens int64
	ReasoningTokens  int64
	CostUSD          *float64
	Raw              any
}
