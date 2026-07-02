package llm

// History is a minimal in-memory conversation builder. It is not goroutine-safe.
type History struct {
	msgs                   []Message
	foreignReasoningAsText bool
}

// HistoryOption configures a History.
type HistoryOption func(*History)

// NewHistory returns a configured history builder.
func NewHistory(opts ...HistoryOption) *History {
	h := &History{}
	for _, opt := range opts {
		opt(h)
	}
	return h
}

// WithForeignReasoningAsText preserves foreign reasoning as text on replay.
func WithForeignReasoningAsText() HistoryOption {
	return func(h *History) {
		h.foreignReasoningAsText = true
	}
}

// ForeignReasoningAsText reports whether foreign reasoning should be replayed as text.
func (h *History) ForeignReasoningAsText() bool {
	return h.foreignReasoningAsText
}

// Add appends messages to the history.
func (h *History) Add(msgs ...Message) {
	h.msgs = append(h.msgs, cloneMessages(msgs)...)
}

// AddUserText appends a user text message.
func (h *History) AddUserText(text string) {
	h.Add(UserText(text))
}

// AddResponse appends an assistant turn from a response.
func (h *History) AddResponse(resp *Response) {
	if resp == nil {
		return
	}
	h.Add(Message{
		Role:     RoleAssistant,
		Parts:    cloneParts(resp.Parts),
		Provider: resp.Provider,
		Model:    resp.Model,
	})
}

// AddToolResults appends all tool results as one grouped tool message.
func (h *History) AddToolResults(results ...ToolResultPart) {
	if len(results) == 0 {
		return
	}
	parts := make([]Part, len(results))
	for i, result := range results {
		parts[i] = clonePart(result)
	}
	h.Add(Message{Role: RoleTool, Parts: parts})
}

// Messages returns a defensive copy of the accumulated messages.
func (h *History) Messages() []Message {
	return cloneMessages(h.msgs)
}
