package llm

// History is a minimal in-memory conversation builder. It is not goroutine-safe.
type History struct {
	msgs []Message
}

// NewHistory returns an empty history builder.
func NewHistory() *History {
	return &History{}
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

// Messages returns a defensive copy of the accumulated messages. Core part
// types are deep-copied; provider extension parts are shared by reference
// (their concrete types are unknown to this package), so treat extension
// parts as immutable.
func (h *History) Messages() []Message {
	return cloneMessages(h.msgs)
}

func (h *History) len() int {
	if h == nil {
		return 0
	}
	return len(h.msgs)
}

func (h *History) truncate(n int) {
	if h == nil {
		return
	}
	if n < 0 {
		n = 0
	}
	if n > len(h.msgs) {
		return
	}
	for i := n; i < len(h.msgs); i++ {
		h.msgs[i] = Message{}
	}
	h.msgs = h.msgs[:n]
}
