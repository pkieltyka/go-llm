package llm

import (
	"fmt"
	"slices"
	"strings"
)

// SystemDeltaPart sets one named section of the system prompt from its
// position in the conversation on. Empty Text deletes the section. It may
// appear only in RoleSystem messages (FS §7A).
type SystemDeltaPart struct {
	Section string
	Text    string
	Cache   *CacheHint
}

func (SystemDeltaPart) part() {}

// ToolsDeltaPart changes the offered tools from its position in the
// conversation on: the Removed names are withdrawn, then the Added tools
// are offered, replacing a tool of the same name. It may appear only in
// RoleSystem messages (FS §7A).
type ToolsDeltaPart struct {
	Added   []Tool
	Removed []string
	Cache   *CacheHint
}

func (ToolsDeltaPart) part() {}

// PositionalSupport says which positional changes a target model accepts in
// place, from ModelInfo.Behavior.
type PositionalSupport struct {
	// Instructions: RoleSystem messages (text and section updates) after
	// the conversation has started.
	Instructions bool
	// Tools: ToolsDeltaPart changes in place. It requires Instructions.
	Tools bool
}

// FoldPositional returns req with the positional changes the target cannot
// send in place folded into Request.System and Request.Tools, which then
// hold the state after the last change. The leading run of RoleSystem
// messages is always folded, since it describes the conversation's start.
// Folding is deterministic and keeps the meaning; only the prompt-cache
// benefit of in-place changes is lost. The result shares no slices with req
// that it changes; req is not modified. Call it after ValidateRequest.
func FoldPositional(req *Request, support PositionalSupport) *Request {
	if req == nil || !hasSystemMessages(req.Messages) {
		return req
	}
	if !support.Instructions {
		support.Tools = false
	}
	state := newTranscriptState(req)
	out := *req
	out.Messages = make([]Message, 0, len(req.Messages))
	leading := true
	for _, msg := range req.Messages {
		if msg.Role != RoleSystem {
			leading = false
			out.Messages = append(out.Messages, msg)
			continue
		}
		if leading || !support.Instructions {
			state.apply(msg.Parts, true)
			continue
		}
		var kept []Part
		for _, part := range msg.Parts {
			if delta, ok := derefPart(part).(ToolsDeltaPart); ok && !support.Tools {
				state.applyTools(delta)
				continue
			}
			kept = append(kept, part)
		}
		if len(kept) > 0 {
			msg.Parts = kept
			out.Messages = append(out.Messages, msg)
		}
	}
	out.System = state.prompt()
	out.Tools = state.toolList()
	return &out
}

func hasSystemMessages(messages []Message) bool {
	for _, msg := range messages {
		if msg.Role == RoleSystem {
			return true
		}
	}
	return false
}

// HasToolsDelta reports whether any message carries a ToolsDeltaPart.
func HasToolsDelta(messages []Message) bool {
	for _, msg := range messages {
		if msg.Role != RoleSystem {
			continue
		}
		for _, part := range msg.Parts {
			if _, ok := derefPart(part).(ToolsDeltaPart); ok {
				return true
			}
		}
	}
	return false
}

// transcriptState replays the system prompt and tool set through a
// conversation, as pi's getCurrentSystemMessage and getCurrentTools do with
// JavaScript Maps: later text is appended to the base prompt, and sections
// and tools are patched by name. An update keeps its position; only a
// deleted (or removed) and re-added entry moves to the end.
type transcriptState struct {
	content  []string
	sections []string
	section  map[string]string
	tools    []Tool
	toolAt   map[string]int
}

func newTranscriptState(req *Request) *transcriptState {
	s := &transcriptState{section: map[string]string{}, toolAt: map[string]int{}}
	if req.System != "" {
		s.content = append(s.content, req.System)
	}
	for _, tool := range req.Tools {
		s.addTool(tool)
	}
	return s
}

// apply folds one system message; withText adds its text to the prompt.
func (s *transcriptState) apply(parts []Part, withText bool) {
	for _, part := range parts {
		switch p := derefPart(part).(type) {
		case TextPart:
			if withText && p.Text != "" {
				s.content = append(s.content, p.Text)
			}
		case SystemDeltaPart:
			s.setSection(p.Section, p.Text)
		case ToolsDeltaPart:
			s.applyTools(p)
		}
	}
}

func (s *transcriptState) setSection(name, text string) {
	_, exists := s.section[name]
	switch {
	case text != "" && exists:
		s.section[name] = text
	case text != "":
		s.section[name] = text
		s.sections = append(s.sections, name)
	case exists:
		delete(s.section, name)
		s.sections = slices.DeleteFunc(s.sections, func(existing string) bool { return existing == name })
	}
}

func (s *transcriptState) applyTools(delta ToolsDeltaPart) {
	for _, name := range delta.Removed {
		s.removeTool(name)
	}
	for _, tool := range delta.Added {
		if i, ok := s.toolAt[tool.Name]; ok {
			s.tools[i] = tool
			continue
		}
		s.addTool(tool)
	}
}

func (s *transcriptState) addTool(tool Tool) {
	s.toolAt[tool.Name] = len(s.tools)
	s.tools = append(s.tools, tool)
}

func (s *transcriptState) removeTool(name string) {
	i, ok := s.toolAt[name]
	if !ok {
		return
	}
	s.tools = append(s.tools[:i:i], s.tools[i+1:]...)
	delete(s.toolAt, name)
	for j := i; j < len(s.tools); j++ {
		s.toolAt[s.tools[j].Name] = j
	}
}

func (s *transcriptState) hasTool(name string) bool {
	_, ok := s.toolAt[name]
	return ok
}

func (s *transcriptState) prompt() string {
	parts := append([]string(nil), s.content...)
	for _, name := range s.sections {
		parts = append(parts, s.section[name])
	}
	return strings.Join(parts, "\n\n")
}

func (s *transcriptState) toolList() []Tool {
	if len(s.tools) == 0 {
		return nil
	}
	return append([]Tool(nil), s.tools...)
}

// validateTranscript checks RoleSystem messages and, when the request
// changes tools in place, that every tool call names a tool offered at its
// position. It returns the tools offered after the last message.
func validateTranscript(caps capabilitySet, req *Request) ([]Tool, error) {
	state := newTranscriptState(req)
	positional := HasToolsDelta(req.Messages)
	for _, msg := range req.Messages {
		if msg.Role != RoleSystem {
			for _, part := range msg.Parts {
				switch p := derefPart(part).(type) {
				case SystemDeltaPart, ToolsDeltaPart:
					return nil, fmt.Errorf("%w: %T is allowed only in system messages", ErrBadRequest, p)
				case ToolCallPart:
					if positional && msg.Role == RoleAssistant && !state.hasTool(p.Name) {
						return nil, fmt.Errorf("%w: tool call names %q, which is not offered at that point", ErrBadRequest, p.Name)
					}
				}
			}
			continue
		}
		for _, part := range msg.Parts {
			switch p := derefPart(part).(type) {
			case TextPart, UnknownPart:
			case SystemDeltaPart:
				if p.Section == "" {
					return nil, fmt.Errorf("%w: system delta section name is required", ErrBadRequest)
				}
			case ToolsDeltaPart:
				if err := validateToolsDelta(caps, state, p); err != nil {
					return nil, err
				}
			case nil:
				return nil, fmt.Errorf("%w: nil part", ErrBadRequest)
			default:
				return nil, fmt.Errorf("%w: system messages cannot carry %T", ErrBadRequest, p)
			}
			// Each delta applies before the next is checked, as replay
			// applies them (§7A).
			state.apply([]Part{part}, false)
		}
	}
	return state.tools, nil
}

func validateToolsDelta(caps capabilitySet, state *transcriptState, delta ToolsDeltaPart) error {
	for _, name := range delta.Removed {
		if !state.hasTool(name) {
			return fmt.Errorf("%w: tools delta removes %q, which is not offered", ErrBadRequest, name)
		}
	}
	added := make(map[string]struct{}, len(delta.Added))
	for _, tool := range delta.Added {
		if tool.Name == "" {
			return fmt.Errorf("%w: tool name is required", ErrBadRequest)
		}
		if _, dup := added[tool.Name]; dup {
			return fmt.Errorf("%w: duplicate tool name %q", ErrBadRequest, tool.Name)
		}
		added[tool.Name] = struct{}{}
		if !caps.has(CapabilityTools) {
			return unsupported(CapabilityTools)
		}
		if tool.Strict && !caps.has(CapabilityStrictTools) {
			return unsupported(CapabilityStrictTools)
		}
	}
	return nil
}
