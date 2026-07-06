package llm

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"iter"
	"sync"
)

// Session is a thin, not-goroutine-safe conversation convenience wrapper.
type Session struct {
	provider   Provider
	model      string
	system     string
	effort     Effort
	maxTokens  int
	tools      []Tool
	toolChoice ToolChoice
	sessionID  string
	history    *History
	usage      Usage
	lastUsage  Usage
}

// SessionOption configures a Session.
type SessionOption func(*Session)

// NewSession creates a session for p and model.
func NewSession(p Provider, model string, opts ...SessionOption) *Session {
	s := &Session{
		provider:  p,
		model:     model,
		sessionID: randomSessionID(),
		history:   NewHistory(),
	}
	for _, opt := range opts {
		opt(s)
	}
	if s.sessionID == "" {
		s.sessionID = randomSessionID()
	}
	return s
}

// WithSessionSystem sets the system prompt sent with every session request.
func WithSessionSystem(system string) SessionOption {
	return func(s *Session) { s.system = system }
}

// WithSessionEffort sets the reasoning Effort sent with every session request.
func WithSessionEffort(effort Effort) SessionOption {
	return func(s *Session) { s.effort = effort }
}

// WithSessionMaxTokens caps output tokens for every session request.
func WithSessionMaxTokens(maxTokens int) SessionOption {
	return func(s *Session) { s.maxTokens = maxTokens }
}

// WithSessionTools sets the tools sent with every session request. The
// session never executes tools: handle Response.ToolCalls yourself, append
// results with AddToolResults, then call Continue.
func WithSessionTools(tools ...Tool) SessionOption {
	return func(s *Session) { s.tools = append([]Tool(nil), tools...) }
}

// WithSessionToolChoice sets the tool choice sent with every session request.
func WithSessionToolChoice(choice ToolChoice) SessionOption {
	return func(s *Session) { s.toolChoice = choice }
}

// WithSessionID overrides the generated session identifier used for provider
// session affinity (Request.SessionID). Empty values keep the generated ID.
func WithSessionID(id string) SessionOption {
	return func(s *Session) { s.sessionID = id }
}

// Chat appends a user turn, calls the provider, and appends the response.
// On error the appended user turn is rolled back, leaving history unchanged.
func (s *Session) Chat(ctx context.Context, parts ...Part) (*Response, error) {
	if s == nil || s.provider == nil {
		return nil, fmt.Errorf("%w: nil session provider", ErrBadRequest)
	}
	s.ensureHistory()
	rollbackTo := s.history.len()
	s.history.Add(UserParts(parts...))
	resp, err := s.provider.Chat(ctx, s.request())
	if err != nil {
		s.history.truncate(rollbackTo)
		return resp, err
	}
	s.appendResponse(resp)
	return resp, nil
}

// ChatText is Chat with one text part.
func (s *Session) ChatText(ctx context.Context, text string) (*Response, error) {
	return s.Chat(ctx, Text(text))
}

// Continue calls the provider on the current history WITHOUT appending a
// user turn — typically right after AddToolResults, to hand tool results
// back and collect the model's next turn. The assistant response is appended
// on success; on error history is left unchanged (matching Chat's rollback
// contract — nothing new remains in history after a failed call).
func (s *Session) Continue(ctx context.Context) (*Response, error) {
	if s == nil || s.provider == nil {
		return nil, fmt.Errorf("%w: nil session provider", ErrBadRequest)
	}
	s.ensureHistory()
	resp, err := s.provider.Chat(ctx, s.request())
	if err != nil {
		return resp, err
	}
	s.appendResponse(resp)
	return resp, nil
}

// ChatStream appends a user turn and returns a stream that appends the
// collected assistant response when the stream completes without error.
func (s *Session) ChatStream(ctx context.Context, parts ...Part) iter.Seq2[Event, error] {
	if s == nil || s.provider == nil {
		return func(yield func(Event, error) bool) {
			yield(nil, fmt.Errorf("%w: nil session provider", ErrBadRequest))
		}
	}
	s.ensureHistory()
	rollbackTo := s.history.len()
	s.history.Add(UserParts(parts...))
	stream := s.provider.ChatStream(ctx, s.request())
	return s.collectingStream(stream, rollbackTo)
}

func (s *Session) collectingStream(stream iter.Seq2[Event, error], rollbackTo int) iter.Seq2[Event, error] {
	var consumedMu sync.Mutex
	consumed := false
	return func(yield func(Event, error) bool) {
		consumedMu.Lock()
		if consumed {
			consumedMu.Unlock()
			yield(nil, fmt.Errorf("%w: session stream already consumed", ErrBadRequest))
			return
		}
		consumed = true
		consumedMu.Unlock()

		committed := false
		defer func() {
			if !committed {
				s.history.truncate(rollbackTo)
			}
		}()

		resp := &Response{}
		blocks := map[int]Part{}
		for event, err := range stream {
			if err != nil {
				yield(event, err)
				return
			}
			event, err = normalizeEvent(event)
			if err != nil {
				yield(nil, err)
				return
			}
			if err := applyCollectEvent(resp, blocks, event); err != nil {
				yield(nil, err)
				return
			}
			if !yield(event, nil) {
				return
			}
		}
		s.appendResponse(finalizeCollectedResponse(resp, blocks))
		committed = true
	}
}

// AddToolResults appends grouped tool results to the session history.
// Follow it with Continue to hand the results back to the model.
func (s *Session) AddToolResults(results ...ToolResultPart) {
	if s == nil {
		return
	}
	s.history.AddToolResults(results...)
}

// History returns the mutable underlying history builder.
func (s *Session) History() *History {
	if s == nil {
		return nil
	}
	return s.history
}

// Messages returns a defensive copy of session messages.
func (s *Session) Messages() []Message {
	if s == nil || s.history == nil {
		return nil
	}
	return s.history.Messages()
}

// Usage returns cumulative session usage.
func (s *Session) Usage() Usage {
	if s == nil {
		return Usage{}
	}
	return cloneUsage(s.usage)
}

// ContextUsage returns last-turn context occupancy using embedded model data.
func (s *Session) ContextUsage() (ContextUsage, bool) {
	if s == nil {
		return ContextUsage{}, false
	}
	provider := ""
	if s.provider != nil {
		provider = s.provider.Name()
	}
	info, ok := LookupModelInfo(provider, s.model)
	if !ok || info.ContextWindow <= 0 {
		return ContextUsage{}, false
	}
	return s.lastUsage.ContextUsage(int64(info.ContextWindow)), true
}

func (s *Session) request() *Request {
	return &Request{
		Model:      s.model,
		Messages:   s.history.Messages(),
		System:     s.system,
		MaxTokens:  s.maxTokens,
		Effort:     s.effort,
		Tools:      append([]Tool(nil), s.tools...),
		ToolChoice: s.toolChoice,
		SessionID:  s.sessionID,
	}
}

func (s *Session) ensureHistory() {
	if s.history == nil {
		s.history = NewHistory()
	}
}

func (s *Session) appendResponse(resp *Response) {
	if resp == nil {
		return
	}
	if resp.Provider == "" && s.provider != nil {
		resp.Provider = s.provider.Name()
	}
	if resp.Model == "" {
		resp.Model = s.model
	}
	s.history.AddResponse(resp)
	s.lastUsage = cloneUsage(resp.Usage)
	s.usage = sumUsage(s.usage, resp.Usage)
}

func randomSessionID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return ""
	}
	return "sess_" + hex.EncodeToString(b[:])
}
