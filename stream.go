package llm

import (
	"fmt"
	"iter"
	"sort"
	"strings"
	"time"
)

// Event is the marker interface for stream events.
type Event interface {
	event()
}

// MessageStart begins a streamed response.
type MessageStart struct {
	ID       string
	Provider string
	Model    string
}

func (MessageStart) event() {}

// TextDelta contains a streamed text fragment.
type TextDelta struct {
	Index int
	Text  string
}

func (TextDelta) event() {}

// ReasoningDelta contains a streamed reasoning fragment.
type ReasoningDelta struct {
	Index int
	Text  string
}

func (ReasoningDelta) event() {}

// ToolCallStart begins a streamed tool call.
type ToolCallStart struct {
	Index int
	ID    string
	Name  string
}

func (ToolCallStart) event() {}

// ToolCallDelta contains a streamed tool-call argument fragment.
type ToolCallDelta struct {
	Index        int
	ArgsFragment string
}

func (ToolCallDelta) event() {}

// ToolCallEnd closes a streamed tool call.
type ToolCallEnd struct {
	Index int
}

func (ToolCallEnd) event() {}

// MessageEnd closes a streamed response.
type MessageEnd struct {
	StopReason    StopReason
	StopReasonRaw string
	Usage         Usage
}

func (MessageEnd) event() {}

// Collect drains a stream into a complete Response.
func Collect(events iter.Seq2[Event, error]) (*Response, error) {
	if events == nil {
		return nil, fmt.Errorf("%w: nil stream", ErrBadRequest)
	}

	resp := &Response{}
	blocks := map[int]Part{}
	seenStart := false

	for event, err := range events {
		if err != nil {
			if seenStart || len(blocks) > 0 {
				return finalizeCollectedResponse(resp, blocks), err
			}
			return nil, err
		}
		event, err = normalizeEvent(event)
		if err != nil {
			return finalizeCollectedResponse(resp, blocks), err
		}

		switch e := event.(type) {
		case MessageStart:
			seenStart = true
			resp.ID = e.ID
			resp.Provider = e.Provider
			resp.Model = e.Model
		case TextDelta:
			if err := appendTextDelta(blocks, e.Index, e.Text); err != nil {
				return finalizeCollectedResponse(resp, blocks), err
			}
		case ReasoningDelta:
			if err := appendReasoningDelta(blocks, e.Index, e.Text); err != nil {
				return finalizeCollectedResponse(resp, blocks), err
			}
		case ToolCallStart:
			if err := startToolCall(blocks, e); err != nil {
				return finalizeCollectedResponse(resp, blocks), err
			}
		case ToolCallDelta:
			if err := appendToolCallDelta(blocks, e); err != nil {
				return finalizeCollectedResponse(resp, blocks), err
			}
		case ToolCallEnd:
			if err := endToolCall(blocks, e.Index); err != nil {
				return finalizeCollectedResponse(resp, blocks), err
			}
		case MessageEnd:
			resp.StopReason = e.StopReason
			resp.StopReasonRaw = e.StopReasonRaw
			resp.Usage = e.Usage
		default:
			return finalizeCollectedResponse(resp, blocks), fmt.Errorf("%w: unknown stream event %T", ErrBadRequest, event)
		}
	}

	return finalizeCollectedResponse(resp, blocks), nil
}

func normalizeEvent(event Event) (Event, error) {
	switch e := event.(type) {
	case nil:
		return nil, fmt.Errorf("%w: nil stream event", ErrBadRequest)
	case MessageStart:
		return e, nil
	case *MessageStart:
		if e == nil {
			return nil, fmt.Errorf("%w: nil message start event", ErrBadRequest)
		}
		return *e, nil
	case TextDelta:
		return e, nil
	case *TextDelta:
		if e == nil {
			return nil, fmt.Errorf("%w: nil text delta event", ErrBadRequest)
		}
		return *e, nil
	case ReasoningDelta:
		return e, nil
	case *ReasoningDelta:
		if e == nil {
			return nil, fmt.Errorf("%w: nil reasoning delta event", ErrBadRequest)
		}
		return *e, nil
	case ToolCallStart:
		return e, nil
	case *ToolCallStart:
		if e == nil {
			return nil, fmt.Errorf("%w: nil tool call start event", ErrBadRequest)
		}
		return *e, nil
	case ToolCallDelta:
		return e, nil
	case *ToolCallDelta:
		if e == nil {
			return nil, fmt.Errorf("%w: nil tool call delta event", ErrBadRequest)
		}
		return *e, nil
	case ToolCallEnd:
		return e, nil
	case *ToolCallEnd:
		if e == nil {
			return nil, fmt.Errorf("%w: nil tool call end event", ErrBadRequest)
		}
		return *e, nil
	case MessageEnd:
		return e, nil
	case *MessageEnd:
		if e == nil {
			return nil, fmt.Errorf("%w: nil message end event", ErrBadRequest)
		}
		return *e, nil
	default:
		return nil, fmt.Errorf("%w: unknown stream event %T", ErrBadRequest, event)
	}
}

func appendTextDelta(blocks map[int]Part, index int, text string) error {
	if text == "" {
		return nil
	}
	if index < 0 {
		return fmt.Errorf("%w: negative stream index %d", ErrBadRequest, index)
	}
	part, ok := blocks[index]
	if !ok {
		blocks[index] = Text(text)
		return nil
	}
	textPart, ok := part.(TextPart)
	if !ok {
		return fmt.Errorf("%w: stream index %d changed from %T to TextDelta", ErrBadRequest, index, part)
	}
	textPart.Text += text
	blocks[index] = textPart
	return nil
}

func appendReasoningDelta(blocks map[int]Part, index int, text string) error {
	if text == "" {
		return nil
	}
	if index < 0 {
		return fmt.Errorf("%w: negative stream index %d", ErrBadRequest, index)
	}
	part, ok := blocks[index]
	if !ok {
		blocks[index] = ReasoningPart{Text: text}
		return nil
	}
	reasoningPart, ok := part.(ReasoningPart)
	if !ok {
		return fmt.Errorf("%w: stream index %d changed from %T to ReasoningDelta", ErrBadRequest, index, part)
	}
	if len(reasoningPart.Raw) != 0 || reasoningPart.Provider != "" {
		return fmt.Errorf("%w: stream index %d cannot append to raw reasoning part", ErrBadRequest, index)
	}
	reasoningPart.Text += text
	blocks[index] = reasoningPart
	return nil
}

func startToolCall(blocks map[int]Part, event ToolCallStart) error {
	if event.Index < 0 {
		return fmt.Errorf("%w: negative stream index %d", ErrBadRequest, event.Index)
	}
	if part, ok := blocks[event.Index]; ok {
		return fmt.Errorf("%w: stream index %d already started as %T", ErrBadRequest, event.Index, part)
	}
	blocks[event.Index] = ToolCallPart{ID: event.ID, Name: event.Name}
	return nil
}

func appendToolCallDelta(blocks map[int]Part, event ToolCallDelta) error {
	part, ok := blocks[event.Index]
	if !ok {
		return fmt.Errorf("%w: tool call delta before start for index %d", ErrBadRequest, event.Index)
	}
	call, ok := part.(ToolCallPart)
	if !ok {
		return fmt.Errorf("%w: stream index %d is %T, not ToolCallPart", ErrBadRequest, event.Index, part)
	}
	call.Args = append(call.Args, event.ArgsFragment...)
	blocks[event.Index] = call
	return nil
}

func endToolCall(blocks map[int]Part, index int) error {
	part, ok := blocks[index]
	if !ok {
		return fmt.Errorf("%w: tool call end before start for index %d", ErrBadRequest, index)
	}
	if _, ok := part.(ToolCallPart); !ok {
		return fmt.Errorf("%w: stream index %d is %T, not ToolCallPart", ErrBadRequest, index, part)
	}
	return nil
}

func finalizeCollectedResponse(resp *Response, blocks map[int]Part) *Response {
	indexes := make([]int, 0, len(blocks))
	for index := range blocks {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)

	resp.Parts = make([]Part, 0, len(indexes))
	for _, index := range indexes {
		resp.Parts = append(resp.Parts, clonePart(blocks[index]))
	}
	return resp
}

// StreamTextOption configures StreamText.
type StreamTextOption func(*streamTextOptions)

type streamTextOptions struct {
	debounce time.Duration
}

// WithDebounce delays text emission by window and flushes pending text on stream end or error.
func WithDebounce(window time.Duration) StreamTextOption {
	return func(opts *streamTextOptions) {
		opts.debounce = window
	}
}

// StreamText filters a stream to plain text deltas.
func StreamText(events iter.Seq2[Event, error], opts ...StreamTextOption) iter.Seq2[string, error] {
	if events == nil {
		return func(yield func(string, error) bool) {
			yield("", fmt.Errorf("%w: nil stream", ErrBadRequest))
		}
	}

	options := streamTextOptions{}
	for _, opt := range opts {
		opt(&options)
	}

	if options.debounce <= 0 {
		return func(yield func(string, error) bool) {
			for event, err := range events {
				if err != nil {
					yield("", err)
					return
				}
				event, err = normalizeEvent(event)
				if err != nil {
					yield("", err)
					return
				}
				if delta, ok := event.(TextDelta); ok && delta.Text != "" {
					if !yield(delta.Text, nil) {
						return
					}
				}
			}
		}
	}

	return func(yield func(string, error) bool) {
		next, stop := iter.Pull2(events)
		defer stop()

		var buffer strings.Builder

		flush := func() bool {
			if buffer.Len() == 0 {
				return true
			}
			text := buffer.String()
			buffer.Reset()
			return yield(text, nil)
		}

		waitAndFlush := func() bool {
			// Keep upstream pulls synchronous: racing a timer against a blocked
			// provider read requires a concurrent next/stop, which iter.Pull2
			// explicitly disallows and would weaken stream cleanup on early exit.
			timer := time.NewTimer(options.debounce)
			<-timer.C
			return flush()
		}

		for {
			event, err, ok := next()
			if !ok {
				flush()
				return
			}
			if err != nil {
				if !flush() {
					return
				}
				yield("", err)
				return
			}
			event, err = normalizeEvent(event)
			if err != nil {
				if !flush() {
					return
				}
				yield("", err)
				return
			}

			switch e := event.(type) {
			case TextDelta:
				if e.Text == "" {
					continue
				}
				buffer.WriteString(e.Text)
				if !waitAndFlush() {
					return
				}
			case MessageEnd:
				if !flush() {
					return
				}
			}
		}
	}
}
