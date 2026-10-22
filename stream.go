package llm

import (
	"encoding/json"
	"fmt"
	"iter"
	"sort"
	"strings"
	"time"
)

// Event is the marker interface for stream events.
//
// Events are value types: construct and pass them as values. Pointer events
// satisfy the interface inherently and are accepted at package boundaries,
// but they are normalized to their value form on entry; a typed nil pointer
// event is rejected as a nil event.
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
	Index    int
	Text     string
	Raw      json.RawMessage
	Provider string
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

// ToolCallIDChanged corrects the ID of an active provisional tool call.
// Direct stream consumers must replace OldID with NewID for the call at Index.
// OldID must match the ID established by ToolCallStart or the preceding
// ToolCallIDChanged event. Collect rejects mismatches as malformed streams;
// an identity change never creates DroppedToolCalls metadata.
type ToolCallIDChanged struct {
	Index int
	OldID string
	NewID string
}

func (ToolCallIDChanged) event() {}

// ToolCallEnd closes a streamed tool call.
type ToolCallEnd struct {
	Index int
}

func (ToolCallEnd) event() {}

// ToolCallDropped reports an actual malformed tool call that an adapter could
// not rescue. It is visible in streams and collected onto Response.
type ToolCallDropped struct {
	Index  int
	Reason string
}

func (ToolCallDropped) event() {}

// MessageEnd closes a streamed response.
type MessageEnd struct {
	StopReason    StopReason
	StopReasonRaw string
	Usage         Usage
	// Raw optionally carries the terminal raw SDK payload when the adapter
	// provides one. Collect installs it as Response.Raw so provider extras
	// accessors (e.g. openai.Extras) work on collected streams too.
	Raw any
}

func (MessageEnd) event() {}

// Collect drains a stream into a complete Response.
//
// On an in-stream error (including a malformed event) Collect returns the
// partial Response accumulated so far alongside the error — never nil once
// MessageStart was seen — so aborted turns can be persisted. When the error
// arrives before MessageStart and before any content, the Response is nil.
// Content accumulates keyed by block Index; blocks may interleave and are
// never assumed contiguous.
func Collect(events iter.Seq2[Event, error]) (*Response, error) {
	if events == nil {
		return nil, fmt.Errorf("%w: nil stream", ErrBadRequest)
	}

	var c Collector
	for event, err := range events {
		if err != nil {
			return c.Finish(err)
		}
		if err := c.Add(event); err != nil {
			return c.Finish(err)
		}
	}
	return c.Finish(nil)
}

// Collector accumulates stream events into a Response one event at a time,
// with exactly the semantics of Collect: Collect is a loop over Add followed
// by Finish. Use it when a consumer must act on events as they arrive — for
// example to forward deltas while also persisting periodic Snapshots of the
// response so far.
//
// The zero value is ready to use. A Collector is not safe for concurrent use
// (streams are single-use and consumed by one goroutine) and must not be
// copied after first use.
type Collector struct {
	resp        Response
	blocks      map[int]Part
	activeTools map[int]struct{}
	seenStart   bool
	err         error
	finished    bool
	final       *Response
}

// Add normalizes and folds one event. Pointer events are accepted and
// normalized to their value form, as in Collect.
//
// Add returns the canonical malformed-stream error for an event that
// violates the stream grammar (unknown or nil event, index type collision,
// mismatched ToolCallIDChanged, ...). The error is sticky: once Add fails,
// later Add calls return the same error without applying anything, and
// Finish reports it. Add after Finish returns ErrBadRequest.
func (c *Collector) Add(event Event) error {
	if c.finished {
		return fmt.Errorf("%w: collector already finished", ErrBadRequest)
	}
	if c.err != nil {
		return c.err
	}
	if c.blocks == nil {
		c.blocks = map[int]Part{}
		c.activeTools = map[int]struct{}{}
	}
	event, err := normalizeEvent(event)
	if err != nil {
		c.err = err
		return err
	}
	if _, ok := event.(MessageStart); ok {
		c.seenStart = true
	}
	if err := applyCollectEvent(&c.resp, c.blocks, c.activeTools, event); err != nil {
		c.err = err
		return err
	}
	return nil
}

// Snapshot returns an owned deep copy of the Response that Collect would
// return if the stream failed now: nil before MessageStart and before any
// content, otherwise the partial response (StopReason stays empty until
// MessageEnd). After Finish it returns a copy of the response Finish
// returned, including any changes the caller made to it. Later Add calls never mutate a returned snapshot, and
// mutating a snapshot never affects the Collector. Raw SDK payloads
// (Response.Raw, Usage.Raw) are shared by reference, as with Response.Clone.
func (c *Collector) Snapshot() *Response {
	if c.finished {
		return c.final.Clone()
	}
	if !c.seenStart && len(c.blocks) == 0 {
		return nil
	}
	snapshot := c.resp
	snapshot.DroppedToolCalls = append([]DroppedToolCall(nil), c.resp.DroppedToolCalls...)
	snapshot.Usage = cloneUsage(c.resp.Usage)
	return finalizeCollectedResponse(&snapshot, c.blocks)
}

// Finish completes collection and returns what Collect would return for the
// same event sequence. Pass the terminal stream error, or nil when the
// stream ended cleanly:
//
//   - clean end after successful Adds: the complete Response and nil;
//   - a non-nil streamErr, or an earlier Add failure: the partial Response
//     (nil before MessageStart and any content) and the error. An earlier
//     Add failure takes precedence over streamErr, because Collect stops at
//     the first error.
//
// The caller owns the returned Response. Finish is idempotent: later calls
// return the same values.
func (c *Collector) Finish(streamErr error) (*Response, error) {
	if c.finished {
		return c.final, c.err
	}
	c.finished = true
	if c.err == nil {
		c.err = streamErr
	}
	if c.err != nil && !c.seenStart && len(c.blocks) == 0 {
		c.final = nil
	} else {
		c.final = finalizeCollectedResponse(&c.resp, c.blocks)
	}
	c.blocks = nil
	c.activeTools = nil
	return c.final, c.err
}

// NormalizeEvent applies the Event value-type doctrine: pointer events
// dereference to their value form, while nil (typed or untyped) and unknown
// event types return ErrBadRequest. Collect, Collector, and the built-in
// middleware normalize this way; stream decorators can use it to see the same
// value types.
func NormalizeEvent(event Event) (Event, error) {
	return normalizeEvent(event)
}

// applyCollectEvent folds one normalized event into an accumulating response.
// It is Collector's single event-application path; Collect and Session's
// stream collection both run through Collector, so they can never diverge.
func applyCollectEvent(resp *Response, blocks map[int]Part, activeTools map[int]struct{}, event Event) error {
	switch e := event.(type) {
	case MessageStart:
		resp.ID = e.ID
		resp.Provider = e.Provider
		resp.Model = e.Model
	case TextDelta:
		return appendTextDelta(blocks, e.Index, e.Text)
	case ReasoningDelta:
		return appendReasoningDelta(blocks, e)
	case ToolCallStart:
		if err := startToolCall(blocks, e); err != nil {
			return err
		}
		activeTools[e.Index] = struct{}{}
		return nil
	case ToolCallDelta:
		return appendToolCallDelta(blocks, e)
	case ToolCallIDChanged:
		return changeToolCallID(blocks, activeTools, e)
	case ToolCallEnd:
		if err := endToolCall(blocks, e.Index); err != nil {
			return err
		}
		delete(activeTools, e.Index)
		return nil
	case ToolCallDropped:
		if e.Index < 0 {
			return fmt.Errorf("%w: negative dropped tool call index %d", ErrBadRequest, e.Index)
		}
		if _, active := activeTools[e.Index]; active {
			if _, ok := blocks[e.Index].(ToolCallPart); ok {
				delete(blocks, e.Index)
				delete(activeTools, e.Index)
			}
		}
		resp.DroppedToolCalls = append(resp.DroppedToolCalls, DroppedToolCall{Index: e.Index, Reason: e.Reason})
	case MessageEnd:
		resp.StopReason = e.StopReason
		resp.StopReasonRaw = e.StopReasonRaw
		resp.Usage = e.Usage
		if e.Raw != nil {
			resp.Raw = e.Raw
		}
	default:
		return fmt.Errorf("%w: unknown stream event %T", ErrBadRequest, event)
	}
	return nil
}

// normalizeEvent applies the Event value-type doctrine at consumer entry:
// pointer events dereference to their value form, nil (typed or untyped)
// and unknown event types error.
func normalizeEvent(event Event) (Event, error) {
	event = derefEvent(event)
	switch event.(type) {
	case nil:
		return nil, fmt.Errorf("%w: nil stream event", ErrBadRequest)
	case MessageStart, TextDelta, ReasoningDelta, ToolCallStart, ToolCallDelta, ToolCallIDChanged, ToolCallEnd, ToolCallDropped, MessageEnd:
		return event, nil
	default:
		return nil, fmt.Errorf("%w: unknown stream event %T", ErrBadRequest, event)
	}
}

// derefEvent normalizes a pointer event to its value form (Event doctrine:
// events are value types; pointer events are normalized on entry). A typed
// nil pointer becomes an untyped nil Event. Unknown events pass through.
func derefEvent(event Event) Event {
	switch e := event.(type) {
	case *MessageStart:
		if e == nil {
			return nil
		}
		return *e
	case *TextDelta:
		if e == nil {
			return nil
		}
		return *e
	case *ReasoningDelta:
		if e == nil {
			return nil
		}
		return *e
	case *ToolCallStart:
		if e == nil {
			return nil
		}
		return *e
	case *ToolCallDelta:
		if e == nil {
			return nil
		}
		return *e
	case *ToolCallIDChanged:
		if e == nil {
			return nil
		}
		return *e
	case *ToolCallEnd:
		if e == nil {
			return nil
		}
		return *e
	case *ToolCallDropped:
		if e == nil {
			return nil
		}
		return *e
	case *MessageEnd:
		if e == nil {
			return nil
		}
		return *e
	default:
		return event
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

func appendReasoningDelta(blocks map[int]Part, event ReasoningDelta) error {
	if event.Index < 0 {
		return fmt.Errorf("%w: negative stream index %d", ErrBadRequest, event.Index)
	}
	if event.Text == "" && len(event.Raw) == 0 && event.Provider == "" {
		return nil
	}
	if len(event.Raw) != 0 && !json.Valid(event.Raw) {
		return fmt.Errorf("%w: invalid reasoning raw JSON", ErrBadRequest)
	}
	part, ok := blocks[event.Index]
	if !ok {
		blocks[event.Index] = ReasoningPart{
			Text:     event.Text,
			Raw:      append(json.RawMessage(nil), event.Raw...),
			Provider: event.Provider,
		}
		return nil
	}
	reasoningPart, ok := part.(ReasoningPart)
	if !ok {
		return fmt.Errorf("%w: stream index %d changed from %T to ReasoningDelta", ErrBadRequest, event.Index, part)
	}
	reasoningPart.Text += event.Text
	if len(event.Raw) != 0 {
		reasoningPart.Raw = append(json.RawMessage(nil), event.Raw...)
	}
	if event.Provider != "" {
		reasoningPart.Provider = event.Provider
	}
	blocks[event.Index] = reasoningPart
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

func changeToolCallID(blocks map[int]Part, activeTools map[int]struct{}, event ToolCallIDChanged) error {
	if event.Index < 0 {
		return fmt.Errorf("%w: negative stream index %d", ErrBadRequest, event.Index)
	}
	if event.OldID == "" || event.NewID == "" || event.OldID == event.NewID {
		return fmt.Errorf("%w: invalid tool call ID change at index %d", ErrBadRequest, event.Index)
	}
	if _, active := activeTools[event.Index]; !active {
		return fmt.Errorf("%w: tool call ID change outside active call at index %d", ErrBadRequest, event.Index)
	}
	part, ok := blocks[event.Index]
	if !ok {
		return fmt.Errorf("%w: tool call ID change before start for index %d", ErrBadRequest, event.Index)
	}
	call, ok := part.(ToolCallPart)
	if !ok {
		return fmt.Errorf("%w: stream index %d is %T, not ToolCallPart", ErrBadRequest, event.Index, part)
	}
	if call.ID != event.OldID {
		return fmt.Errorf("%w: tool call ID change at index %d expected %q, got %q", ErrBadRequest, event.Index, call.ID, event.OldID)
	}
	call.ID = event.NewID
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
	sort.Slice(resp.DroppedToolCalls, func(i, j int) bool {
		left, right := resp.DroppedToolCalls[i], resp.DroppedToolCalls[j]
		if left.Index != right.Index {
			return left.Index < right.Index
		}
		return left.Reason < right.Reason
	})
	return resp
}

// StreamTextOption configures StreamText.
type StreamTextOption func(*streamTextOptions)

type streamTextOptions struct {
	debounce time.Duration
	now      func() time.Time // clock override for tests; nil = time.Now
}

// WithDebounce buffers text deltas and emits the accumulated text at most
// once per window, rate-limiting UI re-renders without slowing the upstream
// pull. The first delta flushes immediately; pending text always flushes on
// MessageEnd, stream end, or error.
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

	if options.now == nil {
		options.now = time.Now
	}

	// Debounced path: pull events as fast as upstream provides them,
	// accumulate text deltas, and flush the buffer only when at least one
	// window has elapsed since the previous flush. The zero lastFlush makes
	// the first delta flush immediately (leading edge), so time-to-first-text
	// is unaffected. All pulls stay on this goroutine: racing a timer against
	// a blocked provider read would require a concurrent next/stop, which
	// iter.Pull2 explicitly disallows and would weaken stream cleanup on
	// early exit. Trade-off: if upstream goes quiet while text is buffered,
	// the tail flushes on the next event or at stream end/error rather than
	// on a mid-quiet-period timer.
	return func(yield func(string, error) bool) {
		next, stop := iter.Pull2(events)
		defer stop()

		var buffer strings.Builder
		var lastFlush time.Time

		flush := func() bool {
			if buffer.Len() == 0 {
				return true
			}
			text := buffer.String()
			buffer.Reset()
			lastFlush = options.now()
			return yield(text, nil)
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
				if options.now().Sub(lastFlush) >= options.debounce {
					if !flush() {
						return
					}
				}
			case MessageEnd:
				if !flush() {
					return
				}
			}
		}
	}
}
