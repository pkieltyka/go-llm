package llm

import (
	"errors"
	"iter"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestCollectAccumulatesResponse(t *testing.T) {
	usage := Usage{InputTokens: 3, OutputTokens: 5, TotalTokens: 8}
	resp, err := Collect(eventSeq(
		MessageStart{ID: "msg_1", Provider: "openai", Model: "model-a"},
		TextDelta{Index: 1, Text: "hel"},
		ReasoningDelta{Index: 0, Text: "think "},
		TextDelta{Index: 1, Text: "lo"},
		ReasoningDelta{Index: 0, Text: "more"},
		ToolCallStart{Index: 2, ID: "call_1", Name: "lookup"},
		ToolCallDelta{Index: 2, ArgsFragment: `{"q":`},
		ToolCallDelta{Index: 2, ArgsFragment: `"go"}`},
		ToolCallEnd{Index: 2},
		MessageEnd{StopReason: StopReasonToolUse, StopReasonRaw: "tool_use", Usage: usage},
	))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}

	if resp.ID != "msg_1" || resp.Provider != "openai" || resp.Model != "model-a" {
		t.Fatalf("response identity = (%q, %q, %q), want (msg_1, openai, model-a)", resp.ID, resp.Provider, resp.Model)
	}
	if got := resp.Reasoning(); got != "think more" {
		t.Fatalf("Reasoning() = %q, want %q", got, "think more")
	}
	if got := resp.Text(); got != "hello" {
		t.Fatalf("Text() = %q, want %q", got, "hello")
	}
	if resp.StopReason != StopReasonToolUse || resp.StopReasonRaw != "tool_use" {
		t.Fatalf("stop = (%q, %q), want (tool_use, tool_use)", resp.StopReason, resp.StopReasonRaw)
	}
	if resp.Usage != usage {
		t.Fatalf("usage = %+v, want %+v", resp.Usage, usage)
	}
	if _, ok := resp.Parts[0].(ReasoningPart); !ok {
		t.Fatalf("part 0 = %T, want ReasoningPart", resp.Parts[0])
	}
	if _, ok := resp.Parts[1].(TextPart); !ok {
		t.Fatalf("part 1 = %T, want TextPart", resp.Parts[1])
	}
	if _, ok := resp.Parts[2].(ToolCallPart); !ok {
		t.Fatalf("part 2 = %T, want ToolCallPart", resp.Parts[2])
	}

	calls := resp.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(calls))
	}
	if calls[0].ID != "call_1" || calls[0].Name != "lookup" || string(calls[0].Args) != `{"q":"go"}` {
		t.Fatalf("tool call = %+v", calls[0])
	}
}

func TestCollectReturnsPartialResponseOnStreamError(t *testing.T) {
	streamErr := &ProviderError{Provider: "test", Kind: ErrRateLimited, Message: "slow down"}
	resp, err := Collect(func(yield func(Event, error) bool) {
		if !yield(MessageStart{ID: "msg_1", Provider: "test", Model: "model-a"}, nil) {
			return
		}
		if !yield(TextDelta{Index: 0, Text: "partial"}, nil) {
			return
		}
		yield(nil, streamErr)
	})
	if !errors.Is(err, ErrRateLimited) {
		t.Fatalf("error = %v, want ErrRateLimited", err)
	}
	if resp == nil {
		t.Fatalf("response is nil, want partial response")
	}
	if resp.ID != "msg_1" || resp.Provider != "test" || resp.Text() != "partial" {
		t.Fatalf("partial response = %+v, text %q", resp, resp.Text())
	}
}

func TestCollectToolCallDropped(t *testing.T) {
	resp, err := Collect(eventSeq(
		MessageStart{ID: "msg_1", Provider: "test", Model: "model-a"},
		ToolCallDropped{Index: 3, Reason: "missing name"},
		MessageEnd{StopReason: StopReasonEndTurn},
	))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if len(resp.DroppedToolCalls) != 1 {
		t.Fatalf("DroppedToolCalls len = %d, want 1", len(resp.DroppedToolCalls))
	}
	if got := resp.DroppedToolCalls[0]; got.Index != 3 || got.Reason != "missing name" {
		t.Fatalf("DroppedToolCalls[0] = %+v", got)
	}
}

func TestCollectReasoningDeltaRaw(t *testing.T) {
	resp, err := Collect(eventSeq(
		MessageStart{ID: "msg_1", Provider: "anthropic", Model: "model-a"},
		ReasoningDelta{Index: 0, Text: "think "},
		ReasoningDelta{Index: 0, Text: "more"},
		ReasoningDelta{Index: 0, Raw: []byte(`{"type":"thinking","thinking":"think more","signature":"sig"}`), Provider: "anthropic"},
		MessageEnd{StopReason: StopReasonEndTurn},
	))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	reasoning, ok := resp.Parts[0].(ReasoningPart)
	if !ok {
		t.Fatalf("part 0 = %T, want ReasoningPart", resp.Parts[0])
	}
	if reasoning.Text != "think more" || reasoning.Provider != "anthropic" || string(reasoning.Raw) != `{"type":"thinking","thinking":"think more","signature":"sig"}` {
		t.Fatalf("reasoning = %+v raw %s", reasoning, reasoning.Raw)
	}
}

func TestCollectAcceptsPointerEvents(t *testing.T) {
	usage := Usage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3}
	resp, err := Collect(eventSeq(
		&MessageStart{ID: "msg_1", Provider: "openai", Model: "model-a"},
		&TextDelta{Index: 0, Text: "hi"},
		&ReasoningDelta{Index: 1, Text: "why"},
		&ToolCallStart{Index: 2, ID: "call_1", Name: "lookup"},
		&ToolCallDelta{Index: 2, ArgsFragment: `{"q":"go"}`},
		&ToolCallEnd{Index: 2},
		&ToolCallDropped{Index: 4, Reason: "truncated arguments"},
		&MessageEnd{StopReason: StopReasonToolUse, Usage: usage},
	))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if resp.Provider != "openai" || resp.Text() != "hi" || resp.Reasoning() != "why" {
		t.Fatalf("response = %+v, text %q, reasoning %q", resp, resp.Text(), resp.Reasoning())
	}
	if resp.Usage != usage {
		t.Fatalf("usage = %+v, want %+v", resp.Usage, usage)
	}
	if calls := resp.ToolCalls(); len(calls) != 1 || calls[0].ID != "call_1" {
		t.Fatalf("ToolCalls = %+v, want call_1", calls)
	}
	if len(resp.DroppedToolCalls) != 1 || resp.DroppedToolCalls[0].Index != 4 {
		t.Fatalf("DroppedToolCalls = %+v, want index 4", resp.DroppedToolCalls)
	}
}

func TestCollectNilPointerEventReturnsBadRequest(t *testing.T) {
	var event *TextDelta
	resp, err := Collect(eventSeq(event))
	if resp != nil {
		t.Fatalf("response = %#v, want nil before MessageStart or content", resp)
	}
	if !errors.Is(err, ErrBadRequest) {
		t.Fatalf("error = %v, want ErrBadRequest", err)
	}
}

func TestCollectMalformedEventAfterStartReturnsPartialResponse(t *testing.T) {
	resp, err := Collect(eventSeq(
		MessageStart{ID: "msg_1", Provider: "test", Model: "model-a"},
		TextDelta{Index: 0, Text: "partial"},
		TextDelta{Index: -1, Text: "bad"},
	))
	if !errors.Is(err, ErrBadRequest) {
		t.Fatalf("error = %v, want ErrBadRequest", err)
	}
	if resp == nil || resp.ID != "msg_1" || resp.Text() != "partial" {
		t.Fatalf("partial response = %+v, want msg_1 with text %q", resp, "partial")
	}
}

func TestCollectNilStreamReturnsBadRequest(t *testing.T) {
	resp, err := Collect(nil)
	if resp != nil {
		t.Fatalf("response = %#v, want nil", resp)
	}
	if !errors.Is(err, ErrBadRequest) {
		t.Fatalf("error = %v, want ErrBadRequest", err)
	}
}

func TestStreamTextNilStreamYieldsBadRequest(t *testing.T) {
	var count int
	for text, err := range StreamText(nil) {
		count++
		if text != "" {
			t.Fatalf("text = %q, want empty", text)
		}
		if !errors.Is(err, ErrBadRequest) {
			t.Fatalf("error = %v, want ErrBadRequest", err)
		}
	}
	if count != 1 {
		t.Fatalf("yield count = %d, want 1", count)
	}
}

func TestStreamTextAcceptsPointerTextDeltas(t *testing.T) {
	var got []string
	for text, err := range StreamText(eventSeq(&TextDelta{Index: 0, Text: "hello"})) {
		if err != nil {
			t.Fatalf("StreamText returned error: %v", err)
		}
		got = append(got, text)
	}
	if !slices.Equal(got, []string{"hello"}) {
		t.Fatalf("text chunks = %#v, want [hello]", got)
	}
}

func TestStreamTextDebounce(t *testing.T) {
	// With a window far larger than the stream duration, the first delta
	// flushes immediately (leading edge) and every later delta coalesces
	// into the single MessageEnd flush.
	var got []string
	for text, err := range StreamText(eventSeq(
		MessageStart{ID: "msg_1", Model: "model-a"},
		TextDelta{Index: 0, Text: "hel"},
		ReasoningDelta{Index: 1, Text: "ignored"},
		TextDelta{Index: 0, Text: "lo "},
		TextDelta{Index: 0, Text: "wor"},
		TextDelta{Index: 0, Text: "ld"},
		MessageEnd{StopReason: StopReasonEndTurn},
	), WithDebounce(time.Hour)) {
		if err != nil {
			t.Fatalf("StreamText error: %v", err)
		}
		got = append(got, text)
	}

	if !slices.Equal(got, []string{"hel", "lo world"}) {
		t.Fatalf("text chunks = %#v, want [hel, lo world]", got)
	}
}

func TestStreamTextDebounceFlushesOncePerWindow(t *testing.T) {
	// The fake clock advances between yields, which is safe because
	// StreamText pulls events on the same goroutine that reads the clock.
	current := time.Unix(1000, 0)
	clock := func(o *streamTextOptions) {
		o.now = func() time.Time { return current }
	}

	stream := func(yield func(Event, error) bool) {
		if !yield(TextDelta{Index: 0, Text: "a"}, nil) { // leading-edge flush
			return
		}
		current = current.Add(5 * time.Millisecond) // within window
		if !yield(TextDelta{Index: 0, Text: "b"}, nil) {
			return
		}
		current = current.Add(20 * time.Millisecond) // window elapsed
		if !yield(TextDelta{Index: 0, Text: "c"}, nil) {
			return
		}
		current = current.Add(3 * time.Millisecond) // within window
		if !yield(TextDelta{Index: 0, Text: "d"}, nil) {
			return
		}
		yield(MessageEnd{StopReason: StopReasonEndTurn}, nil) // terminal flush
	}

	var got []string
	for text, err := range StreamText(stream, WithDebounce(10*time.Millisecond), clock) {
		if err != nil {
			t.Fatalf("StreamText error: %v", err)
		}
		got = append(got, text)
	}

	if !slices.Equal(got, []string{"a", "bc", "d"}) {
		t.Fatalf("text chunks = %#v, want [a, bc, d]", got)
	}
}

func TestStreamTextDebounceFlushesBufferedTextBeforeError(t *testing.T) {
	streamErr := &ProviderError{Provider: "test", Kind: ErrOverloaded, Message: "busy"}
	stream := func(yield func(Event, error) bool) {
		if !yield(TextDelta{Index: 0, Text: "a"}, nil) {
			return
		}
		if !yield(TextDelta{Index: 0, Text: "b"}, nil) {
			return
		}
		yield(nil, streamErr)
	}

	var got []string
	var gotErr error
	for text, err := range StreamText(stream, WithDebounce(time.Hour)) {
		if err != nil {
			gotErr = err
			continue
		}
		got = append(got, text)
	}

	if !slices.Equal(got, []string{"a", "b"}) {
		t.Fatalf("text chunks = %#v, want [a, b]", got)
	}
	if !errors.Is(gotErr, ErrOverloaded) {
		t.Fatalf("error = %v, want ErrOverloaded", gotErr)
	}
}

func TestStreamTextDebounceEarlyExitDoesNotResumeUpstream(t *testing.T) {
	resumed := make(chan struct{}, 1)
	stream := func(yield func(Event, error) bool) {
		if !yield(TextDelta{Index: 0, Text: "hello"}, nil) {
			return
		}
		resumed <- struct{}{}
		select {}
	}

	for text, err := range StreamText(stream, WithDebounce(10*time.Millisecond)) {
		if err != nil {
			t.Fatalf("StreamText returned error: %v", err)
		}
		if text != "hello" {
			t.Fatalf("text = %q, want hello", text)
		}
		break
	}

	select {
	case <-resumed:
		t.Fatalf("upstream resumed after caller stopped")
	case <-time.After(25 * time.Millisecond):
	}
}

// FuzzCollectEventSequences fuzzes Collect over random event SEQUENCES: the
// input bytes are decoded into a mixed stream of MessageStart, TextDelta,
// ToolCallStart/Delta/End, and MessageEnd events, and Collect's observable
// invariants are checked against an independent oracle that applies the
// documented accumulation rules — no panic on any sequence, text
// concatenation per block index matches the deltas, and the last MessageEnd
// wins.
func FuzzCollectEventSequences(f *testing.F) {
	f.Add([]byte{0, 1, 'h', 2, 'i', 5, 0})
	f.Add([]byte{1, 'a', 1, 'b', 5, 1, 5, 2})
	f.Add([]byte{3, 4, '{', 4, '}', 2, 'x', 0})
	f.Fuzz(func(t *testing.T, data []byte) {
		events, oracle := decodeFuzzEvents(data)
		resp, err := Collect(eventSeq(events...))

		if oracle.errExpected {
			if err == nil {
				t.Fatalf("Collect returned nil error for invalid sequence %+v", events)
			}
			if resp == nil && oracle.hadContentBeforeError {
				t.Fatalf("Collect dropped the partial response accumulated before the error")
			}
			return
		}
		if err != nil {
			t.Fatalf("Collect returned error for valid sequence: %v (%+v)", err, events)
		}
		if resp == nil {
			t.Fatalf("Collect returned nil response without error")
		}
		if got, want := resp.Text(), oracle.text(); got != want {
			t.Fatalf("Text() = %q, want per-index delta concatenation %q", got, want)
		}
		if resp.StopReasonRaw != oracle.lastStopRaw {
			t.Fatalf("StopReasonRaw = %q, want last MessageEnd raw %q (single MessageEnd wins)", resp.StopReasonRaw, oracle.lastStopRaw)
		}
	})
}

// collectOracle independently applies Collect's documented accumulation
// rules so the fuzzer can predict outcomes without reusing Collect itself.
type collectOracle struct {
	blocks                map[int]string // index -> "text" or "tool"
	texts                 map[int]string
	lastStopRaw           string
	seenStart             bool
	errExpected           bool
	hadContentBeforeError bool
}

func (o *collectOracle) fail() {
	o.errExpected = true
	o.hadContentBeforeError = o.seenStart || len(o.blocks) > 0
}

func (o *collectOracle) text() string {
	indexes := make([]int, 0, len(o.texts))
	for index := range o.texts {
		indexes = append(indexes, index)
	}
	slices.Sort(indexes)
	var b strings.Builder
	for _, index := range indexes {
		b.WriteString(o.texts[index])
	}
	return b.String()
}

// decodeFuzzEvents turns fuzz bytes into an event sequence plus the oracle's
// predicted outcome. Opcodes: 0 MessageStart, 1 TextDelta(idx 0), 2
// TextDelta(idx 1), 3 ToolCallStart(idx 2), 4 ToolCallDelta(idx 2), 5
// MessageEnd(raw varies). Text/args payloads consume the following byte.
func decodeFuzzEvents(data []byte) ([]Event, *collectOracle) {
	oracle := &collectOracle{blocks: map[int]string{}, texts: map[int]string{}}
	var events []Event
	apply := func(event Event) {
		events = append(events, event)
		if oracle.errExpected {
			return
		}
		switch e := event.(type) {
		case MessageStart:
			oracle.seenStart = true
		case TextDelta:
			if e.Text == "" {
				return
			}
			if kind, ok := oracle.blocks[e.Index]; ok && kind != "text" {
				oracle.fail()
				return
			}
			oracle.blocks[e.Index] = "text"
			oracle.texts[e.Index] += e.Text
		case ToolCallStart:
			if _, ok := oracle.blocks[e.Index]; ok {
				oracle.fail()
				return
			}
			oracle.blocks[e.Index] = "tool"
		case ToolCallDelta:
			if kind, ok := oracle.blocks[e.Index]; !ok || kind != "tool" {
				oracle.fail()
				return
			}
		case MessageEnd:
			oracle.lastStopRaw = e.StopReasonRaw
		}
	}

	for i := 0; i < len(data); i++ {
		switch data[i] % 6 {
		case 0:
			apply(MessageStart{ID: "msg_fuzz", Provider: "fuzz", Model: "fuzz-model"})
		case 1, 2:
			index := int(data[i]%6) - 1 // 0 or 1
			text := ""
			if i+1 < len(data) {
				i++
				text = string(rune(data[i]))
			}
			apply(TextDelta{Index: index, Text: text})
		case 3:
			apply(ToolCallStart{Index: 2, ID: "call_fuzz", Name: "lookup"})
		case 4:
			fragment := ""
			if i+1 < len(data) {
				i++
				fragment = string(rune(data[i]))
			}
			apply(ToolCallDelta{Index: 2, ArgsFragment: fragment})
		case 5:
			raw := "end_turn"
			if i+1 < len(data) && data[i+1]%2 == 1 {
				raw = "stop_sequence"
			}
			apply(MessageEnd{StopReason: StopReasonEndTurn, StopReasonRaw: raw})
		}
	}
	return events, oracle
}

func eventSeq(events ...Event) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		for _, event := range events {
			if !yield(event, nil) {
				return
			}
		}
	}
}

// TestApplyCollectEventInstallsMessageEndRaw pins the B2 fix at the shared
// apply function: MessageEnd.Raw must be installed on the accumulating
// Response. Collect and Session's stream collection both fold events through
// applyCollectEvent, so this single assertion covers both paths.
func TestApplyCollectEventInstallsMessageEndRaw(t *testing.T) {
	resp := &Response{}
	blocks := map[int]Part{}
	raw := map[string]string{"upstream": "extras"}
	end := MessageEnd{
		StopReason: StopReasonEndTurn,
		Usage:      Usage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3},
		Raw:        raw,
	}
	if err := applyCollectEvent(resp, blocks, end); err != nil {
		t.Fatalf("applyCollectEvent returned error: %v", err)
	}
	got, ok := resp.Raw.(map[string]string)
	if !ok || got["upstream"] != "extras" {
		t.Fatalf("resp.Raw = %#v, want MessageEnd.Raw installed", resp.Raw)
	}
	if resp.Usage.TotalTokens != 3 || resp.StopReason != StopReasonEndTurn {
		t.Fatalf("resp = %+v, want MessageEnd fields applied", resp)
	}
}
