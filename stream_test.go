package llm

import (
	"errors"
	"iter"
	"slices"
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

func TestCollectAcceptsPointerEvents(t *testing.T) {
	usage := Usage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3}
	resp, err := Collect(eventSeq(
		&MessageStart{ID: "msg_1", Provider: "openai", Model: "model-a"},
		&TextDelta{Index: 0, Text: "hi"},
		&ReasoningDelta{Index: 1, Text: "why"},
		&ToolCallStart{Index: 2, ID: "call_1", Name: "lookup"},
		&ToolCallDelta{Index: 2, ArgsFragment: `{"q":"go"}`},
		&ToolCallEnd{Index: 2},
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

func FuzzCollectTextDeltas(f *testing.F) {
	f.Add("hello", " world")
	f.Fuzz(func(t *testing.T, a, b string) {
		resp, err := Collect(eventSeq(TextDelta{Index: 0, Text: a}, TextDelta{Index: 0, Text: b}))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		if got, want := resp.Text(), a+b; got != want {
			t.Fatalf("Text() = %q, want %q", got, want)
		}
	})
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
