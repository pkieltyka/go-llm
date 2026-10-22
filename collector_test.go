package llm

import (
	"encoding/json"
	"errors"
	"iter"
	"reflect"
	"testing"
)

// referenceCollect is the pre-Collector Collect implementation, retained
// verbatim as an independent oracle: Collect is now built on Collector, so
// comparing the two directly would be circular.
func referenceCollect(events iter.Seq2[Event, error]) (*Response, error) {
	resp := &Response{}
	blocks := map[int]Part{}
	activeTools := map[int]struct{}{}
	seenStart := false

	partial := func() *Response {
		if !seenStart && len(blocks) == 0 {
			return nil
		}
		return finalizeCollectedResponse(resp, blocks)
	}

	for event, err := range events {
		if err != nil {
			return partial(), err
		}
		event, err = normalizeEvent(event)
		if err != nil {
			return partial(), err
		}
		if _, ok := event.(MessageStart); ok {
			seenStart = true
		}
		if err := applyCollectEvent(resp, blocks, activeTools, event); err != nil {
			return partial(), err
		}
	}
	return finalizeCollectedResponse(resp, blocks), nil
}

var errCollectorTestStream = errors.New("stream failed")

func seqWithError(events []Event, streamErr error) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		for _, event := range events {
			if !yield(event, nil) {
				return
			}
		}
		if streamErr != nil {
			yield(nil, streamErr)
		}
	}
}

func assertSameCollect(t *testing.T, label string, gotResp *Response, gotErr error, wantResp *Response, wantErr error) {
	t.Helper()
	if (gotErr == nil) != (wantErr == nil) || (gotErr != nil && gotErr.Error() != wantErr.Error()) {
		t.Fatalf("%s: err = %v, want %v", label, gotErr, wantErr)
	}
	if !reflect.DeepEqual(gotResp, wantResp) {
		t.Fatalf("%s: response = %#v, want %#v", label, gotResp, wantResp)
	}
}

func FuzzCollectorMatchesReferenceCollect(f *testing.F) {
	f.Add([]byte{0, 1, 'h', 2, 'i', 5, 0}, false)
	f.Add([]byte{1, 'a', 1, 'b', 5, 1, 5, 2}, true)
	f.Add([]byte{3, 4, '{', 4, '}', 2, 'x', 0}, true)
	f.Add([]byte{4, 'x'}, false)
	f.Fuzz(func(t *testing.T, data []byte, failAtEnd bool) {
		events, _ := decodeFuzzEvents(data)
		var streamErr error
		if failAtEnd {
			streamErr = errCollectorTestStream
		}

		gotResp, gotErr := Collect(seqWithError(events, streamErr))
		wantResp, wantErr := referenceCollect(seqWithError(events, streamErr))
		assertSameCollect(t, "Collect", gotResp, gotErr, wantResp, wantErr)

		// Every prefix snapshot equals what the reference returns when the
		// stream fails right after that prefix — unless an Add already
		// failed, in which case Collect would have stopped there.
		var c Collector
		var snapshots []*Response
		var frozen []*Response
		addFailed := false
		for i, event := range events {
			if err := c.Add(event); err != nil {
				addFailed = true
				break
			}
			snapshot := c.Snapshot()
			want, _ := referenceCollect(seqWithError(events[:i+1], errCollectorTestStream))
			assertSameCollect(t, "Snapshot", snapshot, nil, want, nil)
			snapshots = append(snapshots, snapshot)
			frozen = append(frozen, snapshot.Clone())
		}
		finishErr := streamErr
		if addFailed {
			finishErr = nil // the sticky Add error must win
		}
		finalResp, finalErr := c.Finish(finishErr)
		assertSameCollect(t, "Finish", finalResp, finalErr, wantResp, wantErr)

		// Later Adds never mutated earlier snapshots.
		for i := range snapshots {
			if !reflect.DeepEqual(snapshots[i], frozen[i]) {
				t.Fatalf("snapshot %d mutated by later Add: %#v, want %#v", i, snapshots[i], frozen[i])
			}
		}
	})
}

func TestCollectorZeroValueCleanEmptyStream(t *testing.T) {
	var c Collector
	if got := c.Snapshot(); got != nil {
		t.Fatalf("Snapshot before any event = %#v, want nil", got)
	}
	resp, err := c.Finish(nil)
	want, wantErr := referenceCollect(seqWithError(nil, nil))
	assertSameCollect(t, "empty Finish", resp, err, want, wantErr)
	if resp == nil {
		t.Fatal("clean empty stream returned nil response, want empty Response as Collect does")
	}
}

func TestCollectorStickyAddErrorWinsOverStreamError(t *testing.T) {
	var c Collector
	if err := c.Add(MessageStart{ID: "m"}); err != nil {
		t.Fatal(err)
	}
	addErr := c.Add(ToolCallDelta{Index: 3, ArgsFragment: "{"})
	if !errors.Is(addErr, ErrBadRequest) {
		t.Fatalf("Add(malformed) = %v, want ErrBadRequest", addErr)
	}
	if again := c.Add(TextDelta{Index: 0, Text: "ignored"}); again != addErr {
		t.Fatalf("Add after failure = %v, want the sticky error %v", again, addErr)
	}
	resp, err := c.Finish(errCollectorTestStream)
	if err != addErr {
		t.Fatalf("Finish err = %v, want sticky Add error %v", err, addErr)
	}
	if resp == nil || resp.ID != "m" || len(resp.Parts) != 0 {
		t.Fatalf("Finish resp = %#v, want partial with ID m and no parts", resp)
	}
}

func TestCollectorFinishIsIdempotentAndTerminal(t *testing.T) {
	var c Collector
	_ = c.Add(MessageStart{ID: "m"})
	_ = c.Add(TextDelta{Index: 0, Text: "hi"})
	first, err := c.Finish(nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err2 := c.Finish(errCollectorTestStream)
	if second != first || err2 != nil {
		t.Fatalf("second Finish = (%p, %v), want (%p, nil)", second, err2, first)
	}
	if err := c.Add(TextDelta{Index: 0, Text: "late"}); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("Add after Finish = %v, want ErrBadRequest", err)
	}
	snapshot := c.Snapshot()
	if snapshot == first || !reflect.DeepEqual(snapshot, first) {
		t.Fatalf("Snapshot after Finish must be an equal copy, got %p vs %p", snapshot, first)
	}
}

func TestCollectorSnapshotIsolationBothDirections(t *testing.T) {
	cost := 0.5
	var c Collector
	_ = c.Add(MessageStart{ID: "m"})
	_ = c.Add(ReasoningDelta{Index: 0, Text: "think", Raw: json.RawMessage(`{"sig":"a"}`), Provider: "p"})
	_ = c.Add(ToolCallStart{Index: 1, ID: "call", Name: "lookup"})
	_ = c.Add(ToolCallDelta{Index: 1, ArgsFragment: `{"q":`})
	_ = c.Add(ToolCallDropped{Index: 7, Reason: "bad"})
	_ = c.Add(MessageEnd{StopReason: StopReasonToolUse, Usage: Usage{CostUSD: &cost}})

	snapshot := c.Snapshot()
	reasoning := snapshot.Parts[0].(ReasoningPart)
	reasoning.Raw[2] = 'X'
	call := snapshot.Parts[1].(ToolCallPart)
	call.Args[0] = 'X'
	snapshot.DroppedToolCalls[0].Reason = "mutated"
	*snapshot.Usage.CostUSD = 99

	_ = c.Add(ToolCallDelta{Index: 1, ArgsFragment: `"x"}`})
	resp, err := c.Finish(nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := string(resp.Parts[0].(ReasoningPart).Raw); got != `{"sig":"a"}` {
		t.Fatalf("collector reasoning raw = %s, mutated through snapshot", got)
	}
	if got := string(resp.Parts[1].(ToolCallPart).Args); got != `{"q":"x"}` {
		t.Fatalf("collector args = %s, want {\"q\":\"x\"}", got)
	}
	if resp.DroppedToolCalls[0].Reason != "bad" || *resp.Usage.CostUSD != 0.5 {
		t.Fatalf("collector state mutated through snapshot: %#v", resp)
	}
	if got := string(snapshot.Parts[1].(ToolCallPart).Args); got != `X"q":` {
		t.Fatalf("snapshot args = %s, changed by a later Add", got)
	}
}

func TestResponseCloneDeepCopies(t *testing.T) {
	cost := 1.25
	raw := map[string]any{"sdk": true}
	original := &Response{
		ID:               "r",
		Parts:            []Part{Text("a"), &ToolCallPart{ID: "c", Name: "n", Args: json.RawMessage(`{}`)}},
		DroppedToolCalls: []DroppedToolCall{{Index: 1, Reason: "x"}},
		Usage:            Usage{CostUSD: &cost, Raw: raw},
		Raw:              raw,
	}
	cloned := original.Clone()
	if _, ok := cloned.Parts[1].(ToolCallPart); !ok {
		t.Fatalf("Clone kept pointer part %T, want normalized value part", cloned.Parts[1])
	}
	cloned.Parts[1].(ToolCallPart).Args[0] = 'X'
	cloned.DroppedToolCalls[0].Reason = "y"
	*cloned.Usage.CostUSD = 2
	if string(original.Parts[1].(*ToolCallPart).Args) != `{}` || original.DroppedToolCalls[0].Reason != "x" || cost != 1.25 {
		t.Fatalf("Clone shares mutable state with the original: %#v", original)
	}
	if !reflect.DeepEqual(cloned.Raw, raw) || !reflect.DeepEqual(cloned.Usage.Raw, raw) {
		t.Fatal("Clone must keep SDK Raw payloads by reference")
	}
	if (*Response)(nil).Clone() != nil {
		t.Fatal("nil Clone must return nil")
	}
	if empty := (&Response{Parts: []Part{}, DroppedToolCalls: []DroppedToolCall{}}).Clone(); empty.Parts == nil || empty.DroppedToolCalls == nil {
		t.Fatal("Clone turned an empty non-nil slice into nil")
	}
	if bare := (&Response{}).Clone(); bare.Parts != nil || bare.DroppedToolCalls != nil {
		t.Fatal("Clone turned a nil slice into an empty one")
	}
}
