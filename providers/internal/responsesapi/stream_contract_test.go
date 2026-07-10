package responsesapi_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"iter"
	"reflect"
	"strings"
	"testing"

	sdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
	"github.com/pkieltyka/go-llm/providers/internal/responsesapi"
)

func TestStreamStateEmitsLongPreCreatedSequenceImmediately(t *testing.T) {
	state := replayAdapter().NewStreamState("requested-model")
	first, err := state.MapEvent(streamEvent(t, `{
		"type":"response.output_text.delta",
		"output_index":7,
		"content_index":0,
		"delta":"hello"
	}`))
	if err != nil {
		t.Fatalf("MapEvent(delta) returned error: %v", err)
	}
	if len(first) != 2 {
		t.Fatalf("first pre-created events = %#v, want immediate start and delta", first)
	}
	start, ok := providerutil.DerefEvent(first[0]).(llm.MessageStart)
	if !ok {
		t.Fatalf("first event = %T, want MessageStart", first[0])
	}
	if start.ID != "" || start.Model != "requested-model" {
		t.Fatalf("MessageStart = %+v, want request-model fallback without invented ID", start)
	}
	delta, ok := providerutil.DerefEvent(first[1]).(llm.TextDelta)
	if !ok || delta.Index != 7 || delta.Text != "hello" {
		t.Fatalf("pre-created event = %#v, want stable provider TextDelta index 7", first[1])
	}
	for i := 0; i < 1024; i++ {
		events, err := state.MapEvent(streamEvent(t, `{"type":"response.output_text.delta","output_index":7,"delta":"x"}`))
		if err != nil || len(events) != 1 {
			t.Fatalf("pre-created delta %d = %#v, %v; want one immediate event", i, events, err)
		}
	}
	created, err := state.MapEvent(streamEvent(t, `{"type":"response.created","response":{"id":"resp_late","model":"response-model","status":"in_progress","output":[]}}`))
	if err != nil || len(created) != 0 {
		t.Fatalf("late response.created = %#v, %v; want idempotent no-op", created, err)
	}
	terminal, err := state.MapEvent(streamEvent(t, `{"type":"response.completed","response":{"id":"resp_late","model":"response-model","status":"completed","output":[]}}`))
	if err != nil || len(terminal) != 1 {
		t.Fatalf("terminal events = %#v, %v; want MessageEnd only", terminal, err)
	}
	if _, ok := providerutil.DerefEvent(terminal[0]).(llm.MessageEnd); !ok {
		t.Fatalf("terminal event = %T, want MessageEnd", terminal[0])
	}
}

func TestToolCallStreamsIncrementallyAndCollectsLikeBlocking(t *testing.T) {
	const finalRaw = `{
		"id":"resp_1",
		"model":"response-model",
		"status":"completed",
		"output":[
			{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}
		],
		"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}
	}`
	adapter := replayAdapter()
	state := adapter.NewStreamState("requested-model")

	created, err := state.MapEvent(streamEvent(t, `{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`))
	if err != nil {
		t.Fatalf("MapEvent(created) returned error: %v", err)
	}
	added, err := state.MapEvent(streamEvent(t, `{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"","status":"in_progress"}}`))
	if err != nil {
		t.Fatalf("MapEvent(added) returned error: %v", err)
	}
	if len(added) != 1 {
		t.Fatalf("added events = %#v, want immediate ToolCallStart", added)
	}
	start, ok := providerutil.DerefEvent(added[0]).(llm.ToolCallStart)
	if !ok || start.Index != 0 || start.ID != "call_1" || start.Name != "lookup" {
		t.Fatalf("added event = %#v, want ToolCallStart index 0", added[0])
	}

	firstDelta, err := state.MapEvent(streamEvent(t, `{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":"}`))
	if err != nil {
		t.Fatalf("MapEvent(first delta) returned error: %v", err)
	}
	secondDelta, err := state.MapEvent(streamEvent(t, `{"type":"response.function_call_arguments.delta","output_index":0,"delta":"\"go\"}"}`))
	if err != nil {
		t.Fatalf("MapEvent(second delta) returned error: %v", err)
	}
	for i, events := range [][]llm.Event{firstDelta, secondDelta} {
		if len(events) != 1 {
			t.Fatalf("delta %d events = %#v, want one observable ToolCallDelta", i, events)
		}
		if delta, ok := providerutil.DerefEvent(events[0]).(llm.ToolCallDelta); !ok || delta.Index != 0 || delta.ArgsFragment == "" {
			t.Fatalf("delta %d event = %#v, want ToolCallDelta index 0", i, events[0])
		}
	}

	done, err := state.MapEvent(streamEvent(t, `{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}}`))
	if err != nil {
		t.Fatalf("MapEvent(done) returned error: %v", err)
	}
	if len(done) != 1 {
		t.Fatalf("done events = %#v, want immediate ToolCallEnd", done)
	}
	if end, ok := providerutil.DerefEvent(done[0]).(llm.ToolCallEnd); !ok || end.Index != 0 {
		t.Fatalf("done event = %#v, want ToolCallEnd index 0", done[0])
	}

	terminal, err := state.MapEvent(streamEvent(t, `{"type":"response.completed","response":`+finalRaw+`}`))
	if err != nil {
		t.Fatalf("MapEvent(completed) returned error: %v", err)
	}
	if len(terminal) != 1 {
		t.Fatalf("terminal events = %#v, want idempotent MessageEnd only", terminal)
	}
	if _, ok := providerutil.DerefEvent(terminal[0]).(llm.MessageEnd); !ok {
		t.Fatalf("terminal event = %#v, want MessageEnd", terminal[0])
	}
	mapped := append(created, added...)
	mapped = append(mapped, firstDelta...)
	mapped = append(mapped, secondDelta...)
	mapped = append(mapped, done...)
	mapped = append(mapped, terminal...)
	streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	var final responses.Response
	if err := json.Unmarshal([]byte(finalRaw), &final); err != nil {
		t.Fatalf("unmarshal final response: %v", err)
	}
	blocking, err := adapter.MapResponse(&final)
	if err != nil {
		t.Fatalf("MapResponse returned error: %v", err)
	}
	if !reflect.DeepEqual(streamed.Parts, blocking.Parts) || len(streamed.DroppedToolCalls) != len(blocking.DroppedToolCalls) {
		t.Fatalf("stream/blocking mismatch:\nstream:   %#v %#v\nblocking: %#v %#v", streamed.Parts, streamed.DroppedToolCalls, blocking.Parts, blocking.DroppedToolCalls)
	}
}

func TestAuthoritativeContentDoneEvents(t *testing.T) {
	types := []struct {
		name      string
		deltaType string
		doneType  string
		extra     string
		reasoning bool
	}{
		{name: "output text", deltaType: "response.output_text.delta", doneType: "response.output_text.done", extra: `,"content_index":0`},
		{name: "reasoning summary", deltaType: "response.reasoning_summary_text.delta", doneType: "response.reasoning_summary_text.done", extra: `,"summary_index":0`, reasoning: true},
	}

	for _, tt := range types {
		t.Run(tt.name+" suffix and contradiction", func(t *testing.T) {
			state := replayAdapter().NewStreamState("requested-model")
			if _, err := state.MapEvent(streamEvent(t, `{"type":"`+tt.deltaType+`","output_index":0`+tt.extra+`,"delta":"hel"}`)); err != nil {
				t.Fatalf("MapEvent(delta) returned error: %v", err)
			}
			events, err := state.MapEvent(streamEvent(t, `{"type":"`+tt.doneType+`","output_index":0`+tt.extra+`,"text":"hello"}`))
			if err != nil || len(events) != 1 {
				t.Fatalf("MapEvent(done) = %#v, %v; want one suffix event", events, err)
			}
			if tt.reasoning {
				delta, ok := providerutil.DerefEvent(events[0]).(llm.ReasoningDelta)
				if !ok || delta.Index != 0 || delta.Text != "lo" {
					t.Fatalf("done event = %#v, want reasoning suffix", events[0])
				}
			} else {
				delta, ok := providerutil.DerefEvent(events[0]).(llm.TextDelta)
				if !ok || delta.Index != 0 || delta.Text != "lo" {
					t.Fatalf("done event = %#v, want text suffix", events[0])
				}
			}

			contradictory := replayAdapter().NewStreamState("requested-model")
			if _, err := contradictory.MapEvent(streamEvent(t, `{"type":"`+tt.deltaType+`","output_index":0`+tt.extra+`,"delta":"hello"}`)); err != nil {
				t.Fatalf("MapEvent(delta) returned error: %v", err)
			}
			_, err = contradictory.MapEvent(streamEvent(t, `{"type":"`+tt.doneType+`","output_index":0`+tt.extra+`,"text":"different"}`))
			if !errors.Is(err, llm.ErrServer) {
				t.Fatalf("contradiction error = %v, want ErrServer", err)
			}
		})

		for _, errorClass := range []string{"semantic", "transport"} {
			t.Run(tt.name+" before "+errorClass+" error", func(t *testing.T) {
				state := replayAdapter().NewStreamState("requested-model")
				var mapped []llm.Event
				for _, raw := range []string{
					`{"type":"` + tt.deltaType + `","output_index":0` + tt.extra + `,"delta":"hel"}`,
					`{"type":"` + tt.doneType + `","output_index":0` + tt.extra + `,"text":"hello"}`,
				} {
					events, err := state.MapEvent(streamEvent(t, raw))
					if err != nil {
						t.Fatalf("MapEvent returned error: %v", err)
					}
					mapped = append(mapped, events...)
				}

				var streamErr error
				if errorClass == "semantic" {
					events, err := state.MapEvent(streamEvent(t, `{"type":"error","code":"server_error","message":"failed"}`))
					mapped = append(mapped, events...)
					streamErr = err
				} else {
					mapped = append(mapped, state.Finish()...)
					streamErr = providerutil.NormalizeRemoteError(replayAdapterName, errors.New("connection reset"))
				}
				partial, err := llm.Collect(eventsThenErrorSeq(mapped, streamErr))
				if !errors.Is(err, llm.ErrServer) || partial == nil {
					t.Fatalf("Collect = %#v, %v; want partial response and ErrServer", partial, err)
				}
				got := partial.Text()
				if tt.reasoning {
					got = partial.Reasoning()
				}
				if got != "hello" {
					t.Fatalf("partial content = %q, want authoritative done content", got)
				}
			})
		}
	}
}

func TestRefusalDoneIsAuthoritative(t *testing.T) {
	t.Run("suffix and contradiction", func(t *testing.T) {
		state := replayAdapter().NewStreamState("requested-model")
		if _, err := state.MapEvent(streamEvent(t, `{"type":"response.refusal.delta","output_index":0,"content_index":0,"delta":"refu"}`)); err != nil {
			t.Fatalf("MapEvent(delta) returned error: %v", err)
		}
		events, err := state.MapEvent(streamEvent(t, `{"type":"response.refusal.done","output_index":0,"content_index":0,"refusal":"refused"}`))
		if err != nil || len(events) != 1 {
			t.Fatalf("MapEvent(done) = %#v, %v; want one suffix event", events, err)
		}
		if delta, ok := providerutil.DerefEvent(events[0]).(llm.TextDelta); !ok || delta.Index != 0 || delta.Text != "sed" {
			t.Fatalf("done event = %#v, want refusal suffix", events[0])
		}

		contradictory := replayAdapter().NewStreamState("requested-model")
		if _, err := contradictory.MapEvent(streamEvent(t, `{"type":"response.refusal.delta","output_index":0,"content_index":0,"delta":"refused"}`)); err != nil {
			t.Fatalf("MapEvent(delta) returned error: %v", err)
		}
		_, err = contradictory.MapEvent(streamEvent(t, `{"type":"response.refusal.done","output_index":0,"content_index":0,"refusal":"different"}`))
		if !errors.Is(err, llm.ErrServer) {
			t.Fatalf("contradiction error = %v, want ErrServer", err)
		}
	})

	for _, errorClass := range []string{"semantic", "transport"} {
		t.Run("before "+errorClass+" error", func(t *testing.T) {
			state := replayAdapter().NewStreamState("requested-model")
			var mapped []llm.Event
			for _, raw := range []string{
				`{"type":"response.refusal.delta","output_index":0,"content_index":0,"delta":"refu"}`,
				`{"type":"response.refusal.done","output_index":0,"content_index":0,"refusal":"refused"}`,
			} {
				events, err := state.MapEvent(streamEvent(t, raw))
				if err != nil {
					t.Fatalf("MapEvent returned error: %v", err)
				}
				mapped = append(mapped, events...)
			}
			var streamErr error
			if errorClass == "semantic" {
				events, err := state.MapEvent(streamEvent(t, `{"type":"error","code":"server_error","message":"failed"}`))
				mapped = append(mapped, events...)
				streamErr = err
			} else {
				mapped = append(mapped, state.Finish()...)
				streamErr = providerutil.NormalizeRemoteError(replayAdapterName, errors.New("connection reset"))
			}
			partial, err := llm.Collect(eventsThenErrorSeq(mapped, streamErr))
			if !errors.Is(err, llm.ErrServer) || partial == nil || partial.Text() != "refused" {
				t.Fatalf("Collect = %#v, %v; want full refusal partial", partial, err)
			}
		})
	}
}

func TestFunctionCallArgumentsDoneIsAuthoritative(t *testing.T) {
	build := func(t *testing.T, fragment string) (*responsesapi.StreamState, []llm.Event) {
		t.Helper()
		state := replayAdapter().NewStreamState("requested-model")
		var mapped []llm.Event
		for _, raw := range []string{
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"","status":"in_progress"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"` + strings.ReplaceAll(fragment, `"`, `\"`) + `"}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		return state, mapped
	}

	t.Run("repairs suffix without ending tool", func(t *testing.T) {
		state, _ := build(t, `{"q":`)
		events, err := state.MapEvent(streamEvent(t, `{"type":"response.function_call_arguments.done","output_index":0,"name":"lookup","arguments":"{\"q\":\"go\"}"}`))
		if err != nil || len(events) != 1 {
			t.Fatalf("MapEvent(done) = %#v, %v; want one repaired delta", events, err)
		}
		if delta, ok := providerutil.DerefEvent(events[0]).(llm.ToolCallDelta); !ok || delta.ArgsFragment != `"go"}` {
			t.Fatalf("done event = %#v, want repaired argument suffix", events[0])
		}
	})

	t.Run("rejects contradiction", func(t *testing.T) {
		state, _ := build(t, `{"x":`)
		_, err := state.MapEvent(streamEvent(t, `{"type":"response.function_call_arguments.done","output_index":0,"name":"lookup","arguments":"{\"q\":\"go\"}"}`))
		if !errors.Is(err, llm.ErrServer) {
			t.Fatalf("contradiction error = %v, want ErrServer", err)
		}
	})

	t.Run("normalizes empty arguments", func(t *testing.T) {
		state := replayAdapter().NewStreamState("requested-model")
		if _, err := state.MapEvent(streamEvent(t, `{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"","status":"in_progress"}}`)); err != nil {
			t.Fatalf("MapEvent(added) returned error: %v", err)
		}
		events, err := state.MapEvent(streamEvent(t, `{"type":"response.function_call_arguments.done","output_index":0,"name":"lookup","arguments":""}`))
		if err != nil || len(events) != 1 {
			t.Fatalf("MapEvent(done) = %#v, %v; want canonical empty delta", events, err)
		}
		if delta, ok := providerutil.DerefEvent(events[0]).(llm.ToolCallDelta); !ok || delta.ArgsFragment != `{}` {
			t.Fatalf("done event = %#v, want canonical empty object", events[0])
		}
	})

	for _, errorClass := range []string{"semantic", "transport"} {
		t.Run("repaired call before "+errorClass+" error", func(t *testing.T) {
			state, mapped := build(t, `{"q":`)
			events, err := state.MapEvent(streamEvent(t, `{"type":"response.function_call_arguments.done","output_index":0,"name":"lookup","arguments":"{\"q\":\"go\"}"}`))
			if err != nil {
				t.Fatalf("MapEvent(done) returned error: %v", err)
			}
			mapped = append(mapped, events...)
			var streamErr error
			if errorClass == "semantic" {
				events, err := state.MapEvent(streamEvent(t, `{"type":"error","code":"server_error","message":"failed"}`))
				mapped = append(mapped, events...)
				streamErr = err
			} else {
				mapped = append(mapped, state.Finish()...)
				streamErr = providerutil.NormalizeRemoteError(replayAdapterName, errors.New("connection reset"))
			}
			partial, err := llm.Collect(eventsThenErrorSeq(mapped, streamErr))
			if !errors.Is(err, llm.ErrServer) || partial == nil {
				t.Fatalf("Collect = %#v, %v; want repaired partial call", partial, err)
			}
			calls := partial.ToolCalls()
			if len(calls) != 1 || calls[0].ID != "call_0" || string(calls[0].Args) != `{"q":"go"}` {
				t.Fatalf("partial calls = %#v, want repaired arguments", calls)
			}
		})
	}
}

func TestOutputItemDoneToolSurvivesFollowingError(t *testing.T) {
	for _, errorClass := range []string{"semantic", "transport"} {
		t.Run(errorClass, func(t *testing.T) {
			state := replayAdapter().NewStreamState("requested-model")
			var mapped []llm.Event
			for _, raw := range []string{
				`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"","status":"in_progress"}}`,
				`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":\"go\"}"}`,
			} {
				events, err := state.MapEvent(streamEvent(t, raw))
				if err != nil {
					t.Fatalf("MapEvent returned error: %v", err)
				}
				mapped = append(mapped, events...)
			}
			done, err := state.MapEvent(streamEvent(t, `{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}}`))
			if err != nil || len(done) != 1 {
				t.Fatalf("MapEvent(done) = %#v, %v; want immediate ToolCallEnd", done, err)
			}
			if end, ok := providerutil.DerefEvent(done[0]).(llm.ToolCallEnd); !ok || end.Index != 0 {
				t.Fatalf("done event = %#v, want ToolCallEnd", done[0])
			}
			mapped = append(mapped, done...)

			var streamErr error
			if errorClass == "semantic" {
				events, err := state.MapEvent(streamEvent(t, `{"type":"error","code":"server_error","message":"failed"}`))
				mapped = append(mapped, events...)
				streamErr = err
			} else {
				mapped = append(mapped, state.Finish()...)
				streamErr = providerutil.NormalizeRemoteError(replayAdapterName, errors.New("connection reset"))
			}
			partial, err := llm.Collect(eventsThenErrorSeq(mapped, streamErr))
			if !errors.Is(err, llm.ErrServer) || partial == nil {
				t.Fatalf("Collect = %#v, %v; want ended partial tool and ErrServer", partial, err)
			}
			calls := partial.ToolCalls()
			if len(calls) != 1 || calls[0].ID != "call_0" || calls[0].Name != "lookup" || string(calls[0].Args) != `{"q":"go"}` {
				t.Fatalf("partial calls = %#v, want ended tool retained", calls)
			}
		})
	}
}

func TestOutputItemDoneRepairsToolBeforeImmediateEnd(t *testing.T) {
	state := replayAdapter().NewStreamState("requested-model")
	for _, raw := range []string{
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"","status":"in_progress"}}`,
		`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":"}`,
	} {
		if _, err := state.MapEvent(streamEvent(t, raw)); err != nil {
			t.Fatalf("MapEvent returned error: %v", err)
		}
	}
	events, err := state.MapEvent(streamEvent(t, `{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}}`))
	if err != nil || len(events) != 2 {
		t.Fatalf("MapEvent(done) = %#v, %v; want repaired delta then end", events, err)
	}
	delta, ok := providerutil.DerefEvent(events[0]).(llm.ToolCallDelta)
	if !ok || delta.Index != 0 || delta.ArgsFragment != `"go"}` {
		t.Fatalf("first event = %#v, want authoritative argument suffix", events[0])
	}
	if end, ok := providerutil.DerefEvent(events[1]).(llm.ToolCallEnd); !ok || end.Index != 0 {
		t.Fatalf("second event = %#v, want immediate ToolCallEnd", events[1])
	}
}

func TestDuplicateExplicitToolIDLowestOutputIndexWinsReversedArrival(t *testing.T) {
	const finalRaw = `{
		"id":"resp_1","model":"response-model","status":"completed",
		"output":[
			{"id":"fc_0","type":"function_call","call_id":"duplicate","name":"first","arguments":"{\"n\":0}","status":"completed"},
			{"id":"fc_1","type":"function_call","call_id":"duplicate","name":"second","arguments":"{\"n\":1}","status":"completed"}
		]
	}`
	adapter := replayAdapter()

	t.Run("success", func(t *testing.T) {
		state := adapter.NewStreamState("requested-model")
		var mapped []llm.Event
		for _, raw := range []string{
			`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
			`{"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"duplicate","name":"second","arguments":"","status":"in_progress"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"n\":1}"}`,
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"duplicate","name":"first","arguments":"","status":"in_progress"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"n\":0}"}`,
			`{"type":"response.output_item.done","output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"duplicate","name":"second","arguments":"{\"n\":1}","status":"completed"}}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"duplicate","name":"first","arguments":"{\"n\":0}","status":"completed"}}`,
			`{"type":"response.completed","response":` + finalRaw + `}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		var changes []llm.ToolCallIDChanged
		for _, event := range mapped {
			switch event := providerutil.DerefEvent(event).(type) {
			case llm.ToolCallIDChanged:
				changes = append(changes, event)
			case llm.ToolCallDropped:
				t.Fatalf("identity reconciliation emitted a false drop: %#v", event)
			}
		}
		if len(changes) != 1 || changes[0].Index != 1 || changes[0].OldID != "duplicate" || changes[0].NewID == "" {
			t.Fatalf("identity changes = %#v, want one in-place correction for index 1", changes)
		}
		streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		var final responses.Response
		if err := json.Unmarshal([]byte(finalRaw), &final); err != nil {
			t.Fatalf("unmarshal final response: %v", err)
		}
		blocking, err := adapter.MapResponse(&final)
		if err != nil {
			t.Fatalf("MapResponse returned error: %v", err)
		}
		if !reflect.DeepEqual(streamed.ToolCalls(), blocking.ToolCalls()) {
			t.Fatalf("stream/blocking calls = %#v / %#v", streamed.ToolCalls(), blocking.ToolCalls())
		}
	})

	t.Run("partial error", func(t *testing.T) {
		state := adapter.NewStreamState("requested-model")
		var mapped []llm.Event
		for _, raw := range []string{
			`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
			`{"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"duplicate","name":"second","arguments":"","status":"in_progress"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"n\":1}"}`,
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"duplicate","name":"first","arguments":"","status":"in_progress"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"n\":0}"}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		errorEvents, streamErr := state.MapEvent(streamEvent(t, `{"type":"error","code":"server_error","message":"failed"}`))
		mapped = append(mapped, errorEvents...)
		partial, err := llm.Collect(eventsThenErrorSeq(mapped, streamErr))
		if err == nil || partial == nil {
			t.Fatalf("Collect = %#v, %v; want partial response and error", partial, err)
		}
		calls := partial.ToolCalls()
		if len(calls) != 2 || calls[0].ID != "duplicate" || calls[1].ID == "duplicate" {
			t.Fatalf("partial duplicate resolution = %#v", calls)
		}
	})
}

func TestUniqueExplicitToolAtNonzeroIndexStreamsImmediately(t *testing.T) {
	state := replayAdapter().NewStreamState("requested-model")
	if _, err := state.MapEvent(streamEvent(t, `{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`)); err != nil {
		t.Fatalf("MapEvent(created) returned error: %v", err)
	}
	events, err := state.MapEvent(streamEvent(t, `{"type":"response.output_item.added","output_index":2,"item":{"id":"fc_2","type":"function_call","call_id":"unique","name":"lookup","arguments":"","status":"in_progress"}}`))
	if err != nil {
		t.Fatalf("MapEvent(added) returned error: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("events = %#v, want immediate ToolCallStart", events)
	}
	start, ok := providerutil.DerefEvent(events[0]).(llm.ToolCallStart)
	if !ok || start.Index != 2 || start.ID != "unique" {
		t.Fatalf("event = %#v, want unique explicit ID at index 2", events[0])
	}
}

func TestMissingNameToolNeverOwnsExplicitID(t *testing.T) {
	const finalRaw = `{
		"id":"resp_1","model":"response-model","status":"completed",
		"output":[
			{"id":"fc_bad","type":"function_call","call_id":"shared","arguments":"{}","status":"completed"},
			{"id":"fc_good","type":"function_call","call_id":"shared","name":"lookup","arguments":"{}","status":"completed"}
		]
	}`
	adapter := replayAdapter()
	build := func(t *testing.T) (*responsesapi.StreamState, []llm.Event) {
		t.Helper()
		state := adapter.NewStreamState("requested-model")
		var mapped []llm.Event
		for i, raw := range []string{
			`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_bad","type":"function_call","call_id":"shared","arguments":"","status":"in_progress"}}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_bad","type":"function_call","call_id":"shared","arguments":"{}","status":"completed"}}`,
			`{"type":"response.output_item.added","output_index":1,"item":{"id":"fc_good","type":"function_call","call_id":"shared","name":"lookup","arguments":"","status":"in_progress"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{}"}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			if i == 1 && len(events) != 0 {
				t.Fatalf("missing-name item emitted events = %#v, want no ID reservation/start", events)
			}
			mapped = append(mapped, events...)
		}
		return state, mapped
	}

	t.Run("success", func(t *testing.T) {
		state, mapped := build(t)
		for _, raw := range []string{
			`{"type":"response.output_item.done","output_index":1,"item":{"id":"fc_good","type":"function_call","call_id":"shared","name":"lookup","arguments":"{}","status":"completed"}}`,
			`{"type":"response.completed","response":` + finalRaw + `}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		var final responses.Response
		if err := json.Unmarshal([]byte(finalRaw), &final); err != nil {
			t.Fatalf("unmarshal final response: %v", err)
		}
		blocking, err := adapter.MapResponse(&final)
		if err != nil {
			t.Fatalf("MapResponse returned error: %v", err)
		}
		if !reflect.DeepEqual(streamed.ToolCalls(), blocking.ToolCalls()) {
			t.Fatalf("stream/blocking calls = %#v / %#v", streamed.ToolCalls(), blocking.ToolCalls())
		}
		if calls := streamed.ToolCalls(); len(calls) != 1 || calls[0].ID != "shared" {
			t.Fatalf("calls = %#v, want valid call to retain explicit ID", calls)
		}
	})

	t.Run("immediate error", func(t *testing.T) {
		state, mapped := build(t)
		events, streamErr := state.MapEvent(streamEvent(t, `{"type":"error","code":"server_error","message":"failed"}`))
		mapped = append(mapped, events...)
		partial, err := llm.Collect(eventsThenErrorSeq(mapped, streamErr))
		if err == nil || partial == nil {
			t.Fatalf("Collect = %#v, %v; want partial response and error", partial, err)
		}
		if calls := partial.ToolCalls(); len(calls) != 1 || calls[0].ID != "shared" {
			t.Fatalf("partial calls = %#v, want valid call to retain explicit ID", calls)
		}
	})
}

func TestTerminalToolArgumentsRepairAndContradiction(t *testing.T) {
	const finalRaw = `{
		"id":"resp_1","model":"response-model","status":"completed",
		"output":[{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}]
	}`
	build := func(t *testing.T, fragment string) (*responsesapi.StreamState, []llm.Event) {
		t.Helper()
		state := replayAdapter().NewStreamState("requested-model")
		var mapped []llm.Event
		for _, raw := range []string{
			`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"","status":"in_progress"}}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		deltaRaw, _ := json.Marshal(map[string]any{
			"type":         "response.function_call_arguments.delta",
			"output_index": 0,
			"delta":        fragment,
		})
		events, err := state.MapEvent(streamEvent(t, string(deltaRaw)))
		if err != nil {
			t.Fatalf("MapEvent(delta) returned error: %v", err)
		}
		mapped = append(mapped, events...)
		return state, mapped
	}

	t.Run("repairs suffix", func(t *testing.T) {
		state, mapped := build(t, `{"q":`)
		events, err := state.MapEvent(streamEvent(t, `{"type":"response.completed","response":`+finalRaw+`}`))
		if err != nil {
			t.Fatalf("MapEvent(completed) returned error: %v", err)
		}
		mapped = append(mapped, events...)
		streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		if calls := streamed.ToolCalls(); len(calls) != 1 || string(calls[0].Args) != `{"q":"go"}` {
			t.Fatalf("repaired calls = %#v", calls)
		}
	})

	t.Run("rejects contradiction", func(t *testing.T) {
		state, _ := build(t, `{"x":`)
		_, err := state.MapEvent(streamEvent(t, `{"type":"response.completed","response":`+finalRaw+`}`))
		if !errors.Is(err, llm.ErrServer) {
			t.Fatalf("error = %v, want ErrServer", err)
		}
	})
}

func TestAuthoritativeToolArgumentsPreserveExactStreamedBytes(t *testing.T) {
	adapter := replayAdapter()

	t.Run("outer and whitespace-only deltas", func(t *testing.T) {
		const args = " \n{\"q\":\"go\"}\t "
		item := map[string]any{
			"id": "fc_0", "type": "function_call", "call_id": "call_0",
			"name": "lookup", "arguments": args, "status": "completed",
		}
		response := map[string]any{
			"id": "resp_1", "model": "response-model", "status": "completed",
			"output": []any{item},
		}
		state := adapter.NewStreamState("requested-model")
		var mapped []llm.Event
		for _, event := range []any{
			map[string]any{"type": "response.created", "response": map[string]any{"id": "resp_1", "model": "response-model", "status": "in_progress", "output": []any{}}},
			map[string]any{"type": "response.output_item.added", "output_index": 0, "item": map[string]any{"id": "fc_0", "type": "function_call", "call_id": "call_0", "name": "lookup", "arguments": "", "status": "in_progress"}},
			map[string]any{"type": "response.function_call_arguments.delta", "output_index": 0, "delta": " \n"},
			map[string]any{"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{\"q\":\"go\"}\t "},
			map[string]any{"type": "response.output_item.done", "output_index": 0, "item": item},
			map[string]any{"type": "response.completed", "response": response},
		} {
			events, err := state.MapEvent(streamEvent(t, mustJSON(t, event)))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		var final responses.Response
		if err := json.Unmarshal([]byte(mustJSON(t, response)), &final); err != nil {
			t.Fatalf("unmarshal final response: %v", err)
		}
		blocking, err := adapter.MapResponse(&final)
		if err != nil {
			t.Fatalf("MapResponse returned error: %v", err)
		}
		if !reflect.DeepEqual(streamed.ToolCalls(), blocking.ToolCalls()) {
			t.Fatalf("stream/blocking calls = %#v / %#v", streamed.ToolCalls(), blocking.ToolCalls())
		}
		if calls := streamed.ToolCalls(); len(calls) != 1 || string(calls[0].Args) != args {
			t.Fatalf("args = %q, want exact provider bytes %q", calls[0].Args, args)
		}
	})

	t.Run("empty started call emits canonical object", func(t *testing.T) {
		state := adapter.NewStreamState("requested-model")
		var mapped []llm.Event
		for _, raw := range []string{
			`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"","status":"in_progress"}}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"","status":"completed"}}`,
			`{"type":"response.completed","response":{"id":"resp_1","model":"response-model","status":"completed","output":[{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"","status":"completed"}]}}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		if calls := streamed.ToolCalls(); len(calls) != 1 || string(calls[0].Args) != `{}` {
			t.Fatalf("calls = %#v, want canonical empty args", calls)
		}
	})

	t.Run("whitespace cannot become empty object", func(t *testing.T) {
		state := adapter.NewStreamState("requested-model")
		for _, event := range []any{
			map[string]any{"type": "response.output_item.added", "output_index": 0, "item": map[string]any{"id": "fc_0", "type": "function_call", "call_id": "call_0", "name": "lookup", "arguments": "", "status": "in_progress"}},
			map[string]any{"type": "response.function_call_arguments.delta", "output_index": 0, "delta": " "},
		} {
			if _, err := state.MapEvent(streamEvent(t, mustJSON(t, event))); err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
		}
		done := map[string]any{"type": "response.output_item.done", "output_index": 0, "item": map[string]any{"id": "fc_0", "type": "function_call", "call_id": "call_0", "name": "lookup", "arguments": "", "status": "completed"}}
		_, err := state.MapEvent(streamEvent(t, mustJSON(t, done)))
		if !errors.Is(err, llm.ErrServer) {
			t.Fatalf("error = %v, want ErrServer contradiction", err)
		}
	})
}

func TestOutputItemDoneRepairsAuthoritativeToolArguments(t *testing.T) {
	const finalRaw = `{
		"id":"resp_1","model":"response-model","status":"completed",
		"output":[{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}]
	}`
	adapter := replayAdapter()

	t.Run("repairs streamed prefix", func(t *testing.T) {
		state := adapter.NewStreamState("requested-model")
		var mapped []llm.Event
		for _, raw := range []string{
			`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"","status":"in_progress"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":"}`,
			`{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}}`,
			`{"type":"response.completed","response":` + finalRaw + `}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		if calls := streamed.ToolCalls(); len(calls) != 1 || string(calls[0].Args) != `{"q":"go"}` {
			t.Fatalf("repaired calls = %#v", calls)
		}
		if len(streamed.DroppedToolCalls) != 0 {
			t.Fatalf("dropped calls = %#v, want none", streamed.DroppedToolCalls)
		}
	})

	t.Run("rejects contradiction", func(t *testing.T) {
		state := adapter.NewStreamState("requested-model")
		for _, raw := range []string{
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"","status":"in_progress"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"x\":"}`,
		} {
			if _, err := state.MapEvent(streamEvent(t, raw)); err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
		}
		_, err := state.MapEvent(streamEvent(t, `{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}}`))
		if !errors.Is(err, llm.ErrServer) {
			t.Fatalf("error = %v, want ErrServer", err)
		}
	})
}

func TestOutputItemDoneReconcilesLateLowerDuplicateBeforeTransportError(t *testing.T) {
	const finalRaw = `{
		"id":"resp_1","model":"response-model","status":"completed",
		"output":[
			{"id":"fc_0","type":"function_call","call_id":"duplicate","name":"first","arguments":"{\"n\":0}","status":"completed"},
			{"id":"fc_1","type":"function_call","call_id":"duplicate","name":"second","arguments":"{\"n\":1}","status":"completed"}
		]
	}`
	adapter := replayAdapter()
	state := adapter.NewStreamState("requested-model")
	var mapped []llm.Event
	for _, raw := range []string{
		`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"duplicate","name":"second","arguments":"","status":"in_progress"}}`,
		`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"n\":1}"}`,
	} {
		events, err := state.MapEvent(streamEvent(t, raw))
		if err != nil {
			t.Fatalf("MapEvent returned error: %v", err)
		}
		mapped = append(mapped, events...)
	}
	lowerDone, err := state.MapEvent(streamEvent(t, `{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"duplicate","name":"first","arguments":"{\"n\":0}","status":"completed"}}`))
	if err != nil {
		t.Fatalf("MapEvent(lower done) returned error: %v", err)
	}
	if len(lowerDone) != 4 {
		t.Fatalf("lower done events = %#v, want ID change, start, delta, end", lowerDone)
	}
	change, ok := providerutil.DerefEvent(lowerDone[0]).(llm.ToolCallIDChanged)
	if !ok || change.Index != 1 || change.OldID != "duplicate" || change.NewID == "" {
		t.Fatalf("first lower-done event = %#v, want higher-owner ID correction", lowerDone[0])
	}
	start, ok := providerutil.DerefEvent(lowerDone[1]).(llm.ToolCallStart)
	if !ok || start.Index != 0 || start.ID != "duplicate" {
		t.Fatalf("second lower-done event = %#v, want lower ToolCallStart", lowerDone[1])
	}
	if delta, ok := providerutil.DerefEvent(lowerDone[2]).(llm.ToolCallDelta); !ok || delta.Index != 0 || delta.ArgsFragment != `{"n":0}` {
		t.Fatalf("third lower-done event = %#v, want lower ToolCallDelta", lowerDone[2])
	}
	if end, ok := providerutil.DerefEvent(lowerDone[3]).(llm.ToolCallEnd); !ok || end.Index != 0 {
		t.Fatalf("fourth lower-done event = %#v, want lower ToolCallEnd", lowerDone[3])
	}
	mapped = append(mapped, lowerDone...)

	transportErr := providerutil.NormalizeRemoteError(replayAdapterName, errors.New("connection reset"))
	partial, err := llm.Collect(eventsThenErrorSeq(mapped, transportErr))
	if !errors.Is(err, llm.ErrServer) || partial == nil {
		t.Fatalf("Collect = %#v, %v; want partial response and ErrServer", partial, err)
	}
	var final responses.Response
	if err := json.Unmarshal([]byte(finalRaw), &final); err != nil {
		t.Fatalf("unmarshal final response: %v", err)
	}
	blocking, err := adapter.MapResponse(&final)
	if err != nil {
		t.Fatalf("MapResponse returned error: %v", err)
	}
	if !reflect.DeepEqual(partial.ToolCalls(), blocking.ToolCalls()) {
		t.Fatalf("partial/blocking calls = %#v / %#v", partial.ToolCalls(), blocking.ToolCalls())
	}
}

func TestTerminalResponseReconcilesLowerDuplicateBeforeStart(t *testing.T) {
	const output = `[
		{"id":"fc_0","type":"function_call","call_id":"duplicate","name":"first","arguments":"{\"n\":0}","status":"completed"},
		{"id":"fc_1","type":"function_call","call_id":"duplicate","name":"second","arguments":"{\"n\":1}","status":"completed"}
	]`
	adapter := replayAdapter()
	for _, status := range []string{"completed", "failed"} {
		t.Run(status, func(t *testing.T) {
			state := adapter.NewStreamState("requested-model")
			var mapped []llm.Event
			for _, raw := range []string{
				`{"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"duplicate","name":"second","arguments":"","status":"in_progress"}}`,
				`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"n\":1}"}`,
			} {
				events, err := state.MapEvent(streamEvent(t, raw))
				if err != nil {
					t.Fatalf("MapEvent returned error: %v", err)
				}
				mapped = append(mapped, events...)
			}

			response := `{"id":"resp_1","model":"response-model","status":"` + status + `","output":` + output
			if status == "failed" {
				response += `,"error":{"code":"server_error","message":"failed"}`
			}
			response += `}`
			terminal, streamErr := state.MapEvent(streamEvent(t, `{"type":"response.`+status+`","response":`+response+`}`))
			if status == "completed" && streamErr != nil {
				t.Fatalf("MapEvent(completed) returned error: %v", streamErr)
			}
			if status == "failed" && streamErr == nil {
				t.Fatal("MapEvent(failed) returned nil error")
			}
			wantLen := 6
			if status == "failed" {
				wantLen = 5
			}
			if len(terminal) != wantLen {
				t.Fatalf("terminal events = %#v, want five reconciliation events plus optional MessageEnd", terminal)
			}
			change, ok := providerutil.DerefEvent(terminal[0]).(llm.ToolCallIDChanged)
			if !ok || change.Index != 1 || change.OldID != "duplicate" || change.NewID == "" {
				t.Fatalf("first terminal event = %#v, want higher-owner ID correction", terminal[0])
			}
			start, ok := providerutil.DerefEvent(terminal[1]).(llm.ToolCallStart)
			if !ok || start.Index != 0 || start.ID != "duplicate" {
				t.Fatalf("second terminal event = %#v, want lower ToolCallStart", terminal[1])
			}
			if delta, ok := providerutil.DerefEvent(terminal[2]).(llm.ToolCallDelta); !ok || delta.Index != 0 || delta.ArgsFragment != `{"n":0}` {
				t.Fatalf("third terminal event = %#v, want lower ToolCallDelta", terminal[2])
			}
			if end, ok := providerutil.DerefEvent(terminal[3]).(llm.ToolCallEnd); !ok || end.Index != 0 {
				t.Fatalf("fourth terminal event = %#v, want lower ToolCallEnd", terminal[3])
			}
			if end, ok := providerutil.DerefEvent(terminal[4]).(llm.ToolCallEnd); !ok || end.Index != 1 {
				t.Fatalf("fifth terminal event = %#v, want higher ToolCallEnd", terminal[4])
			}

			mapped = append(mapped, terminal...)
			var streamed *llm.Response
			var err error
			if status == "completed" {
				streamed, err = llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
			} else {
				streamed, err = llm.Collect(eventsThenErrorSeq(mapped, streamErr))
				if err == nil {
					t.Fatal("Collect(failed) returned nil error")
				}
			}
			if status == "completed" && err != nil {
				t.Fatalf("Collect returned error: %v", err)
			}
			var final responses.Response
			if err := json.Unmarshal([]byte(response), &final); err != nil {
				t.Fatalf("unmarshal terminal response: %v", err)
			}
			blocking, err := adapter.MapResponse(&final)
			if err != nil {
				t.Fatalf("MapResponse returned error: %v", err)
			}
			if !reflect.DeepEqual(streamed.ToolCalls(), blocking.ToolCalls()) || len(streamed.DroppedToolCalls) != 0 {
				t.Fatalf("stream/blocking calls = %#v / %#v, drops %#v", streamed.ToolCalls(), blocking.ToolCalls(), streamed.DroppedToolCalls)
			}
		})
	}
}

func TestResponseFailedRetainsAuthoritativeOutputAllKinds(t *testing.T) {
	const failedRaw = `{
		"id":"resp_failed","model":"response-model","status":"failed",
		"output":[
			{"id":"msg_0","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"terminal text","annotations":[]},{"type":"refusal","refusal":" refused"}]},
			{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}","status":"completed"},
			{"id":"rs_2","type":"reasoning","summary":[{"type":"summary_text","text":"terminal reasoning"}],"encrypted_content":"enc","status":"completed"}
		],
		"error":{"code":"server_error","message":"failed"}
	}`
	state := replayAdapter().NewStreamState("requested-model")
	events, streamErr := state.MapEvent(streamEvent(t, `{"type":"response.failed","response":`+failedRaw+`}`))
	if streamErr == nil {
		t.Fatal("response.failed returned nil error")
	}
	resp, err := llm.Collect(eventsThenErrorSeq(events, streamErr))
	if err == nil || resp == nil {
		t.Fatalf("Collect = %#v, %v; want authoritative partial response and error", resp, err)
	}
	if resp.Text() != "terminal text refused" || resp.Reasoning() != "terminal reasoning" || len(resp.ToolCalls()) != 1 {
		t.Fatalf("failed authoritative response = %#v", resp)
	}
	if len(resp.Parts) != 3 {
		t.Fatalf("parts = %#v, want text, tool, reasoning", resp.Parts)
	}
}

func TestTerminalReasoningContradictionReturnsServerError(t *testing.T) {
	state := replayAdapter().NewStreamState("requested-model")
	for _, raw := range []string{
		`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
		`{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"streamed"}`,
	} {
		if _, err := state.MapEvent(streamEvent(t, raw)); err != nil {
			t.Fatalf("MapEvent returned error: %v", err)
		}
	}
	_, err := state.MapEvent(streamEvent(t, `{
		"type":"response.completed",
		"response":{"id":"resp_1","model":"response-model","status":"completed","output":[{"id":"rs_0","type":"reasoning","summary":[{"type":"summary_text","text":"different"}],"encrypted_content":"enc","status":"completed"}]}
	}`))
	if !errors.Is(err, llm.ErrServer) {
		t.Fatalf("error = %v, want ErrServer", err)
	}
}

func TestTerminalReasoningRevalidatesOutputItemDone(t *testing.T) {
	const doneItem = `{"id":"rs_0","type":"reasoning","summary":[{"type":"summary_text","text":"streamed"}],"encrypted_content":"enc","status":"completed"}`

	t.Run("matching terminal is idempotent", func(t *testing.T) {
		state := replayAdapter().NewStreamState("requested-model")
		var mapped []llm.Event
		for _, raw := range []string{
			`{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"streamed"}`,
			`{"type":"response.output_item.done","output_index":0,"item":` + doneItem + `}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		terminal, err := state.MapEvent(streamEvent(t, `{"type":"response.completed","response":{"id":"resp_1","model":"response-model","status":"completed","output":[`+doneItem+`]}}`))
		if err != nil {
			t.Fatalf("MapEvent(completed) returned error: %v", err)
		}
		if len(terminal) != 1 {
			t.Fatalf("terminal events = %#v, want MessageEnd without duplicate reasoning", terminal)
		}
		if _, ok := providerutil.DerefEvent(terminal[0]).(llm.MessageEnd); !ok {
			t.Fatalf("terminal event = %T, want MessageEnd", terminal[0])
		}
		mapped = append(mapped, terminal...)
		resp, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
		if err != nil || resp == nil || resp.Reasoning() != "streamed" {
			t.Fatalf("Collect = %#v, %v; want one validated reasoning block", resp, err)
		}
	})

	t.Run("semantic match installs final wire raw", func(t *testing.T) {
		state := replayAdapter().NewStreamState("requested-model")
		var mapped []llm.Event
		for _, raw := range []string{
			`{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"streamed"}`,
			`{"type":"response.output_item.done","output_index":0,"item":` + doneItem + `}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		const terminalItem = `{"type":"reasoning","id":"rs_0","encrypted_content":"enc","status":"completed","summary":[{"text":"streamed","type":"summary_text"}]}`
		terminal, err := state.MapEvent(streamEvent(t, `{"type":"response.completed","response":{"id":"resp_1","model":"response-model","status":"completed","output":[`+terminalItem+`]}}`))
		if err != nil {
			t.Fatalf("MapEvent(completed) returned error: %v", err)
		}
		if len(terminal) != 2 {
			t.Fatalf("terminal events = %#v, want Raw replacement then MessageEnd", terminal)
		}
		replacement, ok := providerutil.DerefEvent(terminal[0]).(llm.ReasoningDelta)
		if !ok || replacement.Text != "" || replacement.Provider != replayAdapterName || !bytes.Equal(replacement.Raw, []byte(terminalItem)) {
			t.Fatalf("replacement = %#v, want final wire-verbatim Raw", terminal[0])
		}
		mapped = append(mapped, terminal...)
		resp, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
		if err != nil || resp == nil {
			t.Fatalf("Collect = %#v, %v", resp, err)
		}
		reasoning, ok := resp.Parts[0].(llm.ReasoningPart)
		if !ok || !bytes.Equal(reasoning.Raw, []byte(terminalItem)) {
			t.Fatalf("reasoning = %#v, want terminal Raw", resp.Parts)
		}
	})

	t.Run("contradiction retains completed partial", func(t *testing.T) {
		state := replayAdapter().NewStreamState("requested-model")
		var mapped []llm.Event
		for _, raw := range []string{
			`{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"streamed"}`,
			`{"type":"response.output_item.done","output_index":0,"item":` + doneItem + `}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		const contradictory = `{"id":"rs_0","type":"reasoning","summary":[{"type":"summary_text","text":"streamed"}],"encrypted_content":"other","status":"completed"}`
		events, streamErr := state.MapEvent(streamEvent(t, `{"type":"response.completed","response":{"id":"resp_1","model":"response-model","status":"completed","output":[`+contradictory+`]}}`))
		if !errors.Is(streamErr, llm.ErrServer) {
			t.Fatalf("terminal error = %v, want ErrServer", streamErr)
		}
		mapped = append(mapped, events...)
		partial, err := llm.Collect(eventsThenErrorSeq(mapped, streamErr))
		if !errors.Is(err, llm.ErrServer) || partial == nil || partial.Reasoning() != "streamed" {
			t.Fatalf("Collect = %#v, %v; want safe completed reasoning partial", partial, err)
		}
		reasoning, ok := partial.Parts[0].(llm.ReasoningPart)
		if !ok || !bytes.Equal(reasoning.Raw, []byte(doneItem)) {
			t.Fatalf("partial reasoning = %#v, want output_item.done raw retained", partial.Parts)
		}
	})
}

func TestTerminalReasoningContradictionRetainsBufferedText(t *testing.T) {
	state := replayAdapter().NewStreamState("requested-model")
	var mapped []llm.Event
	events, err := state.MapEvent(streamEvent(t, `{"type":"response.reasoning_text.delta","output_index":0,"delta":"safe prefix"}`))
	if err != nil {
		t.Fatalf("MapEvent(delta) returned error: %v", err)
	}
	mapped = append(mapped, events...)
	const contradictory = `{"id":"rs_0","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"different"}],"encrypted_content":"enc","status":"completed"}`
	events, streamErr := state.MapEvent(streamEvent(t, `{"type":"response.completed","response":{"id":"resp_1","model":"response-model","status":"completed","output":[`+contradictory+`]}}`))
	if !errors.Is(streamErr, llm.ErrServer) {
		t.Fatalf("terminal error = %v, want ErrServer", streamErr)
	}
	mapped = append(mapped, events...)
	partial, err := llm.Collect(eventsThenErrorSeq(mapped, streamErr))
	if !errors.Is(err, llm.ErrServer) || partial == nil || partial.Reasoning() != "safe prefix" {
		t.Fatalf("Collect = %#v, %v; want safe buffered reasoning text", partial, err)
	}
	reasoning, ok := partial.Parts[0].(llm.ReasoningPart)
	if !ok || len(reasoning.Raw) != 0 || reasoning.Provider != "" {
		t.Fatalf("partial reasoning = %#v, want text without fabricated replay metadata", partial.Parts)
	}
}

func TestTerminalFinalizesProvisionalToolCalls(t *testing.T) {
	t.Run("valid call is ended", func(t *testing.T) {
		const finalRaw = `{
			"id":"resp_1","model":"response-model","status":"completed",
			"output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}]
		}`
		adapter := replayAdapter()
		state := adapter.NewStreamState("requested-model")
		var mapped []llm.Event
		for _, raw := range []string{
			`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"","status":"in_progress"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":\"go\"}"}`,
			`{"type":"response.completed","response":` + finalRaw + `}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		if len(mapped) < 2 {
			t.Fatalf("events = %#v", mapped)
		}
		if _, ok := providerutil.DerefEvent(mapped[len(mapped)-2]).(llm.ToolCallEnd); !ok {
			t.Fatalf("penultimate event = %T, want rescued ToolCallEnd", mapped[len(mapped)-2])
		}
		if _, ok := providerutil.DerefEvent(mapped[len(mapped)-1]).(llm.MessageEnd); !ok {
			t.Fatalf("last event = %T, want MessageEnd", mapped[len(mapped)-1])
		}
		streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		var final responses.Response
		if err := json.Unmarshal([]byte(finalRaw), &final); err != nil {
			t.Fatalf("unmarshal final response: %v", err)
		}
		blocking, err := adapter.MapResponse(&final)
		if err != nil {
			t.Fatalf("MapResponse returned error: %v", err)
		}
		if !reflect.DeepEqual(streamed.Parts, blocking.Parts) {
			t.Fatalf("stream/blocking parts = %#v / %#v", streamed.Parts, blocking.Parts)
		}
	})

	t.Run("malformed call is dropped", func(t *testing.T) {
		const finalRaw = `{
			"id":"resp_1","model":"response-model","status":"completed",
			"output":[{"id":"fc_bad","type":"function_call","call_id":"call_bad","name":"lookup","arguments":"{\"q\":","status":"completed"}]
		}`
		state := replayAdapter().NewStreamState("requested-model")
		var mapped []llm.Event
		for _, raw := range []string{
			`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_bad","type":"function_call","call_id":"call_bad","name":"lookup","arguments":"","status":"in_progress"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":"}`,
			`{"type":"response.completed","response":` + finalRaw + `}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		if len(mapped) < 2 {
			t.Fatalf("events = %#v", mapped)
		}
		if drop, ok := providerutil.DerefEvent(mapped[len(mapped)-2]).(llm.ToolCallDropped); !ok || drop.Index != 0 {
			t.Fatalf("penultimate event = %#v, want ToolCallDropped index 0", mapped[len(mapped)-2])
		}
		streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		if len(streamed.ToolCalls()) != 0 || len(streamed.DroppedToolCalls) != 1 {
			t.Fatalf("collected calls/drops = %#v / %#v", streamed.ToolCalls(), streamed.DroppedToolCalls)
		}
	})
}

func TestEmptyAuthoritativeOutputFinalizesEveryProvisionalTool(t *testing.T) {
	build := func(t *testing.T, fragment string) (*responsesapi.StreamState, []llm.Event) {
		t.Helper()
		state := replayAdapter().NewStreamState("requested-model")
		var mapped []llm.Event
		for _, raw := range []string{
			`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"lookup","arguments":"","status":"in_progress"}}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		deltaRaw, err := json.Marshal(map[string]any{
			"type":         "response.function_call_arguments.delta",
			"output_index": 0,
			"delta":        fragment,
		})
		if err != nil {
			t.Fatalf("marshal delta: %v", err)
		}
		events, err := state.MapEvent(streamEvent(t, string(deltaRaw)))
		if err != nil {
			t.Fatalf("MapEvent(delta) returned error: %v", err)
		}
		return state, append(mapped, events...)
	}

	t.Run("valid tool ends", func(t *testing.T) {
		state, mapped := build(t, `{"q":"go"}`)
		events, err := state.MapEvent(streamEvent(t, `{"type":"response.completed","response":{"id":"resp_1","model":"response-model","status":"completed","output":[]}}`))
		if err != nil {
			t.Fatalf("MapEvent(completed) returned error: %v", err)
		}
		mapped = append(mapped, events...)
		if len(events) != 2 {
			t.Fatalf("terminal events = %#v, want ToolCallEnd and MessageEnd", events)
		}
		if _, ok := providerutil.DerefEvent(events[0]).(llm.ToolCallEnd); !ok {
			t.Fatalf("first terminal event = %T, want ToolCallEnd", events[0])
		}
		resp, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		if calls := resp.ToolCalls(); len(calls) != 1 || string(calls[0].Args) != `{"q":"go"}` || len(resp.DroppedToolCalls) != 0 {
			t.Fatalf("calls/drops = %#v / %#v", calls, resp.DroppedToolCalls)
		}
	})

	t.Run("malformed tool drops", func(t *testing.T) {
		state, mapped := build(t, `{"q":`)
		events, err := state.MapEvent(streamEvent(t, `{"type":"response.completed","response":{"id":"resp_1","model":"response-model","status":"completed","output":[]}}`))
		if err != nil {
			t.Fatalf("MapEvent(completed) returned error: %v", err)
		}
		mapped = append(mapped, events...)
		if len(events) != 2 {
			t.Fatalf("terminal events = %#v, want ToolCallDropped and MessageEnd", events)
		}
		if drop, ok := providerutil.DerefEvent(events[0]).(llm.ToolCallDropped); !ok || drop.Index != 0 {
			t.Fatalf("first terminal event = %#v, want ToolCallDropped index 0", events[0])
		}
		resp, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		if len(resp.ToolCalls()) != 0 || len(resp.DroppedToolCalls) != 1 {
			t.Fatalf("calls/drops = %#v / %#v", resp.ToolCalls(), resp.DroppedToolCalls)
		}
	})
}

func TestFailedEmptyAuthoritativeOutputFinalizesEveryProvisionalTool(t *testing.T) {
	state := replayAdapter().NewStreamState("requested-model")
	var mapped []llm.Event
	for _, raw := range []string{
		`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","name":"rescued","arguments":"","status":"in_progress"}}`,
		`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{}"}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"malformed","arguments":"","status":"in_progress"}}`,
		`{"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"q\":"}`,
	} {
		events, err := state.MapEvent(streamEvent(t, raw))
		if err != nil {
			t.Fatalf("MapEvent returned error: %v", err)
		}
		mapped = append(mapped, events...)
	}
	events, streamErr := state.MapEvent(streamEvent(t, `{
		"type":"response.failed",
		"response":{"id":"resp_1","model":"response-model","status":"failed","output":[],"error":{"code":"server_error","message":"failed"}}
	}`))
	if streamErr == nil {
		t.Fatal("response.failed returned nil error")
	}
	mapped = append(mapped, events...)
	resp, err := llm.Collect(eventsThenErrorSeq(mapped, streamErr))
	if err == nil || resp == nil {
		t.Fatalf("Collect = %#v, %v; want partial response and error", resp, err)
	}
	if calls := resp.ToolCalls(); len(calls) != 1 || calls[0].ID != "call_0" || string(calls[0].Args) != `{}` {
		t.Fatalf("rescued calls = %#v", calls)
	}
	if len(resp.DroppedToolCalls) != 1 || resp.DroppedToolCalls[0].Index != 1 {
		t.Fatalf("dropped calls = %#v, want malformed call at index 1", resp.DroppedToolCalls)
	}
}

func TestTerminalSettlesToolsOmittedFromAuthoritativeOutput(t *testing.T) {
	const output = `[
		{"id":"msg_0","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"terminal text","annotations":[]}]},
		{"id":"fc_1","type":"function_call","call_id":"call_1","name":"authoritative","arguments":"{}","status":"completed"}
	]`
	for _, status := range []string{"completed", "failed"} {
		t.Run(status, func(t *testing.T) {
			state := replayAdapter().NewStreamState("requested-model")
			var mapped []llm.Event
			for _, raw := range []string{
				`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
				`{"type":"response.output_item.added","output_index":2,"item":{"id":"fc_2","type":"function_call","call_id":"call_2","name":"rescued","arguments":"","status":"in_progress"}}`,
				`{"type":"response.function_call_arguments.delta","output_index":2,"delta":"{}"}`,
				`{"type":"response.output_item.added","output_index":3,"item":{"id":"fc_3","type":"function_call","call_id":"call_3","name":"malformed","arguments":"","status":"in_progress"}}`,
				`{"type":"response.function_call_arguments.delta","output_index":3,"delta":"{\"q\":"}`,
			} {
				events, err := state.MapEvent(streamEvent(t, raw))
				if err != nil {
					t.Fatalf("MapEvent returned error: %v", err)
				}
				mapped = append(mapped, events...)
			}

			response := `{"id":"resp_1","model":"response-model","status":"` + status + `","output":` + output
			if status == "failed" {
				response += `,"error":{"code":"server_error","message":"failed"}`
			}
			response += `}`
			events, streamErr := state.MapEvent(streamEvent(t, `{"type":"response.`+status+`","response":`+response+`}`))
			mapped = append(mapped, events...)

			var rescuedEnd, malformedDrop bool
			for _, event := range events {
				switch event := providerutil.DerefEvent(event).(type) {
				case llm.ToolCallEnd:
					rescuedEnd = rescuedEnd || event.Index == 2
				case llm.ToolCallDropped:
					malformedDrop = malformedDrop || event.Index == 3
				}
			}
			if !rescuedEnd || !malformedDrop {
				t.Fatalf("terminal events = %#v, want rescued end at 2 and malformed drop at 3", events)
			}

			var resp *llm.Response
			var err error
			if status == "completed" {
				if streamErr != nil {
					t.Fatalf("completed terminal error: %v", streamErr)
				}
				resp, err = llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
			} else {
				if streamErr == nil {
					t.Fatal("failed terminal returned nil error")
				}
				resp, err = llm.Collect(eventsThenErrorSeq(mapped, streamErr))
				if err == nil {
					t.Fatal("Collect returned nil error for failed terminal")
				}
			}
			if status == "completed" && err != nil {
				t.Fatalf("Collect returned error: %v", err)
			}
			if resp == nil || resp.Text() != "terminal text" {
				t.Fatalf("response = %#v, want authoritative text", resp)
			}
			calls := resp.ToolCalls()
			if len(calls) != 2 || calls[0].ID != "call_1" || calls[1].ID != "call_2" {
				t.Fatalf("calls = %#v, want authoritative and rescued tools", calls)
			}
			if len(resp.DroppedToolCalls) != 1 || resp.DroppedToolCalls[0].Index != 3 {
				t.Fatalf("dropped calls = %#v, want malformed omitted tool", resp.DroppedToolCalls)
			}
		})
	}
}

func TestTerminalReconciliationReturnsSafeEventsBeforeLaterContradiction(t *testing.T) {
	const output = `[
		{"id":"msg_0","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"terminal text","annotations":[]}]},
		{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"terminal reasoning"}],"encrypted_content":"enc","status":"completed"},
		{"id":"fc_2","type":"function_call","call_id":"call_2","name":"lookup","arguments":"{}","status":"completed"},
		{"id":"msg_3","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"different","annotations":[]}]}
	]`
	for _, status := range []string{"completed", "failed"} {
		t.Run(status, func(t *testing.T) {
			state := replayAdapter().NewStreamState("requested-model")
			var mapped []llm.Event
			for _, raw := range []string{
				`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
				`{"type":"response.output_text.delta","output_index":3,"delta":"streamed"}`,
			} {
				events, err := state.MapEvent(streamEvent(t, raw))
				if err != nil {
					t.Fatalf("MapEvent returned error: %v", err)
				}
				mapped = append(mapped, events...)
			}

			response := `{"id":"resp_1","model":"response-model","status":"` + status + `","output":` + output
			if status == "failed" {
				response += `,"error":{"code":"server_error","message":"failed"}`
			}
			response += `}`
			events, streamErr := state.MapEvent(streamEvent(t, `{"type":"response.`+status+`","response":`+response+`}`))
			if !errors.Is(streamErr, llm.ErrServer) {
				t.Fatalf("terminal error = %v, want ErrServer", streamErr)
			}
			mapped = append(mapped, events...)
			partial, err := llm.Collect(eventsThenErrorSeq(mapped, streamErr))
			if !errors.Is(err, llm.ErrServer) || partial == nil {
				t.Fatalf("Collect = %#v, %v; want partial response and ErrServer", partial, err)
			}
			if partial.Text() != "terminal textstreamed" || partial.Reasoning() != "terminal reasoning" {
				t.Fatalf("partial text/reasoning = %q/%q", partial.Text(), partial.Reasoning())
			}
			if calls := partial.ToolCalls(); len(calls) != 1 || calls[0].ID != "call_2" || string(calls[0].Args) != `{}` {
				t.Fatalf("partial calls = %#v", calls)
			}
			if len(partial.Parts) != 4 {
				t.Fatalf("partial parts = %#v, want all earlier terminal items and streamed contradiction prefix", partial.Parts)
			}
		})
	}
}

func TestTerminalFinalizesReasoningWithoutOutputItemDone(t *testing.T) {
	const reasoningItem = `{"id":"rs_1","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"private thought"}],"encrypted_content":"enc","status":"completed"}`
	const finalRaw = `{
		"id":"resp_1","model":"response-model","status":"completed",
		"output":[` + reasoningItem + `]
	}`
	adapter := replayAdapter()
	state := adapter.NewStreamState("requested-model")
	var mapped []llm.Event
	for _, raw := range []string{
		`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
		`{"type":"response.reasoning_text.delta","output_index":0,"delta":"private thought"}`,
		`{"type":"response.completed","response":` + finalRaw + `}`,
	} {
		events, err := state.MapEvent(streamEvent(t, raw))
		if err != nil {
			t.Fatalf("MapEvent returned error: %v", err)
		}
		mapped = append(mapped, events...)
	}
	if len(mapped) < 2 {
		t.Fatalf("events = %#v", mapped)
	}
	reasoningEvent, ok := providerutil.DerefEvent(mapped[len(mapped)-2]).(llm.ReasoningDelta)
	if !ok || reasoningEvent.Index != 0 || reasoningEvent.Text != "private thought" || len(reasoningEvent.Raw) == 0 {
		t.Fatalf("penultimate event = %#v, want finalized reasoning before MessageEnd", mapped[len(mapped)-2])
	}
	streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	var final responses.Response
	if err := json.Unmarshal([]byte(finalRaw), &final); err != nil {
		t.Fatalf("unmarshal final response: %v", err)
	}
	blocking, err := adapter.MapResponse(&final)
	if err != nil {
		t.Fatalf("MapResponse returned error: %v", err)
	}
	if !reflect.DeepEqual(streamed.Parts, blocking.Parts) {
		t.Fatalf("stream/blocking reasoning differ:\nstream:   %#v\nblocking: %#v", streamed.Parts, blocking.Parts)
	}
}

func TestCompleteTerminalReasoningMapping(t *testing.T) {
	const reasoningItem = `{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"terminal summary"}],"encrypted_content":"enc","status":"completed"}`

	t.Run("terminal only", func(t *testing.T) {
		const finalRaw = `{"id":"resp_1","model":"response-model","status":"completed","output":[` + reasoningItem + `]}`
		adapter := replayAdapter()
		state := adapter.NewStreamState("requested-model")
		var mapped []llm.Event
		for _, raw := range []string{
			`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
			`{"type":"response.completed","response":` + finalRaw + `}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		var final responses.Response
		if err := json.Unmarshal([]byte(finalRaw), &final); err != nil {
			t.Fatalf("unmarshal final response: %v", err)
		}
		blocking, err := adapter.MapResponse(&final)
		if err != nil {
			t.Fatalf("MapResponse returned error: %v", err)
		}
		if !reflect.DeepEqual(streamed.Parts, blocking.Parts) {
			t.Fatalf("terminal-only stream/blocking reasoning = %#v / %#v", streamed.Parts, blocking.Parts)
		}
	})

	t.Run("output item done without deltas", func(t *testing.T) {
		state := replayAdapter().NewStreamState("requested-model")
		if _, err := state.MapEvent(streamEvent(t, `{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`)); err != nil {
			t.Fatalf("MapEvent(created) returned error: %v", err)
		}
		events, err := state.MapEvent(streamEvent(t, `{"type":"response.output_item.done","output_index":0,"item":`+reasoningItem+`}`))
		if err != nil {
			t.Fatalf("MapEvent(done) returned error: %v", err)
		}
		if len(events) != 1 {
			t.Fatalf("events = %#v, want one ReasoningDelta", events)
		}
		reasoning, ok := providerutil.DerefEvent(events[0]).(llm.ReasoningDelta)
		if !ok || reasoning.Text != "terminal summary" || reasoning.Provider != replayAdapterName || !bytes.Equal(reasoning.Raw, []byte(reasoningItem)) {
			t.Fatalf("reasoning event = %#v, want summary text and verbatim raw", events[0])
		}
	})

	t.Run("summary delta without terminal item", func(t *testing.T) {
		state := replayAdapter().NewStreamState("requested-model")
		for _, raw := range []string{
			`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
			`{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"streamed summary"}`,
		} {
			_, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
		}
		events, err := state.MapEvent(streamEvent(t, `{"type":"response.completed","response":{"id":"resp_1","model":"response-model","status":"completed","output":[]}}`))
		if !errors.Is(err, llm.ErrServer) {
			t.Fatalf("terminal error = %v, want ErrServer", err)
		}
		if len(events) != 0 {
			t.Fatalf("terminal events = %#v, want no fabricated reasoning metadata", events)
		}
	})
}

func TestIncrementalEmptyToolArgumentsMatchBlockingBytes(t *testing.T) {
	const finalRaw = `{
		"id":"resp_1","model":"response-model","status":"completed",
		"output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"","status":"completed"}]
	}`
	adapter := replayAdapter()
	state := adapter.NewStreamState("requested-model")
	var mapped []llm.Event
	for _, raw := range []string{
		`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"","status":"in_progress"}}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"","status":"completed"}}`,
		`{"type":"response.completed","response":` + finalRaw + `}`,
	} {
		events, err := state.MapEvent(streamEvent(t, raw))
		if err != nil {
			t.Fatalf("MapEvent returned error: %v", err)
		}
		mapped = append(mapped, events...)
	}
	streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	var final responses.Response
	if err := json.Unmarshal([]byte(finalRaw), &final); err != nil {
		t.Fatalf("unmarshal final response: %v", err)
	}
	blocking, err := adapter.MapResponse(&final)
	if err != nil {
		t.Fatalf("MapResponse returned error: %v", err)
	}
	streamCalls, blockingCalls := streamed.ToolCalls(), blocking.ToolCalls()
	if len(streamCalls) != 1 || len(blockingCalls) != 1 {
		t.Fatalf("stream/blocking calls = %#v/%#v, want one each", streamCalls, blockingCalls)
	}
	if !reflect.DeepEqual(streamCalls[0].Args, blockingCalls[0].Args) || string(streamCalls[0].Args) != `{}` {
		t.Fatalf("stream/blocking args = %q/%q, want byte-equal {}", streamCalls[0].Args, blockingCalls[0].Args)
	}
}

func TestInterleavedDropsCollectInStableProviderOrder(t *testing.T) {
	const finalRaw = `{
		"id":"resp_1","model":"response-model","status":"completed",
		"output":[
			{"id":"fc_0","type":"function_call","call_id":"call_0","arguments":"{}","status":"completed"},
			{"id":"fc_1","type":"function_call","call_id":"call_1","arguments":"{}","status":"completed"}
		]
	}`
	adapter := replayAdapter()
	state := adapter.NewStreamState("requested-model")
	var mapped []llm.Event
	for _, raw := range []string{
		`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
		`{"type":"response.output_item.added","output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","arguments":"","status":"in_progress"}}`,
		`{"type":"response.output_item.done","output_index":1,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","arguments":"{}","status":"completed"}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","arguments":"","status":"in_progress"}}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_0","type":"function_call","call_id":"call_0","arguments":"{}","status":"completed"}}`,
		`{"type":"response.completed","response":` + finalRaw + `}`,
	} {
		events, err := state.MapEvent(streamEvent(t, raw))
		if err != nil {
			t.Fatalf("MapEvent returned error: %v", err)
		}
		mapped = append(mapped, events...)
	}
	streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	var final responses.Response
	if err := json.Unmarshal([]byte(finalRaw), &final); err != nil {
		t.Fatalf("unmarshal final response: %v", err)
	}
	blocking, err := adapter.MapResponse(&final)
	if err != nil {
		t.Fatalf("MapResponse returned error: %v", err)
	}
	if !reflect.DeepEqual(streamed.DroppedToolCalls, blocking.DroppedToolCalls) {
		t.Fatalf("stream/blocking drops = %#v / %#v", streamed.DroppedToolCalls, blocking.DroppedToolCalls)
	}
	if len(streamed.DroppedToolCalls) != 2 || streamed.DroppedToolCalls[0].Index != 0 || streamed.DroppedToolCalls[1].Index != 1 {
		t.Fatalf("sorted drops = %#v, want stable indexes 0,1", streamed.DroppedToolCalls)
	}
}

func TestInterleavedMalformedToolStreamsLaterStablePartImmediately(t *testing.T) {
	const finalRaw = `{
		"id":"resp_1","model":"response-model","status":"completed",
		"output":[
			{"id":"fc_bad","type":"function_call","call_id":"call_bad","arguments":"{}","status":"completed"},
			{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"retained","annotations":[]}]}
		]
	}`
	adapter := replayAdapter()
	state := adapter.NewStreamState("requested-model")
	var mapped []llm.Event
	for i, raw := range []string{
		`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_bad","type":"function_call","call_id":"call_bad","arguments":"","status":"in_progress"}}`,
		`{"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"retained"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_bad","type":"function_call","call_id":"call_bad","arguments":"{}","status":"completed"}}`,
		`{"type":"response.completed","response":` + finalRaw + `}`,
	} {
		events, err := state.MapEvent(streamEvent(t, raw))
		if err != nil {
			t.Fatalf("MapEvent %d returned error: %v", i, err)
		}
		if i == 2 {
			if len(events) != 1 {
				t.Fatalf("interleaved text events = %#v, want immediate TextDelta", events)
			}
			if text, ok := providerutil.DerefEvent(events[0]).(llm.TextDelta); !ok || text.Index != 1 {
				t.Fatalf("interleaved event = %#v, want stable text index 1", events[0])
			}
		}
		mapped = append(mapped, events...)
	}
	var dropAt, textAt, dropIndex, textIndex = -1, -1, -1, -1
	for i, event := range mapped {
		switch event := providerutil.DerefEvent(event).(type) {
		case llm.ToolCallDropped:
			dropAt, dropIndex = i, event.Index
		case llm.TextDelta:
			textAt, textIndex = i, event.Index
		}
	}
	if textAt < 0 || dropAt <= textAt || dropIndex != 0 || textIndex != 1 {
		t.Fatalf("text/drop order and indexes = %d/%d and %d/%d, want immediate text 1 before late drop 0", textAt, dropAt, textIndex, dropIndex)
	}
	streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	var final responses.Response
	if err := json.Unmarshal([]byte(finalRaw), &final); err != nil {
		t.Fatalf("unmarshal final response: %v", err)
	}
	blocking, err := adapter.MapResponse(&final)
	if err != nil {
		t.Fatalf("MapResponse returned error: %v", err)
	}
	if !reflect.DeepEqual(streamed.Parts, blocking.Parts) || !reflect.DeepEqual(streamed.DroppedToolCalls, blocking.DroppedToolCalls) {
		t.Fatalf("stream/blocking mismatch:\nstream:   %#v %#v\nblocking: %#v %#v", streamed.Parts, streamed.DroppedToolCalls, blocking.Parts, blocking.DroppedToolCalls)
	}
}

func TestResponseFailedPreservesInterleavedContentInStableOrder(t *testing.T) {
	state := replayAdapter().NewStreamState("requested-model")
	var mapped []llm.Event
	for i, raw := range []string{
		`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_bad","type":"function_call","call_id":"call_bad","arguments":"","status":"in_progress"}}`,
		`{"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"deferred text"}`,
		`{"type":"response.reasoning_text.delta","output_index":2,"delta":"deferred reasoning"}`,
	} {
		events, err := state.MapEvent(streamEvent(t, raw))
		if err != nil {
			t.Fatalf("MapEvent %d returned error: %v", i, err)
		}
		if i == 2 {
			if len(events) != 1 {
				t.Fatalf("text event = %#v, want immediate output", events)
			}
			if text, ok := providerutil.DerefEvent(events[0]).(llm.TextDelta); !ok || text.Index != 1 {
				t.Fatalf("text event = %#v, want stable index 1", events[0])
			}
		}
		mapped = append(mapped, events...)
	}
	failureEvents, streamErr := state.MapEvent(streamEvent(t, `{
		"type":"response.failed",
		"response":{"id":"resp_1","model":"response-model","status":"failed","output":[],"error":{"code":"server_error","message":"failed"}}
	}`))
	if streamErr == nil {
		t.Fatal("response.failed returned nil error")
	}
	mapped = append(mapped, failureEvents...)
	var dropAt, textAt, reasoningAt = -1, -1, -1
	for i, event := range mapped {
		switch providerutil.DerefEvent(event).(type) {
		case llm.ToolCallDropped:
			dropAt = i
		case llm.TextDelta:
			textAt = i
		case llm.ReasoningDelta:
			reasoningAt = i
		}
	}
	if textAt < 0 || dropAt <= textAt || reasoningAt <= dropAt {
		t.Fatalf("text/drop/reasoning order = %d/%d/%d, want immediate text then error finalization", textAt, dropAt, reasoningAt)
	}
	resp, err := llm.Collect(eventsThenErrorSeq(mapped, streamErr))
	if err == nil {
		t.Fatal("Collect returned nil error")
	}
	if resp == nil || resp.Text() != "deferred text" || resp.Reasoning() != "deferred reasoning" {
		t.Fatalf("partial response = %#v, want all deferred content", resp)
	}
}

func TestResponseErrorRescuesSynthesizableMissingIDTool(t *testing.T) {
	state := replayAdapter().NewStreamState("requested-model")
	var mapped []llm.Event
	for _, raw := range []string{
		`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","name":"lookup","arguments":"","status":"in_progress"}}`,
		`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":\"go\"}"}`,
	} {
		events, err := state.MapEvent(streamEvent(t, raw))
		if err != nil {
			t.Fatalf("MapEvent returned error: %v", err)
		}
		mapped = append(mapped, events...)
	}
	errorEvents, streamErr := state.MapEvent(streamEvent(t, `{"type":"error","code":"server_error","message":"failed"}`))
	if streamErr == nil {
		t.Fatal("error event returned nil error")
	}
	mapped = append(mapped, errorEvents...)
	resp, err := llm.Collect(eventsThenErrorSeq(mapped, streamErr))
	if err == nil {
		t.Fatal("Collect returned nil error")
	}
	if resp == nil {
		t.Fatal("Collect returned nil partial response")
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID == "" || calls[0].Name != "lookup" || string(calls[0].Args) != `{"q":"go"}` {
		t.Fatalf("rescued partial calls = %#v", calls)
	}
}

func TestStartedPartialToolSurvivesSemanticAndTruncatedErrors(t *testing.T) {
	build := func(t *testing.T) (*responsesapi.StreamState, []llm.Event) {
		t.Helper()
		state := replayAdapter().NewStreamState("requested-model")
		var mapped []llm.Event
		for _, raw := range []string{
			`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
			`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_partial","type":"function_call","call_id":"call_partial","name":"lookup","arguments":"","status":"in_progress"}}`,
			`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":"}`,
		} {
			events, err := state.MapEvent(streamEvent(t, raw))
			if err != nil {
				t.Fatalf("MapEvent returned error: %v", err)
			}
			mapped = append(mapped, events...)
		}
		return state, mapped
	}
	assertPartial := func(t *testing.T, resp *llm.Response, err error) {
		t.Helper()
		if err == nil || resp == nil {
			t.Fatalf("Collect = %#v, %v; want partial response and error", resp, err)
		}
		calls := resp.ToolCalls()
		if len(calls) != 1 || calls[0].ID != "call_partial" || string(calls[0].Args) != `{"q":` || len(resp.DroppedToolCalls) != 0 {
			t.Fatalf("partial calls/drops = %#v / %#v", calls, resp.DroppedToolCalls)
		}
	}

	t.Run("semantic", func(t *testing.T) {
		state, mapped := build(t)
		events, streamErr := state.MapEvent(streamEvent(t, `{"type":"error","code":"server_error","message":"failed"}`))
		mapped = append(mapped, events...)
		resp, err := llm.Collect(eventsThenErrorSeq(mapped, streamErr))
		assertPartial(t, resp, err)
	})

	t.Run("truncated", func(t *testing.T) {
		state, mapped := build(t)
		mapped = append(mapped, state.Finish()...)
		resp, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
		assertPartial(t, resp, err)
	})
}

func TestErrorOnlyFirstEventDoesNotSynthesizeMessageStart(t *testing.T) {
	state := replayAdapter().NewStreamState("requested-model")
	events, err := state.MapEvent(streamEvent(t, `{"type":"error","code":"server_error","message":"failed"}`))
	if err == nil {
		t.Fatal("error event returned nil error")
	}
	if len(events) != 0 {
		t.Fatalf("events = %#v, want no synthetic MessageStart", events)
	}
	resp, gotErr := llm.Collect(eventsThenErrorSeq(events, err))
	if gotErr == nil || resp != nil {
		t.Fatalf("Collect = %#v, %v; want nil response plus error", resp, gotErr)
	}
}

func TestDroppedToolCallPreservesStableProviderIndexes(t *testing.T) {
	const finalRaw = `{
		"id":"resp_1",
		"model":"response-model",
		"status":"completed",
		"output":[
			{"id":"fc_bad","type":"function_call","call_id":"call_bad","name":"lookup","arguments":"{\"q\":","status":"completed"},
			{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"after drop","annotations":[]}]}
		],
		"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}
	}`
	var final responses.Response
	if err := json.Unmarshal([]byte(finalRaw), &final); err != nil {
		t.Fatalf("unmarshal final response: %v", err)
	}
	adapter := replayAdapter()
	blocking, err := adapter.MapResponse(&final)
	if err != nil {
		t.Fatalf("MapResponse returned error: %v", err)
	}

	state := adapter.NewStreamState("requested-model")
	rawEvents := []string{
		`{"type":"response.created","response":{"id":"resp_1","model":"response-model","status":"in_progress","output":[]}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_bad","type":"function_call","call_id":"call_bad","name":"lookup","arguments":"","status":"in_progress"}}`,
		`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_bad","type":"function_call","call_id":"call_bad","name":"lookup","arguments":"{\"q\":","status":"completed"}}`,
		`{"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"after drop"}`,
		`{"type":"response.completed","response":` + finalRaw + `}`,
	}
	var mapped []llm.Event
	for _, raw := range rawEvents {
		events, err := state.MapEvent(streamEvent(t, raw))
		if err != nil {
			t.Fatalf("MapEvent(%s) returned error: %v", raw, err)
		}
		mapped = append(mapped, events...)
	}

	var dropIndex, textIndex = -1, -1
	var startAt, deltaAt, dropAt = -1, -1, -1
	for i, event := range mapped {
		switch event := providerutil.DerefEvent(event).(type) {
		case llm.ToolCallStart:
			startAt = i
		case llm.ToolCallDelta:
			deltaAt = i
		case llm.ToolCallDropped:
			dropIndex = event.Index
			dropAt = i
		case llm.TextDelta:
			textIndex = event.Index
		}
	}
	if dropIndex != 0 || textIndex != 1 {
		t.Fatalf("drop/text indexes = %d/%d, want stable provider indexes with a retained gap", dropIndex, textIndex)
	}
	if startAt < 0 || deltaAt <= startAt || dropAt <= deltaAt {
		t.Fatalf("tool event order start/delta/drop = %d/%d/%d, want observable partial call before drop", startAt, deltaAt, dropAt)
	}

	streamed, err := llm.Collect(providerutil.StreamContract(replayAdapterName, eventsSeq(mapped)))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if !reflect.DeepEqual(streamed.Parts, blocking.Parts) {
		t.Fatalf("stream/blocking parts differ:\nstream:   %#v\nblocking: %#v", streamed.Parts, blocking.Parts)
	}
	if !reflect.DeepEqual(streamed.DroppedToolCalls, blocking.DroppedToolCalls) {
		t.Fatalf("stream/blocking drops differ: %#v / %#v", streamed.DroppedToolCalls, blocking.DroppedToolCalls)
	}
	if len(blocking.DroppedToolCalls) != 1 || blocking.DroppedToolCalls[0].Index != 0 {
		t.Fatalf("blocking drops = %#v, want one drop at canonical index 0", blocking.DroppedToolCalls)
	}
}

func TestBuildParamsOmitsParallelToolCallsWhenUnsupported(t *testing.T) {
	adapter := responsesapi.Adapter{
		ProviderName: "narrow-responses",
		Capabilities: []llm.Capability{llm.CapabilityTools},
		ApplyOptions: func(_ *llm.Request, params *responses.ResponseNewParams) error {
			// A provider option cannot reintroduce a field excluded by the
			// provider's declared capability surface.
			params.ParallelToolCalls = sdk.Bool(false)
			return nil
		},
	}
	params, err := adapter.BuildParams(&llm.Request{
		Model:    "model",
		Messages: []llm.Message{llm.UserText("use a tool")},
		Tools: []llm.Tool{{
			Name:        "lookup",
			InputSchema: json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`),
		}},
	}, false)
	if err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	wire := marshalWire(t, params)
	if strings.Contains(wire, "parallel_tool_calls") {
		t.Fatalf("unsupported parallel_tool_calls must be omitted, got: %s", wire)
	}
}

func eventsSeq(events []llm.Event) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		for _, event := range events {
			if !yield(event, nil) {
				return
			}
		}
	}
}

func eventsThenErrorSeq(events []llm.Event, streamErr error) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		for _, event := range events {
			if !yield(event, nil) {
				return
			}
		}
		yield(nil, streamErr)
	}
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal JSON: %v", err)
	}
	return string(raw)
}
