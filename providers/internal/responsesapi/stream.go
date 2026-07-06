package responsesapi

import (
	"encoding/json"
	"strings"

	"github.com/openai/openai-go/v3/responses"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// StreamState maps semantic Responses stream events into go-llm events.
type StreamState struct {
	adapter Adapter
	id      string
	model   string
	started bool
	// sawFunctionCall/sawRefusal track output observed mid-stream: some
	// backends (the codex subscription endpoint) send response.completed with
	// an EMPTY output array, so the terminal stop-reason mapping cannot rely
	// on rescanning the terminal response alone.
	sawFunctionCall bool
	sawRefusal      bool
	tools           map[int]*streamToolCall
	refusals        map[int]*strings.Builder
	reasoning       map[int]*streamReasoning
	seenIDs         map[string]struct{}
}

type streamToolCall struct {
	id      string
	name    string
	args    strings.Builder
	started bool
	emitted bool
}

// streamReasoning tracks one reasoning output item so its final Text can be
// composed at block end: summary text streams live, raw reasoning_text is
// buffered and used only when no summary text arrived (matching the
// non-stream mapper's prefer-summary rule).
type streamReasoning struct {
	summarySeen bool
	text        strings.Builder
}

// NewStreamState creates stream mapping state for one stream.
func (a Adapter) NewStreamState() *StreamState {
	return &StreamState{
		adapter:   a,
		tools:     map[int]*streamToolCall{},
		refusals:  map[int]*strings.Builder{},
		reasoning: map[int]*streamReasoning{},
		seenIDs:   map[string]struct{}{},
	}
}

// Model returns the model observed in the stream.
func (s *StreamState) Model() string {
	if s == nil {
		return ""
	}
	return s.model
}

// itemIndex converts a stream event's output_index into the unified block
// index. Blocks are keyed by response output-item position: every content
// piece of one message item (multiple output_text/refusal segments) folds
// into a single text block, exactly like the non-stream mapper, so part
// boundaries and DroppedToolCall indices are identical on both paths.
func itemIndex(outputIndex int64) int {
	return int(outputIndex)
}

// MapEvent maps one SDK stream event.
func (s *StreamState) MapEvent(event responses.ResponseStreamEventUnion) ([]llm.Event, error) {
	switch event.Type {
	case "response.created":
		resp := event.Response
		s.id = resp.ID
		s.model = string(resp.Model)
		s.started = true
		return []llm.Event{llm.MessageStart{ID: resp.ID, Provider: s.adapter.ProviderName, Model: string(resp.Model)}}, nil
	case "response.output_text.delta":
		if event.Delta == "" {
			return nil, nil
		}
		return []llm.Event{llm.TextDelta{Index: itemIndex(event.OutputIndex), Text: event.Delta}}, nil
	case "response.refusal.delta":
		return s.mapRefusalDelta(event), nil
	case "response.refusal.done":
		return s.mapRefusalDone(event), nil
	case "response.reasoning_summary_text.delta":
		if event.Delta == "" {
			return nil, nil
		}
		s.reasoningBlock(itemIndex(event.OutputIndex)).summarySeen = true
		return []llm.Event{llm.ReasoningDelta{Index: itemIndex(event.OutputIndex), Text: event.Delta}}, nil
	case "response.reasoning_text.delta":
		// Raw reasoning_text is buffered and composed at block end: when the
		// block also carries summary text, the non-stream mapper prefers the
		// summary, so emitting both here would diverge.
		if event.Delta != "" {
			s.reasoningBlock(itemIndex(event.OutputIndex)).text.WriteString(event.Delta)
		}
		return nil, nil
	case "response.output_item.added":
		return s.mapOutputItemAdded(event)
	case "response.function_call_arguments.delta":
		return s.mapFunctionCallArgumentsDelta(event)
	case "response.function_call_arguments.done":
		return s.mapFunctionCallArgumentsDone(event), nil
	case "response.output_item.done":
		return s.mapOutputItemDone(event)
	case "response.completed", "response.incomplete":
		return s.mapTerminalResponse(event.Response), nil
	case "response.failed":
		return nil, s.adapter.MapResponseError(event.Response.Error)
	case "error":
		return nil, s.adapter.MapStreamError(event.Code, event.Message, event.Param)
	default:
		return nil, nil
	}
}

func (s *StreamState) mapOutputItemAdded(event responses.ResponseStreamEventUnion) ([]llm.Event, error) {
	item := event.Item
	if item.Type != "function_call" {
		return nil, nil
	}
	s.sawFunctionCall = true
	index := itemIndex(event.OutputIndex)
	call := s.toolCall(index)
	if call.name == "" {
		call.name = item.Name
	}
	s.assignToolCallID(call, index, item.CallID, false)
	return startToolCallEvents(index, call), nil
}

func (s *StreamState) mapFunctionCallArgumentsDelta(event responses.ResponseStreamEventUnion) ([]llm.Event, error) {
	s.sawFunctionCall = true
	index := itemIndex(event.OutputIndex)
	call := s.toolCall(index)
	call.args.WriteString(event.Delta)
	if !call.started {
		if call.name == "" {
			return nil, nil
		}
		return startToolCallEvents(index, call), nil
	}
	call.emitted = true
	return []llm.Event{llm.ToolCallDelta{Index: index, ArgsFragment: event.Delta}}, nil
}

func (s *StreamState) mapFunctionCallArgumentsDone(event responses.ResponseStreamEventUnion) []llm.Event {
	s.sawFunctionCall = true
	index := itemIndex(event.OutputIndex)
	call := s.toolCall(index)
	if call.args.Len() == 0 && event.Arguments != "" {
		call.args.WriteString(event.Arguments)
	}
	if call.name == "" {
		call.name = event.Name
	}
	return startToolCallEvents(index, call)
}

func (s *StreamState) mapOutputItemDone(event responses.ResponseStreamEventUnion) ([]llm.Event, error) {
	item := event.Item
	index := itemIndex(event.OutputIndex)
	switch item.Type {
	case "function_call":
		s.sawFunctionCall = true
		return s.mapFunctionCallDone(index, item), nil
	case "reasoning":
		raw := json.RawMessage(item.RawJSON())
		if len(raw) == 0 {
			encoded, err := json.Marshal(item)
			if err != nil {
				return nil, err
			}
			raw = encoded
		}
		text := ""
		if block := s.reasoning[index]; block != nil {
			delete(s.reasoning, index)
			// Compose the block's final Text: summary deltas already streamed
			// as they arrived; buffered reasoning_text is used only when no
			// summary text ever arrived (non-stream prefer-summary rule).
			if !block.summarySeen {
				text = block.text.String()
			}
		}
		return []llm.Event{llm.ReasoningDelta{Index: index, Text: text, Raw: raw, Provider: s.adapter.ProviderName}}, nil
	default:
		return nil, nil
	}
}

func (s *StreamState) mapFunctionCallDone(index int, item responses.ResponseOutputItemUnion) []llm.Event {
	call := s.toolCall(index)
	if call.name == "" {
		call.name = item.Name
	}
	s.assignToolCallID(call, index, item.CallID, true)
	if call.args.Len() == 0 {
		call.args.WriteString(item.Arguments.OfString)
	}
	if call.name == "" {
		return []llm.Event{llm.ToolCallDropped{Index: index, Reason: "missing tool name"}}
	}
	args := strings.TrimSpace(call.args.String())
	if args == "" {
		args = "{}"
	}
	if !json.Valid([]byte(args)) {
		return []llm.Event{llm.ToolCallDropped{Index: index, Reason: "invalid tool arguments JSON"}}
	}
	var events []llm.Event
	if !call.started {
		events = append(events, llm.ToolCallStart{Index: index, ID: call.id, Name: call.name})
	}
	if !call.emitted {
		events = append(events, llm.ToolCallDelta{Index: index, ArgsFragment: args})
		call.emitted = true
	}
	events = append(events, llm.ToolCallEnd{Index: index})
	delete(s.tools, index)
	return events
}

func startToolCallEvents(index int, call *streamToolCall) []llm.Event {
	if call.name == "" || call.id == "" {
		return nil
	}
	var events []llm.Event
	if !call.started {
		call.started = true
		events = append(events, llm.ToolCallStart{Index: index, ID: call.id, Name: call.name})
	}
	if call.args.Len() > 0 && !call.emitted {
		call.emitted = true
		events = append(events, llm.ToolCallDelta{Index: index, ArgsFragment: call.args.String()})
	}
	return events
}

func (s *StreamState) mapRefusalDelta(event responses.ResponseStreamEventUnion) []llm.Event {
	s.sawRefusal = true
	if event.Delta == "" {
		return nil
	}
	index := itemIndex(event.OutputIndex)
	refusal := s.refusal(index)
	refusal.WriteString(event.Delta)
	return []llm.Event{llm.TextDelta{Index: index, Text: event.Delta}}
}

func (s *StreamState) mapRefusalDone(event responses.ResponseStreamEventUnion) []llm.Event {
	s.sawRefusal = true
	if event.Refusal == "" {
		return nil
	}
	index := itemIndex(event.OutputIndex)
	refusal := s.refusal(index)
	if refusal.Len() > 0 {
		delete(s.refusals, index)
		return nil
	}
	refusal.WriteString(event.Refusal)
	delete(s.refusals, index)
	return []llm.Event{llm.TextDelta{Index: index, Text: event.Refusal}}
}

func (s *StreamState) mapTerminalResponse(resp responses.Response) []llm.Event {
	events := s.ensureStarted(resp)
	_, _, sawRefusal, sawFunctionCall := s.adapter.mapOutput(resp.Output)
	stop, raw := mapResponseStop(resp, sawRefusal || s.sawRefusal, sawFunctionCall || s.sawFunctionCall)
	events = append(events, llm.MessageEnd{
		StopReason:    stop,
		StopReasonRaw: raw,
		Usage:         s.adapter.mapUsage(resp.Model, resp.Usage),
		// The terminal SDK response rides on MessageEnd so Collect installs
		// it as Response.Raw — extras (output_text annotations, ...) stay
		// reachable on the streaming path too.
		Raw: &resp,
	})
	return events
}

func (s *StreamState) ensureStarted(resp responses.Response) []llm.Event {
	if s.started {
		return nil
	}
	s.started = true
	s.id = resp.ID
	s.model = string(resp.Model)
	return []llm.Event{llm.MessageStart{ID: resp.ID, Provider: s.adapter.ProviderName, Model: string(resp.Model)}}
}

func (s *StreamState) toolCall(index int) *streamToolCall {
	call := s.tools[index]
	if call != nil {
		return call
	}
	call = &streamToolCall{}
	s.tools[index] = call
	return call
}

func (s *StreamState) reasoningBlock(index int) *streamReasoning {
	block := s.reasoning[index]
	if block != nil {
		return block
	}
	block = &streamReasoning{}
	s.reasoning[index] = block
	return block
}

func (s *StreamState) assignToolCallID(call *streamToolCall, key int, providerID string, allowSynthetic bool) {
	if call.id != "" {
		return
	}
	if providerID != "" {
		if _, exists := s.seenIDs[providerID]; !exists {
			call.id = providerID
			s.seenIDs[providerID] = struct{}{}
			return
		}
	}
	if allowSynthetic {
		call.id = providerutil.UniqueSyntheticToolCallID(key, s.seenIDs)
		s.seenIDs[call.id] = struct{}{}
	}
}

func (s *StreamState) refusal(index int) *strings.Builder {
	refusal := s.refusals[index]
	if refusal != nil {
		return refusal
	}
	refusal = &strings.Builder{}
	s.refusals[index] = refusal
	return refusal
}
