package responsesapi

import (
	"bytes"
	"encoding/json"
	"sort"
	"strings"

	"github.com/openai/openai-go/v3/responses"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// StreamState maps semantic Responses stream events into go-llm events.
type StreamState struct {
	adapter      Adapter
	requestModel string
	model        string
	started      bool
	ended        bool
	// sawFunctionCall/sawRefusal track output observed mid-stream: some
	// backends (the codex subscription endpoint) send response.completed with
	// an EMPTY output array, so the terminal stop-reason mapping cannot rely
	// on rescanning the terminal response alone.
	sawFunctionCall bool
	sawRefusal      bool
	tools           map[int]*streamToolCall
	texts           map[int]*strings.Builder
	textSegments    map[int]map[int]*strings.Builder
	refusals        map[int]map[int]*strings.Builder
	reasoning       map[int]*streamReasoning
	seenIDs         map[string]struct{}
	explicitOwners  map[string]int
	droppedTools    map[int]struct{}
}

type streamToolCall struct {
	id         string
	providerID string
	name       string
	args       strings.Builder
	index      int
	started    bool
	emitted    int
	completed  bool
}

// streamReasoning tracks one reasoning output item so its final Text can be
// composed at block end: summary text streams live, raw reasoning_text is
// buffered and used only when no summary text arrived (matching the
// non-stream mapper's prefer-summary rule).
type streamReasoning struct {
	summarySeen    bool
	partialEmitted bool
	summary        strings.Builder
	summaryParts   map[int]*strings.Builder
	text           strings.Builder
	completed      bool
	authoritative  string
	raw            json.RawMessage
}

// NewStreamState creates stream mapping state for one stream. requestModel is
// used when the provider omits response.created or its model identity.
func (a Adapter) NewStreamState(requestModel ...string) *StreamState {
	state := &StreamState{
		adapter:        a,
		tools:          map[int]*streamToolCall{},
		texts:          map[int]*strings.Builder{},
		textSegments:   map[int]map[int]*strings.Builder{},
		refusals:       map[int]map[int]*strings.Builder{},
		reasoning:      map[int]*streamReasoning{},
		seenIDs:        map[string]struct{}{},
		explicitOwners: map[string]int{},
		droppedTools:   map[int]struct{}{},
	}
	if len(requestModel) > 0 {
		state.requestModel = requestModel[0]
	}
	return state
}

// Model returns the model observed in the stream.
func (s *StreamState) Model() string {
	if s == nil {
		return ""
	}
	return s.model
}

// Finish flushes buffered pre-start content when the remote stream ends
// without response.created. The stream contract remains responsible for
// reporting a missing MessageEnd after these partial events are emitted.
func (s *StreamState) Finish() []llm.Event {
	if s == nil || s.ended {
		return nil
	}
	return s.drainBeforeError(responses.Response{}, false)
}

// MapEvent maps one SDK stream event.
func (s *StreamState) MapEvent(event responses.ResponseStreamEventUnion) ([]llm.Event, error) {
	return s.mapEvent(event)
}

func (s *StreamState) mapEvent(event responses.ResponseStreamEventUnion) ([]llm.Event, error) {
	switch event.Type {
	case "response.created":
		return s.ensureStarted(event.Response), nil
	case "response.output_text.delta":
		if event.Delta == "" {
			return nil, nil
		}
		outputIndex := int(event.OutputIndex)
		s.textBlock(outputIndex).WriteString(event.Delta)
		s.textSegment(outputIndex, int(event.ContentIndex)).WriteString(event.Delta)
		return s.emit(llm.TextDelta{Index: outputIndex, Text: event.Delta}), nil
	case "response.output_text.done":
		events, err := s.mapOutputTextDone(event)
		return s.emit(events...), err
	case "response.refusal.delta":
		return s.emit(s.mapRefusalDelta(event)...), nil
	case "response.refusal.done":
		events, err := s.mapRefusalDone(event)
		return s.emit(events...), err
	case "response.reasoning_summary_text.delta":
		if event.Delta == "" {
			return nil, nil
		}
		outputIndex := int(event.OutputIndex)
		block := s.reasoningBlock(outputIndex)
		block.summarySeen = true
		block.summary.WriteString(event.Delta)
		block.summaryPart(int(event.SummaryIndex)).WriteString(event.Delta)
		return s.emit(llm.ReasoningDelta{Index: outputIndex, Text: event.Delta}), nil
	case "response.reasoning_summary_text.done":
		events, err := s.mapReasoningSummaryDone(event)
		return s.emit(events...), err
	case "response.reasoning_text.delta":
		// Raw reasoning_text is buffered and composed at block end: when the
		// block also carries summary text, the non-stream mapper prefers the
		// summary, so emitting both here would diverge.
		if event.Delta != "" {
			outputIndex := int(event.OutputIndex)
			block := s.reasoningBlock(outputIndex)
			block.text.WriteString(event.Delta)
		}
		if !s.started {
			return s.ensureStarted(responses.Response{}), nil
		}
		return nil, nil
	case "response.output_item.added":
		events, err := s.mapOutputItemAdded(event)
		return s.emit(events...), err
	case "response.function_call_arguments.delta":
		events, err := s.mapFunctionCallArgumentsDelta(event)
		return s.emit(events...), err
	case "response.function_call_arguments.done":
		events, err := s.mapFunctionCallArgumentsDone(event)
		return s.emit(events...), err
	case "response.output_item.done":
		events, err := s.mapOutputItemDone(event)
		return s.emit(events...), err
	case "response.completed", "response.incomplete":
		return s.mapTerminalResponse(event.Response)
	case "response.failed":
		events, err := s.mapFailedResponse(event.Response)
		if err != nil {
			return events, err
		}
		return events, s.adapter.MapResponseError(event.Response.Error)
	case "error":
		return s.eventsBeforeError(responses.Response{}), s.adapter.MapStreamError(event.Code, event.Message, event.Param)
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
	index := int(event.OutputIndex)
	call := s.toolCall(index)
	if call.name == "" {
		call.name = item.Name
	}
	if call.providerID == "" {
		call.providerID = item.CallID
	}
	return s.activateIncrementalToolCall(index, call), nil
}

func (s *StreamState) mapFunctionCallArgumentsDelta(event responses.ResponseStreamEventUnion) ([]llm.Event, error) {
	s.sawFunctionCall = true
	index := int(event.OutputIndex)
	call := s.toolCall(index)
	call.args.WriteString(event.Delta)
	events := s.activateIncrementalToolCall(index, call)
	return append(events, pendingToolArgumentEvents(call)...), nil
}

func (s *StreamState) mapFunctionCallArgumentsDone(event responses.ResponseStreamEventUnion) ([]llm.Event, error) {
	s.sawFunctionCall = true
	index := int(event.OutputIndex)
	call := s.toolCall(index)
	if call.name != "" && event.Name != "" && call.name != event.Name {
		return nil, s.malformedStreamError("tool arguments done event contradicts streamed name")
	}
	if call.name == "" {
		call.name = event.Name
	}
	wantArgs := event.Arguments
	if wantArgs == "" {
		wantArgs = "{}"
	}
	gotArgs := call.args.String()
	if !strings.HasPrefix(wantArgs, gotArgs) {
		return nil, s.malformedStreamError("tool arguments done event contradicts streamed prefix")
	}
	if suffix := wantArgs[len(gotArgs):]; suffix != "" {
		call.args.WriteString(suffix)
	}
	events := s.activateIncrementalToolCall(index, call)
	return append(events, pendingToolArgumentEvents(call)...), nil
}

func (s *StreamState) mapOutputItemDone(event responses.ResponseStreamEventUnion) ([]llm.Event, error) {
	item := event.Item
	outputIndex := int(event.OutputIndex)
	switch item.Type {
	case "function_call":
		s.sawFunctionCall = true
		return s.mapFunctionCallDone(outputIndex, item, true)
	case "reasoning":
		return s.mapReasoningDone(outputIndex, item)
	default:
		return nil, nil
	}
}

func (s *StreamState) mapOutputTextDone(event responses.ResponseStreamEventUnion) ([]llm.Event, error) {
	outputIndex := int(event.OutputIndex)
	segment := s.textSegment(outputIndex, int(event.ContentIndex))
	suffix, err := s.authoritativeSuffix(segment.String(), event.Text, "output text")
	if err != nil || suffix == "" {
		return nil, err
	}
	segment.WriteString(suffix)
	s.textBlock(outputIndex).WriteString(suffix)
	return []llm.Event{llm.TextDelta{Index: outputIndex, Text: suffix}}, nil
}

func (s *StreamState) mapReasoningSummaryDone(event responses.ResponseStreamEventUnion) ([]llm.Event, error) {
	outputIndex := int(event.OutputIndex)
	block := s.reasoningBlock(outputIndex)
	block.summarySeen = true
	part := block.summaryPart(int(event.SummaryIndex))
	suffix, err := s.authoritativeSuffix(part.String(), event.Text, "reasoning summary")
	if err != nil || suffix == "" {
		return nil, err
	}
	part.WriteString(suffix)
	block.summary.WriteString(suffix)
	return []llm.Event{llm.ReasoningDelta{Index: outputIndex, Text: suffix}}, nil
}

func (s *StreamState) authoritativeSuffix(streamed, authoritative, kind string) (string, error) {
	if !strings.HasPrefix(authoritative, streamed) {
		return "", s.malformedStreamError(kind + " done event contradicts streamed prefix")
	}
	return authoritative[len(streamed):], nil
}

func (s *StreamState) mapFunctionCallDone(outputIndex int, item responses.ResponseOutputItemUnion, finalize bool) ([]llm.Event, error) {
	call := s.toolCall(outputIndex)
	if call.name != "" && item.Name != "" && call.name != item.Name {
		return nil, s.malformedStreamError("terminal tool name contradicts streamed metadata")
	}
	if item.Name != "" {
		call.name = item.Name
	}
	if item.CallID != "" {
		call.providerID = item.CallID
	}
	if call.name == "" {
		if finalize {
			return []llm.Event{s.dropToolCall(outputIndex, "missing tool name", true)}, nil
		}
		return nil, nil
	}

	displacedIndex, displaced := s.observeExplicitToolID(outputIndex, call.providerID)
	events := s.displaceExplicitToolOwner(displacedIndex, call.providerID, displaced)
	events = append(events, s.reconcileExplicitToolIdentity(outputIndex, call)...)

	gotArgs := call.args.String()
	wantArgs := gotArgs
	if item.Type == "function_call" {
		wantArgs = item.Arguments.OfString
	}
	if wantArgs == "" {
		wantArgs = "{}"
	}
	if !strings.HasPrefix(wantArgs, gotArgs) {
		return events, s.malformedStreamError("terminal tool arguments contradict streamed prefix")
	}
	events = append(events, s.startFinalToolCall(outputIndex, call)...)
	if suffix := wantArgs[len(gotArgs):]; suffix != "" {
		call.args.WriteString(suffix)
	}
	events = append(events, pendingToolArgumentEvents(call)...)

	if !json.Valid([]byte(wantArgs)) {
		if finalize {
			events = append(events, s.dropToolCall(outputIndex, "invalid tool arguments JSON", false))
		}
		return events, nil
	}

	if finalize && !call.completed {
		call.completed = true
		events = append(events, llm.ToolCallEnd{Index: call.index})
	}
	return events, nil
}

func (s *StreamState) mapRefusalDelta(event responses.ResponseStreamEventUnion) []llm.Event {
	s.sawRefusal = true
	if event.Delta == "" {
		return nil
	}
	outputIndex := int(event.OutputIndex)
	refusal := s.refusal(outputIndex, int(event.ContentIndex))
	refusal.WriteString(event.Delta)
	s.textBlock(outputIndex).WriteString(event.Delta)
	return []llm.Event{llm.TextDelta{Index: outputIndex, Text: event.Delta}}
}

func (s *StreamState) mapRefusalDone(event responses.ResponseStreamEventUnion) ([]llm.Event, error) {
	s.sawRefusal = true
	outputIndex := int(event.OutputIndex)
	refusal := s.refusal(outputIndex, int(event.ContentIndex))
	suffix, err := s.authoritativeSuffix(refusal.String(), event.Refusal, "refusal")
	if err != nil || suffix == "" {
		return nil, err
	}
	refusal.WriteString(suffix)
	s.textBlock(outputIndex).WriteString(suffix)
	return []llm.Event{llm.TextDelta{Index: outputIndex, Text: suffix}}, nil
}

func (s *StreamState) mapTerminalResponse(resp responses.Response) ([]llm.Event, error) {
	events := s.ensureStarted(resp)
	finalized, err := s.finalizeProvisionalTools(resp.Output)
	events = append(events, finalized...)
	if err != nil {
		return events, err
	}
	authoritative, err := s.reconcileAuthoritativeOutput(resp.Output, true)
	events = append(events, authoritative...)
	if err != nil {
		events = append(events, s.dropIncompleteTerminalTools("terminal tool reconciliation failed")...)
		return events, err
	}
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
	s.ended = true
	return events, nil
}

func (s *StreamState) mapFailedResponse(resp responses.Response) ([]llm.Event, error) {
	var events []llm.Event
	if s.started || resp.ID != "" || resp.Model != "" || len(resp.Output) > 0 {
		events = append(events, s.ensureStarted(resp)...)
	}
	finalized, err := s.finalizeProvisionalTools(resp.Output)
	events = append(events, finalized...)
	if err != nil {
		return events, err
	}
	authoritative, err := s.reconcileAuthoritativeOutput(resp.Output, false)
	events = append(events, authoritative...)
	if err != nil {
		events = append(events, s.dropIncompleteTerminalTools("terminal tool reconciliation failed")...)
		return events, err
	}
	events = append(events, s.eventsBeforeError(resp)...)
	return events, nil
}

func (s *StreamState) reconcileAuthoritativeOutput(output []responses.ResponseOutputItemUnion, complete bool) ([]llm.Event, error) {
	canonicalCalls := map[int]llm.ToolCallPart{}
	droppedCalls := map[int]llm.DroppedToolCall{}
	seenIDs := map[string]struct{}{}
	for outputIndex, item := range output {
		if item.Type != "function_call" {
			continue
		}
		call, drop := mapFunctionCall(outputIndex, item, seenIDs)
		if drop != nil {
			droppedCalls[outputIndex] = *drop
			continue
		}
		canonicalCalls[outputIndex] = call
	}

	var events []llm.Event
	for outputIndex, item := range output {
		switch item.Type {
		case "message":
			mapped, err := s.reconcileMessage(outputIndex, item)
			events = append(events, mapped...)
			if err != nil {
				return events, err
			}
		case "function_call":
			if drop, ok := droppedCalls[outputIndex]; ok {
				if _, alreadyDropped := s.droppedTools[outputIndex]; alreadyDropped {
					continue
				}
				// mapFunctionCall reports "missing tool name" exactly when the
				// authoritative item has no name, so derive the owner-release
				// signal from item.Name rather than matching the display text.
				event := s.dropToolCall(outputIndex, drop.Reason, item.Name == "")
				events = append(events, event)
				continue
			}
			mapped, err := s.reconcileFunctionCall(outputIndex, canonicalCalls[outputIndex], item.CallID)
			events = append(events, mapped...)
			if err != nil {
				return events, err
			}
		case "reasoning":
			mapped, err := s.mapReasoningDone(outputIndex, item)
			events = append(events, mapped...)
			if err != nil {
				return events, err
			}
		}
	}

	if complete {
		indexes := make([]int, 0, len(s.reasoning))
		for outputIndex := range s.reasoning {
			indexes = append(indexes, outputIndex)
		}
		sort.Ints(indexes)
		for _, outputIndex := range indexes {
			if s.reasoning[outputIndex].completed {
				continue
			}
			mapped, err := s.mapReasoningDone(outputIndex, responses.ResponseOutputItemUnion{})
			events = append(events, mapped...)
			if err != nil {
				return events, err
			}
		}
	}
	return events, nil
}

func (s *StreamState) reconcileMessage(outputIndex int, item responses.ResponseOutputItemUnion) ([]llm.Event, error) {
	var expected strings.Builder
	for _, content := range item.Content {
		switch content.Type {
		case "output_text":
			expected.WriteString(content.Text)
		case "refusal":
			s.sawRefusal = true
			expected.WriteString(content.Refusal)
		}
	}
	want := expected.String()
	got := s.textBlock(outputIndex).String()
	if !strings.HasPrefix(want, got) {
		return nil, s.malformedStreamError("terminal text contradicts streamed prefix")
	}
	suffix := want[len(got):]
	if suffix == "" {
		return nil, nil
	}
	s.textBlock(outputIndex).WriteString(suffix)
	return []llm.Event{llm.TextDelta{Index: outputIndex, Text: suffix}}, nil
}

func (s *StreamState) reconcileFunctionCall(outputIndex int, expected llm.ToolCallPart, providerID string) ([]llm.Event, error) {
	call := s.toolCall(outputIndex)
	if call.name != "" && call.name != expected.Name {
		return nil, s.malformedStreamError("terminal tool name contradicts streamed metadata")
	}
	call.name = expected.Name
	wantArgs := string(expected.Args)
	gotArgs := call.args.String()
	if !strings.HasPrefix(wantArgs, gotArgs) {
		return nil, s.malformedStreamError("terminal tool arguments contradict streamed prefix")
	}

	var events []llm.Event
	if providerID != "" {
		call.providerID = providerID
		displacedIndex, displaced := s.observeExplicitToolID(outputIndex, providerID)
		events = append(events, s.displaceExplicitToolOwner(displacedIndex, providerID, displaced)...)
	}
	if call.started && call.id != expected.ID {
		if call.completed {
			return nil, s.malformedStreamError("terminal tool ID contradicts ended streamed call")
		}
		events = append(events, llm.ToolCallIDChanged{Index: outputIndex, OldID: call.id, NewID: expected.ID})
		delete(s.seenIDs, call.id)
		s.seenIDs[expected.ID] = struct{}{}
		call.id = expected.ID
	}
	call.id = expected.ID
	if call.id != "" {
		s.seenIDs[call.id] = struct{}{}
	}
	if call.providerID == "" {
		call.providerID = expected.ID
	}
	if !call.started {
		call.index = outputIndex
		call.started = true
		events = append(events, llm.ToolCallStart{Index: outputIndex, ID: expected.ID, Name: expected.Name})
	}
	if suffix := wantArgs[len(gotArgs):]; suffix != "" {
		call.args.WriteString(suffix)
		call.emitted = call.args.Len()
		events = append(events, llm.ToolCallDelta{Index: outputIndex, ArgsFragment: suffix})
	}
	if !call.completed {
		call.completed = true
		events = append(events, llm.ToolCallEnd{Index: outputIndex})
	}
	return events, nil
}

func (s *StreamState) malformedStreamError(message string) error {
	return &llm.ProviderError{Provider: s.adapter.ProviderName, Message: message, Kind: llm.ErrServer}
}

func (s *StreamState) finalizeProvisionalTools(output []responses.ResponseOutputItemUnion) ([]llm.Event, error) {
	represented := make(map[int]struct{})
	for outputIndex, item := range output {
		if item.Type == "function_call" {
			represented[outputIndex] = struct{}{}
		}
	}
	indexes := make([]int, 0, len(s.tools))
	for outputIndex, call := range s.tools {
		if _, authoritative := represented[outputIndex]; !authoritative && !call.completed {
			indexes = append(indexes, outputIndex)
		}
	}
	sort.Ints(indexes)
	var events []llm.Event
	for _, outputIndex := range indexes {
		mapped, err := s.mapFunctionCallDone(outputIndex, responses.ResponseOutputItemUnion{}, true)
		events = append(events, mapped...)
		if err != nil {
			return events, err
		}
	}
	return events, nil
}

func (s *StreamState) dropIncompleteTerminalTools(reason string) []llm.Event {
	indexes := make([]int, 0, len(s.tools))
	for outputIndex, call := range s.tools {
		if !call.completed {
			indexes = append(indexes, outputIndex)
		}
	}
	sort.Ints(indexes)
	events := make([]llm.Event, 0, len(indexes))
	for _, outputIndex := range indexes {
		// Terminal reconciliation failures are not name-absence drops, so the
		// explicit-owner reservation is released only for genuinely unnamed
		// calls (call.name == "" inside dropToolCall).
		events = append(events, s.dropToolCall(outputIndex, reason, false))
	}
	return events
}

func (s *StreamState) mapReasoningDone(outputIndex int, item responses.ResponseOutputItemUnion) ([]llm.Event, error) {
	block := s.reasoning[outputIndex]
	if block == nil {
		if item.Type != "reasoning" {
			return nil, nil
		}
		block = &streamReasoning{}
	}

	if item.Type != "reasoning" {
		return s.partialReasoningEvent(outputIndex, block), s.malformedStreamError("reasoning stream has no authoritative terminal item")
	}
	itemPart := s.adapter.mapReasoningItem(item)
	itemText := itemPart.Text
	raw := json.RawMessage(item.RawJSON())
	if len(raw) == 0 {
		return s.partialReasoningEvent(outputIndex, block), s.malformedStreamError("terminal reasoning item has no verbatim raw payload")
	}
	if !json.Valid(raw) {
		return s.partialReasoningEvent(outputIndex, block), s.malformedStreamError("malformed terminal reasoning item")
	}
	if block.completed {
		if block.authoritative != itemText || !providerutil.JSONEqual(block.raw, raw) {
			return nil, s.malformedStreamError("terminal reasoning item contradicts previously completed reasoning")
		}
		if !bytes.Equal(block.raw, raw) {
			block.raw = append(block.raw[:0], raw...)
			return []llm.Event{llm.ReasoningDelta{
				Index:    outputIndex,
				Raw:      raw,
				Provider: s.adapter.ProviderName,
			}}, nil
		}
		return nil, nil
	}

	text := ""
	if block.summarySeen {
		streamed := block.summary.String()
		if !strings.HasPrefix(itemText, streamed) {
			return s.partialReasoningEvent(outputIndex, block), s.malformedStreamError("terminal reasoning summary contradicts streamed prefix")
		}
		text = itemText[len(streamed):]
	} else if block.text.Len() > 0 && len(item.Summary) == 0 {
		streamed := block.text.String()
		if !strings.HasPrefix(itemText, streamed) {
			return s.partialReasoningEvent(outputIndex, block), s.malformedStreamError("terminal reasoning text contradicts streamed content")
		}
		// reasoning_text deltas are buffered until authoritative completion,
		// so emit the complete terminal text once.
		text = itemText
	} else {
		text = itemText
	}
	block.completed = true
	block.authoritative = itemText
	block.raw = append(block.raw[:0], raw...)
	if !block.summarySeen && block.text.Len() > 0 {
		block.partialEmitted = true
	}
	s.reasoning[outputIndex] = block
	return []llm.Event{llm.ReasoningDelta{
		Index:    outputIndex,
		Text:     text,
		Raw:      raw,
		Provider: s.adapter.ProviderName,
	}}, nil
}

func (s *StreamState) partialReasoningEvent(outputIndex int, block *streamReasoning) []llm.Event {
	if block == nil || block.summarySeen || block.partialEmitted || block.text.Len() == 0 {
		return nil
	}
	block.partialEmitted = true
	return []llm.Event{llm.ReasoningDelta{
		Index: outputIndex,
		Text:  block.text.String(),
	}}
}

func (s *StreamState) eventsBeforeError(resp responses.Response) []llm.Event {
	hasIdentity := resp.ID != "" || resp.Model != ""
	return s.drainBeforeError(resp, hasIdentity)
}

func (s *StreamState) drainBeforeError(resp responses.Response, hasIdentity bool) []llm.Event {
	events := s.emit(s.settleProvisionalToolsForError()...)
	events = append(events, s.emit(s.partialReasoningEvents()...)...)
	if !s.started {
		if !hasIdentity {
			return nil
		}
		return s.ensureStarted(resp)
	}
	return events
}

func (s *StreamState) settleProvisionalToolsForError() []llm.Event {
	indexes := make([]int, 0, len(s.tools))
	for outputIndex := range s.tools {
		indexes = append(indexes, outputIndex)
	}
	sort.Ints(indexes)
	var events []llm.Event
	for _, outputIndex := range indexes {
		call := s.tools[outputIndex]
		if call.started {
			// Actual stream failures preserve visible provisional calls exactly
			// as received, including incomplete argument JSON.
			delete(s.tools, outputIndex)
			continue
		}
		if call.name == "" {
			events = append(events, s.dropToolCall(outputIndex, "missing tool name", true))
			continue
		}
		args := call.args.String()
		if args == "" {
			args = "{}"
			call.args.WriteString(args)
		}
		events = append(events, s.startFinalToolCall(outputIndex, call)...)
		events = append(events, pendingToolArgumentEvents(call)...)
		if !json.Valid([]byte(args)) {
			events = append(events, s.dropToolCall(outputIndex, "invalid tool arguments JSON", false))
			continue
		}
		delete(s.tools, outputIndex)
	}
	return events
}

func (s *StreamState) partialReasoningEvents() []llm.Event {
	indexes := make([]int, 0, len(s.reasoning))
	for outputIndex, block := range s.reasoning {
		if !block.completed && !block.summarySeen && !block.partialEmitted && block.text.Len() > 0 {
			indexes = append(indexes, outputIndex)
		}
	}
	sort.Ints(indexes)
	events := make([]llm.Event, 0, len(indexes))
	for _, outputIndex := range indexes {
		block := s.reasoning[outputIndex]
		block.partialEmitted = true
		events = append(events, llm.ReasoningDelta{
			Index:    outputIndex,
			Text:     block.text.String(),
			Provider: s.adapter.ProviderName,
		})
	}
	return events
}

func (s *StreamState) ensureStarted(resp responses.Response) []llm.Event {
	if s.started {
		return nil
	}
	s.started = true
	s.model = string(resp.Model)
	if s.model == "" {
		s.model = s.requestModel
	}
	return []llm.Event{llm.MessageStart{ID: resp.ID, Provider: s.adapter.ProviderName, Model: s.model}}
}

func (s *StreamState) emit(events ...llm.Event) []llm.Event {
	if len(events) == 0 {
		return nil
	}
	if !s.started {
		started := s.ensureStarted(responses.Response{})
		return append(started, events...)
	}
	return events
}

func (s *StreamState) startIncrementalToolCall(outputIndex int, call *streamToolCall) []llm.Event {
	if call.started || call.name == "" || call.providerID == "" {
		return nil
	}
	return s.startToolCall(outputIndex, call)
}

func (s *StreamState) activateIncrementalToolCall(outputIndex int, call *streamToolCall) []llm.Event {
	if call.started || call.name == "" || call.providerID == "" {
		return nil
	}
	displacedIndex, displaced := s.observeExplicitToolID(outputIndex, call.providerID)
	events := s.displaceExplicitToolOwner(displacedIndex, call.providerID, displaced)
	events = append(events, s.startIncrementalToolCall(outputIndex, call)...)
	return events
}

func (s *StreamState) observeExplicitToolID(outputIndex int, providerID string) (int, bool) {
	if providerID == "" {
		return 0, false
	}
	owner, ok := s.explicitOwners[providerID]
	if !ok || outputIndex < owner {
		s.explicitOwners[providerID] = outputIndex
		return owner, ok
	}
	return 0, false
}

func (s *StreamState) displaceExplicitToolOwner(outputIndex int, providerID string, displaced bool) []llm.Event {
	if !displaced {
		return nil
	}
	call := s.tools[outputIndex]
	if call == nil || !call.started || call.id != providerID {
		return nil
	}
	delete(s.seenIDs, providerID)
	newID := providerutil.UniqueSyntheticToolCallID(outputIndex, s.seenIDs)
	s.seenIDs[newID] = struct{}{}
	call.id = newID
	return []llm.Event{llm.ToolCallIDChanged{Index: outputIndex, OldID: providerID, NewID: newID}}
}

func (s *StreamState) reconcileExplicitToolIdentity(outputIndex int, call *streamToolCall) []llm.Event {
	if !call.started || call.providerID == "" {
		return nil
	}
	owner, known := s.explicitOwners[call.providerID]
	usesExplicitID := call.id == call.providerID
	shouldUseExplicitID := known && owner == outputIndex
	if usesExplicitID == shouldUseExplicitID {
		return nil
	}

	oldID := call.id
	delete(s.seenIDs, call.id)
	if shouldUseExplicitID {
		call.id = call.providerID
	} else {
		call.id = providerutil.UniqueSyntheticToolCallID(outputIndex, s.seenIDs)
	}
	s.seenIDs[call.id] = struct{}{}
	return []llm.Event{llm.ToolCallIDChanged{Index: outputIndex, OldID: oldID, NewID: call.id}}
}

func (s *StreamState) startFinalToolCall(outputIndex int, call *streamToolCall) []llm.Event {
	if call.started {
		return nil
	}
	return s.startToolCall(outputIndex, call)
}

func (s *StreamState) startToolCall(outputIndex int, call *streamToolCall) []llm.Event {
	call.index = outputIndex
	s.assignToolCallID(call, call.index, call.providerID)
	call.started = true
	return []llm.Event{llm.ToolCallStart{Index: call.index, ID: call.id, Name: call.name}}
}

func pendingToolArgumentEvents(call *streamToolCall) []llm.Event {
	if !call.started || call.emitted >= call.args.Len() {
		return nil
	}
	args := call.args.String()
	fragment := args[call.emitted:]
	call.emitted = len(args)
	if fragment == "" {
		return nil
	}
	return []llm.Event{llm.ToolCallDelta{Index: call.index, ArgsFragment: fragment}}
}

// dropToolCall drops the tool at outputIndex and emits ToolCallDropped with
// reason. missingName reports that the drop is caused by an absent tool name
// (either the provisional call never received one, or the authoritative output
// item carried none); in that case the explicit-owner reservation is released
// so a later stable position can claim the same provider ID. reason is display
// text only and never drives control flow.
func (s *StreamState) dropToolCall(outputIndex int, reason string, missingName bool) llm.ToolCallDropped {
	if call := s.tools[outputIndex]; call != nil {
		if call.id != "" {
			delete(s.seenIDs, call.id)
		}
		if (call.name == "" || missingName) && call.providerID != "" && s.explicitOwners[call.providerID] == outputIndex {
			delete(s.explicitOwners, call.providerID)
		}
	}
	delete(s.tools, outputIndex)
	s.droppedTools[outputIndex] = struct{}{}
	return llm.ToolCallDropped{Index: outputIndex, Reason: reason}
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

func (s *streamReasoning) summaryPart(index int) *strings.Builder {
	if s.summaryParts == nil {
		s.summaryParts = map[int]*strings.Builder{}
	}
	part := s.summaryParts[index]
	if part == nil {
		part = &strings.Builder{}
		s.summaryParts[index] = part
	}
	return part
}

func (s *StreamState) textBlock(index int) *strings.Builder {
	text := s.texts[index]
	if text != nil {
		return text
	}
	text = &strings.Builder{}
	s.texts[index] = text
	return text
}

func (s *StreamState) textSegment(outputIndex, contentIndex int) *strings.Builder {
	segments := s.textSegments[outputIndex]
	if segments == nil {
		segments = map[int]*strings.Builder{}
		s.textSegments[outputIndex] = segments
	}
	segment := segments[contentIndex]
	if segment == nil {
		segment = &strings.Builder{}
		segments[contentIndex] = segment
	}
	return segment
}

func (s *StreamState) assignToolCallID(call *streamToolCall, key int, providerID string) {
	if call.id != "" {
		return
	}
	if providerID != "" {
		owner, known := s.explicitOwners[providerID]
		if known && owner == key {
			if _, exists := s.seenIDs[providerID]; !exists {
				call.id = providerID
				s.seenIDs[providerID] = struct{}{}
				return
			}
		} else if !known {
			call.id = providerID
			s.seenIDs[providerID] = struct{}{}
			return
		}
	}
	call.id = providerutil.UniqueSyntheticToolCallID(key, s.seenIDs)
	s.seenIDs[call.id] = struct{}{}
}

func (s *StreamState) refusal(outputIndex, contentIndex int) *strings.Builder {
	refusals := s.refusals[outputIndex]
	if refusals == nil {
		refusals = map[int]*strings.Builder{}
		s.refusals[outputIndex] = refusals
	}
	refusal := refusals[contentIndex]
	if refusal == nil {
		refusal = &strings.Builder{}
		refusals[contentIndex] = refusal
	}
	return refusal
}
