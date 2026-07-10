package anthropic

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

type streamState struct {
	provider *Provider
	model    string
	usage    llm.Usage
	tools    map[int]*streamToolCall
	thinking map[int]*streamThinkingBlock
	seenIDs  map[string]struct{}
	terminal llm.MessageEnd
}

type streamToolCall struct {
	id      string
	name    string
	args    strings.Builder
	index   int
	started bool
}

type streamThinkingBlock struct {
	typ       string
	thinking  strings.Builder
	signature string
	data      string
	rawStart  json.RawMessage
	dirty     bool
	sawText   bool
}

func newStreamState(p *Provider) *streamState {
	return &streamState{
		provider: p,
		tools:    map[int]*streamToolCall{},
		thinking: map[int]*streamThinkingBlock{},
		seenIDs:  map[string]struct{}{},
	}
}

func (s *streamState) mapEvent(event sdk.MessageStreamEventUnion) ([]llm.Event, error) {
	switch event.Type {
	case "message_start":
		msg := event.Message
		s.model = string(msg.Model)
		s.usage = s.provider.mapUsage(msg.Model, msg.Usage)
		return []llm.Event{llm.MessageStart{ID: msg.ID, Provider: providerName, Model: string(msg.Model)}}, nil
	case "content_block_start":
		return s.mapContentBlockStart(event)
	case "content_block_delta":
		return s.mapContentBlockDelta(event)
	case "content_block_stop":
		return s.mapContentBlockStop(int(event.Index))
	case "message_delta":
		usage := s.provider.mergeStreamUsage(s.model, s.usage, event.Usage)
		s.usage = usage
		s.terminal = llm.MessageEnd{
			StopReason:    mapStopReason(string(event.Delta.StopReason)),
			StopReasonRaw: string(event.Delta.StopReason),
			Usage:         usage,
		}
		return nil, nil
	case "message_stop":
		end := s.terminal
		end.Usage = s.usage
		events, err := s.finalizeOpenBlocks()
		if err != nil {
			return nil, err
		}
		return append(events, end), nil
	default:
		return nil, nil
	}
}

func (s *streamState) mapContentBlockStart(event sdk.MessageStreamEventUnion) ([]llm.Event, error) {
	index := int(event.Index)
	block := event.ContentBlock
	switch block.Type {
	case "text":
		if block.Text == "" {
			return nil, nil
		}
		return []llm.Event{llm.TextDelta{Index: s.canonicalIndex(index), Text: block.Text}}, nil
	case "thinking", "redacted_thinking":
		thinking := s.thinking[index]
		if thinking == nil {
			thinking = &streamThinkingBlock{}
			s.thinking[index] = thinking
		}
		if thinking.typ != "" && thinking.typ != block.Type {
			return nil, fmt.Errorf("anthropic reasoning block %d changed type from %q to %q", index, thinking.typ, block.Type)
		}
		thinking.typ = block.Type
		thinking.signature = firstNonEmpty(thinking.signature, block.Signature)
		thinking.data = firstNonEmpty(thinking.data, block.Data)
		thinking.sawText = thinking.sawText || block.Thinking != "" || block.JSON.Thinking.Valid()
		if raw := block.RawJSON(); raw != "" && json.Valid([]byte(raw)) {
			thinking.rawStart = append(json.RawMessage(nil), raw...)
		}
		emitted := ""
		if block.Thinking != "" && thinking.thinking.Len() == 0 {
			thinking.thinking.WriteString(block.Thinking)
			emitted = block.Thinking
		}
		if emitted == "" {
			return nil, nil
		}
		return []llm.Event{llm.ReasoningDelta{Index: s.canonicalIndex(index), Text: emitted}}, nil
	case "tool_use":
		call := s.tools[index]
		if call == nil {
			call = &streamToolCall{}
			s.tools[index] = call
		}
		if block.ID != "" {
			call.id = block.ID
		}
		if block.Name != "" {
			call.name = block.Name
		}
		if block.Input != nil {
			raw, err := json.Marshal(block.Input)
			if err != nil {
				return nil, err
			}
			if call.args.Len() == 0 && string(raw) != "null" && string(raw) != "{}" {
				call.args.Write(raw)
			}
		}
		return s.startToolCall(index, call), nil
	default:
		return nil, nil
	}
}

func (s *streamState) mapContentBlockDelta(event sdk.MessageStreamEventUnion) ([]llm.Event, error) {
	index := int(event.Index)
	delta := event.Delta
	switch delta.Type {
	case "text_delta":
		if delta.Text == "" {
			return nil, nil
		}
		return []llm.Event{llm.TextDelta{Index: s.canonicalIndex(index), Text: delta.Text}}, nil
	case "thinking_delta":
		if delta.Thinking == "" {
			return nil, nil
		}
		thinking := s.thinking[index]
		if thinking == nil {
			thinking = &streamThinkingBlock{typ: "thinking"}
			s.thinking[index] = thinking
		}
		thinking.thinking.WriteString(delta.Thinking)
		thinking.dirty = true
		thinking.sawText = true
		return []llm.Event{llm.ReasoningDelta{Index: s.canonicalIndex(index), Text: delta.Thinking}}, nil
	case "signature_delta":
		thinking := s.thinking[index]
		if thinking == nil {
			thinking = &streamThinkingBlock{typ: "thinking"}
			s.thinking[index] = thinking
		}
		thinking.signature += delta.Signature
		if delta.Signature != "" {
			thinking.dirty = true
		}
		return nil, nil
	case "input_json_delta":
		call := s.tools[index]
		if call == nil {
			call = &streamToolCall{}
			s.tools[index] = call
		}
		call.args.WriteString(delta.PartialJSON)
		if call.started && delta.PartialJSON != "" {
			return []llm.Event{llm.ToolCallDelta{Index: call.index, ArgsFragment: delta.PartialJSON}}, nil
		}
		return nil, nil
	default:
		return nil, nil
	}
}

func (s *streamState) mapContentBlockStop(index int) ([]llm.Event, error) {
	if s.thinking[index] != nil {
		event, err := s.finalizeReasoningBlock(index)
		if err != nil {
			return nil, err
		}
		return []llm.Event{event}, nil
	}

	return s.finalizeToolCall(index), nil
}

func (s *streamState) finalizeOpenBlocks() ([]llm.Event, error) {
	sourceSet := make(map[int]struct{}, len(s.thinking)+len(s.tools))
	preparedReasoning := make(map[int]json.RawMessage, len(s.thinking))
	for source, thinking := range s.thinking {
		if s.tools[source] != nil {
			return nil, fmt.Errorf("anthropic stream block %d has both reasoning and tool state", source)
		}
		raw, err := thinking.raw()
		if err != nil {
			return nil, fmt.Errorf("anthropic reasoning block %d: %w", source, err)
		}
		preparedReasoning[source] = raw
		sourceSet[source] = struct{}{}
	}
	for source := range s.tools {
		sourceSet[source] = struct{}{}
	}
	sources := make([]int, 0, len(sourceSet))
	for source := range sourceSet {
		sources = append(sources, source)
	}
	sort.Ints(sources)

	var events []llm.Event
	for _, source := range sources {
		if raw, ok := preparedReasoning[source]; ok {
			delete(s.thinking, source)
			events = append(events, llm.ReasoningDelta{
				Index:    s.canonicalIndex(source),
				Raw:      raw,
				Provider: providerName,
			})
		}
		events = append(events, s.finalizeToolCall(source)...)
	}
	return events, nil
}

func (s *streamState) finalizeReasoningBlock(source int) (llm.ReasoningDelta, error) {
	thinking := s.thinking[source]
	if thinking == nil {
		return llm.ReasoningDelta{}, fmt.Errorf("missing reasoning state")
	}
	raw, err := thinking.raw()
	if err != nil {
		return llm.ReasoningDelta{}, fmt.Errorf("anthropic reasoning block %d: %w", source, err)
	}
	delete(s.thinking, source)
	return llm.ReasoningDelta{Index: s.canonicalIndex(source), Raw: raw, Provider: providerName}, nil
}

func (s *streamState) settleBlocksOnError() []llm.Event {
	sourceSet := make(map[int]struct{}, len(s.thinking)+len(s.tools))
	for source := range s.thinking {
		sourceSet[source] = struct{}{}
	}
	for source := range s.tools {
		sourceSet[source] = struct{}{}
	}
	sources := make([]int, 0, len(sourceSet))
	for source := range sourceSet {
		sources = append(sources, source)
	}
	sort.Ints(sources)

	var events []llm.Event
	for _, source := range sources {
		if thinking := s.thinking[source]; thinking != nil {
			if raw, err := thinking.raw(); err == nil {
				delete(s.thinking, source)
				events = append(events, llm.ReasoningDelta{
					Index:    s.canonicalIndex(source),
					Raw:      raw,
					Provider: providerName,
				})
			}
			continue
		}
		events = append(events, s.settleToolCallOnError(source)...)
	}
	return events
}

func (s *streamState) settleToolCallOnError(source int) []llm.Event {
	call := s.tools[source]
	if call == nil || call.started {
		return nil
	}
	if call.name != "" {
		return s.startToolCall(source, call)
	}
	delete(s.tools, source)
	return []llm.Event{llm.ToolCallDropped{
		Index:  s.canonicalIndex(source),
		Reason: "missing tool name before stream error",
	}}
}

func (s *streamState) finalizeToolCall(source int) []llm.Event {
	call := s.tools[source]
	if call == nil {
		return nil
	}
	delete(s.tools, source)
	if call.name == "" {
		return []llm.Event{llm.ToolCallDropped{Index: s.canonicalIndex(source), Reason: "missing tool name"}}
	}
	events := s.startToolCall(source, call)
	args := call.args.String()
	emptyArgs := args == ""
	if emptyArgs {
		args = "{}"
	}
	if !json.Valid([]byte(args)) {
		events = append(events, llm.ToolCallDropped{Index: call.index, Reason: "invalid tool arguments JSON"})
		return events
	}
	if emptyArgs {
		events = append(events, llm.ToolCallDelta{Index: call.index, ArgsFragment: args})
	}
	events = append(events, llm.ToolCallEnd{Index: call.index})
	return events
}

func (s *streamState) startToolCall(source int, call *streamToolCall) []llm.Event {
	if call.started || call.name == "" {
		return nil
	}
	call.index = s.canonicalIndex(source)
	if call.id == "" {
		call.id = providerutil.UniqueSyntheticToolCallID(call.index, s.seenIDs)
	}
	if _, exists := s.seenIDs[call.id]; exists {
		call.id = providerutil.UniqueSyntheticToolCallID(call.index, s.seenIDs)
	}
	s.seenIDs[call.id] = struct{}{}
	call.started = true

	events := []llm.Event{llm.ToolCallStart{Index: call.index, ID: call.id, Name: call.name}}
	if call.args.Len() > 0 {
		events = append(events, llm.ToolCallDelta{Index: call.index, ArgsFragment: call.args.String()})
	}
	return events
}

func (s *streamState) canonicalIndex(source int) int {
	return source
}

func (b *streamThinkingBlock) raw() (json.RawMessage, error) {
	switch b.typ {
	case "redacted_thinking":
		if b.data == "" {
			return nil, fmt.Errorf("redacted thinking block is missing data")
		}
		if !b.dirty && b.verbatimMatches() {
			return append(json.RawMessage(nil), b.rawStart...), nil
		}
		return json.Marshal(struct {
			Type string `json:"type"`
			Data string `json:"data"`
		}{Type: "redacted_thinking", Data: b.data})
	case "thinking":
		if !b.sawText {
			return nil, fmt.Errorf("thinking block is missing thinking data")
		}
		if b.signature == "" {
			return nil, fmt.Errorf("thinking block is missing signature")
		}
		if !b.dirty && b.verbatimMatches() {
			return append(json.RawMessage(nil), b.rawStart...), nil
		}
		return json.Marshal(struct {
			Type      string `json:"type"`
			Thinking  string `json:"thinking"`
			Signature string `json:"signature"`
		}{Type: "thinking", Thinking: b.thinking.String(), Signature: b.signature})
	default:
		return nil, fmt.Errorf("unsupported reasoning block type %q", b.typ)
	}
}

func (b *streamThinkingBlock) verbatimMatches() bool {
	if len(b.rawStart) == 0 || !json.Valid(b.rawStart) {
		return false
	}
	var wire struct {
		Type      string `json:"type"`
		Thinking  string `json:"thinking"`
		Signature string `json:"signature"`
		Data      string `json:"data"`
	}
	if json.Unmarshal(b.rawStart, &wire) != nil || wire.Type != b.typ {
		return false
	}
	if b.typ == "redacted_thinking" {
		return wire.Data == b.data
	}
	return wire.Thinking == b.thinking.String() && wire.Signature == b.signature
}

func firstNonEmpty(current, next string) string {
	if current != "" {
		return current
	}
	return next
}

func (p *Provider) mergeStreamUsage(model string, current llm.Usage, delta sdk.MessageDeltaUsage) llm.Usage {
	merged := current
	if delta.JSON.InputTokens.Valid() {
		merged.InputTokens = delta.InputTokens
	}
	if delta.JSON.CacheReadInputTokens.Valid() {
		merged.CacheReadTokens = delta.CacheReadInputTokens
	}
	if delta.JSON.CacheCreationInputTokens.Valid() {
		merged.CacheWriteTokens = delta.CacheCreationInputTokens
	}
	if delta.JSON.OutputTokens.Valid() {
		merged.OutputTokens = delta.OutputTokens
	}
	if delta.JSON.OutputTokensDetails.Valid() {
		merged.ReasoningTokens = delta.OutputTokensDetails.ThinkingTokens
	}
	merged.Raw = delta
	return p.finalizeUsage(model, merged)
}

func (p *Provider) finalizeUsage(model string, out llm.Usage) llm.Usage {
	out.TotalTokens = out.InputTokens + out.CacheReadTokens + out.CacheWriteTokens + out.OutputTokens
	out.CostUSD = nil
	if p.priceTable != nil {
		return llm.EstimateCostWithTable(p.priceTable, providerName, model, out)
	}
	return llm.EstimateCostForModel(providerName, model, out)
}
