package anthropic

import (
	"encoding/json"
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
}

type streamToolCall struct {
	id      string
	name    string
	args    strings.Builder
	started bool
}

type streamThinkingBlock struct {
	typ       string
	thinking  strings.Builder
	signature string
	data      string
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
		return []llm.Event{llm.MessageEnd{
			StopReason:    mapStopReason(string(event.Delta.StopReason)),
			StopReasonRaw: string(event.Delta.StopReason),
			Usage:         usage,
		}}, nil
	case "message_stop":
		return nil, nil
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
		return []llm.Event{llm.TextDelta{Index: index, Text: block.Text}}, nil
	case "thinking", "redacted_thinking":
		thinking := &streamThinkingBlock{
			typ:       block.Type,
			signature: block.Signature,
			data:      block.Data,
		}
		thinking.thinking.WriteString(block.Thinking)
		s.thinking[index] = thinking
		if block.Thinking == "" {
			return nil, nil
		}
		return []llm.Event{llm.ReasoningDelta{Index: index, Text: block.Thinking}}, nil
	case "tool_use":
		id := block.ID
		if id == "" {
			id = providerutil.UniqueSyntheticToolCallID(index, s.seenIDs)
		}
		if _, exists := s.seenIDs[id]; exists {
			id = providerutil.UniqueSyntheticToolCallID(index, s.seenIDs)
		}
		if id != "" {
			s.seenIDs[id] = struct{}{}
		}
		call := &streamToolCall{id: id, name: block.Name}
		s.tools[index] = call

		var events []llm.Event
		if call.name != "" {
			call.started = true
			events = append(events, llm.ToolCallStart{Index: index, ID: call.id, Name: call.name})
		}
		if block.Input != nil {
			raw, err := json.Marshal(block.Input)
			if err != nil {
				return nil, err
			}
			if string(raw) != "null" && string(raw) != "{}" {
				call.args.Write(raw)
				if call.started {
					events = append(events, llm.ToolCallDelta{Index: index, ArgsFragment: string(raw)})
				}
			}
		}
		return events, nil
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
		return []llm.Event{llm.TextDelta{Index: index, Text: delta.Text}}, nil
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
		return []llm.Event{llm.ReasoningDelta{Index: index, Text: delta.Thinking}}, nil
	case "signature_delta":
		thinking := s.thinking[index]
		if thinking == nil {
			thinking = &streamThinkingBlock{typ: "thinking"}
			s.thinking[index] = thinking
		}
		thinking.signature += delta.Signature
		return nil, nil
	case "input_json_delta":
		call := s.tools[index]
		if call == nil {
			call = &streamToolCall{id: providerutil.UniqueSyntheticToolCallID(index, s.seenIDs)}
			s.seenIDs[call.id] = struct{}{}
			s.tools[index] = call
		}
		call.args.WriteString(delta.PartialJSON)
		if !call.started {
			if call.name == "" {
				return nil, nil
			}
			call.started = true
			return []llm.Event{
				llm.ToolCallStart{Index: index, ID: call.id, Name: call.name},
				llm.ToolCallDelta{Index: index, ArgsFragment: delta.PartialJSON},
			}, nil
		}
		return []llm.Event{llm.ToolCallDelta{Index: index, ArgsFragment: delta.PartialJSON}}, nil
	default:
		return nil, nil
	}
}

func (s *streamState) mapContentBlockStop(index int) ([]llm.Event, error) {
	if thinking := s.thinking[index]; thinking != nil {
		delete(s.thinking, index)
		raw, err := thinking.raw()
		if err != nil {
			return nil, err
		}
		return []llm.Event{llm.ReasoningDelta{Index: index, Raw: raw, Provider: providerName}}, nil
	}

	call := s.tools[index]
	if call == nil {
		return nil, nil
	}
	delete(s.tools, index)
	if call.name == "" {
		return []llm.Event{llm.ToolCallDropped{Index: index, Reason: "missing tool name"}}, nil
	}
	args := call.args.String()
	if args == "" {
		args = "{}"
	}
	if !json.Valid([]byte(args)) {
		return []llm.Event{llm.ToolCallDropped{Index: index, Reason: "invalid tool arguments JSON"}}, nil
	}
	if !call.started {
		return []llm.Event{
			llm.ToolCallStart{Index: index, ID: call.id, Name: call.name},
			llm.ToolCallDelta{Index: index, ArgsFragment: args},
			llm.ToolCallEnd{Index: index},
		}, nil
	}
	return []llm.Event{llm.ToolCallEnd{Index: index}}, nil
}

func (b *streamThinkingBlock) raw() (json.RawMessage, error) {
	switch b.typ {
	case "redacted_thinking":
		return json.Marshal(struct {
			Type string `json:"type"`
			Data string `json:"data"`
		}{Type: "redacted_thinking", Data: b.data})
	default:
		return json.Marshal(struct {
			Type      string `json:"type"`
			Thinking  string `json:"thinking"`
			Signature string `json:"signature"`
		}{Type: "thinking", Thinking: b.thinking.String(), Signature: b.signature})
	}
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
