package anthropic

import (
	"encoding/json"
	"fmt"

	sdk "github.com/anthropics/anthropic-sdk-go"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

func (p *Provider) mapMessage(msg *sdk.Message) (*llm.Response, error) {
	if msg == nil {
		return nil, fmt.Errorf("%w: nil Anthropic message", llm.ErrServer)
	}
	parts, dropped := mapContentBlocks(msg.Content)
	usage := p.mapUsage(msg.Model, msg.Usage)
	return &llm.Response{
		ID:               msg.ID,
		Provider:         providerName,
		Model:            string(msg.Model),
		Parts:            parts,
		StopReason:       mapStopReason(string(msg.StopReason)),
		StopReasonRaw:    string(msg.StopReason),
		Usage:            usage,
		DroppedToolCalls: dropped,
		Raw:              msg,
	}, nil
}

func mapContentBlocks(blocks []sdk.ContentBlockUnion) ([]llm.Part, []llm.DroppedToolCall) {
	parts := make([]llm.Part, 0, len(blocks))
	dropped := make([]llm.DroppedToolCall, 0)
	seenIDs := map[string]struct{}{}
	for index, block := range blocks {
		switch block.Type {
		case "text":
			if block.Text != "" {
				parts = append(parts, llm.Text(block.Text))
			}
		case "thinking", "redacted_thinking":
			parts = append(parts, llm.ReasoningPart{
				Text:     block.Thinking,
				Raw:      json.RawMessage(block.RawJSON()),
				Provider: providerName,
			})
		case "tool_use":
			call, drop := mapToolUseBlock(index, block, seenIDs)
			if drop != nil {
				dropped = append(dropped, *drop)
				continue
			}
			parts = append(parts, call)
		}
	}
	return parts, dropped
}

func mapToolUseBlock(index int, block sdk.ContentBlockUnion, seenIDs map[string]struct{}) (llm.ToolCallPart, *llm.DroppedToolCall) {
	if block.Name == "" {
		return llm.ToolCallPart{}, &llm.DroppedToolCall{Index: index, Reason: "missing tool name"}
	}
	id := block.ID
	if id == "" {
		id = providerutil.UniqueSyntheticToolCallID(index, seenIDs)
	}
	if _, exists := seenIDs[id]; exists {
		id = providerutil.UniqueSyntheticToolCallID(index, seenIDs)
	}
	seenIDs[id] = struct{}{}

	args := json.RawMessage("{}")
	if len(block.Input) != 0 {
		if !json.Valid(block.Input) {
			return llm.ToolCallPart{}, &llm.DroppedToolCall{Index: index, Reason: "invalid tool arguments JSON"}
		}
		args = append(json.RawMessage(nil), block.Input...)
	}
	return llm.ToolCallPart{ID: id, Name: block.Name, Args: args}, nil
}

func (p *Provider) mapUsage(model sdk.Model, usage sdk.Usage) llm.Usage {
	out := llm.Usage{
		InputTokens:      usage.InputTokens,
		OutputTokens:     usage.OutputTokens,
		CacheReadTokens:  usage.CacheReadInputTokens,
		CacheWriteTokens: usage.CacheCreationInputTokens,
		ReasoningTokens:  usage.OutputTokensDetails.ThinkingTokens,
		Raw:              usage,
	}
	out.TotalTokens = out.InputTokens + out.CacheReadTokens + out.CacheWriteTokens + out.OutputTokens
	if p.priceTable != nil {
		return llm.EstimateCostWithTable(p.priceTable, providerName, string(model), out)
	}
	return llm.EstimateCostForModel(providerName, string(model), out)
}

func mapStopReason(raw string) llm.StopReason {
	switch raw {
	case "end_turn":
		return llm.StopReasonEndTurn
	case "max_tokens":
		return llm.StopReasonMaxTokens
	case "stop_sequence":
		return llm.StopReasonStopSequence
	case "tool_use":
		return llm.StopReasonToolUse
	case "refusal":
		return llm.StopReasonRefusal
	case "model_context_window_exceeded":
		return llm.StopReasonContextOverflow
	case "pause_turn":
		return llm.StopReasonPaused
	case "":
		return ""
	default:
		return llm.StopReasonOther
	}
}
