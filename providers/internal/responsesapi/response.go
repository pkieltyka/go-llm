package responsesapi

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// MapResponse converts an SDK Responses response to a go-llm response.
func (a Adapter) MapResponse(resp *responses.Response) (*llm.Response, error) {
	if resp == nil {
		return nil, fmt.Errorf("%w: nil %s response", llm.ErrServer, a.ProviderName)
	}
	parts, dropped, sawRefusal, sawFunctionCall := a.mapOutput(resp.Output)
	usage := a.mapUsage(resp.Model, resp.Usage)
	stop, raw := mapResponseStop(*resp, sawRefusal, sawFunctionCall)
	return &llm.Response{
		ID:               resp.ID,
		Provider:         a.ProviderName,
		Model:            string(resp.Model),
		Parts:            parts,
		StopReason:       stop,
		StopReasonRaw:    raw,
		Usage:            usage,
		DroppedToolCalls: dropped,
		Raw:              resp,
	}, nil
}

func (a Adapter) mapOutput(output []responses.ResponseOutputItemUnion) ([]llm.Part, []llm.DroppedToolCall, bool, bool) {
	parts := make([]llm.Part, 0, len(output))
	dropped := make([]llm.DroppedToolCall, 0)
	seenIDs := map[string]struct{}{}
	sawRefusal := false
	sawFunctionCall := false
	for i, item := range output {
		switch item.Type {
		case "message":
			// All content pieces of one message item fold into a single
			// TextPart, mirroring the stream path where deltas accumulate
			// per output-item index — part boundaries and DroppedToolCall
			// indices (output-item positions) then match on both paths.
			var text strings.Builder
			for _, content := range item.Content {
				switch content.Type {
				case "output_text":
					text.WriteString(content.Text)
				case "refusal":
					sawRefusal = true
					text.WriteString(content.Refusal)
				}
			}
			if text.Len() > 0 {
				parts = append(parts, llm.Text(text.String()))
			}
		case "function_call":
			sawFunctionCall = true
			call, drop := mapFunctionCall(i, item, seenIDs)
			if drop != nil {
				dropped = append(dropped, *drop)
				continue
			}
			parts = append(parts, call)
		case "reasoning":
			parts = append(parts, a.mapReasoningItem(item))
		}
	}
	return parts, dropped, sawRefusal, sawFunctionCall
}

func mapFunctionCall(index int, item responses.ResponseOutputItemUnion, seenIDs map[string]struct{}) (llm.ToolCallPart, *llm.DroppedToolCall) {
	if item.Name == "" {
		return llm.ToolCallPart{}, &llm.DroppedToolCall{Index: index, Reason: "missing tool name"}
	}
	id := item.CallID
	if id == "" {
		id = providerutil.UniqueSyntheticToolCallID(index, seenIDs)
	}
	if _, exists := seenIDs[id]; exists {
		id = providerutil.UniqueSyntheticToolCallID(index, seenIDs)
	}
	seenIDs[id] = struct{}{}

	args := strings.TrimSpace(item.Arguments.OfString)
	if args == "" {
		args = "{}"
	}
	if !json.Valid([]byte(args)) {
		return llm.ToolCallPart{}, &llm.DroppedToolCall{Index: index, Reason: "invalid tool arguments JSON"}
	}
	return llm.ToolCallPart{ID: id, Name: item.Name, Args: json.RawMessage(args)}, nil
}

func (a Adapter) mapReasoningItem(item responses.ResponseOutputItemUnion) llm.ReasoningPart {
	var b strings.Builder
	for _, summary := range item.Summary {
		b.WriteString(summary.Text)
	}
	if b.Len() == 0 {
		var reasoning responses.ResponseReasoningItem
		if err := json.Unmarshal([]byte(item.RawJSON()), &reasoning); err == nil {
			for _, content := range reasoning.Content {
				b.WriteString(content.Text)
			}
		}
	}
	return llm.ReasoningPart{
		Text:     b.String(),
		Raw:      json.RawMessage(item.RawJSON()),
		Provider: a.ProviderName,
	}
}

func mapResponseStop(resp responses.Response, sawRefusal, sawFunctionCall bool) (llm.StopReason, string) {
	raw := string(resp.Status)
	switch resp.Status {
	case responses.ResponseStatusCompleted:
		if sawRefusal {
			return llm.StopReasonRefusal, raw
		}
		if sawFunctionCall {
			return llm.StopReasonToolUse, raw
		}
		return llm.StopReasonEndTurn, raw
	case responses.ResponseStatusIncomplete:
		if resp.IncompleteDetails.Reason != "" {
			raw += ":" + resp.IncompleteDetails.Reason
		}
		switch resp.IncompleteDetails.Reason {
		case "max_output_tokens":
			return llm.StopReasonMaxTokens, raw
		case "content_filter":
			return llm.StopReasonContentFilter, raw
		default:
			return llm.StopReasonOther, raw
		}
	case responses.ResponseStatusFailed:
		return llm.StopReasonError, raw
	case "":
		return "", raw
	default:
		return llm.StopReasonOther, raw
	}
}

func (a Adapter) mapUsage(model shared.ResponsesModel, usage responses.ResponseUsage) llm.Usage {
	cacheRead := usage.InputTokensDetails.CachedTokens
	inputTokens := usage.InputTokens
	if cacheRead > 0 && inputTokens >= cacheRead {
		inputTokens -= cacheRead
	}
	out := llm.Usage{
		InputTokens:     inputTokens,
		CacheReadTokens: cacheRead,
		OutputTokens:    usage.OutputTokens,
		ReasoningTokens: usage.OutputTokensDetails.ReasoningTokens,
		TotalTokens:     usage.TotalTokens,
		Raw:             usage,
	}
	if out.TotalTokens == 0 {
		out.TotalTokens = out.InputTokens + out.CacheReadTokens + out.OutputTokens
	}
	if a.PriceTable != nil {
		return llm.EstimateCostWithTable(a.PriceTable, a.ProviderName, string(model), out)
	}
	return llm.EstimateCostForModel(a.ProviderName, string(model), out)
}
