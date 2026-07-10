package chatcompletions

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	sdk "github.com/openai/openai-go/v3"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// mapResponse converts a chat completion response into the normalized shape.
func (p *Provider) mapResponse(resp *sdk.ChatCompletion) (*llm.Response, error) {
	if resp == nil {
		return nil, providerutil.NormalizeRemoteError(p.Name(), fmt.Errorf("%w: nil %s response", llm.ErrServer, p.Name()))
	}
	var raw rawChatCompletion
	if err := json.Unmarshal([]byte(resp.RawJSON()), &raw); err != nil {
		return nil, providerutil.NormalizeRemoteError(p.Name(), err)
	}
	choice, ok := choiceAtIndexZero(raw.Choices)
	if !ok {
		return nil, &llm.ProviderError{
			Provider: p.Name(),
			Code:     "empty_choices",
			Message:  "chat completion returned no choice at index 0",
			Kind:     llm.ErrServer,
			RawBody:  []byte(resp.RawJSON()),
		}
	}
	if choice.Error != nil {
		return nil, p.mapChunkError(choice.Error, []byte(resp.RawJSON()))
	}
	parts, dropped, err := p.partsFromChoice(raw.Raw, choice)
	if err != nil {
		return nil, err
	}
	usage := p.dialect.MapUsage(raw.Model, raw.Usage, p.priceTable)
	return &llm.Response{
		ID:               raw.ID,
		Provider:         p.Name(),
		Model:            raw.Model,
		Parts:            parts,
		StopReason:       p.normalizeToolUseStop(p.dialect.MapStopReason(choice.FinishReason), hasToolCalls(parts)),
		StopReasonRaw:    choice.FinishReason,
		Usage:            usage,
		DroppedToolCalls: dropped,
		Raw:              p.dialect.ExtractExtras(raw.Raw, choice),
	}, nil
}

func choiceAtIndexZero(choices []rawChoice) (rawChoice, bool) {
	for _, choice := range choices {
		if choice.Index == 0 {
			return choice, true
		}
	}
	return rawChoice{}, false
}

const (
	reasoningBlockIndex = iota
	contentBlockIndex
	refusalBlockIndex
	toolBlockBase
)

func toolBlockIndex(position int) int { return toolBlockBase + position }

// normalizeToolUseStop implements Compat.NormalizeToolUseStop: only an
// end-turn mapping is upgraded (truncation and error finishes are
// preserved), and StopReasonRaw keeps the wire value either way.
func (p *Provider) normalizeToolUseStop(mapped llm.StopReason, toolCalls bool) llm.StopReason {
	if p.compat.NormalizeToolUseStop && toolCalls && mapped == llm.StopReasonEndTurn {
		return llm.StopReasonToolUse
	}
	return mapped
}

func hasToolCalls(parts []llm.Part) bool {
	for _, part := range parts {
		if _, ok := providerutil.DerefPart(part).(llm.ToolCallPart); ok {
			return true
		}
	}
	return false
}

func (p *Provider) partsFromChoice(root jsonObject, choice rawChoice) ([]llm.Part, []llm.DroppedToolCall, error) {
	parts, dropped, err := p.dialect.ExtractParts(root, choice.Message)
	if err != nil {
		return nil, nil, err
	}
	if parts != nil || dropped != nil {
		return parts, dropped, nil
	}
	return defaultParts(p.Name(), choice.Message)
}

// defaultParts is the standard chat-completions message mapping used when a
// dialect's ExtractParts defers (returns nil, nil, nil). ReasoningPart.Raw
// holds the wire reasoning_details ARRAY verbatim, tagged with the dialect's
// provider name for same-provider replay. Reasoning text tolerates both the
// current `reasoning` and legacy `reasoning_content` field spellings.
func defaultParts(provider string, message rawMessage) ([]llm.Part, []llm.DroppedToolCall, error) {
	var parts []llm.Part
	if reasoning := message.reasoningText(); reasoning != "" || len(message.ReasoningDetails) > 0 {
		parts = append(parts, llm.ReasoningPart{Text: reasoning, Raw: append([]byte(nil), message.ReasoningDetails...), Provider: provider})
	}
	if message.Content != "" {
		parts = append(parts, llm.Text(message.Content))
	}
	if message.Refusal != "" {
		parts = append(parts, llm.Text(message.Refusal))
	}
	// nil (not a non-nil empty slice) when nothing is dropped, keeping the
	// blocking path DeepEqual-symmetric with Collect over the stream path.
	var dropped []llm.DroppedToolCall
	seenIDs := map[string]struct{}{}
	type candidate struct {
		position int
		call     rawToolCall
	}
	candidates := make([]candidate, 0, len(message.ToolCalls))
	for position, call := range message.ToolCalls {
		wirePosition := position
		if call.Index != nil && *call.Index >= 0 {
			wirePosition = *call.Index
		}
		candidates = append(candidates, candidate{position: wirePosition, call: call})
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		return candidates[i].position < candidates[j].position
	})
	for _, candidate := range candidates {
		index := toolBlockIndex(candidate.position)
		part, drop := mapToolCall(index, candidate.call, seenIDs)
		if drop != nil {
			dropped = append(dropped, *drop)
			continue
		}
		parts = append(parts, part)
	}
	return parts, dropped, nil
}

func mapToolCall(index int, call rawToolCall, seenIDs map[string]struct{}) (llm.ToolCallPart, *llm.DroppedToolCall) {
	if call.Function.Name == "" {
		return llm.ToolCallPart{}, &llm.DroppedToolCall{Index: index, Reason: "missing tool name"}
	}
	id := reserveToolCallID(index, call.ID, seenIDs)
	args := strings.TrimSpace(call.Function.Arguments)
	if args == "" {
		args = "{}"
	}
	if !json.Valid([]byte(args)) {
		return llm.ToolCallPart{}, &llm.DroppedToolCall{Index: index, Reason: "invalid tool arguments JSON"}
	}
	return llm.ToolCallPart{ID: id, Name: call.Function.Name, Args: json.RawMessage(args)}, nil
}

func reserveToolCallID(index int, candidate string, seenIDs map[string]struct{}) string {
	id := candidate
	if id == "" {
		id = providerutil.UniqueSyntheticToolCallID(index, seenIDs)
	}
	if _, exists := seenIDs[id]; exists {
		id = providerutil.UniqueSyntheticToolCallID(index, seenIDs)
	}
	seenIDs[id] = struct{}{}
	return id
}
