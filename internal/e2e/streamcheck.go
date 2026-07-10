package e2e

import (
	"fmt"
	"iter"

	llm "github.com/pkieltyka/go-llm"
)

type capturedStreamItem struct {
	event llm.Event
	err   error
}

// CollectLiveStream inspects the raw provider iterator before replaying the
// exact event/error sequence through llm.Collect. This makes live tests pin
// stream grammar instead of validating only the final folded response.
func CollectLiveStream(providerID string, events iter.Seq2[llm.Event, error]) (*llm.Response, error) {
	if providerID == "" {
		return nil, fmt.Errorf("live stream grammar: empty expected provider ID")
	}
	if events == nil {
		return nil, fmt.Errorf("live stream grammar: nil iterator")
	}
	items := make([]capturedStreamItem, 0, 32)
	for event, err := range events {
		items = append(items, capturedStreamItem{event: event, err: err})
	}
	grammarErr := validateLiveStreamGrammar(providerID, items)
	response, collectErr := llm.Collect(replayCapturedStream(items))
	if grammarErr != nil {
		if collectErr != nil {
			return response, fmt.Errorf("%v; collect: %w", grammarErr, collectErr)
		}
		return response, grammarErr
	}
	return response, collectErr
}

func replayCapturedStream(items []capturedStreamItem) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		for _, item := range items {
			if !yield(item.event, item.err) {
				return
			}
		}
	}
}

func validateLiveStreamGrammar(providerID string, items []capturedStreamItem) error {
	if len(items) == 0 {
		return fmt.Errorf("live stream grammar: empty stream")
	}
	started := false
	ended := false
	sawError := false
	activeTools := map[int]bool{}

	for index, item := range items {
		if sawError {
			return fmt.Errorf("live stream grammar: item %d followed a terminal error", index)
		}
		if item.err != nil {
			if item.event != nil {
				return fmt.Errorf("live stream grammar: item %d yielded an event and error together", index)
			}
			if ended {
				return fmt.Errorf("live stream grammar: error followed MessageEnd")
			}
			if index != len(items)-1 {
				return fmt.Errorf("live stream grammar: error at item %d was not terminal", index)
			}
			sawError = true
			continue
		}
		if item.event == nil {
			return fmt.Errorf("live stream grammar: nil event at item %d", index)
		}
		if ended {
			return fmt.Errorf("live stream grammar: event %T followed MessageEnd", item.event)
		}

		switch event := item.event.(type) {
		case llm.MessageStart:
			if index != 0 {
				return fmt.Errorf("live stream grammar: MessageStart at item %d, want first", index)
			}
			if started {
				return fmt.Errorf("live stream grammar: duplicate MessageStart")
			}
			if event.Provider != providerID {
				return fmt.Errorf("live stream grammar: MessageStart provider = %q, want %q", event.Provider, providerID)
			}
			started = true
		case *llm.MessageStart:
			if event == nil {
				return fmt.Errorf("live stream grammar: nil MessageStart at item %d", index)
			}
			if index != 0 || started {
				return fmt.Errorf("live stream grammar: invalid MessageStart at item %d", index)
			}
			if event.Provider != providerID {
				return fmt.Errorf("live stream grammar: MessageStart provider = %q, want %q", event.Provider, providerID)
			}
			started = true
		case llm.MessageEnd:
			if !started {
				return fmt.Errorf("live stream grammar: MessageEnd preceded MessageStart")
			}
			if len(activeTools) != 0 {
				return fmt.Errorf("live stream grammar: MessageEnd with %d active tool calls", len(activeTools))
			}
			if index != len(items)-1 {
				return fmt.Errorf("live stream grammar: MessageEnd at item %d was not terminal", index)
			}
			ended = true
		case *llm.MessageEnd:
			if event == nil {
				return fmt.Errorf("live stream grammar: nil MessageEnd at item %d", index)
			}
			if !started || len(activeTools) != 0 || index != len(items)-1 {
				return fmt.Errorf("live stream grammar: invalid MessageEnd at item %d", index)
			}
			ended = true
		default:
			if !started {
				return fmt.Errorf("live stream grammar: %T preceded MessageStart", item.event)
			}
			if err := validateLiveStreamPartEvent(item.event, activeTools); err != nil {
				return fmt.Errorf("live stream grammar at item %d: %w", index, err)
			}
		}
	}

	if sawError {
		return nil
	}
	if !started {
		return fmt.Errorf("live stream grammar: missing MessageStart")
	}
	if !ended {
		return fmt.Errorf("live stream grammar: missing MessageEnd")
	}
	return nil
}

func validateLiveStreamPartEvent(event llm.Event, activeTools map[int]bool) error {
	switch event := event.(type) {
	case llm.TextDelta:
		return validateLiveBlockIndex(event.Index)
	case *llm.TextDelta:
		if event == nil {
			return fmt.Errorf("nil TextDelta")
		}
		return validateLiveBlockIndex(event.Index)
	case llm.ReasoningDelta:
		return validateLiveBlockIndex(event.Index)
	case *llm.ReasoningDelta:
		if event == nil {
			return fmt.Errorf("nil ReasoningDelta")
		}
		return validateLiveBlockIndex(event.Index)
	case llm.ToolCallStart:
		return startLiveTool(activeTools, event.Index)
	case *llm.ToolCallStart:
		if event == nil {
			return fmt.Errorf("nil ToolCallStart")
		}
		return startLiveTool(activeTools, event.Index)
	case llm.ToolCallDelta:
		return requireLiveTool(activeTools, event.Index, "ToolCallDelta")
	case *llm.ToolCallDelta:
		if event == nil {
			return fmt.Errorf("nil ToolCallDelta")
		}
		return requireLiveTool(activeTools, event.Index, "ToolCallDelta")
	case llm.ToolCallIDChanged:
		return requireLiveTool(activeTools, event.Index, "ToolCallIDChanged")
	case *llm.ToolCallIDChanged:
		if event == nil {
			return fmt.Errorf("nil ToolCallIDChanged")
		}
		return requireLiveTool(activeTools, event.Index, "ToolCallIDChanged")
	case llm.ToolCallEnd:
		return endLiveTool(activeTools, event.Index, "ToolCallEnd")
	case *llm.ToolCallEnd:
		if event == nil {
			return fmt.Errorf("nil ToolCallEnd")
		}
		return endLiveTool(activeTools, event.Index, "ToolCallEnd")
	case llm.ToolCallDropped:
		return dropLiveTool(activeTools, event.Index)
	case *llm.ToolCallDropped:
		if event == nil {
			return fmt.Errorf("nil ToolCallDropped")
		}
		return dropLiveTool(activeTools, event.Index)
	default:
		return fmt.Errorf("unknown event %T", event)
	}
}

func validateLiveBlockIndex(index int) error {
	if index < 0 {
		return fmt.Errorf("negative block index %d", index)
	}
	return nil
}

func startLiveTool(activeTools map[int]bool, index int) error {
	if err := validateLiveBlockIndex(index); err != nil {
		return err
	}
	if activeTools[index] {
		return fmt.Errorf("duplicate ToolCallStart for index %d", index)
	}
	activeTools[index] = true
	return nil
}

func requireLiveTool(activeTools map[int]bool, index int, kind string) error {
	if !activeTools[index] {
		return fmt.Errorf("%s for inactive tool index %d", kind, index)
	}
	return nil
}

func endLiveTool(activeTools map[int]bool, index int, kind string) error {
	if err := requireLiveTool(activeTools, index, kind); err != nil {
		return err
	}
	delete(activeTools, index)
	return nil
}

func dropLiveTool(activeTools map[int]bool, index int) error {
	if err := validateLiveBlockIndex(index); err != nil {
		return err
	}
	delete(activeTools, index)
	return nil
}
