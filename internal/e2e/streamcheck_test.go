package e2e

import (
	"errors"
	"iter"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestCollectLiveStreamValidatesAndCollectsRawGrammar(t *testing.T) {
	response, err := CollectLiveStream("test", streamItems(
		capturedStreamItem{event: llm.MessageStart{ID: "msg", Provider: "test", Model: "model"}},
		capturedStreamItem{event: llm.TextDelta{Index: 0, Text: "ok"}},
		capturedStreamItem{event: llm.ToolCallStart{Index: 3, ID: "call", Name: "lookup"}},
		capturedStreamItem{event: llm.ToolCallDelta{Index: 3, ArgsFragment: `{}`}},
		capturedStreamItem{event: llm.ToolCallEnd{Index: 3}},
		capturedStreamItem{event: llm.MessageEnd{StopReason: llm.StopReasonEndTurn}},
	))
	if err != nil {
		t.Fatalf("CollectLiveStream returned error: %v", err)
	}
	if response == nil || response.ID != "msg" || response.Text() != "ok" || len(response.ToolCalls()) != 1 {
		t.Fatalf("response = %+v", response)
	}
}

func TestCollectLiveStreamRejectsMalformedGrammar(t *testing.T) {
	tests := []struct {
		name  string
		items []capturedStreamItem
		want  string
	}{
		{
			name: "empty",
			want: "empty stream",
		},
		{
			name: "content before start",
			items: []capturedStreamItem{
				{event: llm.TextDelta{Index: 0, Text: "bad"}},
				{event: llm.MessageEnd{}},
			},
			want: "preceded MessageStart",
		},
		{
			name: "missing end",
			items: []capturedStreamItem{
				{event: llm.MessageStart{Provider: "test"}},
				{event: llm.TextDelta{Index: 0, Text: "partial"}},
			},
			want: "missing MessageEnd",
		},
		{
			name: "event after end",
			items: []capturedStreamItem{
				{event: llm.MessageStart{Provider: "test"}},
				{event: llm.MessageEnd{}},
				{event: llm.TextDelta{Index: 0, Text: "late"}},
			},
			want: "not terminal",
		},
		{
			name: "tool delta before start",
			items: []capturedStreamItem{
				{event: llm.MessageStart{Provider: "test"}},
				{event: llm.ToolCallDelta{Index: 2, ArgsFragment: "{}"}},
				{event: llm.MessageEnd{}},
			},
			want: "inactive tool",
		},
		{
			name: "active tool at end",
			items: []capturedStreamItem{
				{event: llm.MessageStart{Provider: "test"}},
				{event: llm.ToolCallStart{Index: 2, ID: "call", Name: "tool"}},
				{event: llm.MessageEnd{}},
			},
			want: "active tool",
		},
		{
			name: "event after error",
			items: []capturedStreamItem{
				{event: llm.MessageStart{Provider: "test"}},
				{err: errors.New("remote")},
				{event: llm.TextDelta{Index: 0, Text: "late"}},
			},
			want: "not terminal",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := CollectLiveStream("test", streamItems(test.items...))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want substring %q", err, test.want)
			}
		})
	}
}

func TestCollectLiveStreamPreservesTerminalProviderErrorAndPartialResponse(t *testing.T) {
	providerErr := errors.New("remote stream failed")
	response, err := CollectLiveStream("test", streamItems(
		capturedStreamItem{event: llm.MessageStart{ID: "partial", Provider: "test"}},
		capturedStreamItem{event: llm.TextDelta{Index: 0, Text: "safe"}},
		capturedStreamItem{err: providerErr},
	))
	if !errors.Is(err, providerErr) {
		t.Fatalf("error = %v, want provider error", err)
	}
	if response == nil || response.ID != "partial" || response.Text() != "safe" {
		t.Fatalf("partial response = %+v", response)
	}
}

func TestCollectLiveStreamRejectsWrongProviderID(t *testing.T) {
	_, err := CollectLiveStream("openai", streamItems(
		capturedStreamItem{event: llm.MessageStart{Provider: "openrouter"}},
		capturedStreamItem{event: llm.MessageEnd{}},
	))
	if err == nil || !strings.Contains(err.Error(), "want \"openai\"") {
		t.Fatalf("error = %v", err)
	}
}

func streamItems(items ...capturedStreamItem) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		for _, item := range items {
			if !yield(item.event, item.err) {
				return
			}
		}
	}
}
