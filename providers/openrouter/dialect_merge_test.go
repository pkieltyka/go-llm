package openrouter

import (
	"reflect"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
)

// Pins the FS §14 unified-wins rule at the exact seam the v0.3 MapEffort
// change moved: the adapter pre-populates extras["reasoning"] from the
// unified Effort, and ApplyRequest merges Options.Reasoning beneath it.
func TestApplyRequestMergesEffortWithReasoningOptions(t *testing.T) {
	extras := chatcompletions.JSONObject{
		// As pre-populated by the adapter for Effort: high.
		"reasoning": map[string]any{"effort": "high"},
	}
	req := &llm.Request{
		Model:    "qwen/qwen3.6-27b",
		Messages: []llm.Message{llm.UserText("hi")},
		Effort:   llm.EffortHigh,
		ProviderOptions: Options{
			Reasoning: map[string]any{
				"exclude": true,
				// Conflicting key: the unified Effort must win (FS §14).
				"effort": "low",
			},
		},
	}
	if err := (dialect{}).ApplyRequest(req, nil, extras); err != nil {
		t.Fatalf("ApplyRequest: %v", err)
	}
	want := map[string]any{"effort": "high", "exclude": true}
	if got := extras["reasoning"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("merged reasoning = %#v, want %#v", got, want)
	}
}
