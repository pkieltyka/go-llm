package e2e

import (
	"context"
	"errors"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestOpenRouterReasoningProbeRetriesMissingStructuredEvidence(t *testing.T) {
	responses := []*llm.Response{
		{Parts: []llm.Part{llm.Text("unstructured thinking")}},
		{Parts: []llm.Part{llm.ReasoningPart{Provider: "other", Raw: []byte(`[{"type":"reasoning.text"}]`)}}},
		{Parts: []llm.Part{llm.ReasoningPart{Provider: "openrouter", Raw: []byte(`[{"type":"reasoning.text","text":"why"}]`)}}},
	}
	calls := 0
	resp, err := probeOpenRouterReasoningEvidence(context.Background(), 3, func(context.Context) (*llm.Response, error) {
		response := responses[calls]
		calls++
		return response, nil
	})
	if err != nil {
		t.Fatalf("probeOpenRouterReasoningEvidence returned error: %v", err)
	}
	if calls != 3 || resp != responses[2] {
		t.Fatalf("calls=%d response=%+v", calls, resp)
	}
}

func TestOpenRouterReasoningProbeFailsAfterBound(t *testing.T) {
	calls := 0
	resp, err := probeOpenRouterReasoningEvidence(context.Background(), 3, func(context.Context) (*llm.Response, error) {
		calls++
		return &llm.Response{
			Parts:      []llm.Part{llm.Text("plain response")},
			StopReason: llm.StopReasonEndTurn,
		}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "no structured reasoning after 3 attempts") {
		t.Fatalf("error = %v", err)
	}
	if calls != 3 || resp == nil {
		t.Fatalf("calls=%d response=%+v", calls, resp)
	}
}

func TestOpenRouterReasoningProbeFailsFastOnProviderError(t *testing.T) {
	wantErr := errors.New("upstream failed")
	calls := 0
	_, err := probeOpenRouterReasoningEvidence(context.Background(), 3, func(context.Context) (*llm.Response, error) {
		calls++
		return nil, wantErr
	})
	if !errors.Is(err, wantErr) || calls != 1 {
		t.Fatalf("error=%v calls=%d", err, calls)
	}
}
