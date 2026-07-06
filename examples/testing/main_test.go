// Hermetic unit tests for code that consumes go-llm, using the llmtest fake:
// enqueue scripted responses, call your function, then assert on both the
// output and the requests your code actually sent.
package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

func TestSummarize(t *testing.T) {
	ctx := context.Background()
	p := llmtest.New()
	p.EnqueueResponse(&llm.Response{
		Provider: "llmtest",
		Model:    "test-model",
		Parts:    []llm.Part{llm.Text("Go gained generics in 1.18.")},
	})

	got, err := Summarize(ctx, p, "test-model", "Go 1.18 added generics via type parameters.")
	if err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	if want := "Go gained generics in 1.18."; got != want {
		t.Fatalf("Summarize = %q, want %q", got, want)
	}

	// llmtest records every request, so tests can assert on exactly what the
	// code under test sent to the provider.
	reqs := p.Requests()
	if len(reqs) != 1 {
		t.Fatalf("recorded %d requests, want 1", len(reqs))
	}
	req := reqs[0]
	if req.Model != "test-model" {
		t.Errorf("request model = %q, want %q", req.Model, "test-model")
	}
	if !strings.Contains(req.System, "one short sentence") {
		t.Errorf("system prompt = %q, missing summary instruction", req.System)
	}
	if len(req.Messages) != 1 || req.Messages[0].Role != llm.RoleUser {
		t.Fatalf("messages = %+v, want a single user message", req.Messages)
	}
}

func TestStreamSummary(t *testing.T) {
	ctx := context.Background()
	p := llmtest.New()
	p.EnqueueStream(
		llm.MessageStart{Provider: "llmtest", Model: "test-model"},
		llm.TextDelta{Index: 0, Text: "Go gained "},
		llm.TextDelta{Index: 0, Text: "generics in 1.18."},
		llm.MessageEnd{StopReason: llm.StopReasonEndTurn},
	)

	var buf strings.Builder
	if err := StreamSummary(ctx, p, "test-model", "Go 1.18 added generics.", &buf); err != nil {
		t.Fatalf("StreamSummary: %v", err)
	}
	if want := "Go gained generics in 1.18."; buf.String() != want {
		t.Fatalf("StreamSummary wrote %q, want %q", buf.String(), want)
	}
}

func TestSummarizeProviderError(t *testing.T) {
	ctx := context.Background()
	p := llmtest.New()
	p.EnqueueError(fmt.Errorf("%w: slow down", llm.ErrRateLimited))

	_, err := Summarize(ctx, p, "test-model", "anything")
	if !errors.Is(err, llm.ErrRateLimited) {
		t.Fatalf("Summarize error = %v, want llm.ErrRateLimited", err)
	}
}
