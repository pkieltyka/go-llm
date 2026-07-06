package llmtest

import (
	"context"
	"errors"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestProvider(t *testing.T) {
	p := New(
		WithName("fake"),
		WithCapabilities(llm.CapabilityStreaming),
		WithModels(llm.ModelInfo{ID: "model-a", Pricing: &llm.ModelPricing{InputPerMTok: 1}}),
	)
	p.EnqueueResponse(&llm.Response{
		Provider: "fake",
		Model:    "model-a",
		Parts:    []llm.Part{llm.ToolCall("call_1", "lookup", []byte(`{"q":"go"}`))},
	})
	p.EnqueueStream(
		llm.MessageStart{Provider: "fake", Model: "model-a"},
		llm.TextDelta{Index: 0, Text: "hello"},
		llm.MessageEnd{StopReason: llm.StopReasonEndTurn},
	)
	p.EnqueueError(llm.ErrRateLimited)

	if p.Name() != "fake" {
		t.Fatalf("Name = %q", p.Name())
	}
	if caps := p.Capabilities(); len(caps) != 1 || caps[0] != llm.CapabilityStreaming {
		t.Fatalf("Capabilities = %+v", caps)
	}
	models, err := p.Models(context.Background())
	if err != nil || len(models) != 1 || models[0].Pricing == nil {
		t.Fatalf("Models = %+v, %v", models, err)
	}
	models[0].Pricing.InputPerMTok = 99
	models, _ = p.Models(context.Background())
	if models[0].Pricing.InputPerMTok != 1 {
		t.Fatalf("Models did not return defensive pricing copy")
	}

	resp, err := p.Chat(context.Background(), &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("hi")}})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	resp.ToolCalls()[0].Args[0] = '['
	streamResp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("stream")}}))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if streamResp.Text() != "hello" {
		t.Fatalf("stream text = %q", streamResp.Text())
	}

	_, err = p.Chat(context.Background(), &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("err")}})
	if !errors.Is(err, llm.ErrRateLimited) {
		t.Fatalf("error = %v, want ErrRateLimited", err)
	}

	requests := p.Requests()
	if len(requests) != 3 {
		t.Fatalf("requests len = %d, want 3", len(requests))
	}
	requests[0].Messages[0].Parts[0] = llm.Text("mutated")
	again := p.Requests()
	if got := again[0].Messages[0].Parts[0].(llm.TextPart).Text; got != "hi" {
		t.Fatalf("recorded request mutated to %q", got)
	}
}

func TestProviderCanceledStreamStillConsumesIterator(t *testing.T) {
	p := New()
	p.EnqueueStream(llm.TextDelta{Index: 0, Text: "unread"})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	stream := p.ChatStream(ctx, &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("stream")}})
	resp, err := llm.Collect(stream)
	if resp != nil {
		t.Fatalf("canceled response = %+v, want nil", resp)
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Collect error = %v, want context.Canceled", err)
	}

	_, err = llm.Collect(stream)
	if !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("second Collect error = %v, want ErrBadRequest", err)
	}
}
