package llmtest

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

type testProviderOptions struct {
	Value string
}

func (*testProviderOptions) ForProvider() string { return "test" }

func TestProvider(t *testing.T) {
	p := New(
		WithName("fake"),
		WithCapabilities(llm.CapabilityStreaming),
		WithModels(llm.ModelInfo{ID: "model-a", Pricing: &llm.ModelPricing{InputPerMTok: 1}, Raw: map[string]any{"nested": []any{map[string]any{"value": "original"}}}}),
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
	models[0].Raw.(map[string]any)["nested"].([]any)[0].(map[string]any)["value"] = "mutated"
	models, _ = p.Models(context.Background())
	if models[0].Pricing.InputPerMTok != 1 {
		t.Fatalf("Models did not return defensive pricing copy")
	}
	if got := models[0].Raw.(map[string]any)["nested"].([]any)[0].(map[string]any)["value"]; got != "original" {
		t.Fatalf("Models did not return defensive Raw copy: %v", got)
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

func TestProviderRequestCopiesAreDeepAndValueNormalized(t *testing.T) {
	temperature := 0.25
	topP := 0.75
	messagePart := &llm.TextPart{Text: "original", Cache: &llm.CacheHint{}}
	toolSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"items": []any{map[string]any{"type": "string"}},
		},
	}
	formatSchema := map[string]any{
		"enum": []any{map[string]any{"value": json.RawMessage(`{"ok":true}`)}},
	}
	providerOptions := &testProviderOptions{Value: "immutable"}
	req := &llm.Request{
		Model:           "model-a",
		Messages:        []llm.Message{{Role: llm.RoleUser, Parts: []llm.Part{messagePart}}},
		Temperature:     &temperature,
		TopP:            &topP,
		Tools:           []llm.Tool{{Name: "lookup", InputSchema: toolSchema}},
		ResponseFormat:  &llm.ResponseFormat{Type: llm.FormatJSONSchema, Schema: formatSchema},
		ProviderOptions: providerOptions,
	}

	p := New()
	p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text("ok")}})
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}

	temperature = 1
	topP = 1
	messagePart.Text = "mutated source"
	toolSchema["type"] = "array"
	toolSchema["properties"].(map[string]any)["items"].([]any)[0].(map[string]any)["type"] = "number"
	formatSchema["enum"].([]any)[0].(map[string]any)["value"].(json.RawMessage)[0] = '['

	got := p.Requests()[0]
	if got.Temperature == req.Temperature || got.TopP == req.TopP {
		t.Fatal("scalar option pointers were not cloned")
	}
	if *got.Temperature != 0.25 || *got.TopP != 0.75 {
		t.Fatalf("recorded Temperature/TopP = %v/%v, want 0.25/0.75", *got.Temperature, *got.TopP)
	}
	part, ok := got.Messages[0].Parts[0].(llm.TextPart)
	if !ok || part.Text != "original" {
		t.Fatalf("recorded part = %T %+v, want value TextPart(original)", got.Messages[0].Parts[0], got.Messages[0].Parts[0])
	}
	gotToolType := got.Tools[0].InputSchema.(map[string]any)["properties"].(map[string]any)["items"].([]any)[0].(map[string]any)["type"]
	if gotToolType != "string" {
		t.Fatalf("recorded nested tool schema type = %v, want string", gotToolType)
	}
	gotRaw := got.ResponseFormat.Schema.(map[string]any)["enum"].([]any)[0].(map[string]any)["value"].(json.RawMessage)
	if string(gotRaw) != `{"ok":true}` {
		t.Fatalf("recorded nested response schema raw = %s", gotRaw)
	}
	if got.ProviderOptions != providerOptions {
		t.Fatal("opaque ProviderOptions should be shallow-copied")
	}

	*got.Temperature = 0.9
	got.Tools[0].InputSchema.(map[string]any)["properties"].(map[string]any)["items"].([]any)[0].(map[string]any)["type"] = "boolean"
	got.Messages[0].Parts[0] = llm.Text("mutated copy")
	again := p.Requests()[0]
	if *again.Temperature != 0.25 {
		t.Fatalf("mutating Requests result changed stored Temperature to %v", *again.Temperature)
	}
	if nested := again.Tools[0].InputSchema.(map[string]any)["properties"].(map[string]any)["items"].([]any)[0].(map[string]any)["type"]; nested != "string" {
		t.Fatalf("mutating Requests result changed stored schema to %v", nested)
	}
	if text := again.Messages[0].Parts[0].(llm.TextPart).Text; text != "original" {
		t.Fatalf("mutating Requests result changed stored part to %q", text)
	}
}

func TestProviderStreamCopiesPointerEventsAsValues(t *testing.T) {
	start := &llm.MessageStart{ID: "msg_1", Provider: "fake", Model: "model-a"}
	reasoning := &llm.ReasoningDelta{Index: 0, Text: "think", Raw: json.RawMessage(`{"type":"reasoning"}`), Provider: "fake"}
	toolStart := &llm.ToolCallStart{Index: 1, ID: "temporary", Name: "lookup"}
	idChanged := &llm.ToolCallIDChanged{Index: 1, OldID: "temporary", NewID: "call_1"}
	delta := &llm.ToolCallDelta{Index: 1, ArgsFragment: `{"q":"go"}`}
	toolEnd := &llm.ToolCallEnd{Index: 1}
	cost := 0.01
	end := &llm.MessageEnd{StopReason: llm.StopReasonToolUse, Usage: llm.Usage{CostUSD: &cost}}

	p := New()
	p.EnqueueStream(start, reasoning, toolStart, idChanged, delta, toolEnd, end)
	start.ID = "mutated"
	reasoning.Raw[0] = '['
	idChanged.NewID = "mutated"
	delta.ArgsFragment = "{}"
	cost = 1

	var seen []llm.Event
	seq := func(yield func(llm.Event, error) bool) {
		for event, err := range p.ChatStream(context.Background(), &llm.Request{Model: "model-a"}) {
			if err == nil {
				seen = append(seen, event)
			}
			if !yield(event, err) {
				return
			}
		}
	}
	resp, err := llm.Collect(seq)
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	for i, event := range seen {
		if typ := reflect.TypeOf(event); typ == nil || typ.Kind() == reflect.Pointer {
			t.Fatalf("event %d = %T, want non-pointer value", i, event)
		}
	}
	if seen[0].(llm.MessageStart).ID != "msg_1" {
		t.Fatalf("MessageStart ID = %q, want msg_1", seen[0].(llm.MessageStart).ID)
	}
	changed, ok := seen[3].(llm.ToolCallIDChanged)
	if !ok || changed.OldID != "temporary" || changed.NewID != "call_1" {
		t.Fatalf("ToolCallIDChanged = %T %+v", seen[3], seen[3])
	}
	if got := resp.ToolCalls(); len(got) != 1 || got[0].ID != "call_1" || string(got[0].Args) != `{"q":"go"}` {
		t.Fatalf("collected tool calls = %+v", got)
	}
	if reasoningPart, ok := resp.Parts[0].(llm.ReasoningPart); !ok || string(reasoningPart.Raw) != `{"type":"reasoning"}` {
		t.Fatalf("collected reasoning = %T %+v", resp.Parts[0], resp.Parts[0])
	}
	if resp.Usage.CostUSD == nil || *resp.Usage.CostUSD != 0.01 {
		t.Fatalf("collected cost = %v, want 0.01", resp.Usage.CostUSD)
	}
}

func TestProviderResponseCopiesPointerPartsAsValues(t *testing.T) {
	text := &llm.TextPart{Text: "text", Cache: &llm.CacheHint{}}
	image := &llm.ImagePart{Data: []byte{1, 2}, MediaType: "image/png", Cache: &llm.CacheHint{}}
	file := &llm.FilePart{Data: []byte{3, 4}, MediaType: "text/plain", Name: "a.txt", Cache: &llm.CacheHint{}}
	call := &llm.ToolCallPart{ID: "call_1", Name: "lookup", Args: json.RawMessage(`{"q":"go"}`)}
	resultText := &llm.TextPart{Text: "result"}
	result := &llm.ToolResultPart{ToolCallID: "call_1", Content: []llm.Part{resultText}}
	reasoning := &llm.ReasoningPart{Text: "think", Raw: json.RawMessage(`{"type":"thinking"}`), Provider: "fake"}
	unknown := &llm.UnknownPart{Type: "future", Data: json.RawMessage(`{"type":"future","value":1}`)}

	p := New()
	p.EnqueueResponse(&llm.Response{Parts: []llm.Part{text, image, file, call, result, reasoning, unknown}})
	text.Text = "mutated"
	image.Data[0] = 9
	file.Data[0] = 9
	call.Args[0] = '['
	resultText.Text = "mutated"
	reasoning.Raw[0] = '['
	unknown.Data[0] = '['

	resp, err := p.Chat(context.Background(), &llm.Request{Model: "model-a"})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	for i, part := range resp.Parts {
		if typ := reflect.TypeOf(part); typ == nil || typ.Kind() == reflect.Pointer {
			t.Fatalf("part %d = %T, want non-pointer value", i, part)
		}
	}
	if resp.Parts[0].(llm.TextPart).Text != "text" || resp.Parts[1].(llm.ImagePart).Data[0] != 1 || resp.Parts[2].(llm.FilePart).Data[0] != 3 {
		t.Fatalf("pointer part aliases leaked into response: %+v", resp.Parts[:3])
	}
	if string(resp.Parts[3].(llm.ToolCallPart).Args) != `{"q":"go"}` {
		t.Fatalf("tool args = %s", resp.Parts[3].(llm.ToolCallPart).Args)
	}
	nested := resp.Parts[4].(llm.ToolResultPart).Content[0]
	if reflect.TypeOf(nested).Kind() == reflect.Pointer || nested.(llm.TextPart).Text != "result" {
		t.Fatalf("nested result part = %T %+v", nested, nested)
	}
	if string(resp.Parts[5].(llm.ReasoningPart).Raw) != `{"type":"thinking"}` || string(resp.Parts[6].(llm.UnknownPart).Data) != `{"type":"future","value":1}` {
		t.Fatalf("raw pointer part aliases leaked into response: %+v", resp.Parts[5:])
	}
}

func TestProviderCancellationAfterFirstEvent(t *testing.T) {
	p := New()
	p.EnqueueStream(
		&llm.MessageStart{ID: "msg_1", Provider: "fake", Model: "model-a"},
		&llm.TextDelta{Index: 0, Text: "must not be yielded"},
		&llm.MessageEnd{StopReason: llm.StopReasonEndTurn},
	)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var events []llm.Event
	var terminal error
	for event, err := range p.ChatStream(ctx, &llm.Request{Model: "model-a"}) {
		if err != nil {
			terminal = err
			continue
		}
		events = append(events, event)
		cancel()
	}
	if !errors.Is(terminal, context.Canceled) {
		t.Fatalf("terminal error = %v, want context.Canceled", terminal)
	}
	if len(events) != 1 {
		t.Fatalf("events after cancellation = %+v, want one MessageStart", events)
	}
	if _, ok := events[0].(llm.MessageStart); !ok {
		t.Fatalf("first event = %T, want llm.MessageStart", events[0])
	}
}
