package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"iter"
	"os"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

func TestRunChatSavesConversation(t *testing.T) {
	dir := t.TempDir()
	loadPath := dir + "/load.json"
	savePath := dir + "/save.json"
	loaded := []llm.Message{llm.UserText("earlier"), llm.AssistantText("prior")}
	data, err := llm.MarshalMessages(loaded)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(loadPath, data, 0o600); err != nil {
		t.Fatal(err)
	}

	fake := llmtest.New(llmtest.WithName("llmtest"))
	fake.EnqueueResponse(&llm.Response{
		Provider: "llmtest",
		Model:    "model-1",
		Parts:    []llm.Part{llm.Text("answer")},
		Usage:    llm.Usage{InputTokens: 2, OutputTokens: 3, TotalTokens: 5},
	})
	var stdout, stderr bytes.Buffer
	a := testApp(fake, &stdout, &stderr)
	err = a.runChat(context.Background(), chatConfig{
		provider:  "llmtest",
		model:     "model-1",
		noStream:  true,
		usage:     true,
		loadPath:  loadPath,
		savePath:  savePath,
		args:      []string{"current"},
		sessionID: "s1",
		maxTokens: 42,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := stdout.String(); got != "answer" {
		t.Fatalf("stdout = %q", got)
	}
	if got := stderr.String(); !strings.Contains(got, "usage input=2 output=3 total=5") {
		t.Fatalf("stderr missing usage: %q", got)
	}
	requests := fake.Requests()
	if len(requests) != 1 {
		t.Fatalf("requests len = %d, want 1", len(requests))
	}
	if len(requests[0].Messages) != 3 {
		t.Fatalf("request messages len = %d, want 3", len(requests[0].Messages))
	}
	if requests[0].SessionID != "s1" || requests[0].MaxTokens != 42 {
		t.Fatalf("request fields not forwarded: %+v", requests[0])
	}

	savedData, err := os.ReadFile(savePath)
	if err != nil {
		t.Fatal(err)
	}
	saved, err := llm.UnmarshalMessages(savedData)
	if err != nil {
		t.Fatal(err)
	}
	if len(saved) != 4 {
		t.Fatalf("saved messages len = %d, want 4", len(saved))
	}
	if got := saved[2].Parts[0].(llm.TextPart).Text; got != "current" {
		t.Fatalf("saved user text = %q", got)
	}
	if got := saved[3].Parts[0].(llm.TextPart).Text; got != "answer" {
		t.Fatalf("saved assistant text = %q", got)
	}
	if saved[3].Provider != "llmtest" || saved[3].Model != "model-1" {
		t.Fatalf("saved provenance missing: %+v", saved[3])
	}
}

func TestRunChatJSONAndNoStream(t *testing.T) {
	fake := llmtest.New()
	fake.EnqueueResponse(&llm.Response{
		ID:       "resp-1",
		Provider: "llmtest",
		Model:    "model-1",
		Parts:    []llm.Part{llm.Text("json answer")},
	})
	var stdout, stderr bytes.Buffer
	a := testApp(fake, &stdout, &stderr)
	if err := a.runChat(context.Background(), chatConfig{
		provider:   "llmtest",
		model:      "model-1",
		jsonOutput: true,
		args:       []string{"prompt"},
	}); err != nil {
		t.Fatal(err)
	}
	resp, err := llm.UnmarshalResponse(bytes.TrimSpace(stdout.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if resp.ID != "resp-1" || resp.Text() != "json answer" {
		t.Fatalf("unexpected JSON response: %+v", resp)
	}

	fake.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text("plain answer")}})
	stdout.Reset()
	if err := a.runChat(context.Background(), chatConfig{
		provider: "llmtest",
		model:    "model-1",
		noStream: true,
		args:     []string{"prompt"},
	}); err != nil {
		t.Fatal(err)
	}
	if got := stdout.String(); got != "plain answer" {
		t.Fatalf("no-stream stdout = %q", got)
	}
}

func TestRunChatSchemaValidatesBeforeOutput(t *testing.T) {
	dir := t.TempDir()
	schemaPath := dir + "/answer.json"
	if err := os.WriteFile(schemaPath, []byte(`{"type":"object","required":["answer"],"properties":{"answer":{"type":"string"}},"additionalProperties":false}`), 0o600); err != nil {
		t.Fatal(err)
	}

	fake := llmtest.New()
	fake.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(`{
		"answer": "ok"
	}`)}})
	var stdout, stderr bytes.Buffer
	a := testApp(fake, &stdout, &stderr)
	if err := a.runChat(context.Background(), chatConfig{
		provider:   "llmtest",
		model:      "model-1",
		schemaPath: schemaPath,
		args:       []string{"prompt"},
	}); err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(stdout.String()); got != `{"answer":"ok"}` {
		t.Fatalf("validated stdout = %q", got)
	}

	fake.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(`{"nope":"bad"}`)}})
	stdout.Reset()
	err := a.runChat(context.Background(), chatConfig{
		provider:   "llmtest",
		model:      "model-1",
		schemaPath: schemaPath,
		args:       []string{"prompt"},
	})
	if err == nil {
		t.Fatal("expected validation error")
	}
	if got := stdout.String(); got != "" {
		t.Fatalf("stdout should stay empty on invalid structured output, got %q", got)
	}
	if !strings.Contains(err.Error(), "structured output validation failed") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunModelsOutput(t *testing.T) {
	price := &llm.ModelPricing{InputPerMTok: 1.25, OutputPerMTok: 2.5}
	fake := llmtest.New(llmtest.WithModels(llm.ModelInfo{
		ID:              "model-1",
		DisplayName:     "Model One",
		ContextWindow:   1000,
		MaxOutputTokens: 200,
		Pricing:         price,
	}))
	var stdout, stderr bytes.Buffer
	a := testApp(fake, &stdout, &stderr)
	if err := a.runModels(context.Background(), modelsConfig{provider: "llmtest"}); err != nil {
		t.Fatal(err)
	}
	got := stdout.String()
	for _, want := range []string{"ID", "model-1", "Model One", "1.25", "2.5"} {
		if !strings.Contains(got, want) {
			t.Fatalf("table output missing %q: %q", want, got)
		}
	}

	stdout.Reset()
	if err := a.runModels(context.Background(), modelsConfig{provider: "llmtest", jsonOutput: true}); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"id": "model-1"`, `"input_per_mtok": "1.25"`} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("JSON output missing %q: %q", want, stdout.String())
		}
	}
}

func TestRunChatStreamsDeltasInOrder(t *testing.T) {
	fake := llmtest.New(llmtest.WithName("llmtest"))
	fake.EnqueueStream(
		llm.MessageStart{ID: "msg-1", Provider: "llmtest", Model: "model-1"},
		llm.TextDelta{Index: 0, Text: "Hel"},
		llm.TextDelta{Index: 0, Text: "lo "},
		llm.TextDelta{Index: 0, Text: "world"},
		llm.MessageEnd{StopReason: llm.StopReasonEndTurn, Usage: llm.Usage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3}},
	)
	var stdout, stderr bytes.Buffer
	a := testApp(fake, &stdout, &stderr)
	if err := a.runChat(context.Background(), chatConfig{
		provider: "llmtest",
		model:    "model-1",
		usage:    true,
		args:     []string{"prompt"},
	}); err != nil {
		t.Fatal(err)
	}
	if got := stdout.String(); got != "Hello world" {
		t.Fatalf("stdout = %q, want %q", got, "Hello world")
	}
	if got := stderr.String(); !strings.Contains(got, "usage input=1 output=2 total=3") {
		t.Fatalf("stderr missing usage from collected stream: %q", got)
	}
}

func TestRunChatStreamReasoningMirroredToStderr(t *testing.T) {
	streamEvents := []llm.Event{
		llm.MessageStart{ID: "msg-1", Provider: "llmtest", Model: "model-1"},
		llm.ReasoningDelta{Index: 0, Text: "step one, "},
		llm.ReasoningDelta{Index: 0, Text: "step two"},
		llm.TextDelta{Index: 1, Text: "answer"},
		llm.MessageEnd{StopReason: llm.StopReasonEndTurn},
	}

	fake := llmtest.New(llmtest.WithName("llmtest"))
	fake.EnqueueStream(streamEvents...)
	var stdout, stderr bytes.Buffer
	a := testApp(fake, &stdout, &stderr)
	if err := a.runChat(context.Background(), chatConfig{
		provider:  "llmtest",
		model:     "model-1",
		reasoning: true,
		args:      []string{"prompt"},
	}); err != nil {
		t.Fatal(err)
	}
	if got := stdout.String(); got != "answer" {
		t.Fatalf("stdout = %q, want %q", got, "answer")
	}
	if got := stderr.String(); got != "step one, step two" {
		t.Fatalf("stderr = %q, want mirrored reasoning", got)
	}

	// Without --reasoning the deltas must not leak anywhere.
	fake.EnqueueStream(streamEvents...)
	stdout.Reset()
	stderr.Reset()
	if err := a.runChat(context.Background(), chatConfig{
		provider: "llmtest",
		model:    "model-1",
		args:     []string{"prompt"},
	}); err != nil {
		t.Fatal(err)
	}
	if got := stdout.String(); got != "answer" {
		t.Fatalf("stdout = %q, want %q", got, "answer")
	}
	if got := stderr.String(); got != "" {
		t.Fatalf("stderr = %q, want empty without --reasoning", got)
	}
}

// scriptedStreamProvider yields its events then a terminal error, covering
// mid-stream failures llmtest cannot script.
type scriptedStreamProvider struct {
	events []llm.Event
	err    error
}

func (p *scriptedStreamProvider) Name() string                   { return "scripted" }
func (p *scriptedStreamProvider) Capabilities() []llm.Capability { return nil }
func (p *scriptedStreamProvider) Models(context.Context) ([]llm.ModelInfo, error) {
	return nil, nil
}
func (p *scriptedStreamProvider) Chat(context.Context, *llm.Request) (*llm.Response, error) {
	return nil, p.err
}
func (p *scriptedStreamProvider) ChatStream(context.Context, *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		for _, event := range p.events {
			if !yield(event, nil) {
				return
			}
		}
		yield(nil, p.err)
	}
}

func TestRunChatStreamMidStreamErrorFlushesPartialText(t *testing.T) {
	streamErr := &llm.ProviderError{Provider: "scripted", Code: "overloaded", Message: "boom", Kind: llm.ErrServer}
	p := &scriptedStreamProvider{
		events: []llm.Event{
			llm.MessageStart{ID: "msg-1", Provider: "scripted", Model: "model-1"},
			llm.TextDelta{Index: 0, Text: "partial "},
			llm.TextDelta{Index: 0, Text: "text"},
		},
		err: streamErr,
	}
	var stdout, stderr bytes.Buffer
	a := testApp(p, &stdout, &stderr)
	err := a.runChat(context.Background(), chatConfig{
		provider: "scripted",
		model:    "model-1",
		args:     []string{"prompt"},
	})
	if !errors.Is(err, llm.ErrServer) || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("err = %v, want the provider stream error", err)
	}
	// Deltas were flushed unbuffered before the error arrived; main prints
	// the returned error to stderr and exits 1.
	if got := stdout.String(); got != "partial text" {
		t.Fatalf("stdout = %q, want partial text flushed", got)
	}
}

func TestRunChatStreamProviderErrorBeatsCollectError(t *testing.T) {
	streamErr := errors.New("provider stream error")
	p := &scriptedStreamProvider{
		events: []llm.Event{
			llm.MessageStart{ID: "msg-1", Provider: "scripted", Model: "model-1"},
			llm.TextDelta{Index: 0, Text: "x"},
			nil, // makes the re-collect over partial events fail
		},
		err: streamErr,
	}
	var stdout, stderr bytes.Buffer
	a := testApp(p, &stdout, &stderr)
	err := a.runChat(context.Background(), chatConfig{
		provider: "scripted",
		model:    "model-1",
		args:     []string{"prompt"},
	})
	if !errors.Is(err, streamErr) {
		t.Fatalf("err = %v, want the provider error to take precedence over the collect error", err)
	}
}

func TestRunChatStreamPrintsToolCalls(t *testing.T) {
	fake := llmtest.New(llmtest.WithName("llmtest"))
	fake.EnqueueStream(
		llm.MessageStart{ID: "msg-1", Provider: "llmtest", Model: "model-1"},
		llm.TextDelta{Index: 0, Text: "calling"},
		llm.ToolCallStart{Index: 1, ID: "call_1", Name: "lookup"},
		llm.ToolCallDelta{Index: 1, ArgsFragment: `{"q":`},
		llm.ToolCallDelta{Index: 1, ArgsFragment: `"go"}`},
		llm.ToolCallEnd{Index: 1},
		llm.MessageEnd{StopReason: llm.StopReasonToolUse},
	)
	var stdout, stderr bytes.Buffer
	a := testApp(fake, &stdout, &stderr)
	if err := a.runChat(context.Background(), chatConfig{
		provider: "llmtest",
		model:    "model-1",
		args:     []string{"prompt"},
	}); err != nil {
		t.Fatal(err)
	}
	got := stdout.String()
	if !strings.HasPrefix(got, "calling\n") {
		t.Fatalf("stdout should stream text before tool calls: %q", got)
	}
	var printed []llm.ToolCallPart
	if err := json.Unmarshal([]byte(strings.TrimPrefix(got, "calling")), &printed); err != nil {
		t.Fatalf("tool call output is not a JSON array: %v\n%q", err, got)
	}
	if len(printed) != 1 || printed[0].ID != "call_1" || printed[0].Name != "lookup" {
		t.Fatalf("printed tool calls = %+v", printed)
	}
	if got := compactJSON(t, printed[0].Args); got != `{"q":"go"}` {
		t.Fatalf("printed args = %s, want %s", got, `{"q":"go"}`)
	}
}

func compactJSON(t *testing.T, raw json.RawMessage) string {
	t.Helper()
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		t.Fatalf("invalid JSON %q: %v", raw, err)
	}
	return buf.String()
}

func TestPrintToolCallsShape(t *testing.T) {
	var buf bytes.Buffer
	calls := []llm.ToolCallPart{
		{ID: "call_1", Name: "lookup", Args: json.RawMessage(`{"q":"go"}`)},
		{ID: "call_2", Name: "fetch", Args: json.RawMessage(`{}`)},
	}
	if err := printToolCalls(&buf, calls); err != nil {
		t.Fatal(err)
	}
	got := buf.String()
	if !strings.HasPrefix(got, "\n[") || !strings.HasSuffix(got, "]\n") {
		t.Fatalf("output should be a blank line then an indented JSON array: %q", got)
	}
	var round []llm.ToolCallPart
	if err := json.Unmarshal([]byte(got), &round); err != nil {
		t.Fatalf("output is not valid JSON: %v", err)
	}
	if len(round) != len(calls) {
		t.Fatalf("round-tripped %d calls, want %d", len(round), len(calls))
	}
	for i := range calls {
		if round[i].ID != calls[i].ID || round[i].Name != calls[i].Name {
			t.Fatalf("call %d = %+v, want %+v", i, round[i], calls[i])
		}
		// MarshalIndent re-indents nested raw args; compare compacted.
		if got, want := compactJSON(t, round[i].Args), string(calls[i].Args); got != want {
			t.Fatalf("call %d args = %s, want %s", i, got, want)
		}
	}
}

func testApp(provider llm.Provider, stdout, stderr *bytes.Buffer) app {
	return app{
		stdin:  strings.NewReader(""),
		stdout: stdout,
		stderr: stderr,
		providerFactory: func(context.Context, providerConfig) (llm.Provider, error) {
			return provider, nil
		},
	}
}
