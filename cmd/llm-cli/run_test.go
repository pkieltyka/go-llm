package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/providers/openrouter"
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
	price := &llm.ModelPricing{InputPerMTok: 1.25, OutputPerMTok: 2.5, CacheReadPerMTok: 0.25, CacheWritePerMTok: 0.75}
	fake := llmtest.New(llmtest.WithModels(llm.ModelInfo{
		ID:                "model-1",
		DisplayName:       "Model One",
		ContextWindow:     1000,
		MaxOutputTokens:   200,
		Pricing:           price,
		SupportedEfforts:  []llm.Effort{llm.EffortLow, llm.EffortHigh},
		DefaultEffort:     llm.EffortHigh,
		ReasoningRequired: true,
		Capabilities:      []llm.Capability{llm.CapabilityTools, llm.CapabilityReasoning},
	}))
	var stdout, stderr bytes.Buffer
	a := testApp(fake, &stdout, &stderr)
	if err := a.runModels(context.Background(), modelsConfig{provider: "llmtest"}); err != nil {
		t.Fatal(err)
	}
	got := stdout.String()
	lines := strings.Split(strings.TrimSuffix(got, "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("table lines = %d, want header and one row: %q", len(lines), got)
	}
	for _, association := range []struct{ header, value string }{
		{"ID", "model-1"}, {"DISPLAY", "Model One"}, {"CONTEXT", "1000"}, {"MAX OUTPUT", "200"},
		{"INPUT $/M", "1.25"}, {"OUTPUT $/M", "2.5"}, {"CACHE READ $/M", "0.25"}, {"CACHE WRITE $/M", "0.75"},
		{"EFFORTS", "low,high"}, {"DEFAULT EFFORT", "high"}, {"REASONING REQUIRED", "true"}, {"CAPABILITIES", "tools,reasoning"},
	} {
		column := strings.Index(lines[0], association.header)
		if column < 0 || column >= len(lines[1]) || !strings.HasPrefix(lines[1][column:], association.value) {
			t.Fatalf("table column %q is not associated with %q: %q", association.header, association.value, got)
		}
	}

	stdout.Reset()
	if err := a.runModels(context.Background(), modelsConfig{provider: "llmtest", jsonOutput: true}); err != nil {
		t.Fatal(err)
	}
	var gotRows []modelRow
	if err := json.Unmarshal(stdout.Bytes(), &gotRows); err != nil {
		t.Fatalf("decode JSON rows: %v\n%s", err, stdout.String())
	}
	efforts := []llm.Effort{llm.EffortLow, llm.EffortHigh}
	wantRows := []modelRow{{
		ID: "model-1", DisplayName: "Model One", ContextWindow: 1000, MaxOutputTokens: 200,
		InputPerMTok: "1.25", OutputPerMTok: "2.5", CacheReadPerMTok: "0.25", CacheWritePerMTok: "0.75",
		SupportedEfforts: &efforts, DefaultEffort: llm.EffortHigh, ReasoningRequired: true,
		Capabilities: []llm.Capability{llm.CapabilityTools, llm.CapabilityReasoning},
	}}
	if !reflect.DeepEqual(gotRows, wantRows) {
		t.Fatalf("decoded rows = %#v, want %#v", gotRows, wantRows)
	}
}

func TestModelRowsHideInvalidPricesAndPreserveFreePricing(t *testing.T) {
	rows := modelRows([]llm.ModelInfo{
		{ID: "negative", Pricing: &llm.ModelPricing{InputPerMTok: -1, OutputPerMTok: 2}},
		{ID: "non-finite", Pricing: &llm.ModelPricing{InputPerMTok: math.NaN(), OutputPerMTok: math.Inf(1)}},
		{ID: "free", Pricing: &llm.ModelPricing{}},
		{ID: "partial", Pricing: &llm.ModelPricing{OutputPerMTok: 2, Availability: &llm.ModelPricingAvailability{OutputPerMTok: true}}},
		{ID: "free-cache", Pricing: &llm.ModelPricing{Availability: &llm.ModelPricingAvailability{CacheReadPerMTok: true, CacheWritePerMTok: true}}},
	})
	if rows[0].InputPerMTok != "" || rows[0].OutputPerMTok != "2" {
		t.Fatalf("negative row = %+v", rows[0])
	}
	if rows[1].InputPerMTok != "" || rows[1].OutputPerMTok != "" {
		t.Fatalf("non-finite row = %+v", rows[1])
	}
	if rows[2].InputPerMTok != "0" || rows[2].OutputPerMTok != "0" {
		t.Fatalf("free row = %+v", rows[2])
	}
	if rows[3].InputPerMTok != "" || rows[3].OutputPerMTok != "2" {
		t.Fatalf("partial row = %+v", rows[3])
	}
	if rows[4].CacheReadPerMTok != "0" || rows[4].CacheWritePerMTok != "0" {
		t.Fatalf("free cache row = %+v", rows[4])
	}
	raw, err := json.Marshal(rows)
	if err != nil {
		t.Fatalf("marshal rows: %v", err)
	}
	for _, forbidden := range []string{"-1", "NaN", "Inf"} {
		if strings.Contains(string(raw), forbidden) {
			t.Fatalf("JSON leaked %q: %s", forbidden, raw)
		}
	}
}

func TestRunModelsOpenRouterPartialPricingAndEffortStates(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"data":[
			{"id":"partial","pricing":{"prompt":true,"completion":"0.000002","input_cache_read":null,"input_cache_write":"0"},"reasoning":{"supported_efforts":[]}},
			{"id":"future-effort","pricing":{"prompt":"0","completion":{}},"reasoning":{"supported_efforts":["turbo"]}},
			{"id":"unknown"}
		]}`)
	}))
	t.Cleanup(server.Close)
	provider, err := openrouter.New(openrouter.WithAPIKey("test"), openrouter.WithBaseURL(server.URL), openrouter.WithHTTPClient(server.Client()), openrouter.WithMaxRetries(0))
	if err != nil {
		t.Fatalf("openrouter.New: %v", err)
	}
	var stdout, stderr bytes.Buffer
	a := testApp(provider, &stdout, &stderr)
	if err := a.runModels(context.Background(), modelsConfig{provider: "openrouter", jsonOutput: true}); err != nil {
		t.Fatalf("runModels: %v", err)
	}
	var rows []map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &rows); err != nil {
		t.Fatalf("decode rows: %v\n%s", err, stdout.String())
	}
	if len(rows) != 3 {
		t.Fatalf("rows = %d, want 3", len(rows))
	}
	if _, present := rows[0]["input_per_mtok"]; present || rows[0]["output_per_mtok"] != "2" || rows[0]["cache_write_per_mtok"] != "0" {
		t.Fatalf("partial pricing row = %#v", rows[0])
	}
	for _, index := range []int{0, 1} {
		efforts, present := rows[index]["supported_efforts"].([]any)
		if !present || len(efforts) != 0 {
			t.Fatalf("row %d supported_efforts = %#v, want explicit []", index, rows[index]["supported_efforts"])
		}
	}
	if rows[1]["input_per_mtok"] != "0" {
		t.Fatalf("explicit free prompt row = %#v", rows[1])
	}
	if _, present := rows[1]["output_per_mtok"]; present {
		t.Fatalf("invalid output price presented as free: %#v", rows[1])
	}
	if _, present := rows[2]["supported_efforts"]; present {
		t.Fatalf("unknown effort ladder should be omitted: %#v", rows[2])
	}
}

func TestModelRowsPreserveEmptyEffortsWithoutAliasingCapacity(t *testing.T) {
	efforts := make([]llm.Effort, 0, 4)
	rows := modelRows([]llm.ModelInfo{{ID: "explicit-empty", SupportedEfforts: efforts}, {ID: "unknown"}})
	if rows[0].SupportedEfforts == nil || len(*rows[0].SupportedEfforts) != 0 || rows[1].SupportedEfforts != nil {
		t.Fatalf("effort states = %#v", rows)
	}
	efforts = append(efforts, llm.EffortMax)
	if len(*rows[0].SupportedEfforts) != 0 {
		t.Fatalf("modelRows retained caller backing storage: %#v", *rows[0].SupportedEfforts)
	}
	*rows[0].SupportedEfforts = append(*rows[0].SupportedEfforts, llm.EffortLow)
	if efforts[0] != llm.EffortMax {
		t.Fatalf("modelRows shares caller backing storage: %#v", efforts)
	}
	*rows[0].SupportedEfforts = (*rows[0].SupportedEfforts)[:0]
	raw, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"supported_efforts":[]`) || strings.Count(string(raw), "supported_efforts") != 1 {
		t.Fatalf("effort JSON states = %s", raw)
	}
}

func TestModelRowsOmitUnknownReasoningPolicy(t *testing.T) {
	rows := modelRows([]llm.ModelInfo{{ID: "unknown"}})
	raw, err := json.Marshal(rows)
	if err != nil {
		t.Fatalf("marshal rows: %v", err)
	}
	if strings.Contains(string(raw), "default_effort") || strings.Contains(string(raw), "reasoning_required") {
		t.Fatalf("unknown reasoning metadata should be omitted: %s", raw)
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
	resp, err := a.runStreaming(context.Background(), p, &llm.Request{Model: "model-1"}, chatConfig{})
	if !errors.Is(err, llm.ErrServer) || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("err = %v, want the provider stream error", err)
	}
	if resp == nil || resp.Text() != "partial text" {
		t.Fatalf("partial response = %#v, want partial text", resp)
	}
	// Deltas were flushed unbuffered before the error arrived; main prints
	// the returned error to stderr and exits 1.
	if got := stdout.String(); got != "partial text" {
		t.Fatalf("stdout = %q, want partial text flushed", got)
	}
}

func TestRunChatStreamReturnsFirstCollectorErrorImmediately(t *testing.T) {
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
	if !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("err = %v, want the first collector error", err)
	}
}

type brokenWriter struct {
	err error
}

func (w brokenWriter) Write([]byte) (int, error) { return 0, w.err }

func TestRunChatReturnsResponseWriteError(t *testing.T) {
	writeErr := errors.New("stdout failed")
	fake := llmtest.New()
	fake.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text("answer")}})
	a := testApp(fake, brokenWriter{err: writeErr}, io.Discard)
	err := a.runChat(context.Background(), chatConfig{
		provider: "llmtest",
		model:    "model-1",
		noStream: true,
		args:     []string{"prompt"},
	})
	if !errors.Is(err, writeErr) || !strings.Contains(err.Error(), "write response") {
		t.Fatalf("err = %v, want response writer error", err)
	}
}

func TestRunStreamingReturnsEventWriteErrorWithPartialResponse(t *testing.T) {
	writeErr := errors.New("event output failed")
	fake := llmtest.New(llmtest.WithName("llmtest"))
	fake.EnqueueStream(
		llm.MessageStart{ID: "msg-1", Provider: "llmtest", Model: "model-1"},
		llm.TextDelta{Index: 0, Text: "unwritten"},
		llm.MessageEnd{StopReason: llm.StopReasonEndTurn},
	)
	a := testApp(fake, brokenWriter{err: writeErr}, io.Discard)
	resp, err := a.runStreaming(context.Background(), fake, &llm.Request{Model: "model-1"}, chatConfig{})
	if !errors.Is(err, writeErr) || !strings.Contains(err.Error(), "write response event") {
		t.Fatalf("err = %v, want event writer error", err)
	}
	if resp == nil || resp.ID != "msg-1" || resp.Text() != "" {
		t.Fatalf("partial response = %#v, want collected MessageStart only", resp)
	}
}

func TestRunStreamingReturnsReasoningWriteErrorWithPartialResponse(t *testing.T) {
	writeErr := errors.New("reasoning output failed")
	fake := llmtest.New(llmtest.WithName("llmtest"))
	fake.EnqueueStream(
		llm.MessageStart{ID: "msg-1", Provider: "llmtest", Model: "model-1"},
		llm.ReasoningDelta{Index: 0, Text: "unwritten"},
		llm.MessageEnd{StopReason: llm.StopReasonEndTurn},
	)
	a := testApp(fake, io.Discard, brokenWriter{err: writeErr})
	resp, err := a.runStreaming(context.Background(), fake, &llm.Request{Model: "model-1"}, chatConfig{reasoning: true})
	if !errors.Is(err, writeErr) || !strings.Contains(err.Error(), "write reasoning event") {
		t.Fatalf("err = %v, want reasoning writer error", err)
	}
	if resp == nil || resp.ID != "msg-1" {
		t.Fatalf("partial response = %#v, want collected MessageStart", resp)
	}
}

func TestRunChatReturnsUsageWriteError(t *testing.T) {
	writeErr := errors.New("usage output failed")
	fake := llmtest.New()
	fake.EnqueueResponse(&llm.Response{
		Parts: []llm.Part{llm.Text("answer")},
		Usage: llm.Usage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3},
	})
	a := testApp(fake, io.Discard, brokenWriter{err: writeErr})
	err := a.runChat(context.Background(), chatConfig{
		provider: "llmtest",
		model:    "model-1",
		noStream: true,
		usage:    true,
		args:     []string{"prompt"},
	})
	if !errors.Is(err, writeErr) || !strings.Contains(err.Error(), "write usage") {
		t.Fatalf("err = %v, want usage writer error", err)
	}
}

func TestPrintErrorReturnsWriterError(t *testing.T) {
	writeErr := errors.New("stderr failed")
	err := printError(brokenWriter{err: writeErr}, fmt.Errorf("provider failed"))
	if !errors.Is(err, writeErr) || !strings.Contains(err.Error(), "write error") {
		t.Fatalf("err = %v, want error-output writer error", err)
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

func testApp(provider llm.Provider, stdout, stderr io.Writer) app {
	return app{
		stdin:  strings.NewReader(""),
		stdout: stdout,
		stderr: stderr,
		providerFactory: func(context.Context, providerConfig) (llm.Provider, error) {
			return provider, nil
		},
	}
}
