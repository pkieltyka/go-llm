package otelllm_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/otelllm"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

type harness struct {
	spans  *tracetest.InMemoryExporter
	reader *sdkmetric.ManualReader
	tp     *sdktrace.TracerProvider
	mw     llm.Middleware
}

func newHarness(t *testing.T) *harness {
	t.Helper()
	spans := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(spans))
	reader := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	t.Cleanup(func() {
		_ = tp.Shutdown(context.Background())
		_ = mp.Shutdown(context.Background())
	})
	return &harness{spans: spans, reader: reader, tp: tp, mw: otelllm.Middleware(otelllm.WithTracerProvider(tp), otelllm.WithMeterProvider(mp))}
}

func (h *harness) onlySpan(t *testing.T) tracetest.SpanStub {
	t.Helper()
	spans := h.spans.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("got %d spans, want 1", len(spans))
	}
	return spans[0]
}

func (h *harness) metrics(t *testing.T) metricdata.ResourceMetrics {
	t.Helper()
	var rm metricdata.ResourceMetrics
	if err := h.reader.Collect(context.Background(), &rm); err != nil {
		t.Fatal(err)
	}
	return rm
}

func attrMap(kvs []attribute.KeyValue) map[string]attribute.Value {
	out := map[string]attribute.Value{}
	for _, kv := range kvs {
		out[string(kv.Key)] = kv.Value
	}
	return out
}

func findMetric(rm metricdata.ResourceMetrics, name string) (metricdata.Metrics, bool) {
	for _, scope := range rm.ScopeMetrics {
		for _, m := range scope.Metrics {
			if m.Name == name {
				return m, true
			}
		}
	}
	return metricdata.Metrics{}, false
}

func TestChatSpanAttributesEventsAndMetrics(t *testing.T) {
	h := newHarness(t)
	cost := 0.0125
	p := llmtest.New(llmtest.WithName("anthropic"))
	p.EnqueueResponse(&llm.Response{
		ID: "msg_1", Model: "claude-x-20260101",
		Parts: []llm.Part{
			llm.ReasoningPart{Text: "thinking!"},
			llm.Text("héllo"),
			llm.ToolCall("call_1", "lookup", json.RawMessage(`{"q":"go"}`)),
		},
		StopReason: llm.StopReasonToolUse,
		Usage: llm.Usage{InputTokens: 10, CacheReadTokens: 5, CacheWriteTokens: 3, OutputTokens: 7, ReasoningTokens: 2,
			CostUSD: &cost, CostSource: llm.CostSourceEstimated},
	})
	wrapped := llm.Wrap(p, h.mw)
	if _, err := wrapped.Chat(context.Background(), &llm.Request{Model: "claude-x", MaxTokens: 64, Temperature: llm.Ptr(0.5), Effort: llm.EffortLow,
		Tools: []llm.Tool{{Name: "lookup"}}, Messages: []llm.Message{llm.UserText("hi")}}); err != nil {
		t.Fatal(err)
	}

	span := h.onlySpan(t)
	if span.Name != "chat claude-x" || span.SpanKind != trace.SpanKindClient {
		t.Fatalf("span = %q kind %v", span.Name, span.SpanKind)
	}
	attrs := attrMap(span.Attributes)
	want := map[string]any{
		"gen_ai.operation.name":                    "chat",
		"gen_ai.provider.name":                     "anthropic",
		"go_llm.provider":                          "anthropic",
		"gen_ai.request.model":                     "claude-x",
		"gen_ai.request.max_tokens":                int64(64),
		"gen_ai.request.temperature":               0.5,
		"gen_ai.request.stream":                    false,
		"go_llm.request.effort":                    "low",
		"go_llm.request.tools":                     int64(1),
		"gen_ai.response.id":                       "msg_1",
		"gen_ai.response.model":                    "claude-x-20260101",
		"gen_ai.usage.input_tokens":                int64(18),
		"gen_ai.usage.output_tokens":               int64(7),
		"gen_ai.usage.cache_read.input_tokens":     int64(5),
		"gen_ai.usage.cache_creation.input_tokens": int64(3),
		"gen_ai.usage.reasoning.output_tokens":     int64(2),
		"go_llm.usage.cost_usd":                    0.0125,
		"go_llm.usage.cost_source":                 "estimated",
	}
	for key, value := range want {
		got, ok := attrs[key]
		if !ok || got.AsInterface() != value {
			t.Errorf("attr %s = %v (present %v), want %v", key, got.AsInterface(), ok, value)
		}
	}
	if reasons := attrs["gen_ai.response.finish_reasons"].AsStringSlice(); len(reasons) != 1 || reasons[0] != "tool_use" {
		t.Errorf("finish_reasons = %v", reasons)
	}
	if span.Status.Code != codes.Unset {
		t.Errorf("status = %+v, want unset on success", span.Status)
	}
	var blocks []string
	for _, event := range span.Events {
		if event.Name == "go_llm.block" {
			a := attrMap(event.Attributes)
			blocks = append(blocks, fmt.Sprintf("%s/%d/%d/%s", a["go_llm.block.type"].AsString(), a["go_llm.block.index"].AsInt64(), a["go_llm.block.chars"].AsInt64(), a["go_llm.block.tool"].AsString()))
		}
	}
	if got := strings.Join(blocks, " "); got != "reasoning/0/9/ text/1/5/ tool_call/2/10/lookup" {
		t.Errorf("block events = %q", got)
	}

	rm := h.metrics(t)
	duration, ok := findMetric(rm, "gen_ai.client.operation.duration")
	if !ok || duration.Unit != "s" || duration.Data.(metricdata.Histogram[float64]).DataPoints[0].Count != 1 {
		t.Fatalf("duration metric = %+v", duration)
	}
	tokens, ok := findMetric(rm, "gen_ai.client.token.usage")
	if !ok {
		t.Fatal("token usage metric missing")
	}
	byType := map[string]int64{}
	for _, dp := range tokens.Data.(metricdata.Histogram[int64]).DataPoints {
		tokenType, _ := dp.Attributes.Value("gen_ai.token.type")
		byType[tokenType.AsString()] = dp.Sum
	}
	if byType["input"] != 18 || byType["output"] != 7 {
		t.Fatalf("token usage = %v, want input 18 output 7", byType)
	}
	costMetric, ok := findMetric(rm, "go_llm.client.cost")
	if !ok || costMetric.Data.(metricdata.Sum[float64]).DataPoints[0].Value != 0.0125 {
		t.Fatalf("cost metric = %+v", costMetric)
	}
	if _, ok := findMetric(rm, "gen_ai.client.operation.time_to_first_chunk"); ok {
		t.Fatal("blocking Chat must not record time to first chunk")
	}
}

func TestStreamRecordsFirstChunkAndCollectedResponse(t *testing.T) {
	h := newHarness(t)
	p := llmtest.New(llmtest.WithName("openai"))
	p.EnqueueStream(
		llm.MessageStart{ID: "resp_1", Model: "gpt-x"},
		llm.TextDelta{Index: 0, Text: "hel"},
		llm.TextDelta{Index: 0, Text: "lo"},
		llm.MessageEnd{StopReason: llm.StopReasonEndTurn, Usage: llm.Usage{InputTokens: 4, OutputTokens: 2}},
	)
	if _, err := llm.Collect(llm.Wrap(p, h.mw).ChatStream(context.Background(), &llm.Request{Model: "gpt-x"})); err != nil {
		t.Fatal(err)
	}
	span := h.onlySpan(t)
	attrs := attrMap(span.Attributes)
	if attrs["gen_ai.provider.name"].AsString() != "openai" || attrs["go_llm.provider"].AsString() != "openai" {
		t.Fatalf("provider attrs = %v / %v, want openai / openai", attrs["gen_ai.provider.name"].AsString(), attrs["go_llm.provider"].AsString())
	}
	if !attrs["gen_ai.request.stream"].AsBool() || attrs["gen_ai.response.id"].AsString() != "resp_1" || attrs["gen_ai.usage.output_tokens"].AsInt64() != 2 {
		t.Fatalf("stream attrs = %v", span.Attributes)
	}
	if _, ok := attrs["gen_ai.response.time_to_first_chunk"]; !ok || span.Events[0].Name != "go_llm.first_chunk" {
		t.Fatalf("first chunk not recorded: attrs=%v events=%v", span.Attributes, span.Events)
	}
	if _, ok := findMetric(h.metrics(t), "gen_ai.client.operation.time_to_first_chunk"); !ok {
		t.Fatal("time_to_first_chunk metric missing for a stream")
	}
}

func TestErrorsCancellationAndAbandonment(t *testing.T) {
	h := newHarness(t)
	p := llmtest.New(llmtest.WithName("openrouter"))
	p.EnqueueError(&llm.ProviderError{Provider: "openrouter", HTTPStatus: 429, Message: "provider detail", Kind: llm.ErrRateLimited})
	p.EnqueueError(context.Canceled)
	p.EnqueueStream(llm.MessageStart{}, llm.TextDelta{Index: 0, Text: "a"}, llm.TextDelta{Index: 0, Text: "b"}, llm.MessageEnd{StopReason: llm.StopReasonEndTurn})
	wrapped := llm.Wrap(p, h.mw)

	if _, err := wrapped.Chat(context.Background(), &llm.Request{Model: "m"}); !errors.Is(err, llm.ErrRateLimited) {
		t.Fatalf("err = %v", err)
	}
	if _, err := wrapped.Chat(context.Background(), &llm.Request{Model: "m"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	for range wrapped.ChatStream(context.Background(), &llm.Request{Model: "m"}) {
		break
	}

	spans := h.spans.GetSpans()
	if len(spans) != 3 {
		t.Fatalf("got %d spans", len(spans))
	}
	failed := attrMap(spans[0].Attributes)
	if spans[0].Status.Code != codes.Error || failed["error.type"].AsString() != "rate_limited" || strings.Contains(spans[0].Status.Description, "provider detail") {
		t.Fatalf("error span = %+v attrs %v", spans[0].Status, spans[0].Attributes)
	}
	canceled := attrMap(spans[1].Attributes)
	if spans[1].Status.Code == codes.Error || !canceled["go_llm.canceled"].AsBool() {
		t.Fatalf("canceled span = %+v attrs %v, want no error status", spans[1].Status, spans[1].Attributes)
	}
	abandoned := attrMap(spans[2].Attributes)
	if spans[2].Status.Code == codes.Error || !abandoned["go_llm.stream.abandoned"].AsBool() {
		t.Fatalf("abandoned stream span = %+v attrs %v", spans[2].Status, spans[2].Attributes)
	}

	duration, _ := findMetric(h.metrics(t), "gen_ai.client.operation.duration")
	errorTypes := map[string]bool{}
	for _, dp := range duration.Data.(metricdata.Histogram[float64]).DataPoints {
		if v, ok := dp.Attributes.Value("error.type"); ok {
			errorTypes[v.AsString()] = true
		}
	}
	if !errorTypes["rate_limited"] || len(errorTypes) != 1 {
		t.Fatalf("duration error.type values = %v, want only rate_limited", errorTypes)
	}
}

func TestSpanContextPropagatesToProvider(t *testing.T) {
	h := newHarness(t)
	p := llmtest.New()
	p.EnqueueResponse(&llm.Response{StopReason: llm.StopReasonEndTurn})
	var inner trace.SpanContext
	capture := llm.Middleware{Chat: func(next llm.ChatFunc) llm.ChatFunc {
		return func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
			inner = trace.SpanContextFromContext(ctx)
			return next(ctx, req)
		}
	}}
	ctx, parent := h.tp.Tracer("test").Start(context.Background(), "parent")
	if _, err := llm.Wrap(p, h.mw, capture).Chat(ctx, &llm.Request{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	parent.End()
	var chat tracetest.SpanStub
	for _, s := range h.spans.GetSpans() {
		if s.Name == "chat m" {
			chat = s
		}
	}
	if chat.Parent.SpanID() != parent.SpanContext().SpanID() {
		t.Fatal("chat span must be a child of the context's span")
	}
	if inner.SpanID() != chat.SpanContext.SpanID() {
		t.Fatal("the wrapped provider must receive the chat span's context")
	}
}

// TestTelemetryIsContentFree plants canary strings in every content position
// and asserts none reaches span names, attributes, events, status, or metric
// attributes.
func TestTelemetryIsContentFree(t *testing.T) {
	h := newHarness(t)
	canaries := []string{"CANARY-system", "CANARY-user", "CANARY-args", "CANARY-result", "CANARY-text", "CANARY-reasoning", "CANARY-tool-out", "CANARY-error", "CANARY-stop"}
	p := llmtest.New(llmtest.WithName("openrouter"))
	p.EnqueueResponse(&llm.Response{
		Parts:      []llm.Part{llm.Text("CANARY-text"), llm.ReasoningPart{Text: "CANARY-reasoning"}, llm.ToolCall("c", "lookup", json.RawMessage(`{"q":"CANARY-tool-out"}`))},
		StopReason: llm.StopReasonEndTurn, StopReasonRaw: "CANARY-stop",
	})
	p.EnqueueError(&llm.ProviderError{Provider: "openrouter", HTTPStatus: 400, Code: "CANARY-error", Message: "CANARY-error", RawBody: []byte("CANARY-error"), Kind: llm.ErrBadRequest})
	req := &llm.Request{Model: "m", System: "CANARY-system", Messages: []llm.Message{
		llm.UserText("CANARY-user"),
		llm.AssistantParts(llm.ToolCall("c0", "lookup", json.RawMessage(`{"q":"CANARY-args"}`))),
		{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResult("c0", "CANARY-result")}},
	}}
	wrapped := llm.Wrap(p, h.mw)
	_, _ = wrapped.Chat(context.Background(), req)
	_, _ = wrapped.Chat(context.Background(), req)

	var dump strings.Builder
	for _, span := range h.spans.GetSpans() {
		fmt.Fprintf(&dump, "%s|%s|%v|", span.Name, span.Status.Description, span.Attributes)
		for _, event := range span.Events {
			fmt.Fprintf(&dump, "%s|%v|", event.Name, event.Attributes)
		}
	}
	for _, scope := range h.metrics(t).ScopeMetrics {
		for _, m := range scope.Metrics {
			fmt.Fprintf(&dump, "%s|%+v|", m.Name, m.Data)
		}
	}
	for _, canary := range canaries {
		if strings.Contains(dump.String(), canary) {
			t.Errorf("telemetry contains content canary %q", canary)
		}
	}
}

// rawStreamProvider yields a fixed event sequence, bypassing llmtest's
// stream validation, to exercise events the Collector rejects.
type rawStreamProvider struct {
	*llmtest.Provider
	events []llm.Event
}

func (p rawStreamProvider) ChatStream(context.Context, *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		for _, event := range p.events {
			if !yield(event, nil) {
				return
			}
		}
	}
}

func TestStreamCollectorRejectionIsNotTheCallError(t *testing.T) {
	h := newHarness(t)
	p := rawStreamProvider{Provider: llmtest.New(llmtest.WithName("custom")), events: []llm.Event{
		llm.MessageStart{ID: "r"},
		llm.TextDelta{Index: 0, Text: "hi"},
		llm.ToolCallDropped{Index: -1, Reason: "rejected by the collector"},
		llm.MessageEnd{StopReason: llm.StopReasonEndTurn, Usage: llm.Usage{InputTokens: 3, OutputTokens: 1}},
	}}
	for _, err := range llm.Wrap(p, h.mw).ChatStream(context.Background(), &llm.Request{Model: "m"}) {
		if err != nil {
			t.Fatalf("the consumer saw an error: %v", err)
		}
	}
	span := h.onlySpan(t)
	if span.Status.Code == codes.Error {
		t.Fatalf("span status = %+v, want no error for a stream the consumer saw as clean", span.Status)
	}
	if attrMap(span.Attributes)["gen_ai.response.id"].AsString() != "r" {
		t.Fatalf("partial response summary missing: %v", span.Attributes)
	}
}

func TestStatusDescriptionExcludesMiddlewareErrorText(t *testing.T) {
	h := newHarness(t)
	failing := llm.Middleware{Chat: func(llm.ChatFunc) llm.ChatFunc {
		return func(context.Context, *llm.Request) (*llm.Response, error) {
			return nil, fmt.Errorf("%w: could not parse %q", llm.ErrBadRequest, "CANARY-user-content")
		}
	}}
	_, _ = llm.Wrap(llmtest.New(), h.mw, failing).Chat(context.Background(), &llm.Request{Model: "m"})
	span := h.onlySpan(t)
	if strings.Contains(span.Status.Description, "CANARY") || span.Status.Description != "llm: bad_request" {
		t.Fatalf("status description = %q, want only the error class", span.Status.Description)
	}
}

func TestAbandonedStreamRecordsNoZeroUsageAndTokensLackErrorType(t *testing.T) {
	h := newHarness(t)
	p := llmtest.New()
	p.EnqueueStream(llm.MessageStart{}, llm.TextDelta{Index: 0, Text: "a"}, llm.MessageEnd{StopReason: llm.StopReasonEndTurn, Usage: llm.Usage{InputTokens: 2, OutputTokens: 1}})
	for range llm.Wrap(p, h.mw).ChatStream(context.Background(), &llm.Request{Model: "m"}) {
		break
	}
	attrs := attrMap(h.onlySpan(t).Attributes)
	if _, ok := attrs["gen_ai.usage.input_tokens"]; ok {
		t.Fatalf("abandoned stream recorded unmeasured usage: %v", attrs)
	}

	failed := llmtest.New()
	failed.EnqueueError(&llm.ProviderError{Kind: llm.ErrServer})
	_, _ = llm.Wrap(failed, h.mw).Chat(context.Background(), &llm.Request{Model: "m"})
	tokens, ok := findMetric(h.metrics(t), "gen_ai.client.token.usage")
	if ok {
		for _, dp := range tokens.Data.(metricdata.Histogram[int64]).DataPoints {
			if _, has := dp.Attributes.Value("error.type"); has {
				t.Fatal("token usage must not carry error.type")
			}
		}
	}
}

// Pointer events are valid stream events: the first one still records the
// first-chunk telemetry, and the consumer receives it as sent.
func TestPointerEventsRecordFirstChunk(t *testing.T) {
	h := newHarness(t)
	text := &llm.TextDelta{Index: 0, Text: "hello"}
	p := rawStreamProvider{Provider: llmtest.New(llmtest.WithName("custom")), events: []llm.Event{
		&llm.MessageStart{ID: "r"},
		text,
		&llm.MessageEnd{StopReason: llm.StopReasonEndTurn},
	}}
	var sawPointer bool
	for event, err := range llm.Wrap(p, h.mw).ChatStream(context.Background(), &llm.Request{Model: "m"}) {
		if err != nil {
			t.Fatal(err)
		}
		sawPointer = sawPointer || event == llm.Event(text)
	}
	if !sawPointer {
		t.Fatal("the consumer did not receive the pointer event as sent")
	}
	if _, ok := attrMap(h.onlySpan(t).Attributes)["gen_ai.response.time_to_first_chunk"]; !ok {
		t.Fatal("first chunk not recorded for a pointer TextDelta")
	}
}

// Ranging a wrapped stream again never sends another request.
func TestWrappedStreamStaysSingleUse(t *testing.T) {
	h := newHarness(t)
	p := llmtest.New()
	p.EnqueueStream(llm.MessageStart{}, llm.TextDelta{Text: "a"}, llm.MessageEnd{StopReason: llm.StopReasonEndTurn})
	p.EnqueueStream(llm.MessageStart{}, llm.TextDelta{Text: "b"}, llm.MessageEnd{StopReason: llm.StopReasonEndTurn})
	seq := llm.Wrap(p, h.mw).ChatStream(context.Background(), &llm.Request{Model: "m"})
	if _, err := llm.Collect(seq); err != nil {
		t.Fatal(err)
	}
	if _, err := llm.Collect(seq); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("second range = %v, want ErrBadRequest", err)
	}
	if n := len(p.Requests()); n != 1 {
		t.Fatalf("provider requests = %d, want 1", n)
	}
}

func classifyRequest() *llm.ClassifyRequest {
	return &llm.ClassifyRequest{Model: "jev", State: json.RawMessage(`{"secret":"do not record"}`),
		Questions: []llm.Question{llm.BoolQuestion("approved", "Approve?", "", "")}}
}

func TestClassifySpanAndMetrics(t *testing.T) {
	h := newHarness(t)
	fake := llmtest.New(llmtest.WithName("openrouter"))
	cost := 0.00002
	fake.EnqueueClassifyResult(&llm.ClassifyResult{Provider: "openrouter", Model: "typesafe/jev-1.14", ID: "gen-1",
		Answers: map[string]llm.Answer{"approved": {Type: llm.QuestionBool, Probability: 0.8}},
		Usage: llm.Usage{InputTokens: 40, CacheReadTokens: 60, CacheWriteTokens: 10, OutputTokens: 6, ReasoningTokens: 4,
			TotalTokens: 116, CostUSD: &cost, CostSource: llm.CostSourceNative}})
	p := llm.Wrap(fake, h.mw)
	if _, err := llm.Classify(context.Background(), p, classifyRequest()); err != nil {
		t.Fatal(err)
	}
	span := h.onlySpan(t)
	attrs := attrMap(span.Attributes)
	if span.Name != "classify jev" || span.SpanKind != trace.SpanKindClient ||
		attrs["gen_ai.operation.name"].AsString() != "classify" || attrs["gen_ai.request.model"].AsString() != "jev" ||
		attrs["gen_ai.response.model"].AsString() != "typesafe/jev-1.14" || attrs["gen_ai.response.id"].AsString() != "gen-1" ||
		attrs["gen_ai.usage.input_tokens"].AsInt64() != 110 || attrs["gen_ai.usage.output_tokens"].AsInt64() != 6 ||
		attrs["gen_ai.usage.cache_read.input_tokens"].AsInt64() != 60 || attrs["gen_ai.usage.cache_creation.input_tokens"].AsInt64() != 10 ||
		attrs["gen_ai.usage.reasoning.output_tokens"].AsInt64() != 4 || attrs["go_llm.request.questions"].AsInt64() != 1 ||
		attrs["go_llm.usage.cost_usd"].AsFloat64() != cost {
		t.Fatalf("span %s attrs %v", span.Name, attrs)
	}
	for _, kv := range span.Attributes {
		if strings.Contains(kv.Value.Emit(), "do not record") || strings.Contains(kv.Value.Emit(), "Approve?") {
			t.Fatalf("content recorded in %s", kv.Key)
		}
	}
	rm := h.metrics(t)
	for _, name := range []string{"gen_ai.client.operation.duration", "gen_ai.client.token.usage", "go_llm.client.cost"} {
		if _, ok := findMetric(rm, name); !ok {
			t.Fatalf("metric %s missing", name)
		}
	}
}

func TestClassifyErrorKeepsBilledUsage(t *testing.T) {
	h := newHarness(t)
	fake := llmtest.New(llmtest.WithName("openai"))
	fake.EnqueueClassifyError(fmt.Errorf("%w: refused", llm.ErrContentFiltered), &llm.ClassifyResult{Provider: "openai", Model: "gpt-6-luna",
		Usage: llm.Usage{InputTokens: 9, CacheWriteTokens: 3, ReasoningTokens: 2, OutputTokens: 2, TotalTokens: 14}})
	_, err := llm.Classify(context.Background(), llm.Wrap(fake, h.mw), classifyRequest())
	if !errors.Is(err, llm.ErrContentFiltered) {
		t.Fatalf("err = %v", err)
	}
	span := h.onlySpan(t)
	attrs := attrMap(span.Attributes)
	if span.Status.Code != codes.Error || attrs["error.type"].AsString() == "" || attrs["gen_ai.usage.input_tokens"].AsInt64() != 12 ||
		attrs["gen_ai.usage.cache_creation.input_tokens"].AsInt64() != 3 || attrs["gen_ai.usage.reasoning.output_tokens"].AsInt64() != 2 {
		t.Fatalf("status %+v attrs %v", span.Status, attrs)
	}
}
