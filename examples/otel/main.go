// Otel records OpenTelemetry traces and metrics for LLM calls with the
// otelllm middleware.
//
// otelllm.Middleware wraps any provider: one CLIENT span per Chat or
// ChatStream named "chat {model}", with gen_ai.* request, response and usage
// attributes (OpenTelemetry GenAI semantic conventions), plus the
// gen_ai.client.* duration and token-usage metrics (plus time to first chunk
// for ChatStream). It
// is content-free: no prompt, completion or tool bytes are recorded. By
// default it uses the global OpenTelemetry providers; here it gets an
// in-memory exporter and a manual metric reader so the example can print
// what it recorded. In an application, configure your usual exporter (OTLP,
// stdout, ...) and pass nothing.
//
// With ANTHROPIC_API_KEY set it talks to Claude Haiku 4.5; without
// credentials it runs offline on the scripted llmtest provider.
package main

import (
	"context"
	"fmt"
	"os"
	"sort"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/otelllm"
	"github.com/pkieltyka/go-llm/providers/anthropic"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func main() {
	ctx := context.Background()
	spans := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(spans))
	reader := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	defer func() { _ = tp.Shutdown(ctx); _ = mp.Shutdown(ctx) }()

	base, model := newProvider()
	// Outermost, so the span covers any other middleware (retries, guards).
	p := llm.Wrap(base, otelllm.Middleware(otelllm.WithTracerProvider(tp), otelllm.WithMeterProvider(mp)))

	resp, err := p.Chat(ctx, &llm.Request{
		Model:     model,
		Messages:  []llm.Message{llm.UserText("Say hello in one short sentence.")},
		MaxTokens: 100,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println("reply:", resp.Text())

	for _, span := range spans.GetSpans() {
		fmt.Printf("\nspan %q (%s)\n", span.Name, span.SpanKind)
		attrs := make([]string, 0, len(span.Attributes))
		for _, kv := range span.Attributes {
			attrs = append(attrs, fmt.Sprintf("  %s = %s", kv.Key, kv.Value.String()))
		}
		sort.Strings(attrs)
		for _, line := range attrs {
			fmt.Println(line)
		}
	}

	var rm metricdata.ResourceMetrics
	if err := reader.Collect(ctx, &rm); err != nil {
		panic(err)
	}
	fmt.Println("\nmetrics:")
	for _, scope := range rm.ScopeMetrics {
		for _, m := range scope.Metrics {
			fmt.Printf("  %s (%s)\n", m.Name, m.Unit)
		}
	}
}

func newProvider() (llm.Provider, string) {
	if os.Getenv("ANTHROPIC_API_KEY") != "" {
		p, err := anthropic.New() // reads ANTHROPIC_API_KEY
		if err != nil {
			panic(err)
		}
		return p, "claude-haiku-4-5"
	}
	fmt.Println("(offline demo — set ANTHROPIC_API_KEY to run against the real API)")
	p := llmtest.New()
	p.EnqueueResponse(&llm.Response{
		Provider: "llmtest",
		Model:    "demo-model",
		Parts:    []llm.Part{llm.Text("Hello from go-llm.")},
		Usage:    llm.Usage{InputTokens: 12, OutputTokens: 6, TotalTokens: 18},
	})
	return p, "demo-model"
}
