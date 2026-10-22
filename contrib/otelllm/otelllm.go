// Package otelllm instruments go-llm providers with OpenTelemetry traces and
// metrics through the llm.Wrap middleware seam.
//
// Names follow the OpenTelemetry GenAI semantic conventions as published in
// semconv v1.41.0 (development status): one CLIENT span per Chat or
// ChatStream call named "chat {model}", gen_ai.* request, response, and usage
// attributes, and the gen_ai.client.operation.duration,
// gen_ai.client.token.usage, and gen_ai.client.operation.time_to_first_chunk
// metrics. Concepts the conventions lack use go_llm.* names: cost, cost
// provenance, the raw provider name, dropped tool calls, and per-block
// events.
//
// Telemetry is content-free by construction: no prompt, system, completion,
// reasoning, tool-argument, or tool-result bytes are recorded. Error status
// descriptions use llm.SafeError for provider errors, which excludes
// provider-controlled detail, and only the error class for any other error,
// so text from caller middleware is never recorded either.
// Per-block events carry only the block type, index, character count, and,
// for tool calls, the tool name. The parent span comes from the request
// context, and the span's context is passed to the wrapped provider so HTTP
// instrumentation nests beneath it.
package otelllm

import (
	"context"
	"errors"
	"iter"
	"time"
	"unicode/utf8"

	llm "github.com/pkieltyka/go-llm"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

// ScopeName is the instrumentation scope for the tracer and meter.
const ScopeName = "github.com/pkieltyka/go-llm/contrib/otelllm"

// Attribute and metric names. gen_ai.* names follow semconv v1.41.0.
const (
	attrOperationName      = "gen_ai.operation.name"
	attrProviderName       = "gen_ai.provider.name"
	attrRequestModel       = "gen_ai.request.model"
	attrRequestMaxTokens   = "gen_ai.request.max_tokens"
	attrRequestTemperature = "gen_ai.request.temperature"
	attrRequestTopP        = "gen_ai.request.top_p"
	attrRequestStream      = "gen_ai.request.stream"
	attrResponseID         = "gen_ai.response.id"
	attrResponseModel      = "gen_ai.response.model"
	attrFinishReasons      = "gen_ai.response.finish_reasons"
	attrTimeToFirstChunk   = "gen_ai.response.time_to_first_chunk"
	attrInputTokens        = "gen_ai.usage.input_tokens"
	attrOutputTokens       = "gen_ai.usage.output_tokens"
	attrCacheReadTokens    = "gen_ai.usage.cache_read.input_tokens"
	attrCacheWriteTokens   = "gen_ai.usage.cache_creation.input_tokens"
	attrReasoningTokens    = "gen_ai.usage.reasoning.output_tokens"
	attrTokenType          = "gen_ai.token.type"
	attrErrorType          = "error.type"

	attrGoLLMProvider       = "go_llm.provider"
	attrGoLLMEffort         = "go_llm.request.effort"
	attrGoLLMTools          = "go_llm.request.tools"
	attrGoLLMCostUSD        = "go_llm.usage.cost_usd"
	attrGoLLMCostSource     = "go_llm.usage.cost_source"
	attrGoLLMDroppedCalls   = "go_llm.response.dropped_tool_calls"
	attrGoLLMCanceled       = "go_llm.canceled"
	attrGoLLMAbandoned      = "go_llm.stream.abandoned"
	attrGoLLMBlockType      = "go_llm.block.type"
	attrGoLLMBlockIndex     = "go_llm.block.index"
	attrGoLLMBlockChars     = "go_llm.block.chars"
	attrGoLLMBlockTool      = "go_llm.block.tool"
	eventFirstChunk         = "go_llm.first_chunk"
	eventBlock              = "go_llm.block"
	metricOperationDuration = "gen_ai.client.operation.duration"
	metricTokenUsage        = "gen_ai.client.token.usage"
	metricTimeToFirstChunk  = "gen_ai.client.operation.time_to_first_chunk"
	metricCost              = "go_llm.client.cost"

	operationChat     = "chat"
	errorTypeOther    = "_OTHER"
	maxStatusDescSize = 500
)

// Bucket boundaries recommended by the GenAI semantic conventions.
var (
	durationBuckets = []float64{0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92}
	tokenBuckets    = []float64{1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864}
)

// Option configures Middleware.
type Option func(*config)

type config struct {
	tracerProvider trace.TracerProvider
	meterProvider  metric.MeterProvider
}

// WithTracerProvider sets the tracer provider. The default is the global
// provider from otel.GetTracerProvider at Middleware construction.
func WithTracerProvider(tp trace.TracerProvider) Option {
	return func(c *config) { c.tracerProvider = tp }
}

// WithMeterProvider sets the meter provider. The default is the global
// provider from otel.GetMeterProvider at Middleware construction.
func WithMeterProvider(mp metric.MeterProvider) Option {
	return func(c *config) { c.meterProvider = mp }
}

// Middleware returns llm.Wrap middleware that records one span and the GenAI
// client metrics per Chat or ChatStream call. Place it outermost (first among
// the Wrap arguments) to measure the whole call, including other middleware
// such as retries.
func Middleware(opts ...Option) llm.Middleware {
	cfg := config{tracerProvider: otel.GetTracerProvider(), meterProvider: otel.GetMeterProvider()}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	if cfg.tracerProvider == nil {
		cfg.tracerProvider = otel.GetTracerProvider()
	}
	if cfg.meterProvider == nil {
		cfg.meterProvider = otel.GetMeterProvider()
	}
	tracer := cfg.tracerProvider.Tracer(ScopeName)
	meter := cfg.meterProvider.Meter(ScopeName)
	inst := newInstruments(meter)

	return llm.Middleware{Bind: func(p llm.Provider) llm.Middleware {
		t := &telemetry{tracer: tracer, inst: inst, providerName: p.Name()}
		return llm.Middleware{
			Chat: func(next llm.ChatFunc) llm.ChatFunc {
				return func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
					ctx, call := t.start(ctx, req, false)
					resp, err := next(ctx, req)
					call.end(resp, err, false)
					return resp, err
				}
			},
			Stream: func(next llm.StreamFunc) llm.StreamFunc {
				return func(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
					return func(yield func(llm.Event, error) bool) {
						ctx, call := t.start(ctx, req, true)
						var collector llm.Collector
						var streamErr error
						abandoned := false
						for event, err := range next(ctx, req) {
							if err != nil {
								streamErr = err
							} else {
								call.observe(event)
								_ = collector.Add(event)
							}
							if !yield(event, err) {
								abandoned = err == nil
								break
							}
							if err != nil {
								break
							}
						}
						// The collector only supplies the response summary: its own
						// rejection of an event never becomes the call's error, which
						// is exactly what the consumer saw.
						resp, _ := collector.Finish(streamErr)
						err := streamErr
						if abandoned {
							err = nil
						}
						call.end(resp, err, abandoned)
					}
				}
			},
		}
	}}
}

type instruments struct {
	duration metric.Float64Histogram
	tokens   metric.Int64Histogram
	ttfc     metric.Float64Histogram
	cost     metric.Float64Counter
}

func newInstruments(meter metric.Meter) instruments {
	// Instrument creation only fails for invalid names; the no-op instruments
	// returned alongside such an error keep the middleware usable.
	duration, _ := meter.Float64Histogram(metricOperationDuration,
		metric.WithDescription("GenAI operation duration."), metric.WithUnit("s"),
		metric.WithExplicitBucketBoundaries(durationBuckets...))
	tokens, _ := meter.Int64Histogram(metricTokenUsage,
		metric.WithDescription("Number of input and output tokens used."), metric.WithUnit("{token}"),
		metric.WithExplicitBucketBoundaries(tokenBuckets...))
	ttfc, _ := meter.Float64Histogram(metricTimeToFirstChunk,
		metric.WithDescription("Time to receive the first chunk of a streamed response."), metric.WithUnit("s"),
		metric.WithExplicitBucketBoundaries(durationBuckets...))
	cost, _ := meter.Float64Counter(metricCost,
		metric.WithDescription("Request cost in US dollars, where known."), metric.WithUnit("USD"))
	return instruments{duration: duration, tokens: tokens, ttfc: ttfc, cost: cost}
}

type telemetry struct {
	tracer       trace.Tracer
	inst         instruments
	providerName string
}

type call struct {
	t          *telemetry
	ctx        context.Context
	span       trace.Span
	start      time.Time
	model      string
	firstChunk time.Duration
	sawChunk   bool
}

func (t *telemetry) start(ctx context.Context, req *llm.Request, stream bool) (context.Context, *call) {
	model := ""
	if req != nil {
		model = req.Model
	}
	attrs := []attribute.KeyValue{
		attribute.String(attrOperationName, operationChat),
		attribute.String(attrProviderName, t.providerName),
		attribute.String(attrGoLLMProvider, t.providerName),
		attribute.String(attrRequestModel, model),
		attribute.Bool(attrRequestStream, stream),
	}
	if req != nil {
		if req.MaxTokens > 0 {
			attrs = append(attrs, attribute.Int(attrRequestMaxTokens, req.MaxTokens))
		}
		if req.Temperature != nil {
			attrs = append(attrs, attribute.Float64(attrRequestTemperature, *req.Temperature))
		}
		if req.TopP != nil {
			attrs = append(attrs, attribute.Float64(attrRequestTopP, *req.TopP))
		}
		if req.Effort != "" {
			attrs = append(attrs, attribute.String(attrGoLLMEffort, string(req.Effort)))
		}
		if len(req.Tools) > 0 {
			attrs = append(attrs, attribute.Int(attrGoLLMTools, len(req.Tools)))
		}
	}
	name := operationChat
	if model != "" {
		name += " " + model
	}
	ctx, span := t.tracer.Start(ctx, name, trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(attrs...))
	return ctx, &call{t: t, ctx: ctx, span: span, start: time.Now(), model: model}
}

// observe records the first content chunk of a stream. Content is never read
// beyond its presence.
func (c *call) observe(event llm.Event) {
	if c.sawChunk {
		return
	}
	switch e := event.(type) {
	case llm.TextDelta:
		if e.Text == "" {
			return
		}
	case llm.ReasoningDelta:
		if e.Text == "" {
			return
		}
	case llm.ToolCallStart:
	default:
		return
	}
	c.sawChunk = true
	c.firstChunk = time.Since(c.start)
	c.span.AddEvent(eventFirstChunk)
}

func (c *call) end(resp *llm.Response, err error, abandoned bool) {
	elapsed := time.Since(c.start)
	metricAttrs := []attribute.KeyValue{
		attribute.String(attrOperationName, operationChat),
		attribute.String(attrProviderName, c.t.providerName),
		attribute.String(attrRequestModel, c.model),
	}
	if resp != nil && resp.Model != "" {
		metricAttrs = append(metricAttrs, attribute.String(attrResponseModel, resp.Model))
	}
	// error.type is defined for the duration metric only.
	durationAttrs := metricAttrs

	if resp != nil {
		c.recordResponse(resp)
	}
	if c.sawChunk {
		c.span.SetAttributes(attribute.Float64(attrTimeToFirstChunk, c.firstChunk.Seconds()))
	}
	if abandoned {
		c.span.SetAttributes(attribute.Bool(attrGoLLMAbandoned, true))
	}

	switch {
	case err == nil:
	case errors.Is(err, context.Canceled):
		// The caller chose to stop: not an operation error.
		c.span.SetAttributes(attribute.Bool(attrGoLLMCanceled, true))
	default:
		errorType := llm.SafeErrorInfo(err).Class
		if errorType == "" {
			errorType = errorTypeOther
		}
		c.span.SetAttributes(attribute.String(attrErrorType, errorType))
		c.span.SetStatus(codes.Error, statusDescription(err, errorType))
		durationAttrs = append(append([]attribute.KeyValue(nil), metricAttrs...), attribute.String(attrErrorType, errorType))
	}

	set := metric.WithAttributeSet(attribute.NewSet(metricAttrs...))
	c.t.inst.duration.Record(c.ctx, elapsed.Seconds(), metric.WithAttributeSet(attribute.NewSet(durationAttrs...)))
	if c.sawChunk {
		c.t.inst.ttfc.Record(c.ctx, c.firstChunk.Seconds(), set)
	}
	if resp != nil {
		usage := resp.Usage
		prompt := usage.InputTokens + usage.CacheReadTokens + usage.CacheWriteTokens
		if prompt > 0 || usage.OutputTokens > 0 {
			c.t.inst.tokens.Record(c.ctx, prompt, metric.WithAttributeSet(attribute.NewSet(append(metricAttrs, attribute.String(attrTokenType, "input"))...)))
			c.t.inst.tokens.Record(c.ctx, usage.OutputTokens, metric.WithAttributeSet(attribute.NewSet(append(metricAttrs, attribute.String(attrTokenType, "output"))...)))
		}
		if usage.CostUSD != nil {
			c.t.inst.cost.Add(c.ctx, *usage.CostUSD, metric.WithAttributeSet(attribute.NewSet(append(metricAttrs, attribute.String(attrGoLLMCostSource, usage.CostSource))...)))
		}
	}
	c.span.End()
}

func (c *call) recordResponse(resp *llm.Response) {
	usage := resp.Usage
	var attrs []attribute.KeyValue
	if prompt := usage.InputTokens + usage.CacheReadTokens + usage.CacheWriteTokens; prompt > 0 || usage.OutputTokens > 0 {
		attrs = append(attrs, attribute.Int64(attrInputTokens, prompt), attribute.Int64(attrOutputTokens, usage.OutputTokens))
	}
	if resp.ID != "" {
		attrs = append(attrs, attribute.String(attrResponseID, resp.ID))
	}
	if resp.Model != "" {
		attrs = append(attrs, attribute.String(attrResponseModel, resp.Model))
	}
	if resp.StopReason != "" {
		attrs = append(attrs, attribute.StringSlice(attrFinishReasons, []string{string(resp.StopReason)}))
	}
	if usage.CacheReadTokens > 0 {
		attrs = append(attrs, attribute.Int64(attrCacheReadTokens, usage.CacheReadTokens))
	}
	if usage.CacheWriteTokens > 0 {
		attrs = append(attrs, attribute.Int64(attrCacheWriteTokens, usage.CacheWriteTokens))
	}
	if usage.ReasoningTokens > 0 {
		attrs = append(attrs, attribute.Int64(attrReasoningTokens, usage.ReasoningTokens))
	}
	if usage.CostUSD != nil {
		attrs = append(attrs, attribute.Float64(attrGoLLMCostUSD, *usage.CostUSD), attribute.String(attrGoLLMCostSource, usage.CostSource))
	}
	if n := len(resp.DroppedToolCalls); n > 0 {
		attrs = append(attrs, attribute.Int(attrGoLLMDroppedCalls, n))
	}
	c.span.SetAttributes(attrs...)

	for index, part := range resp.Parts {
		var blockType, tool string
		var chars int
		switch p := part.(type) {
		case llm.TextPart:
			blockType, chars = "text", utf8.RuneCountInString(p.Text)
		case llm.ReasoningPart:
			blockType, chars = "reasoning", utf8.RuneCountInString(p.Text)
		case llm.ToolCallPart:
			blockType, chars, tool = "tool_call", utf8.RuneCount(p.Args), p.Name
		default:
			continue
		}
		blockAttrs := []attribute.KeyValue{
			attribute.String(attrGoLLMBlockType, blockType),
			attribute.Int(attrGoLLMBlockIndex, index),
			attribute.Int(attrGoLLMBlockChars, chars),
		}
		if tool != "" {
			blockAttrs = append(blockAttrs, attribute.String(attrGoLLMBlockTool, tool))
		}
		c.span.AddEvent(eventBlock, trace.WithAttributes(blockAttrs...))
	}
}

// statusDescription is content-free for every error: provider errors use
// llm.SafeError, which excludes provider-controlled detail, while any other
// error (for example one authored by caller middleware inside the chain,
// which may quote request content) is reduced to its class.
func statusDescription(err error, errorType string) string {
	var providerErr *llm.ProviderError
	if errors.As(err, &providerErr) {
		return truncate(llm.SafeError(err), maxStatusDescSize)
	}
	return "llm: " + errorType
}

func truncate(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	cut := limit
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut]
}
