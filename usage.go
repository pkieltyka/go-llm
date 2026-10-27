package llm

import (
	"context"
	"iter"
	"sync"
	"time"
)

// UsageTracker aggregates provider usage through middleware.
type UsageTracker struct {
	mu      sync.Mutex
	total   usageAccumulator
	buckets map[string]usageAccumulator
}

type usageAccumulator struct {
	Calls         int64
	Errors        int64
	Usage         Usage
	TotalDuration time.Duration
	streamStats   streamStats
}

// streamStats aggregates streaming latency telemetry. Averages are the
// caller's division: TotalTimeToMessageStart/MessageStartSamples and
// TotalTimeToFirstContent/FirstContentSamples.
type streamStats struct {
	StreamCalls             int64
	MessageStartSamples     int64
	TotalTimeToMessageStart time.Duration
	FirstContentSamples     int64
	TotalTimeToFirstContent time.Duration
}

// streamTimings carries one stream's observed latencies into record. Streams
// that never produced the corresponding event contribute no sample.
type streamTimings struct {
	timeToMessageStart time.Duration
	messageStartSeen   bool
	timeToFirstContent time.Duration
	firstContentSeen   bool
}

// UsageStats is a snapshot of aggregated usage.
type UsageStats struct {
	Calls         int64
	Errors        int64
	Usage         Usage
	TotalDuration time.Duration
	// Streaming latency telemetry (ChatStream calls only; blocking calls
	// leave every stream field zero). Time-to-message-start measures stream
	// open → MessageStart; time-to-first-content measures stream open → the
	// first non-empty TextDelta, non-empty ReasoningDelta, or
	// ToolCallStart. Empty or error-only streams contribute no sample, so
	// sample counts can trail StreamCalls; averages are
	// TotalTimeToMessageStart/MessageStartSamples and
	// TotalTimeToFirstContent/FirstContentSamples.
	StreamCalls             int64
	MessageStartSamples     int64
	TotalTimeToMessageStart time.Duration
	FirstContentSamples     int64
	TotalTimeToFirstContent time.Duration
	ByProviderModel         map[string]UsageStats
}

// NewUsageTracker returns a goroutine-safe usage aggregator.
func NewUsageTracker() *UsageTracker {
	return &UsageTracker{buckets: make(map[string]usageAccumulator)}
}

// Middleware returns middleware that observes Chat, ChatStream and Classify
// calls.
func (t *UsageTracker) Middleware() Middleware {
	if t == nil {
		return Middleware{}
	}
	return Middleware{
		Bind: func(p Provider) Middleware {
			provider := ""
			if p != nil {
				provider = p.Name()
			}
			return t.middlewareForProvider(provider)
		},
	}
}

func (t *UsageTracker) middlewareForProvider(defaultProvider string) Middleware {
	return Middleware{
		Chat: func(next ChatFunc) ChatFunc {
			return func(ctx context.Context, req *Request) (*Response, error) {
				start := time.Now()
				resp, err := next(ctx, req)
				provider, model, usage := identityFromResponse(resp, req)
				if provider == "" {
					provider = defaultProvider
				}
				t.record(provider, model, usage, err != nil, time.Since(start), nil)
				return resp, err
			}
		},
		Stream: func(next StreamFunc) StreamFunc {
			return func(ctx context.Context, req *Request) iter.Seq2[Event, error] {
				events := next(ctx, req)
				return t.trackStream(defaultProvider, req, events)
			}
		},
		Classify: func(next ClassifyFunc) ClassifyFunc {
			return func(ctx context.Context, req *ClassifyRequest) (*ClassifyResult, error) {
				start := time.Now()
				result, err := next(ctx, req)
				provider, model, usage := defaultProvider, "", Usage{}
				if req != nil {
					model = req.Model
				}
				// A failed but billed call (a refusal, a malformed answer)
				// returns its usage beside the error; it is counted.
				if result != nil {
					if result.Provider != "" {
						provider = result.Provider
					}
					if result.Model != "" {
						model = result.Model
					}
					usage = result.Usage
				}
				t.record(provider, model, usage, err != nil, time.Since(start), nil)
				return result, err
			}
		},
	}
}

func (t *UsageTracker) trackStream(defaultProvider string, req *Request, events iter.Seq2[Event, error]) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		start := time.Now()
		provider := defaultProvider
		model := requestModel(req)
		var usage Usage
		var timings streamTimings
		recorded := false
		streamErr := false
		defer func() {
			if !recorded {
				t.record(provider, model, usage, streamErr, time.Since(start), &timings)
			}
		}()

		for event, err := range events {
			if err != nil {
				streamErr = true
				yield(event, err)
				return
			}
			normalized, normalizeErr := normalizeEvent(event)
			if normalizeErr != nil {
				streamErr = true
				yield(nil, normalizeErr)
				return
			}
			switch e := normalized.(type) {
			case MessageStart:
				if e.Provider != "" {
					provider = e.Provider
				}
				if e.Model != "" {
					model = e.Model
				}
				if !timings.messageStartSeen {
					timings.messageStartSeen = true
					timings.timeToMessageStart = time.Since(start)
				}
			case TextDelta:
				if e.Text != "" {
					timings.observeFirstContent(start)
				}
			case ReasoningDelta:
				if e.Text != "" {
					timings.observeFirstContent(start)
				}
			case ToolCallStart:
				timings.observeFirstContent(start)
			case MessageEnd:
				usage = e.Usage
				t.record(provider, model, usage, false, time.Since(start), &timings)
				recorded = true
			}
			if !yield(normalized, nil) {
				return
			}
		}
	}
}

// Stats returns a copy of the current usage counters.
func (t *UsageTracker) Stats() UsageStats {
	if t == nil {
		return UsageStats{}
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	out := t.total.stats()
	out.ByProviderModel = make(map[string]UsageStats, len(t.buckets))
	for key, bucket := range t.buckets {
		out.ByProviderModel[key] = bucket.stats()
	}
	return out
}

func (t *UsageTracker) record(provider, model string, usage Usage, isErr bool, duration time.Duration, timings *streamTimings) {
	key := modelKey(provider, model)
	t.mu.Lock()
	defer t.mu.Unlock()

	t.total.add(usage, isErr, duration, timings)
	bucket := t.buckets[key]
	bucket.add(usage, isErr, duration, timings)
	t.buckets[key] = bucket
}

func (a *usageAccumulator) add(usage Usage, isErr bool, duration time.Duration, timings *streamTimings) {
	a.Calls++
	if isErr {
		a.Errors++
	}
	a.Usage = a.Usage.Add(usage)
	a.TotalDuration += duration
	if timings != nil {
		a.streamStats.StreamCalls++
		if timings.messageStartSeen {
			a.streamStats.MessageStartSamples++
			a.streamStats.TotalTimeToMessageStart += timings.timeToMessageStart
		}
		if timings.firstContentSeen {
			a.streamStats.FirstContentSamples++
			a.streamStats.TotalTimeToFirstContent += timings.timeToFirstContent
		}
	}
}

func (s *streamTimings) observeFirstContent(start time.Time) {
	if !s.firstContentSeen {
		s.firstContentSeen = true
		s.timeToFirstContent = time.Since(start)
	}
}

func (a usageAccumulator) stats() UsageStats {
	return UsageStats{
		Calls:                   a.Calls,
		Errors:                  a.Errors,
		Usage:                   cloneUsage(a.Usage),
		TotalDuration:           a.TotalDuration,
		StreamCalls:             a.streamStats.StreamCalls,
		MessageStartSamples:     a.streamStats.MessageStartSamples,
		TotalTimeToMessageStart: a.streamStats.TotalTimeToMessageStart,
		FirstContentSamples:     a.streamStats.FirstContentSamples,
		TotalTimeToFirstContent: a.streamStats.TotalTimeToFirstContent,
	}
}

func identityFromResponse(resp *Response, req *Request) (string, string, Usage) {
	if resp == nil {
		return "", requestModel(req), Usage{}
	}
	model := resp.Model
	if model == "" {
		model = requestModel(req)
	}
	return resp.Provider, model, resp.Usage
}

func requestModel(req *Request) string {
	if req == nil {
		return ""
	}
	return req.Model
}

// Add returns summed token counters and known costs without mutating either
// input. CacheWrite1hTokens and ReasoningTokens remain informational subsets.
// Token-bearing usage with unknown cost, or an explicit CostIncomplete flag,
// makes the result incomplete; it never estimates the missing charge. A nil
// cost stays nil unless at least one input has a known cost, including zero.
// CostSource describes only the known costs: all-native costs stay native;
// any estimated or unspecified source makes the subtotal estimated.
//
// The result is self-contained for further aggregation or JSON persistence;
// no hidden accumulator state is needed. Adding empty Usage preserves accounting
// while normalizing legacy cost metadata according to the rules above.
// Raw is discarded because a sum has no single provider payload.
func (total Usage) Add(next Usage) Usage {
	allCostsNative := (total.CostUSD == nil || total.CostSource == CostSourceNative) &&
		(next.CostUSD == nil || next.CostSource == CostSourceNative)
	total.CostIncomplete = total.CostIncomplete || next.CostIncomplete ||
		(total.CostUSD == nil && usageHasTokens(total)) ||
		(next.CostUSD == nil && usageHasTokens(next))

	total.InputTokens += next.InputTokens
	total.OutputTokens += next.OutputTokens
	total.TotalTokens += next.TotalTokens
	total.CacheReadTokens += next.CacheReadTokens
	total.CacheWriteTokens += next.CacheWriteTokens
	total.CacheWrite1hTokens += next.CacheWrite1hTokens
	total.ReasoningTokens += next.ReasoningTokens
	if next.CostUSD != nil {
		sum := *next.CostUSD
		if total.CostUSD != nil {
			sum += *total.CostUSD
		}
		total.CostUSD = &sum
	} else {
		// Return independent cost storage even when only total has a cost.
		total = cloneUsage(total)
	}
	if total.CostUSD == nil {
		total.CostSource = ""
	} else if allCostsNative {
		total.CostSource = CostSourceNative
	} else {
		total.CostSource = CostSourceEstimated
	}
	total.Raw = nil
	return total
}

func usageHasTokens(u Usage) bool {
	return u.InputTokens > 0 || u.OutputTokens > 0 || u.TotalTokens > 0 ||
		u.CacheReadTokens > 0 || u.CacheWriteTokens > 0 ||
		u.CacheWrite1hTokens > 0 || u.ReasoningTokens > 0
}

func cloneUsage(usage Usage) Usage {
	if usage.CostUSD != nil {
		cost := *usage.CostUSD
		usage.CostUSD = &cost
	}
	return usage
}
