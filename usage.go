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
	cost          usageCostAggregation
}

type usageCostAggregation struct {
	initialized    bool
	allTokensCost  bool
	allCostsNative bool
}

// UsageStats is a snapshot of aggregated usage.
type UsageStats struct {
	Calls           int64
	Errors          int64
	Usage           Usage
	TotalDuration   time.Duration
	ByProviderModel map[string]UsageStats
}

// NewUsageTracker returns a goroutine-safe usage aggregator.
func NewUsageTracker() *UsageTracker {
	return &UsageTracker{buckets: make(map[string]usageAccumulator)}
}

// Middleware returns middleware that observes Chat and ChatStream calls.
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
				t.record(provider, model, usage, err != nil, time.Since(start))
				return resp, err
			}
		},
		Stream: func(next StreamFunc) StreamFunc {
			return func(ctx context.Context, req *Request) iter.Seq2[Event, error] {
				events := next(ctx, req)
				return t.trackStream(defaultProvider, req, events)
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
		recorded := false
		streamErr := false
		defer func() {
			if !recorded {
				t.record(provider, model, usage, streamErr, time.Since(start))
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
			case MessageEnd:
				usage = e.Usage
				t.record(provider, model, usage, false, time.Since(start))
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

func (t *UsageTracker) record(provider, model string, usage Usage, isErr bool, duration time.Duration) {
	key := modelKey(provider, model)
	t.mu.Lock()
	defer t.mu.Unlock()

	t.total.add(usage, isErr, duration)
	bucket := t.buckets[key]
	bucket.add(usage, isErr, duration)
	t.buckets[key] = bucket
}

func (a *usageAccumulator) add(usage Usage, isErr bool, duration time.Duration) {
	a.Calls++
	if isErr {
		a.Errors++
	}
	a.Usage = sumUsage(a.Usage, usage, &a.cost)
	a.TotalDuration += duration
}

func (a usageAccumulator) stats() UsageStats {
	return UsageStats{
		Calls:         a.Calls,
		Errors:        a.Errors,
		Usage:         cloneUsage(a.Usage),
		TotalDuration: a.TotalDuration,
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

// sumUsage adds next into total. CostSource provenance: a sum of all-native
// costs stays native; mixing native and estimated components (or adding any
// estimated cost) marks the sum estimated — a total is only billing-grade
// when every part of it is.
func sumUsage(total, next Usage, state *usageCostAggregation) Usage {
	totalCostState := *state
	if !totalCostState.initialized {
		totalCostState = usageCostAggregationFor(total)
	}
	nextCostState := usageCostAggregationFor(next)

	total.InputTokens += next.InputTokens
	total.OutputTokens += next.OutputTokens
	total.TotalTokens += next.TotalTokens
	total.CacheReadTokens += next.CacheReadTokens
	total.CacheWriteTokens += next.CacheWriteTokens
	total.ReasoningTokens += next.ReasoningTokens
	if next.CostUSD != nil {
		sum := *next.CostUSD
		if total.CostUSD != nil {
			sum += *total.CostUSD
		}
		total.CostUSD = &sum
	}
	*state = usageCostAggregation{
		initialized:    true,
		allTokensCost:  totalCostState.allTokensCost && nextCostState.allTokensCost,
		allCostsNative: totalCostState.allCostsNative && nextCostState.allCostsNative,
	}
	if total.CostUSD == nil {
		total.CostSource = ""
	} else if state.allTokensCost && state.allCostsNative {
		total.CostSource = CostSourceNative
	} else {
		total.CostSource = CostSourceEstimated
	}
	return total
}

func usageCostAggregationFor(usage Usage) usageCostAggregation {
	hasCost := usage.CostUSD != nil
	return usageCostAggregation{
		initialized:    true,
		allTokensCost:  !usageHasTokens(usage) || hasCost,
		allCostsNative: !hasCost || usage.CostSource == CostSourceNative,
	}
}

func usageHasTokens(u Usage) bool {
	return u.InputTokens > 0 || u.OutputTokens > 0 || u.TotalTokens > 0 ||
		u.CacheReadTokens > 0 || u.CacheWriteTokens > 0
}

func cloneUsage(usage Usage) Usage {
	if usage.CostUSD != nil {
		cost := *usage.CostUSD
		usage.CostUSD = &cost
	}
	return usage
}
