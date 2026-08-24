package llm_test

import (
	"math"
	"sync"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestLookupModelInfoFallbacks(t *testing.T) {
	info, ok := llm.LookupModelInfo("anthropic", "claude-sonnet-4-5-20260701")
	if !ok {
		t.Fatalf("LookupModelInfo prefix fallback returned ok=false")
	}
	if info.ID != "claude-sonnet-4-5-20260701" {
		t.Fatalf("prefix fallback ID = %q", info.ID)
	}
	if info.Pricing == nil || info.Pricing.InputPerMTok != 3 {
		t.Fatalf("prefix fallback pricing = %+v", info.Pricing)
	}

	info, ok = llm.LookupModelInfo("openrouter", "anthropic/claude-sonnet-4-5")
	if !ok {
		t.Fatalf("LookupModelInfo canonical fallback returned ok=false")
	}
	if info.Pricing == nil || info.Pricing.OutputPerMTok != 15 {
		t.Fatalf("canonical fallback pricing = %+v", info.Pricing)
	}

	if _, ok := llm.LookupModelInfo("openai", "unknown-model"); ok {
		t.Fatalf("unknown model returned ok=true")
	}
	if _, ok := llm.LookupModelInfo("openai", "gpt-52"); ok {
		t.Fatalf("boundaryless prefix fallback returned ok=true")
	}
	if _, ok := llm.LookupModelInfo("openai", "gpt-5-chat-latest"); ok {
		t.Fatalf("non-dated gpt-5 variant inherited base-model metadata")
	}
	if llm.PriceTableDate() == "" {
		t.Fatalf("PriceTableDate returned empty string")
	}
}

func TestEstimateCostSelectsHighestStrictlyExceededPricingTier(t *testing.T) {
	pricing := llm.ModelPricing{
		InputPerMTok:      1,
		OutputPerMTok:     2,
		CacheReadPerMTok:  3,
		CacheWritePerMTok: 4,
		Tiers: []llm.ModelPricingTier{
			{InputTokensAbove: 200, InputPerMTok: 20, OutputPerMTok: 21, CacheReadPerMTok: 22, CacheWritePerMTok: 23},
			{InputTokensAbove: 100, InputPerMTok: 10, OutputPerMTok: 11, CacheReadPerMTok: 12, CacheWritePerMTok: 13},
		},
	}
	for _, tt := range []struct {
		name  string
		usage llm.Usage
		want  float64
	}{
		{
			name:  "base",
			usage: llm.Usage{InputTokens: 99, OutputTokens: 1_000_000},
			want:  2.000099,
		},
		{
			name:  "exact threshold stays lower",
			usage: llm.Usage{InputTokens: 40, CacheReadTokens: 30, CacheWriteTokens: 30, OutputTokens: 1_000_000},
			want:  2.00025,
		},
		{
			name:  "cache occupancy crosses first tier",
			usage: llm.Usage{InputTokens: 40, CacheReadTokens: 31, CacheWriteTokens: 30, OutputTokens: 1_000_000},
			want:  11.001162,
		},
		{
			name:  "highest matching unsorted tier",
			usage: llm.Usage{InputTokens: 201, OutputTokens: 1_000_000},
			want:  21.00402,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := llm.EstimateCost(tt.usage, pricing)
			if got.CostUSD == nil || math.Abs(*got.CostUSD-tt.want) > 1e-12 {
				t.Fatalf("cost = %v, want %.12f", got.CostUSD, tt.want)
			}
		})
	}
}

func TestEstimateCostHonorsComponentAvailability(t *testing.T) {
	pricing := llm.ModelPricing{
		OutputPerMTok: 2,
		Availability:  &llm.ModelPricingAvailability{OutputPerMTok: true},
	}

	covered := llm.EstimateCost(llm.Usage{OutputTokens: 500}, pricing)
	if covered.CostUSD == nil || *covered.CostUSD != 0.001 {
		t.Fatalf("covered cost = %v, want 0.001", covered.CostUSD)
	}

	unknown := llm.EstimateCost(llm.Usage{InputTokens: 1, OutputTokens: 500}, pricing)
	if unknown.CostUSD != nil || unknown.CostSource != "" {
		t.Fatalf("unknown component produced estimate: %+v", unknown)
	}

	pricing.Tiers = []llm.ModelPricingTier{{
		InputTokensAbove: 1, InputPerMTok: 3, OutputPerMTok: 4,
		CacheReadPerMTok: 1, CacheWritePerMTok: 2,
	}}
	tiered := llm.EstimateCost(llm.Usage{InputTokens: 2, OutputTokens: 1}, pricing)
	if tiered.CostUSD == nil || *tiered.CostUSD != 0.00001 {
		t.Fatalf("complete tier cost = %v, want 0.00001", tiered.CostUSD)
	}
}

func TestEstimateCostIgnoresInvalidCallerPricingTiers(t *testing.T) {
	pricing := llm.ModelPricing{
		InputPerMTok: 1,
		Tiers: []llm.ModelPricingTier{
			{InputTokensAbove: 100, InputPerMTok: 5},
			{InputTokensAbove: 200, InputPerMTok: -1},
			{InputTokensAbove: 300, InputPerMTok: math.NaN()},
			{InputTokensAbove: 400, InputPerMTok: 9, OutputPerMTok: 9, CacheReadPerMTok: 9, CacheWritePerMTok: 9},
		},
	}
	got := llm.EstimateCost(llm.Usage{InputTokens: 500}, pricing)
	if got.CostUSD == nil || *got.CostUSD != 0.0045 {
		t.Fatalf("cost = %v, want highest valid tier cost 0.0045", got.CostUSD)
	}

	table := llm.PriceTable{"provider/model": pricing}
	got = llm.EstimateCostWithTable(table, "provider", "model-2026", llm.Usage{InputTokens: 500})
	if got.CostUSD == nil || *got.CostUSD != 0.0045 {
		t.Fatalf("table cost = %v, want 0.0045", got.CostUSD)
	}

	baseOnly := llm.EstimateCost(llm.Usage{InputTokens: 500}, llm.ModelPricing{
		InputPerMTok: 1,
		Tiers: []llm.ModelPricingTier{
			{InputTokensAbove: -1, InputPerMTok: 8},
			{InputTokensAbove: 100, InputPerMTok: math.Inf(1)},
		},
	})
	if baseOnly.CostUSD == nil || *baseOnly.CostUSD != 0.0005 {
		t.Fatalf("invalid-tier fallback cost = %v, want base cost 0.0005", baseOnly.CostUSD)
	}
}

func TestPriceTablePrefixFallbackRequiresBoundary(t *testing.T) {
	table := llm.PriceTable{
		"openai/gpt-5": {InputPerMTok: 1},
	}
	if _, ok := table.Lookup("openai", "gpt-52"); ok {
		t.Fatalf("gpt-52 matched gpt-5 pricing")
	}
	if pricing, ok := table.Lookup("openai", "gpt-5.2"); !ok || pricing.InputPerMTok != 1 {
		t.Fatalf("gpt-5.2 lookup = %+v, %v; want boundary prefix match", pricing, ok)
	}
}

func TestEstimateCostForModel(t *testing.T) {
	usage := llm.Usage{
		InputTokens:      1_000_000,
		OutputTokens:     2_000_000,
		CacheReadTokens:  500_000,
		CacheWriteTokens: 250_000,
	}
	estimated := llm.EstimateCostForModel("anthropic", "claude-sonnet-4-5", usage)
	if estimated.CostUSD == nil {
		t.Fatalf("CostUSD is nil")
	}
	want := 1*3.0 + 2*15.0 + 0.5*0.3 + 0.25*3.75
	if *estimated.CostUSD != want {
		t.Fatalf("CostUSD = %v, want %v", *estimated.CostUSD, want)
	}
	if estimated.CostSource != llm.CostSourceEstimated {
		t.Fatalf("CostSource = %q, want %q", estimated.CostSource, llm.CostSourceEstimated)
	}

	reportedCost := 42.0
	usage.CostUSD = &reportedCost
	usage.CostSource = llm.CostSourceNative
	if got := llm.EstimateCostForModel("anthropic", "claude-sonnet-4-5", usage); got.CostUSD == nil || *got.CostUSD != reportedCost || got.CostSource != llm.CostSourceNative {
		t.Fatalf("provider-reported cost was overwritten: %+v (%q)", got.CostUSD, got.CostSource)
	}
}

func TestModelTableConcurrentLookup(t *testing.T) {
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			info, ok := llm.LookupModelInfo("openai", "gpt-5.2")
			if !ok || info.ContextWindow == 0 {
				t.Errorf("LookupModelInfo = %+v, %v", info, ok)
			}
		}()
	}
	wg.Wait()
}

func TestUsageContextUsage(t *testing.T) {
	usage := llm.Usage{InputTokens: 10, CacheReadTokens: 20, CacheWriteTokens: 30, OutputTokens: 40}
	got := usage.ContextUsage(200)
	if got.UsedTokens != 100 || got.Remaining != 100 || got.UsedPercent != 50 {
		t.Fatalf("ContextUsage = %+v, want used=100 remaining=100 percent=50", got)
	}
}

// Pins the embedded-table expectations Phase 3a relies on: the gpt-5.6 family
// rows exist with cache-write rates, dated snapshots resolve by prefix, and
// cache-write tokens cost at the selected cache-write rate rather than the
// input rate.
func TestEmbeddedTableGPT56CacheWritePricing(t *testing.T) {
	info, ok := llm.LookupModelInfo("openai", "gpt-5.6")
	if !ok || info.Pricing == nil {
		t.Fatalf("gpt-5.6 not in embedded table (ok=%v, pricing=%v)", ok, info.Pricing)
	}
	if info.Pricing.CacheWritePerMTok <= 0 {
		t.Fatalf("gpt-5.6 cache-write rate = %v, want > 0", info.Pricing.CacheWritePerMTok)
	}
	if info.Pricing.CacheWritePerMTok == info.Pricing.InputPerMTok {
		t.Fatalf("cache-write rate equals input rate (%v); cost test would be vacuous", info.Pricing.InputPerMTok)
	}
	for _, id := range []string{"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-sol-2026-07-09"} {
		if _, ok := llm.LookupModelInfo("openai", id); !ok {
			t.Fatalf("%s did not resolve in embedded table", id)
		}
	}

	costed := llm.EstimateCost(llm.Usage{CacheWriteTokens: 1_000_000}, *info.Pricing)
	wantRate := info.Pricing.CacheWritePerMTok
	if len(info.Pricing.Tiers) > 0 {
		wantRate = info.Pricing.Tiers[len(info.Pricing.Tiers)-1].CacheWritePerMTok
	}
	if costed.CostUSD == nil || *costed.CostUSD != wantRate {
		t.Fatalf("cache-write-only cost = %v, want %v", costed.CostUSD, wantRate)
	}
	if costed.CostSource != llm.CostSourceEstimated {
		t.Fatalf("cost source = %q, want estimated", costed.CostSource)
	}
}
