package llm_test

import (
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
	if llm.PriceTableDate() == "" {
		t.Fatalf("PriceTableDate returned empty string")
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
