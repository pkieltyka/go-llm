package llm_test

import (
	"context"
	"math"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// knownNoWrite prices input, output and cache reads, but has no
// cache-write rate.
func knownNoWrite() llm.ModelPricing {
	return llm.ModelPricing{
		InputPerMTok: 0.4, OutputPerMTok: 1.6, CacheReadPerMTok: 0.1,
		Availability: &llm.ModelPricingAvailability{InputPerMTok: true, OutputPerMTok: true, CacheReadPerMTok: true},
	}
}

func near(got *float64, want float64) bool {
	return got != nil && math.Abs(*got-want) < 1e-12
}

// Plan 14 A2: without the rule an unknown write rate keeps the bound
// unknown; with it, written tokens cost uncached input.
func TestCacheWritesAsInput(t *testing.T) {
	plain := knownNoWrite()
	if _, ok := llm.EstimateCostBound(8192, 256, plain); ok {
		t.Fatal("an unknown write rate without the rule must leave the bound unknown")
	}
	if got := llm.EstimateCost(llm.Usage{CacheWriteTokens: 1_000_000}, plain); got.CostUSD != nil {
		t.Fatalf("unruled cache writes priced at %v, want unknown", *got.CostUSD)
	}

	ruled := knownNoWrite()
	ruled.CacheWritesAsInput = true
	bound, ok := llm.EstimateCostBound(8192, 256, ruled)
	want := (8192*0.4 + 256*1.6) / 1_000_000
	if !ok || math.Abs(bound-want) > 1e-12 {
		t.Fatalf("bound = %v, %v, want %v (the input interpretation)", bound, ok, want)
	}
	usage := llm.Usage{InputTokens: 100, CacheReadTokens: 200, CacheWriteTokens: 1_000_000, OutputTokens: 10}
	got := llm.EstimateCost(usage, ruled)
	if !near(got.CostUSD, (100*0.4+200*0.1+1_000_000*0.4+10*1.6)/1_000_000) || got.CostSource != llm.CostSourceEstimated {
		t.Fatalf("ruled cost = %v (%v)", got.CostUSD, got.CostSource)
	}
	if ruled.Availability.CacheWritePerMTok || ruled.CacheWritePerMTok != 0 {
		t.Fatalf("estimation mutated the caller's pricing: %+v %+v", ruled, *ruled.Availability)
	}

	t.Run("published write rate wins", func(t *testing.T) {
		p := knownNoWrite()
		p.CacheWritesAsInput = true
		p.CacheWritePerMTok = 0.5
		p.Availability.CacheWritePerMTok = true
		if got := llm.EstimateCost(llm.Usage{CacheWriteTokens: 1_000_000}, p); !near(got.CostUSD, 0.5) {
			t.Fatalf("cost = %v, want 0.5", got.CostUSD)
		}
	})
	t.Run("explicit zero write rate stays zero", func(t *testing.T) {
		p := knownNoWrite()
		p.CacheWritesAsInput = true
		p.Availability.CacheWritePerMTok = true
		if got := llm.EstimateCost(llm.Usage{CacheWriteTokens: 1_000_000}, p); !near(got.CostUSD, 0) {
			t.Fatalf("cost = %v, want 0", got.CostUSD)
		}
	})
	t.Run("unknown input rate stays unknown", func(t *testing.T) {
		p := knownNoWrite()
		p.CacheWritesAsInput = true
		p.Availability.InputPerMTok = false
		if _, ok := llm.EstimateCostBound(10, 10, p); ok {
			t.Fatal("bound known without an input rate")
		}
		if got := llm.EstimateCost(llm.Usage{CacheWriteTokens: 10}, p); got.CostUSD != nil {
			t.Fatalf("cost = %v, want unknown", *got.CostUSD)
		}
	})
	t.Run("one-hour writes stay unknown", func(t *testing.T) {
		if got := llm.EstimateCost(llm.Usage{CacheWriteTokens: 10, CacheWrite1hTokens: 10}, ruled); got.CostUSD != nil {
			t.Fatalf("cost = %v, want unknown", *got.CostUSD)
		}
	})
	t.Run("tier input rate", func(t *testing.T) {
		p := knownNoWrite()
		p.CacheWritesAsInput = true
		p.Tiers = []llm.ModelPricingTier{{
			InputTokensAbove: 1000, InputPerMTok: 0.8, OutputPerMTok: 3.2, CacheReadPerMTok: 0.2,
			Availability: &llm.ModelPricingAvailability{InputPerMTok: true, OutputPerMTok: true, CacheReadPerMTok: true},
		}}
		if got := llm.EstimateCost(llm.Usage{CacheWriteTokens: 1000}, p); !near(got.CostUSD, 1000*0.4/1_000_000) {
			t.Fatalf("at the threshold cost = %v, want the base input rate", got.CostUSD)
		}
		if got := llm.EstimateCost(llm.Usage{CacheWriteTokens: 1001}, p); !near(got.CostUSD, 1001*0.8/1_000_000) {
			t.Fatalf("above the threshold cost = %v, want the tier input rate", got.CostUSD)
		}
		bound, ok := llm.EstimateCostBound(2000, 100, p)
		if want := (2000*0.8 + 100*3.2) / 1_000_000; !ok || math.Abs(bound-want) > 1e-12 {
			t.Fatalf("tier bound = %v, %v, want %v", bound, ok, want)
		}
		p.Tiers[0].Availability.InputPerMTok = false
		if _, ok := llm.EstimateCostBound(2000, 100, p); ok {
			t.Fatal("tier bound known without the tier input rate")
		}
	})
	t.Run("covers every mixture", func(t *testing.T) {
		tiered := knownNoWrite()
		tiered.CacheWritesAsInput = true
		tiered.Tiers = []llm.ModelPricingTier{{
			InputTokensAbove: 30_000, InputPerMTok: 0.8, OutputPerMTok: 3.2, CacheReadPerMTok: 0.2,
			Availability: &llm.ModelPricingAvailability{InputPerMTok: true, OutputPerMTok: true, CacheReadPerMTok: true},
		}}
		for name, pricing := range map[string]llm.ModelPricing{"flat": ruled, "tiered": tiered} {
			for _, prompt := range []int64{20_000, 50_000} {
				const output = 2_000
				bound, ok := llm.EstimateCostBound(prompt, output, pricing)
				if !ok {
					t.Fatalf("%s %d: bound unknown", name, prompt)
				}
				for written := int64(0); written <= prompt; written += 5_000 {
					for read := int64(0); read+written <= prompt; read += 5_000 {
						usage := llm.Usage{InputTokens: prompt - read - written, CacheReadTokens: read, CacheWriteTokens: written, OutputTokens: output}
						if got := llm.EstimateCost(usage, pricing); got.CostUSD == nil || *got.CostUSD > bound+1e-15 {
							t.Fatalf("%s: %+v costs %v, above the bound %v", name, usage, got.CostUSD, bound)
						}
					}
				}
			}
		}
	})
}

// The embedded catalog applies the rule to the documented OpenAI families
// on both endpoints, and nowhere else (plan 14 A0).
func TestEmbeddedCacheWritesAsInputRule(t *testing.T) {
	for _, tt := range []struct {
		provider, id string
		rule         bool
	}{
		{"openrouter", "openai/gpt-4.1-mini", true},
		{"openai", "gpt-4.1-mini", true},
		{"openai", "gpt-4o-2024-08-06", true},
		{"openai", "gpt-5.5", true},
		{"openai", "gpt-5.6-luna", false},
		{"openrouter", "openai/gpt-5.6-luna:batch", false},
		{"openrouter", "openai/gpt-4.1-mini:batch", false},
		{"openrouter", "openai/gpt-oss-120b", false},
		{"openai", "gpt-5.5-pro", false},
		{"openai", "gpt-5.3-codex", false},
		{"openrouter", "openai/gpt-audio", false},
		{"anthropic", "claude-haiku-4-5", false},
	} {
		info, ok := llm.LookupModelInfo(tt.provider, tt.id)
		if !ok || info.Pricing == nil {
			t.Fatalf("%s/%s not in the embedded table", tt.provider, tt.id)
		}
		if info.Pricing.CacheWritesAsInput != tt.rule {
			t.Errorf("%s/%s CacheWritesAsInput = %v, want %v", tt.provider, tt.id, info.Pricing.CacheWritesAsInput, tt.rule)
		}
	}

	// Go-Agent's strict preflight case: an 8,192-token prompt with 256
	// output tokens on OpenRouter GPT-4.1-mini.
	info, _ := llm.LookupModelInfo("openrouter", "openai/gpt-4.1-mini")
	bound, ok := llm.EstimateCostBound(8192, 256, *info.Pricing)
	want := (8192*info.Pricing.InputPerMTok + 256*info.Pricing.OutputPerMTok) / 1_000_000
	if !ok || math.Abs(bound-want) > 1e-12 {
		t.Fatalf("GPT-4.1-mini bound = %v, %v, want %v", bound, ok, want)
	}
	batch, _ := llm.LookupModelInfo("openrouter", "openai/gpt-5.6-luna:batch")
	if _, ok := llm.EstimateCostBound(8192, 256, *batch.Pricing); ok {
		t.Fatal("GPT-5.6 batch has no published write rate; its bound must stay unknown")
	}
}

// Go-Agent resolves pricing through ResolveModelInfo, where OpenRouter's
// live listing wins over the embedded table. The table's rule must still
// reach a live row that publishes no write rate.
func TestResolveModelInfoCarriesCacheWritesAsInput(t *testing.T) {
	ctx := context.Background()
	live := knownNoWrite()
	p := llmtest.New(llmtest.WithName("openrouter"), llmtest.WithModels(
		llm.ModelInfo{ID: "openai/gpt-4.1-mini", Pricing: &live},
		llm.ModelInfo{ID: "openai/gpt-5.6-luna:batch", Pricing: func() *llm.ModelPricing { p := knownNoWrite(); return &p }()},
	))
	info, err := llm.ResolveModelInfo(ctx, p, "openai/gpt-4.1-mini")
	if err != nil {
		t.Fatal(err)
	}
	if !info.Pricing.CacheWritesAsInput || info.Pricing.InputPerMTok != live.InputPerMTok {
		t.Fatalf("resolved pricing = %+v, want the live rates with the table's rule", *info.Pricing)
	}
	if _, ok := llm.EstimateCostBound(8192, 256, *info.Pricing); !ok {
		t.Fatal("resolved GPT-4.1-mini bound is unknown")
	}
	if live.CacheWritesAsInput {
		t.Fatal("ResolveModelInfo mutated the provider's pricing")
	}

	batch, err := llm.ResolveModelInfo(ctx, p, "openai/gpt-5.6-luna:batch")
	if err != nil {
		t.Fatal(err)
	}
	if batch.Pricing.CacheWritesAsInput {
		t.Fatal("an unruled model gained the rule")
	}
	if _, ok := llm.EstimateCostBound(8192, 256, *batch.Pricing); ok {
		t.Fatal("GPT-5.6 batch bound must stay unknown")
	}
}

// Fuzzy lookups keep resolving metadata, but the billing rule only follows
// the identities the generator authorizes: the row's own ID or that ID with
// one -YYYY-MM-DD snapshot date.
func TestCacheWritesAsInputStaysOnAuthorizedIDs(t *testing.T) {
	base, _ := llm.LookupModelInfo("openai", "gpt-4.1-mini")
	for _, tt := range []struct {
		provider, id string
		rule         bool
	}{
		{"openai", "gpt-4.1-mini", true},
		{"openai", "gpt-4.1-mini-2025-04-14", true},
		{"openai", "gpt-4o-2024-08-06", true},
		{"openrouter", "openai/gpt-4.1-mini-2025-04-14", true},
		{"openai", "gpt-4.1-mini-2025-04-14:batch", false},
		{"openai", "gpt-4.1-mini-20250414-preview", false},
		{"openai", "gpt-4.1-mini-20250414", false},
		{"openai", "gpt-4.1-mini-2025-04-14-preview", false},
		{"openai", "gpt-4-1-mini", false},
		{"openai", "GPT-4.1-mini", false},
		{"openrouter", "openai/gpt-4.1-mini-2025-04-14:batch", false},
	} {
		info, ok := llm.LookupModelInfo(tt.provider, tt.id)
		if !ok || info.Pricing == nil {
			t.Fatalf("%s/%s no longer resolves", tt.provider, tt.id)
		}
		if info.Pricing.CacheWritesAsInput != tt.rule {
			t.Errorf("%s/%s CacheWritesAsInput = %v, want %v", tt.provider, tt.id, info.Pricing.CacheWritesAsInput, tt.rule)
		}
		if tt.provider == "openai" && tt.id != "gpt-4o-2024-08-06" && info.Pricing.InputPerMTok != base.Pricing.InputPerMTok {
			t.Errorf("%s/%s input rate = %v, want the base row's %v", tt.provider, tt.id, info.Pricing.InputPerMTok, base.Pricing.InputPerMTok)
		}
		if _, ok := llm.EstimateCostBound(8192, 256, *info.Pricing); ok != tt.rule {
			t.Errorf("%s/%s bound known = %v, want %v", tt.provider, tt.id, ok, tt.rule)
		}
	}
	if again, _ := llm.LookupModelInfo("openai", "gpt-4.1-mini"); !again.Pricing.CacheWritesAsInput {
		t.Fatal("a fuzzy lookup cleared the flag in the shared table")
	}
}

// A live row that SameModel matches but the rule excludes (here :batch,
// standing in for a missing base row) cannot inherit the base model's rule.
func TestResolveModelInfoKeepsRuleOffExcludedLiveRows(t *testing.T) {
	ctx := context.Background()
	for name, pricing := range map[string]*llm.ModelPricing{
		"live pricing":    func() *llm.ModelPricing { p := knownNoWrite(); return &p }(),
		"no live pricing": nil,
	} {
		p := llmtest.New(llmtest.WithName("openrouter"), llmtest.WithModels(
			llm.ModelInfo{ID: "openai/gpt-4.1-mini:batch", Pricing: pricing},
		))
		info, err := llm.ResolveModelInfo(ctx, p, "openai/gpt-4.1-mini")
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if info.ID != "openai/gpt-4.1-mini:batch" || info.Pricing == nil {
			t.Fatalf("%s: resolved %+v, want the live :batch row", name, info)
		}
		if info.Pricing.CacheWritesAsInput {
			t.Errorf("%s: the :batch row inherited the base model's rule", name)
		}
	}

	dated := llmtest.New(llmtest.WithName("openrouter"), llmtest.WithModels(
		llm.ModelInfo{ID: "openai/gpt-4.1-mini-2025-04-14", Pricing: func() *llm.ModelPricing { p := knownNoWrite(); return &p }()},
	))
	info, err := llm.ResolveModelInfo(ctx, dated, "openai/gpt-4.1-mini-2025-04-14")
	if err != nil || !info.Pricing.CacheWritesAsInput {
		t.Fatalf("dated live row = %+v, %v; want the rule", info.Pricing, err)
	}
}
