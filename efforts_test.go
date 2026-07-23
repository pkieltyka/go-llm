package llm_test

import (
	"reflect"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestSupportedEffortsForModel(t *testing.T) {
	gpt5 := []llm.Effort{llm.EffortMinimal, llm.EffortLow, llm.EffortMedium, llm.EffortHigh}
	trio := []llm.Effort{llm.EffortLow, llm.EffortMedium, llm.EffortHigh}
	full := []llm.Effort{llm.EffortNone, llm.EffortMinimal, llm.EffortLow, llm.EffortMedium, llm.EffortHigh, llm.EffortXHigh, llm.EffortMax}

	cases := []struct {
		name     string
		provider string
		modelID  string
		want     []llm.Effort
	}{
		{"curated gpt-5.6", "openai", "gpt-5.6", gpt5},
		{"curated dated snapshot via prefix", "openai", "gpt-5.6-sol-2026-07-09", gpt5},
		{"curated codex id", "openai", "gpt-5.1-codex-max", trio},
		{"curated o-series", "openai", "o4-mini", trio},
		{"curated anthropic full dial", "anthropic", "claude-sonnet-4-5", full},
		{"aggregator via canonical fallback", "openrouter", "openai/gpt-5.6-luna", gpt5},
		{"inferred o-series absent from table", "openai", "o3-ultra", trio},
		{"inferred with vendor slash prefix", "somegateway", "openai/o4-mega", trio},
		{"inferred with provider colon prefix", "somegateway", "openai:gpt-5.9", gpt5},
		{"catalogued chat variant stays uncurated", "openai", "gpt-5-chat-latest", nil},
		{"chat variants never inferred", "openai", "gpt-6-chat-latest", nil},
		{"unrecognized model", "somegateway", "mystery-model", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := llm.SupportedEffortsForModel(tc.provider, tc.modelID)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("SupportedEffortsForModel(%q, %q) = %v, want %v", tc.provider, tc.modelID, got, tc.want)
			}
		})
	}
}

func TestSupportedEffortsReturnsClones(t *testing.T) {
	first := llm.SupportedEffortsForModel("openai", "gpt-5.6")
	if len(first) == 0 {
		t.Fatal("gpt-5.6 efforts missing from embedded table")
	}
	first[0] = llm.EffortMax
	second := llm.SupportedEffortsForModel("openai", "gpt-5.6")
	if second[0] == llm.EffortMax {
		t.Fatal("mutating a returned slice leaked into the embedded table")
	}
	info, ok := llm.LookupModelInfo("openai", "gpt-5.6")
	if !ok || len(info.SupportedEfforts) == 0 {
		t.Fatalf("LookupModelInfo lost efforts: ok=%v info=%+v", ok, info)
	}
	info.SupportedEfforts[0] = llm.EffortMax
	if again, _ := llm.LookupModelInfo("openai", "gpt-5.6"); again.SupportedEfforts[0] == llm.EffortMax {
		t.Fatal("mutating LookupModelInfo result leaked into the embedded table")
	}
}

// Effort-family normalization must not leak into pricing: aggregator price
// rows keep their vendor namespace, so the full aggregator id prices while
// the family-stripped form does not.
func TestEffortFamilyNormalizationNotAppliedToPricing(t *testing.T) {
	withNamespace := llm.EstimateCostForModel("openrouter", "openai/gpt-5.6-luna", llm.Usage{InputTokens: 1000})
	if withNamespace.CostUSD == nil {
		t.Fatal("full aggregator id no longer prices — pricing lookup changed")
	}
	stripped := llm.EstimateCostForModel("openrouter", "gpt-5.6-luna", llm.Usage{InputTokens: 1000})
	if stripped.CostUSD != nil {
		t.Fatal("family-stripped id priced — effort normalization leaked into pricing lookups")
	}
}
