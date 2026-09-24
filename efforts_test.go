package llm_test

import (
	"reflect"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestSupportedEffortsForModel(t *testing.T) {
	gpt5 := []llm.Effort{llm.EffortMinimal, llm.EffortLow, llm.EffortMedium, llm.EffortHigh}
	trio := []llm.Effort{llm.EffortLow, llm.EffortMedium, llm.EffortHigh}

	cases := []struct {
		name     string
		provider string
		modelID  string
		want     []llm.Effort
	}{
		{"inferred codex id absent from table", "openai", "gpt-5.1-codex-max", trio},
		{"inferred o-series absent from table", "openai", "o3-ultra", trio},
		{"inferred with vendor slash prefix", "somegateway", "openai/o4-mega", trio},
		{"inferred with provider colon prefix", "somegateway", "openai:gpt-5.9", gpt5},
		{"catalogued chat variant stays unknown", "openai", "gpt-5-chat-latest", nil},
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

// Table-backed cases compare against the embedded table rather than literal
// ladders, so an upstream models.dev change does not break the test.
func TestSupportedEffortsForModelPrefersEmbeddedTable(t *testing.T) {
	for _, tc := range []struct {
		name              string
		provider, modelID string
		tableModelID      string
	}{
		{"openai row", "openai", "gpt-5.6", "gpt-5.6"},
		{"dated snapshot via prefix", "openai", "gpt-5.6-sol-2026-07-09", "gpt-5.6-sol"},
		{"anthropic row", "anthropic", "claude-sonnet-5", "claude-sonnet-5"},
		{"aggregator row", "openrouter", "openai/gpt-5.6-luna", "openai/gpt-5.6-luna"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			info, ok := llm.LookupModelInfo(tc.provider, tc.tableModelID)
			if !ok || len(info.SupportedEfforts) == 0 {
				t.Fatalf("embedded table has no efforts for %s/%s", tc.provider, tc.tableModelID)
			}
			got := llm.SupportedEffortsForModel(tc.provider, tc.modelID)
			if !reflect.DeepEqual(got, info.SupportedEfforts) {
				t.Fatalf("SupportedEffortsForModel(%q, %q) = %v, want table value %v", tc.provider, tc.modelID, got, info.SupportedEfforts)
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
