package llm_test

import (
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestMatchModelRanksNormalizedModelNames(t *testing.T) {
	models := []llm.ModelInfo{
		{ID: "prefix", DisplayName: "Claude Opus 4.8 Preview"},
		{ID: "substring", DisplayName: "Preview Claude Opus 4.8"},
		{ID: "claude-opus-4-8", DisplayName: "different"},
	}

	matched, ok := llm.MatchModel(models, "CLAUDE-OPUS-4.8")
	if !ok || matched.ID != "claude-opus-4-8" {
		t.Fatalf("MatchModel = (%+v, %v), want normalized exact ID match", matched, ok)
	}

	matched, ok = llm.MatchModel(models[:2], "Claude Opus 4.8")
	if !ok || matched.ID != "prefix" {
		t.Fatalf("MatchModel = (%+v, %v), want display-name prefix match", matched, ok)
	}
}

func TestMatchModelUsesCanonicalIDsAndPrefersLatestVersion(t *testing.T) {
	models := []llm.ModelInfo{
		{ID: "first-opus", CanonicalID: "anthropic/claude-opus-4-8"},
		{ID: "second-opus", CanonicalID: "anthropic/claude-opus-4-9"},
	}

	matched, ok := llm.MatchModel(models, "anthropic/claude-opus-4.8")
	if !ok || matched.ID != "first-opus" {
		t.Fatalf("MatchModel canonical exact = (%+v, %v)", matched, ok)
	}

	matched, ok = llm.MatchModel(models, "opus")
	if !ok || matched.ID != "second-opus" {
		t.Fatalf("MatchModel tie = (%+v, %v), want latest provider model", matched, ok)
	}
}

func TestMatchModelPrefersLatestVersionRegardlessOfCatalogOrder(t *testing.T) {
	for _, models := range [][]llm.ModelInfo{
		{{ID: "gpt-5"}, {ID: "gpt-5.1"}, {ID: "gpt-5.6"}},
		{{ID: "gpt-5.6"}, {ID: "gpt-5.1"}, {ID: "gpt-5"}},
	} {
		matched, ok := llm.MatchModel(models, "gpt-5")
		if !ok || matched.ID != "gpt-5.6" {
			t.Fatalf("MatchModel(%v, gpt-5) = (%+v, %v), want gpt-5.6", models, matched, ok)
		}
	}

	models := []llm.ModelInfo{{ID: "gpt-5.10"}, {ID: "gpt-5.6"}}
	matched, ok := llm.MatchModel(models, "gpt-5")
	if !ok || matched.ID != "gpt-5.10" {
		t.Fatalf("MatchModel natural version = (%+v, %v), want gpt-5.10", matched, ok)
	}

	models = []llm.ModelInfo{{ID: "gpt-5.6-sol"}, {ID: "gpt-5.6"}}
	matched, ok = llm.MatchModel(models, "gpt-5")
	if !ok || matched.ID != "gpt-5.6" {
		t.Fatalf("MatchModel base version = (%+v, %v), want gpt-5.6", matched, ok)
	}
}

func TestMatchModelKeepsExplicitMinorVersionExact(t *testing.T) {
	models := []llm.ModelInfo{{ID: "gpt-5.10"}, {ID: "gpt-5.1"}}
	matched, ok := llm.MatchModel(models, "gpt-5.1")
	if !ok || matched.ID != "gpt-5.1" {
		t.Fatalf("MatchModel explicit version = (%+v, %v), want gpt-5.1", matched, ok)
	}
}

func TestMatchModelDoesNotTreatModelMetadataAsVersion(t *testing.T) {
	tests := []struct {
		name   string
		query  string
		models []llm.ModelInfo
		want   string
	}{
		{
			name:  "parameter counts",
			query: "qwen",
			models: []llm.ModelInfo{
				{ID: "qwen/qwen3-235b-a22b"},
				{ID: "qwen/qwen3.8-max"},
				{ID: "qwen/qwen3.6-27b"},
			},
			want: "qwen/qwen3.8-max",
		},
		{
			name:  "quantization levels and compact dates",
			query: "qwen",
			models: []llm.ModelInfo{
				{ID: "qwen/qwen3-30b-fp8-2507"},
				{ID: "qwen/qwen3.6-27b-nvfp4"},
			},
			want: "qwen/qwen3.6-27b-nvfp4",
		},
		{
			name:  "dated snapshots",
			query: "gpt-4o",
			models: []llm.ModelInfo{
				{ID: "gpt-4o-20241120"},
				{ID: "gpt-4o-2024-11-20"},
				{ID: "gpt-4o"},
			},
			want: "gpt-4o",
		},
		{
			name:  "numeric provider names",
			query: "euryale",
			models: []llm.ModelInfo{
				{ID: "sao10k/l3.1-euryale-70b"},
				{ID: "sao10k/l3.3-euryale-70b"},
			},
			want: "sao10k/l3.3-euryale-70b",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			matched, ok := llm.MatchModel(test.models, test.query)
			if !ok || matched.ID != test.want {
				t.Fatalf("MatchModel(%q) = (%+v, %v), want %q", test.query, matched, ok, test.want)
			}
		})
	}
}

func TestMatchModelDoesNotReorderCatalog(t *testing.T) {
	models := []llm.ModelInfo{{ID: "gpt-5.6"}, {ID: "gpt-5.1"}}
	if _, ok := llm.MatchModel(models, "gpt-5"); !ok {
		t.Fatal("MatchModel returned no match")
	}
	if models[0].ID != "gpt-5.6" || models[1].ID != "gpt-5.1" {
		t.Fatalf("MatchModel reordered input: %+v", models)
	}
}

func TestMatchModelReturnsIndependentMutableMetadata(t *testing.T) {
	models := []llm.ModelInfo{{
		ID:               "gpt-5.6",
		SupportedEfforts: []llm.Effort{llm.EffortLow},
		Capabilities:     []llm.Capability{llm.CapabilityTools},
		Pricing: &llm.ModelPricing{
			InputPerMTok: 1,
			Tiers:        []llm.ModelPricingTier{{InputTokensAbove: 100, InputPerMTok: 2}},
		},
	}}
	matched, ok := llm.MatchModel(models, "gpt-5.6")
	if !ok {
		t.Fatal("MatchModel returned no match")
	}
	matched.SupportedEfforts[0] = llm.EffortMax
	matched.Capabilities[0] = llm.CapabilityStreaming
	matched.Pricing.InputPerMTok = 99
	matched.Pricing.Tiers[0].InputPerMTok = 99

	if models[0].SupportedEfforts[0] != llm.EffortLow {
		t.Fatal("MatchModel result efforts alias the input")
	}
	if models[0].Capabilities[0] != llm.CapabilityTools {
		t.Fatal("MatchModel result capabilities alias the input")
	}
	if models[0].Pricing.InputPerMTok != 1 || models[0].Pricing.Tiers[0].InputPerMTok != 2 {
		t.Fatal("MatchModel result pricing aliases the input")
	}
}

func TestMatchModelRejectsEmptyAndUnrelatedQueries(t *testing.T) {
	models := []llm.ModelInfo{{ID: "claude-opus-4-8"}}
	for _, query := range []string{"", " ", "---", "/", "_", "gpt-5.6"} {
		if matched, ok := llm.MatchModel(models, query); ok {
			t.Fatalf("MatchModel(%q) = %+v, want no match", query, matched)
		}
	}
}

func TestMatchModelTrimsQueryWhitespace(t *testing.T) {
	models := []llm.ModelInfo{{ID: "gpt-5.6"}}
	matched, ok := llm.MatchModel(models, " \tGPT-5.6\n")
	if !ok || matched.ID != "gpt-5.6" {
		t.Fatalf("MatchModel with padded query = (%+v, %v)", matched, ok)
	}
}
