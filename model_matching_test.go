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

func TestSameModel(t *testing.T) {
	tests := []struct {
		a, b string
		want bool
	}{
		{"gpt-5.5", "gpt-5.5", true},
		{" GPT-5.5 ", "gpt-5.5", true},
		{"gpt-5.5", "gpt-5.5-2026-08-15", true},
		{"claude-haiku-4-5", "claude-haiku-4-5-20251001", true},
		{"claude-3-5-sonnet@20240620", "claude-3-5-sonnet", true},
		{"model_20250101", "model", true},
		{"anthropic/claude-haiku-4.5", "anthropic/claude-haiku-4.5", true},
		{"gpt-5.5", "gpt-5.6", false},
		{"gpt-5-mini", "gpt-5-nano", false},
		{"anthropic/claude-haiku-4.5", "anthropic/claude-haiku-5.5", false},
		{"anthropic/claude-haiku-4.5", "claude-haiku-4.5", false},
		// Anthropic aliases are answered with differently named dated IDs.
		{"claude-opus-4-0", "claude-opus-4-20250514", true},
		{"claude-sonnet-4-0", "claude-sonnet-4-20250514", true},
		{"claude-3-7-sonnet-latest", "claude-3-7-sonnet-20250219", true},
		{"claude-haiku-4-5", "claude-haiku-4-5-20251001", true},
		// Routing variants and Bedrock revisions name the same weights.
		{"qwen/qwen3.6-27b:free", "qwen/qwen3.6-27b", true},
		{"anthropic/claude-haiku-4.5:nitro", "anthropic/claude-haiku-4.5", true},
		{"anthropic.claude-3-5-sonnet-20241022-v2:0", "anthropic.claude-3-5-sonnet-20241022-v2", true},
		{"claude-3-5-sonnet-v2@20241022", "claude-3-5-sonnet-v2", true},
		{"anthropic.claude-3-5-sonnet-20240620-v1:0", "anthropic.claude-3-5-sonnet-20241022-v2:0", false},
		{"gpt-5", "gpt-5-mini", false},
		{"claude-opus-4-1", "claude-opus-4-20250514", false},
		{"openrouter/auto", "anthropic/claude-haiku-4.5", false},
		{"model-12345678", "model", false}, // not a plausible date
		{"model-2026-13-01", "model", false},
		{"", "", false},
		{"gpt-5.5", "", false},
	}
	for _, tt := range tests {
		if got := llm.SameModel(tt.a, tt.b); got != tt.want {
			t.Errorf("SameModel(%q, %q) = %v, want %v", tt.a, tt.b, got, tt.want)
		}
		if got := llm.SameModel(tt.b, tt.a); got != tt.want {
			t.Errorf("SameModel(%q, %q) = %v, want %v (symmetry)", tt.b, tt.a, got, tt.want)
		}
	}
}
