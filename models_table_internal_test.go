package llm

import (
	"strings"
	"testing"
)

func TestEmbeddedModelTablePassesStrictValidation(t *testing.T) {
	table, err := parseModelTable(embeddedModelsJSON)
	if err != nil {
		t.Fatalf("embedded models.json failed validation: %v", err)
	}
	if len(table.Rows) == 0 || len(table.ByKey) != len(table.Rows) {
		t.Fatalf("embedded table rows/keys = %d/%d", len(table.Rows), len(table.ByKey))
	}
}

func TestParseModelTableRejectsMalformedCatalogs(t *testing.T) {
	validRow := `{"provider":"openai","id":"gpt","context_window":1,"max_output_tokens":1,"pricing":{"input_per_mtok":1,"tiers":[{"input_tokens_above":100,"input_per_mtok":2,"output_per_mtok":3,"cache_read_per_mtok":4,"cache_write_per_mtok":5}]},"supported_efforts":["none","low","high"]}`
	for _, tt := range []struct {
		name string
		raw  string
		want string
	}{
		{name: "unknown root field", raw: strings.TrimSuffix(catalogJSON(validRow), `}`) + `,"future":true}`, want: "unknown field"},
		{name: "unknown row field", raw: catalogJSON(`{"provider":"openai","id":"gpt","future":true}`), want: "unknown field"},
		{name: "unknown pricing field", raw: catalogJSON(`{"provider":"openai","id":"gpt","pricing":{"future":1}}`), want: "unknown field"},
		{name: "concatenated documents", raw: catalogJSON(validRow) + catalogJSON(validRow), want: "multiple JSON documents"},
		{name: "trailing garbage", raw: catalogJSON(validRow) + ` trailing`, want: "trailing data"},
		{name: "missing schema", raw: `{"generator":"go-llm-model-snapshot/v1","generated_at":"2026-08-05T00:00:00Z","sources":` + modelTableTestSources + `,"models":[` + validRow + `]}`, want: "schema_version"},
		{name: "wrong generator", raw: strings.Replace(catalogJSON(validRow), modelTableGenerator, "future-generator", 1), want: "generator"},
		{name: "missing sources", raw: `{"schema_version":1,"generator":"go-llm-model-snapshot/v1","generated_at":"2026-08-05T00:00:00Z","models":[` + validRow + `]}`, want: "sources"},
		{name: "bad source digest", raw: strings.Replace(catalogJSON(validRow), strings.Repeat("0", 64), "not-a-digest", 1), want: "SHA-256"},
		{name: "missing timestamp", raw: `{"schema_version":1,"generator":"go-llm-model-snapshot/v1","sources":` + modelTableTestSources + `,"models":[` + validRow + `]}`, want: "generated_at"},
		{name: "invalid timestamp", raw: catalogJSONAt("today", validRow), want: "generated_at"},
		{name: "empty models", raw: catalogJSON(``), want: "no models"},
		{name: "empty identity", raw: catalogJSON(`{"provider":"","id":"gpt"}`), want: "empty provider or id"},
		{name: "duplicate", raw: catalogJSON(`{"provider":"openai","id":"gpt"}, {"provider":"openai","id":"gpt"}`), want: "duplicate key"},
		{name: "out of order", raw: catalogJSON(`{"provider":"openai","id":"z"}, {"provider":"openai","id":"a"}`), want: "out of canonical order"},
		{name: "zero context", raw: catalogJSON(`{"provider":"openai","id":"gpt","context_window":0}`), want: "context_window must be positive"},
		{name: "negative output", raw: catalogJSON(`{"provider":"openai","id":"gpt","max_output_tokens":-1}`), want: "max_output_tokens must be positive"},
		{name: "negative base price", raw: catalogJSON(`{"provider":"openai","id":"gpt","pricing":{"input_per_mtok":-1}}`), want: "finite and non-negative"},
		{name: "zero tier threshold", raw: catalogJSON(`{"provider":"openai","id":"gpt","pricing":{"tiers":[{"input_tokens_above":0,"input_per_mtok":1,"output_per_mtok":1,"cache_read_per_mtok":1,"cache_write_per_mtok":1}]}}`), want: "threshold must be positive and ascending"},
		{name: "duplicate tier threshold", raw: catalogJSON(`{"provider":"openai","id":"gpt","pricing":{"tiers":[{"input_tokens_above":100,"input_per_mtok":1,"output_per_mtok":1,"cache_read_per_mtok":1,"cache_write_per_mtok":1},{"input_tokens_above":100,"input_per_mtok":2,"output_per_mtok":2,"cache_read_per_mtok":2,"cache_write_per_mtok":2}]}}`), want: "threshold must be positive and ascending"},
		{name: "incomplete tier", raw: catalogJSON(`{"provider":"openai","id":"gpt","pricing":{"tiers":[{"input_tokens_above":100,"input_per_mtok":1}]}}`), want: "must be present"},
		{name: "unknown effort", raw: catalogJSON(`{"provider":"openai","id":"gpt","supported_efforts":["turbo"]}`), want: "known, unique, and ordered"},
		{name: "duplicate effort", raw: catalogJSON(`{"provider":"openai","id":"gpt","supported_efforts":["low","low"]}`), want: "known, unique, and ordered"},
		{name: "out of order effort", raw: catalogJSON(`{"provider":"openai","id":"gpt","supported_efforts":["high","low"]}`), want: "known, unique, and ordered"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseModelTable([]byte(tt.raw))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("parseModelTable error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestModelTableCanonicalFallbackDeepCopiesPricingTiers(t *testing.T) {
	raw := catalogJSON(
		`{"provider":"anthropic","id":"claude","pricing":{"input_per_mtok":1,"tiers":[{"input_tokens_above":100,"input_per_mtok":2,"output_per_mtok":3,"cache_read_per_mtok":4,"cache_write_per_mtok":5}]}}` +
			`,{"provider":"openrouter","id":"anthropic/claude","canonical_id":"anthropic/claude"}`,
	)
	table, err := parseModelTable([]byte(raw))
	if err != nil {
		t.Fatalf("parseModelTable returned error: %v", err)
	}
	first, ok := table.lookup("openrouter", "anthropic/claude")
	if !ok || first.Pricing == nil || len(first.Pricing.Tiers) != 1 {
		t.Fatalf("canonical fallback = %+v, %v", first, ok)
	}
	first.Pricing.Tiers[0].InputPerMTok = 999
	second, ok := table.lookup("openrouter", "anthropic/claude")
	if !ok || second.Pricing.Tiers[0].InputPerMTok != 2 {
		t.Fatalf("tier mutation leaked into catalog: %+v", second.Pricing)
	}
	costed := EstimateCost(Usage{InputTokens: 101}, *second.Pricing)
	if costed.CostUSD == nil || *costed.CostUSD != 0.000202 {
		t.Fatalf("canonical tier cost = %v, want 0.000202", costed.CostUSD)
	}
}

func TestCloneModelInfoCopiesMutableMetadata(t *testing.T) {
	availability := &ModelPricingAvailability{InputPerMTok: true}
	original := ModelInfo{
		ID:               "model",
		SupportedEfforts: []Effort{EffortLow, EffortHigh},
		Capabilities:     []Capability{CapabilityTools, CapabilityReasoning},
		Pricing:          &ModelPricing{Availability: availability, Tiers: []ModelPricingTier{{InputTokensAbove: 100}}},
	}
	cloned := cloneModelInfo(original)
	cloned.SupportedEfforts[0] = EffortMax
	cloned.Capabilities[0] = CapabilityStreaming
	cloned.Pricing.Tiers[0].InputTokensAbove = 999
	cloned.Pricing.Availability.InputPerMTok = false

	if original.SupportedEfforts[0] != EffortLow {
		t.Fatal("cloned efforts alias the original")
	}
	if original.Capabilities[0] != CapabilityTools {
		t.Fatal("cloned capabilities alias the original")
	}
	if original.Pricing.Tiers[0].InputTokensAbove != 100 {
		t.Fatal("cloned pricing tiers alias the original")
	}
	if !original.Pricing.Availability.InputPerMTok {
		t.Fatal("cloned pricing availability aliases the original")
	}
}

func TestModelTablePreservesExplicitEmptyEffortsThroughFallbackAndClone(t *testing.T) {
	raw := catalogJSON(
		`{"provider":"openai","id":"gpt","supported_efforts":["low","high"]},` +
			`{"provider":"openrouter","id":"openai/gpt","canonical_id":"openai/gpt","supported_efforts":[]}`,
	)
	table, err := parseModelTable([]byte(raw))
	if err != nil {
		t.Fatalf("parseModelTable: %v", err)
	}
	info, ok := table.lookup("openrouter", "openai/gpt")
	if !ok || info.SupportedEfforts == nil || len(info.SupportedEfforts) != 0 {
		t.Fatalf("explicit empty efforts = %#v, found=%v", info.SupportedEfforts, ok)
	}

	capacityBearing := make([]Effort, 0, 4)
	cloned := cloneModelInfo(ModelInfo{SupportedEfforts: capacityBearing})
	capacityBearing = append(capacityBearing, EffortMax)
	if cloned.SupportedEfforts == nil || len(cloned.SupportedEfforts) != 0 {
		t.Fatalf("clone collapsed or aliased empty efforts: %#v", cloned.SupportedEfforts)
	}
	cloned.SupportedEfforts = append(cloned.SupportedEfforts, EffortLow)
	if capacityBearing[0] != EffortMax {
		t.Fatalf("clone shares source backing storage: %#v", capacityBearing)
	}
}

func TestModelTablePricingPreservesComponentAvailability(t *testing.T) {
	raw := catalogJSON(
		`{"provider":"openai","id":"free","pricing":{"input_per_mtok":0,"output_per_mtok":0,"cache_read_per_mtok":0,"cache_write_per_mtok":0}},` +
			`{"provider":"openai","id":"partial","pricing":{"output_per_mtok":2}}`,
	)
	table, err := parseModelTable([]byte(raw))
	if err != nil {
		t.Fatalf("parseModelTable: %v", err)
	}

	partial, ok := table.lookup("openai", "partial")
	if !ok || partial.Pricing == nil {
		t.Fatalf("partial pricing missing: %#v, found=%v", partial.Pricing, ok)
	}
	if partial.Pricing.HasInputPrice() || !partial.Pricing.HasOutputPrice() ||
		partial.Pricing.HasCacheReadPrice() || partial.Pricing.HasCacheWritePrice() {
		t.Fatalf("partial pricing availability = %#v", partial.Pricing.Availability)
	}
	if partial.Pricing.OutputPerMTok != 2 {
		t.Fatalf("partial output price = %v, want 2", partial.Pricing.OutputPerMTok)
	}

	free, ok := table.lookup("openai", "free")
	if !ok || free.Pricing == nil || !free.Pricing.HasInputPrice() || !free.Pricing.HasOutputPrice() ||
		!free.Pricing.HasCacheReadPrice() || !free.Pricing.HasCacheWritePrice() {
		t.Fatalf("explicit-zero pricing availability = %#v, found=%v", free.Pricing, ok)
	}
}

func catalogJSON(rows string) string {
	return catalogJSONAt("2026-08-05T00:00:00Z", rows)
}

const modelTableTestSources = `[{"id":"models.dev","url":"fixture:models.dev","sha256":"0000000000000000000000000000000000000000000000000000000000000000"},{"id":"openrouter","url":"fixture:openrouter","sha256":"1111111111111111111111111111111111111111111111111111111111111111"},{"id":"overrides","url":"fixture:overrides","sha256":"2222222222222222222222222222222222222222222222222222222222222222"}]`

func catalogJSONAt(generatedAt, rows string) string {
	return `{"schema_version":1,"generator":"go-llm-model-snapshot/v1","generated_at":"` + generatedAt + `","sources":` + modelTableTestSources + `,"models":[` + rows + `]}`
}
