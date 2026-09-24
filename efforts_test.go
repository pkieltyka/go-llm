package llm_test

import (
	"reflect"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

// Table-backed cases compare against the embedded table rather than literal
// ladders, so an upstream models.dev or OpenRouter change does not break the
// test.
func TestSupportedEffortsForModelReturnsEmbeddedTableValue(t *testing.T) {
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

// Models without upstream effort metadata stay unknown; nothing is inferred
// from the name. The ids are synthetic and share no prefix with real rows,
// so an upstream addition cannot change the expectation.
func TestSupportedEffortsForModelUnknownWithoutTableMetadata(t *testing.T) {
	for _, tc := range []struct{ provider, modelID string }{
		{"openai", "synthetic-reasoning-codex"},
		{"anthropic", "synthetic-claude"},
		{"openrouter", "synthetic/reasoning-model"},
		{"somegateway", "openai/synthetic-o-series"},
	} {
		if got := llm.SupportedEffortsForModel(tc.provider, tc.modelID); got != nil {
			t.Errorf("SupportedEffortsForModel(%q, %q) = %v, want nil", tc.provider, tc.modelID, got)
		}
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
