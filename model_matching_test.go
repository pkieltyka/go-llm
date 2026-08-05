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

func TestMatchModelUsesCanonicalIDsAndPrefersLaterTies(t *testing.T) {
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
		t.Fatalf("MatchModel tie = (%+v, %v), want later provider model", matched, ok)
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
