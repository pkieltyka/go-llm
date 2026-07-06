package openai_test

import (
	"net/http"
	"path/filepath"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/e2e"
	"github.com/pkieltyka/go-llm/providers/openai"
)

// TestReplayRecordedFixtures drives the OpenAI Responses streaming path
// offline against the recorded openai-codex corpus: the Codex backend speaks
// the genuine Responses SSE wire shape, and no OpenAI-recorded corpus exists
// yet (see docs/release.md — the OpenAI fixture is recorded once an OpenAI
// API key is configured). Replayed responses map through the shared
// responsesapi layer exactly as api.openai.com traffic would.
func TestReplayRecordedFixtures(t *testing.T) {
	root, err := e2e.RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	fixture := filepath.Join(root, "internal", "e2e", "fixtures", "openai-codex", "live.json")
	e2e.ReplayExchanges(t, fixture, e2e.ReplayProfile{
		Provider: "openai",
		New: func(t *testing.T, client *http.Client) llm.Provider {
			p, err := openai.New(
				openai.WithAPIKey("replay-key"),
				openai.WithMaxRetries(0),
				openai.WithHTTPClient(client),
			)
			if err != nil {
				t.Fatalf("openai.New returned error: %v", err)
			}
			return p
		},
		ToolCallMarkers:  []string{`"type":"function_call"`},
		ReasoningMarkers: []string{`"type":"reasoning"`},
	})
}
