package openrouter_test

import (
	"net/http"
	"path/filepath"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/e2e"
	"github.com/pkieltyka/go-llm/providers/openrouter"
)

// TestReplayRecordedFixtures drives the OpenRouter dialect plus the shared
// chat-completions adapter (blocking, SSE streaming, models listing, error
// mapping) offline against the recorded live corpus in
// internal/e2e/fixtures/openrouter.
func TestReplayRecordedFixtures(t *testing.T) {
	root, err := e2e.RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	fixture := filepath.Join(root, "internal", "e2e", "fixtures", "openrouter", "live.json")
	e2e.ReplayExchanges(t, fixture, e2e.ReplayProfile{
		Provider: "openrouter",
		New: func(t *testing.T, client *http.Client) llm.Provider {
			p, err := openrouter.New(
				openrouter.WithAPIKey("replay-key"),
				openrouter.WithMaxRetries(0),
				openrouter.WithHTTPClient(client),
			)
			if err != nil {
				t.Fatalf("openrouter.New returned error: %v", err)
			}
			return p
		},
		ToolCallMarkers:  []string{`"tool_calls"`},
		ReasoningMarkers: []string{`"reasoning_details"`},
	})
}
