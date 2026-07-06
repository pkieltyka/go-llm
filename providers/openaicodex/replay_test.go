package openaicodex_test

import (
	"net/http"
	"path/filepath"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/e2e"
	"github.com/pkieltyka/go-llm/providers/openaicodex"
)

// TestReplayRecordedFixtures drives the Codex direct SSE transport and the
// shared Responses mapping offline against the recorded live corpus in
// internal/e2e/fixtures/openai-codex. The replay credential carries a static
// access token (no expiry), so no OAuth refresh traffic is attempted.
func TestReplayRecordedFixtures(t *testing.T) {
	root, err := e2e.RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	fixture := filepath.Join(root, "internal", "e2e", "fixtures", "openai-codex", "live.json")
	e2e.ReplayExchanges(t, fixture, e2e.ReplayProfile{
		Provider: "openai-codex",
		New: func(t *testing.T, client *http.Client) llm.Provider {
			cred := llm.AuthCredential{Type: "oauth", Access: "replay-access", AccountID: "replay-account"}
			p, err := openaicodex.New(
				openaicodex.WithOAuth(cred, nil),
				openaicodex.WithMaxRetries(0),
				openaicodex.WithHTTPClient(client),
			)
			if err != nil {
				t.Fatalf("openaicodex.New returned error: %v", err)
			}
			return p
		},
		ToolCallMarkers:  []string{`"type":"function_call"`},
		ReasoningMarkers: []string{`"type":"reasoning"`},
	})
}
