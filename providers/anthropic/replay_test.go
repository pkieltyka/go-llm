package anthropic_test

import (
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/e2e"
	"github.com/pkieltyka/go-llm/providers/anthropic"
)

// TestReplayRecordedFixtures drives the full Anthropic mapping paths (blocking
// Messages, SSE streaming, models listing, error mapping) offline against the
// recorded live corpus in internal/e2e/fixtures/anthropic.
func TestReplayRecordedFixtures(t *testing.T) {
	root, err := e2e.RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	fixture := filepath.Join(root, "internal", "e2e", "fixtures", "anthropic", "live.json")
	e2e.ReplayExchanges(t, fixture, e2e.ReplayProfile{
		Provider: "anthropic",
		New: func(t *testing.T, client *http.Client) llm.Provider {
			p, err := anthropic.New(
				anthropic.WithAPIKey("replay-key"),
				anthropic.WithMaxRetries(0),
				anthropic.WithHTTPClient(client),
			)
			if err != nil {
				t.Fatalf("anthropic.New returned error: %v", err)
			}
			return p
		},
		ToolCallMarkers:  []string{`"tool_use"`},
		ReasoningMarkers: []string{`"thinking"`},
	})
}

func TestRecordedFixturesUseAPIKeyAuthentication(t *testing.T) {
	root, err := e2e.RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	fixture := filepath.Join(root, "internal", "e2e", "fixtures", "anthropic", "live.json")
	exchanges, err := e2e.LoadRecordedExchanges(fixture)
	if err != nil {
		t.Fatalf("LoadRecordedExchanges returned error: %v", err)
	}
	for index, exchange := range exchanges {
		if got := exchange.RequestHeaders.Get("X-Api-Key"); got != "[REDACTED]" {
			t.Errorf("exchange %d does not contain a redacted API key", index)
		}
		for _, header := range []string{"Authorization", "X-App"} {
			if got := exchange.RequestHeaders.Get(header); got != "" {
				t.Errorf("exchange %d retained legacy %s header", index, header)
			}
		}
		legacy := exchange.RequestHeaders.Get("Anthropic-Beta") + exchange.RequestHeaders.Get("User-Agent") + exchange.RequestBody
		if strings.Contains(legacy, "oauth-2025-04-20") || strings.Contains(legacy, "claude-code") || strings.Contains(legacy, "Claude Code") {
			t.Errorf("exchange %d retained Claude subscription identity", index)
		}
	}
}
