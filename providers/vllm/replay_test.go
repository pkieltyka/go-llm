package vllm_test

import (
	"net/http"
	"path/filepath"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/e2e"
	"github.com/pkieltyka/go-llm/providers/vllm"
)

// TestReplayRecordedFixtures drives the vLLM dialect plus the shared
// chat-completions engine (blocking, SSE streaming, models listing, error
// mapping) offline against the recorded live corpus in
// internal/e2e/fixtures/vllm. vLLM reasoning is plain text, so the profile
// asserts text preservation instead of raw replay payloads.
func TestReplayRecordedFixtures(t *testing.T) {
	root, err := e2e.RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	fixture := filepath.Join(root, "internal", "e2e", "fixtures", "vllm", "live.json")
	e2e.ReplayExchanges(t, fixture, e2e.ReplayProfile{
		Provider: "vllm",
		New: func(t *testing.T, client *http.Client) llm.Provider {
			p, err := vllm.New("http://vllm.replay/v1",
				vllm.WithMaxRetries(0),
				vllm.WithHTTPClient(client),
			)
			if err != nil {
				t.Fatalf("vllm.New returned error: %v", err)
			}
			return p
		},
		ToolCallMarkers: []string{`"tool_calls"`},
		// vLLM blocking responses always carry `"reasoning":null`; only a
		// string-valued field means reasoning content was produced.
		ReasoningTextMarkers: []string{`"reasoning":"`, `"reasoning_content":"`},
	})
}
