package openaicodex

import (
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestOpenAICodexConformance machine-checks the llm.Provider contract
// against a fixture server speaking the codex SSE wire shape (the codex
// backend streams both blocking and streaming calls).
func TestOpenAICodexConformance(t *testing.T) {
	llmtest.RunConformance(t, func(t *testing.T) llm.Provider {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			writeCodexSSESuccess(w)
		}))
		t.Cleanup(server.Close)

		p, err := New(
			WithOAuth(llm.AuthCredential{Type: "oauth", Access: fakeCodexJWT(t, "acct-conformance"), Refresh: "refresh"}, nil),
			WithBaseURL(server.URL),
			withOAuthTokenURL(server.URL+"/oauth/token"),
			WithHTTPClient(server.Client()),
			WithMaxRetries(0),
		)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		return p
	})
}
