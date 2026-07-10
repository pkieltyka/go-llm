package openaicodex

import (
	"context"
	"io"
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
			switch llmtest.ConformanceScenarioFromRequest(r) {
			case llmtest.ConformanceEmpty:
				w.Header().Set("Content-Type", "text/event-stream")
				return
			case llmtest.ConformanceCancel:
				writeCodexSSEStart(w)
				w.(http.Flusher).Flush()
				<-r.Context().Done()
				return
			case llmtest.ConformanceTruncated:
				writeCodexSSEStart(w)
				return
			}
			writeCodexSSESuccess(w)
		}))
		t.Cleanup(server.Close)

		p, err := New(
			WithOAuth(llm.AuthCredential{Type: "oauth", Access: fakeCodexJWT(t, "acct-conformance"), Refresh: "refresh"}, func(ctx context.Context, _ llm.AuthCredential) error {
				return ctx.Err()
			}),
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

func writeCodexSSEStart(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	_, _ = io.WriteString(w, `data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.4-mini","status":"in_progress","output":[]}}`+"\n\n")
}
