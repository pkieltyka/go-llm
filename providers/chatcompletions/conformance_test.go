package chatcompletions_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
)

// TestChatCompletionsConformance machine-checks the llm.Provider contract
// for the shared chat-completions engine itself, using the quirk-free test
// dialect against a fixture server.
func TestChatCompletionsConformance(t *testing.T) {
	llmtest.RunConformance(t, func(t *testing.T) llm.Provider {
		server := newConformanceServer(t)
		t.Cleanup(server.Close)

		p, err := chatcompletions.NewWithDialect(chatcompletions.Config{
			Dialect:    replayDialect{},
			APIKey:     "conformance-key",
			BaseURL:    server.URL,
			HTTPClient: server.Client(),
			MaxRetries: new(int),
		})
		if err != nil {
			t.Fatalf("chatcompletions.NewWithDialect returned error: %v", err)
		}
		return p
	})
}

// TestGenericChatCompletionsConformance retains executable coverage for the
// public quirk-free shared engine in addition to the advanced Dialect path.
func TestGenericChatCompletionsConformance(t *testing.T) {
	llmtest.RunConformance(t, func(t *testing.T) llm.Provider {
		server := newConformanceServer(t)
		t.Cleanup(server.Close)
		p, err := chatcompletions.New(server.URL,
			chatcompletions.WithAPIKey("conformance-key"),
			chatcompletions.WithHTTPClient(server.Client()),
			chatcompletions.WithMaxRetries(0),
		)
		if err != nil {
			t.Fatalf("chatcompletions.New returned error: %v", err)
		}
		return p
	})
}

func newConformanceServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		scenario := llmtest.ConformanceScenarioFromRequest(r)
		if r.Header.Get("Accept") == "text/event-stream" {
			w.Header().Set("Content-Type", "text/event-stream")
			if scenario == llmtest.ConformanceEmpty {
				return
			}
			_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"replay-model","choices":[{"index":0,"delta":{"role":"assistant"}}]}`+"\n\n")
			switch scenario {
			case llmtest.ConformanceCancel:
				w.(http.Flusher).Flush()
				<-r.Context().Done()
				return
			case llmtest.ConformanceTruncated:
				return
			}
			_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"replay-model","choices":[{"index":0,"delta":{"content":"po"}}]}`+"\n\n")
			_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"replay-model","choices":[{"index":0,"delta":{"content":"ng"},"finish_reason":"stop"}]}`+"\n\n")
			_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"replay-model","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"gen_1","model":"replay-model","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"pong"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
}
