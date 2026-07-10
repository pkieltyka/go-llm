package ollama_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
	"github.com/pkieltyka/go-llm/providers/ollama"
)

// TestOllamaConformance machine-checks the llm.Provider contract through the
// data-only Ollama preset, independently of the shared engine's own suite.
func TestOllamaConformance(t *testing.T) {
	llmtest.RunConformance(t, func(t *testing.T) llm.Provider {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			scenario := llmtest.ConformanceScenarioFromRequest(r)
			if r.Header.Get("Accept") == "text/event-stream" {
				w.Header().Set("Content-Type", "text/event-stream")
				if scenario == llmtest.ConformanceEmpty {
					return
				}
				_, _ = io.WriteString(w, `data: {"id":"c1","model":"qwen3:8b","choices":[{"index":0,"delta":{"role":"assistant"}}]}`+"\n\n")
				switch scenario {
				case llmtest.ConformanceCancel:
					w.(http.Flusher).Flush()
					<-r.Context().Done()
					return
				case llmtest.ConformanceTruncated:
					return
				}
				_, _ = io.WriteString(w, `data: {"id":"c1","model":"qwen3:8b","choices":[{"index":0,"delta":{"content":"po"}}]}`+"\n\n")
				_, _ = io.WriteString(w, `data: {"id":"c1","model":"qwen3:8b","choices":[{"index":0,"delta":{"content":"ng"},"finish_reason":"stop"}]}`+"\n\n")
				_, _ = io.WriteString(w, `data: {"id":"c1","model":"qwen3:8b","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\n")
				_, _ = io.WriteString(w, "data: [DONE]\n\n")
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"c1","model":"qwen3:8b","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"pong"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
		}))
		t.Cleanup(server.Close)

		p, err := ollama.New(server.URL,
			chatcompletions.WithHTTPClient(server.Client()),
			chatcompletions.WithMaxRetries(0),
		)
		if err != nil {
			t.Fatalf("ollama.New returned error: %v", err)
		}
		return p
	})
}
