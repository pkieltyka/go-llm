package openrouter

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestOpenRouterConformance machine-checks the llm.Provider contract
// (single-use streams, cancellation, concurrency, panic-freedom, Collect
// partial shape) against a fixture handler speaking the chat-completions
// wire shape through the shared engine.
func TestOpenRouterConformance(t *testing.T) {
	llmtest.RunConformance(t, func(t *testing.T) llm.Provider {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			scenario := llmtest.ConformanceScenarioFromRequest(r)
			if r.Header.Get("Accept") == "text/event-stream" {
				w.Header().Set("Content-Type", "text/event-stream")
				if scenario == llmtest.ConformanceEmpty {
					return
				}
				_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"role":"assistant"}}]}`+"\n\n")
				switch scenario {
				case llmtest.ConformanceCancel:
					w.(http.Flusher).Flush()
					<-r.Context().Done()
					return
				case llmtest.ConformanceTruncated:
					return
				}
				_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"content":"po"}}]}`+"\n\n")
				_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"content":"ng"},"finish_reason":"stop"}]}`+"\n\n")
				_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"cost":0.00001}}`+"\n\n")
				_, _ = io.WriteString(w, "data: [DONE]\n\n")
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"pong"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"cost":0.00001}}`)
		}))
		t.Cleanup(server.Close)
		p, err := New(
			WithAPIKey("test"),
			WithBaseURL(server.URL),
			WithHTTPClient(server.Client()),
			WithMaxRetries(0),
		)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		return p
	})
}

// TestProviderIdentitySurface pins the trivial identity accessors.
func TestProviderIdentitySurface(t *testing.T) {
	p, err := New(WithAPIKey("test"))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if p.Name() != providerName {
		t.Fatalf("Name = %q, want %q", p.Name(), providerName)
	}
	if p.Client() == nil {
		t.Fatalf("Client() = nil")
	}
	if (*Provider)(nil).Client() != nil {
		t.Fatalf("nil provider Client() should be nil")
	}
	caps := p.Capabilities()
	if len(caps) == 0 || caps[0] != llm.CapabilityStreaming {
		t.Fatalf("capabilities = %+v", caps)
	}
}

var _ llm.Provider = (*Provider)(nil)
