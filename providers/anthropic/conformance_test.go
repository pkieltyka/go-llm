package anthropic

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestAnthropicConformance machine-checks the llm.Provider contract
// (single-use streams, cancellation, concurrency, panic-freedom, Collect
// partial shape) against a fixture server speaking the Messages wire shape.
func TestAnthropicConformance(t *testing.T) {
	llmtest.RunConformance(t, func(t *testing.T) llm.Provider {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			scenario := llmtest.ConformanceScenarioFromRequest(r)
			body, _ := io.ReadAll(r.Body)
			if strings.Contains(string(body), `"stream":true`) {
				w.Header().Set("Content-Type", "text/event-stream")
				start := strings.Join([]string{
					`event: message_start`,
					`data: {"type":"message_start","message":{"id":"msg_1","model":"claude-test","usage":{"input_tokens":1,"output_tokens":0}}}`,
					``,
				}, "\n") + "\n"
				switch scenario {
				case llmtest.ConformanceEmpty:
					return
				case llmtest.ConformanceCancel:
					_, _ = io.WriteString(w, start)
					w.(http.Flusher).Flush()
					<-r.Context().Done()
					return
				case llmtest.ConformanceTruncated:
					_, _ = io.WriteString(w, start)
					return
				}
				_, _ = io.WriteString(w, start+strings.Join([]string{
					`event: content_block_start`,
					`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"po"}}`,
					``,
					`event: content_block_delta`,
					`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ng"}}`,
					``,
					`event: content_block_stop`,
					`data: {"type":"content_block_stop","index":0}`,
					``,
					`event: message_delta`,
					`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
					``,
					`event: message_stop`,
					`data: {"type":"message_stop"}`,
					``,
					``,
				}, "\n"))
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"pong"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		}))
		t.Cleanup(server.Close)

		p, err := New(
			WithAPIKey("test-key"),
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
