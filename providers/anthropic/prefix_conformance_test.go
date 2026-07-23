package anthropic

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestAnthropicAppendOnlyPrefix machine-checks the append-only wire-prefix
// property prompt caches rely on across a two-turn tool session.
func TestAnthropicAppendOnlyPrefix(t *testing.T) {
	var mu sync.Mutex
	var bodies [][]byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, body)
		turn := len(bodies)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if turn == 1 {
			_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"go"}}],"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"msg_2","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":2,"output_tokens":1}}`)
	}))
	t.Cleanup(server.Close)

	llmtest.RunAppendOnlyPrefix(t, llmtest.AppendOnlyPrefixConfig{
		NewProvider: func(t *testing.T) llm.Provider {
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
		},
		Requests: func() [][]byte {
			mu.Lock()
			defer mu.Unlock()
			return append([][]byte(nil), bodies...)
		},
		Model:         "claude-test",
		MessagesField: "messages",
		StableFields:  []string{"model", "tools", "system"},
	})
}
