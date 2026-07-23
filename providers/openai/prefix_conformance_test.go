package openai

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestOpenAIAppendOnlyPrefix machine-checks the append-only wire-prefix
// property prompt caches rely on across a two-turn tool session.
func TestOpenAIAppendOnlyPrefix(t *testing.T) {
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
			_, _ = io.WriteString(w, `{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"resp_2","model":"gpt-test","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],"usage":{"input_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":3}}`)
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
		Model:         "gpt-test",
		MessagesField: "input",
		StableFields:  []string{"model", "tools", "prompt_cache_key", "instructions"},
	})
}
