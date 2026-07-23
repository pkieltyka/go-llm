package vllm

import (
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestVLLMAppendOnlyPrefix machine-checks the append-only wire-prefix
// property prompt caches rely on across a two-turn tool session.
func TestVLLMAppendOnlyPrefix(t *testing.T) {
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
			_, _ = io.WriteString(w, `{"id":"c1","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"go\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"c2","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	}))
	t.Cleanup(server.Close)

	llmtest.RunAppendOnlyPrefix(t, llmtest.AppendOnlyPrefixConfig{
		NewProvider: func(t *testing.T) llm.Provider {
			p, err := New(server.URL,
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
		Model:         "m",
		MessagesField: "messages",
		StableFields:  []string{"model", "tools"},
	})
}
