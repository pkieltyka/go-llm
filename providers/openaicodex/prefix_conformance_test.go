package openaicodex

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestOpenAICodexAppendOnlyPrefix machine-checks the append-only wire-prefix
// property prompt caches rely on across a two-turn tool session. The codex
// backend streams both blocking and streaming calls, so the fixture speaks
// SSE for both turns.
func TestOpenAICodexAppendOnlyPrefix(t *testing.T) {
	var mu sync.Mutex
	var bodies [][]byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, body)
		turn := len(bodies)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.4-mini","status":"in_progress","output":[]}}`+"\n\n")
		if turn == 1 {
			_, _ = io.WriteString(w, `data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"","status":"in_progress"}}`+"\n\n")
			_, _ = io.WriteString(w, `data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"q\":\"go\"}"}`+"\n\n")
			_, _ = io.WriteString(w, `data: {"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}}`+"\n\n")
			_, _ = io.WriteString(w, `data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.4-mini","status":"completed","output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}}`+"\n\n")
			return
		}
		_, _ = io.WriteString(w, `data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"done"}`+"\n\n")
		_, _ = io.WriteString(w, `data: {"type":"response.completed","response":{"id":"resp_2","model":"gpt-5.4-mini","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],"usage":{"input_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":3}}}`+"\n\n")
	}))
	t.Cleanup(server.Close)

	llmtest.RunAppendOnlyPrefix(t, llmtest.AppendOnlyPrefixConfig{
		NewProvider: func(t *testing.T) llm.Provider {
			p, err := New(
				WithOAuth(llm.AuthCredential{Type: "oauth", Access: fakeCodexJWT(t, "acct-prefix"), Refresh: "refresh"}, func(ctx context.Context, _ llm.AuthCredential) error {
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
		},
		Requests: func() [][]byte {
			mu.Lock()
			defer mu.Unlock()
			return append([][]byte(nil), bodies...)
		},
		Model:         "gpt-5.4-mini",
		MessagesField: "input",
		StableFields:  []string{"model", "tools", "prompt_cache_key", "instructions"},
	})
}
