package ollama_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
	"github.com/pkieltyka/go-llm/providers/ollama"
)

func TestNewDefaults(t *testing.T) {
	p, err := ollama.New("")
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if p.Name() != "ollama" {
		t.Fatalf("Name = %q, want ollama", p.Name())
	}
	if !ollama.Compat().StreamIncludeUsage {
		t.Fatalf("preset compat should request usage in streams")
	}
}

// TestChatRoundTrip drives the preset against a fixture server speaking the
// standard chat-completions shape (this tests the preset's wiring, not
// Ollama itself — the preset is data-only and community-verified).
func TestChatRoundTrip(t *testing.T) {
	var streamBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "" {
			t.Fatalf("keyless request sent Authorization %q", got)
		}
		if r.Header.Get("Accept") == "text/event-stream" {
			streamBody, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"id":"c1","model":"qwen3:8b","choices":[{"index":0,"delta":{"role":"assistant","content":"pong"},"finish_reason":"stop"}]}`+"\n\n")
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
		t.Fatalf("New returned error: %v", err)
	}
	ctx := context.Background()
	req := &llm.Request{Model: "qwen3:8b", Messages: []llm.Message{llm.UserText("Say pong")}}
	resp, err := p.Chat(ctx, req)
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Text() != "pong" || resp.Provider != "ollama" {
		t.Fatalf("chat response = %+v", resp)
	}

	streamed, err := llm.Collect(p.ChatStream(ctx, req))
	if err != nil {
		t.Fatalf("stream returned error: %v", err)
	}
	if streamed.Text() != "pong" || streamed.Usage.TotalTokens != 2 {
		t.Fatalf("streamed response = %+v", streamed)
	}
	if want := `"stream_options":{"include_usage":true}`; !strings.Contains(string(streamBody), want) {
		t.Fatalf("stream request missing %s: %s", want, streamBody)
	}
}
