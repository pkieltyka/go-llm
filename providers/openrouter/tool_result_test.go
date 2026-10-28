package openrouter

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

// OpenRouter's Chat Completions tool results stay text-only (plan 14 B0):
// image and file results, alone or mixed with text, fail before any HTTP
// request, on Chat and ChatStream alike.
func TestOpenRouterRichToolResultsAreUnsupported(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		http.Error(w, "unexpected request", http.StatusInternalServerError)
	}))
	defer server.Close()
	p, err := New(WithAPIKey("test"), WithBaseURL(server.URL), WithHTTPClient(server.Client()), WithMaxRetries(0))
	if err != nil {
		t.Fatal(err)
	}

	cached := llm.Text("caption")
	cached.Cache = &llm.CacheHint{}
	image := llm.ImageData([]byte{1, 2, 3}, "image/png")
	for name, result := range map[string]llm.ToolResultPart{
		"image":                 llm.ToolResultParts("call_1", image),
		"image url":             llm.ToolResultParts("call_1", llm.ImageURL("https://example.test/a.png")),
		"pdf":                   llm.ToolResultParts("call_1", llm.FileData([]byte("%PDF"), "application/pdf", "a.pdf")),
		"text then image":       llm.ToolResultParts("call_1", llm.Text("caption"), &image),
		"cached text and image": llm.ToolResultParts("call_1", cached, image),
	} {
		t.Run(name, func(t *testing.T) {
			req := &llm.Request{
				Model: "openai/gpt-4.1-mini",
				Messages: []llm.Message{
					llm.UserText("look"),
					llm.AssistantParts(llm.ToolCall("call_1", "screenshot", nil)),
					{Role: llm.RoleTool, Parts: []llm.Part{result}},
				},
			}
			if _, err := p.Chat(context.Background(), req); !errors.Is(err, llm.ErrUnsupported) {
				t.Fatalf("Chat error = %v, want ErrUnsupported", err)
			}
			var streamErr error
			for _, err := range p.ChatStream(context.Background(), req) {
				streamErr = err
				break
			}
			if !errors.Is(streamErr, llm.ErrUnsupported) {
				t.Fatalf("ChatStream error = %v, want ErrUnsupported", streamErr)
			}
		})
	}
	if n := requests.Load(); n != 0 {
		t.Fatalf("%d requests reached the server", n)
	}
}
