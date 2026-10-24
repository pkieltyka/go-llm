// Self-hosted talks to a model server you run yourself: vLLM through its
// dedicated preset, or any OpenAI Chat Completions-compatible server
// (Ollama, llama.cpp, LM Studio, ...) through chatcompletions.New.
//
// Constructors are host-first and the API key is optional; it is never read
// from the environment.
//
//	LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.2 go run ./examples/self-hosted
//	LLM_SERVER=vllm LLM_BASE_URL=http://gpu-box:8000/v1 LLM_MODEL=Qwen/Qwen3-8B go run ./examples/self-hosted
//
// LLM_API_KEY is sent when set. Without LLM_BASE_URL the example starts a
// tiny in-process OpenAI-compatible server, so it still runs offline over
// real HTTP.
package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
	"github.com/pkieltyka/go-llm/providers/vllm"
)

func main() {
	ctx := context.Background()
	baseURL, model := os.Getenv("LLM_BASE_URL"), os.Getenv("LLM_MODEL")
	if baseURL == "" {
		server := httptest.NewServer(fakeServer())
		defer server.Close()
		fmt.Println("(offline demo against an in-process server — set LLM_BASE_URL and LLM_MODEL to use yours)")
		baseURL, model = server.URL+"/v1", "demo-model"
	}

	p, err := newProvider(baseURL)
	if err != nil {
		panic(err)
	}

	// Most servers list what they serve; Models works the same everywhere.
	models, err := p.Models(ctx)
	if err != nil {
		panic(err)
	}
	for _, m := range models {
		fmt.Printf("serves %s (context window %d)\n", m.ID, m.ContextWindow)
	}
	if model == "" && len(models) > 0 {
		model = models[0].ID
	}

	resp, err := p.Chat(ctx, &llm.Request{
		Model:     model,
		Messages:  []llm.Message{llm.UserText("Say hello in one short sentence.")},
		MaxTokens: 100,
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s: %s\n(%d input, %d output tokens)\n", resp.Model, resp.Text(), resp.Usage.InputTokens, resp.Usage.OutputTokens)
}

// newProvider picks the vLLM preset (reasoning output, typed extensions,
// max_model_len as the context window) or the generic Chat Completions
// engine, and passes LLM_API_KEY only when it is set.
func newProvider(baseURL string) (llm.Provider, error) {
	key := os.Getenv("LLM_API_KEY")
	if os.Getenv("LLM_SERVER") == "vllm" {
		var opts []vllm.Option
		if key != "" {
			opts = append(opts, vllm.WithAPIKey(key))
		}
		return vllm.New(baseURL, opts...)
	}
	var opts []chatcompletions.Option
	if key != "" {
		opts = append(opts, chatcompletions.WithAPIKey(key))
	}
	return chatcompletions.New(baseURL, opts...)
}

// fakeServer answers the two OpenAI-compatible endpoints the example uses.
func fakeServer() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"demo-model","object":"model","owned_by":"me"}]}`)
	})
	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl-1","object":"chat.completion","model":"demo-model",
			"choices":[{"index":0,"message":{"role":"assistant","content":"Hello from your own server."},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":14,"completion_tokens":7,"total_tokens":21}}`)
	})
	return mux
}
