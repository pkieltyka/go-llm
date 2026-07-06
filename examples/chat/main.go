// Chat sends a single blocking chat request and prints the reply.
//
// With ANTHROPIC_API_KEY or OPENROUTER_API_KEY set, it talks to the real
// provider API. Without credentials, it falls back to the scripted llmtest
// provider so the example still runs offline.
package main

import (
	"context"
	"fmt"
	"os"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/providers/anthropic"
	"github.com/pkieltyka/go-llm/providers/openrouter"
)

func main() {
	ctx := context.Background()
	p, model := newProvider()

	resp, err := p.Chat(ctx, &llm.Request{
		Model:    model,
		Messages: []llm.Message{llm.UserText("Say hello in one short sentence.")},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Text())
}

// newProvider returns a real provider when API credentials are present in the
// environment, pinned to a cheap model.
func newProvider() (llm.Provider, string) {
	switch {
	case os.Getenv("ANTHROPIC_API_KEY") != "":
		p, err := anthropic.New() // reads ANTHROPIC_API_KEY
		if err != nil {
			panic(err)
		}
		return p, "claude-haiku-4-5"
	case os.Getenv("OPENROUTER_API_KEY") != "":
		p, err := openrouter.New() // reads OPENROUTER_API_KEY
		if err != nil {
			panic(err)
		}
		return p, "qwen/qwen3.6-27b"
	}
	return offlineProvider()
}

// offlineProvider is the no-credentials fallback: a scripted llmtest fake.
func offlineProvider() (llm.Provider, string) {
	fmt.Println("(offline demo — set ANTHROPIC_API_KEY or OPENROUTER_API_KEY to run against the real API)")
	p := llmtest.New()
	p.EnqueueResponse(&llm.Response{
		Provider: "llmtest",
		Model:    "demo-model",
		Parts:    []llm.Part{llm.Text("Hello from go-llm.")},
	})
	return p, "demo-model"
}
