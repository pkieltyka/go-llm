// Stream prints a chat response incrementally as the provider generates it,
// using ChatStream and the llm.StreamText iterator.
//
// With ANTHROPIC_API_KEY or OPENROUTER_API_KEY set, it streams from the real
// provider API. Without credentials, it falls back to a scripted llmtest
// stream so the example still runs offline.
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

	events := p.ChatStream(ctx, &llm.Request{
		Model:    model,
		Messages: []llm.Message{llm.UserText("Write one sentence about Go's concurrency model.")},
	})
	for text, err := range llm.StreamText(events) {
		if err != nil {
			panic(err)
		}
		fmt.Print(text)
	}
	fmt.Println()
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

// offlineProvider is the no-credentials fallback: a scripted llmtest stream.
func offlineProvider() (llm.Provider, string) {
	fmt.Println("(offline demo — set ANTHROPIC_API_KEY or OPENROUTER_API_KEY to run against the real API)")
	p := llmtest.New()
	p.EnqueueStream(
		llm.MessageStart{Provider: "llmtest", Model: "demo-model"},
		llm.TextDelta{Index: 0, Text: "Goroutines and channels make "},
		llm.TextDelta{Index: 0, Text: "concurrency a language feature."},
		llm.MessageEnd{StopReason: llm.StopReasonEndTurn},
	)
	return p, "demo-model"
}
