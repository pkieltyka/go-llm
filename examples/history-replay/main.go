// History-replay demonstrates cross-provider conversation portability: one
// provider starts a conversation, the messages are persisted to canonical
// JSON with llm.MarshalMessages, reloaded, and a different provider continues
// the same conversation.
//
// With both ANTHROPIC_API_KEY and OPENROUTER_API_KEY set, it performs a real
// Anthropic → OpenRouter handoff. Otherwise it falls back to two scripted
// llmtest providers so the example still runs offline.
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
	providerA, modelA, providerB, modelB := newProviders()
	fmt.Printf("handoff: %s/%s -> %s/%s\n", providerA.Name(), modelA, providerB.Name(), modelB)

	// Provider A starts the conversation.
	h := llm.NewHistory()
	h.AddUserText("Start a release checklist. Give exactly one first step.")
	first, err := providerA.Chat(ctx, &llm.Request{Model: modelA, Messages: h.Messages()})
	if err != nil {
		panic(err)
	}
	h.AddResponse(first)
	fmt.Printf("A: %s\n", first.Text())

	// Persist and reload. Canonical JSON preserves provider/model attribution
	// and reasoning payloads, so any provider can replay the conversation.
	data, err := llm.MarshalMessages(h.Messages())
	if err != nil {
		panic(err)
	}
	messages, err := llm.UnmarshalMessages(data)
	if err != nil {
		panic(err)
	}

	// Provider B continues where A left off.
	messages = append(messages, llm.UserText("Add exactly one next step."))
	resp, err := providerB.Chat(ctx, &llm.Request{Model: modelB, Messages: messages})
	if err != nil {
		panic(err)
	}
	fmt.Printf("B: %s\n", resp.Text())
}

// newProviders returns two distinct real providers when both credentials are
// present in the environment.
func newProviders() (llm.Provider, string, llm.Provider, string) {
	if os.Getenv("ANTHROPIC_API_KEY") != "" && os.Getenv("OPENROUTER_API_KEY") != "" {
		a, err := anthropic.New() // reads ANTHROPIC_API_KEY
		if err != nil {
			panic(err)
		}
		b, err := openrouter.New() // reads OPENROUTER_API_KEY
		if err != nil {
			panic(err)
		}
		return a, "claude-haiku-4-5", b, "qwen/qwen3.6-27b"
	}
	return offlineProviders()
}

// offlineProviders is the fallback: two scripted llmtest fakes standing in
// for the two real providers.
func offlineProviders() (llm.Provider, string, llm.Provider, string) {
	fmt.Println("(offline demo — set ANTHROPIC_API_KEY and OPENROUTER_API_KEY to run against the real APIs)")
	a := llmtest.New(llmtest.WithName("provider-a"))
	a.EnqueueResponse(&llm.Response{
		Provider: "provider-a",
		Model:    "model-a",
		Parts:    []llm.Part{llm.Text("1. Run the tests.")},
	})
	b := llmtest.New(llmtest.WithName("provider-b"))
	b.EnqueueResponse(&llm.Response{
		Provider: "provider-b",
		Model:    "model-b",
		Parts:    []llm.Part{llm.Text("2. Tag the release.")},
	})
	return a, "model-a", b, "model-b"
}
