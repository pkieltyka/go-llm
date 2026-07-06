// Parse decodes a model response directly into a Go struct with llm.Parse,
// which drives the provider's structured-output support (JSON schema, forced
// tools, or JSON mode — whichever the provider advertises).
//
// With ANTHROPIC_API_KEY or OPENROUTER_API_KEY set, it runs against the real
// provider API. Without credentials, it falls back to a scripted llmtest
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

// Answer is the shape we want back; llm.Parse derives a JSON schema from it.
type Answer struct {
	Summary string `json:"summary"`
	Score   int    `json:"score"`
}

func main() {
	ctx := context.Background()
	p, model := newProvider()

	out, _, err := llm.Parse[Answer](ctx, p, &llm.Request{
		Model: model,
		Messages: []llm.Message{
			llm.UserText("Summarize Go's error handling in one sentence, with a 1-10 score for how much you like it."),
		},
	})
	if err != nil {
		panic(err)
	}
	fmt.Printf("summary: %s\nscore: %d\n", out.Summary, out.Score)
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

// offlineProvider is the no-credentials fallback: a scripted llmtest fake
// that returns canned JSON matching the Answer schema.
func offlineProvider() (llm.Provider, string) {
	fmt.Println("(offline demo — set ANTHROPIC_API_KEY or OPENROUTER_API_KEY to run against the real API)")
	p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityJSONSchema))
	p.EnqueueResponse(&llm.Response{
		Provider: "llmtest",
		Model:    "demo-model",
		Parts:    []llm.Part{llm.Text(`{"summary":"Errors are values you check explicitly.","score":8}`)},
	})
	return p, "demo-model"
}
