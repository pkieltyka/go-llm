// Observability wraps a provider with llm.UsageTracker middleware and prints
// aggregated call and token counts after a request.
//
// With ANTHROPIC_API_KEY or OPENROUTER_API_KEY set, it tracks a real API
// call. Without credentials, it falls back to a scripted llmtest provider so
// the example still runs offline.
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

	// UsageTracker aggregates usage across every call made through the
	// wrapped provider, keyed by provider/model.
	tracker := llm.NewUsageTracker()
	wrapped := llm.Wrap(p, tracker.Middleware())

	resp, err := wrapped.Chat(ctx, &llm.Request{
		Model:    model,
		Messages: []llm.Message{llm.UserText("Reply with the single word: pong")},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Text())

	stats := tracker.Stats()
	fmt.Printf("calls=%d input_tokens=%d output_tokens=%d total_tokens=%d\n",
		stats.Calls, stats.Usage.InputTokens, stats.Usage.OutputTokens, stats.Usage.TotalTokens)
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
// with canned usage numbers.
func offlineProvider() (llm.Provider, string) {
	fmt.Println("(offline demo — set ANTHROPIC_API_KEY or OPENROUTER_API_KEY to run against the real API)")
	p := llmtest.New(llmtest.WithName("demo"))
	p.EnqueueResponse(&llm.Response{
		Provider: "demo",
		Model:    "demo-model",
		Parts:    []llm.Part{llm.Text("pong")},
		Usage:    llm.Usage{InputTokens: 8, OutputTokens: 3, TotalTokens: 11},
	})
	return p, "demo-model"
}
