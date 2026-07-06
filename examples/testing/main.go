// Testing shows how to test YOUR code that consumes go-llm.
//
// Summarize and StreamSummary below take llm.Provider as a dependency — that
// interface is the seam that makes them testable. main_test.go unit-tests
// them against the scripted llmtest fake: no network, no credentials, no
// flakes. Run it with:
//
//	go test ./examples/testing/
//
// The same functions run unchanged against a real provider: main() uses
// Anthropic when ANTHROPIC_API_KEY is set, and the llmtest fake otherwise.
package main

import (
	"context"
	"fmt"
	"io"
	"os"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/providers/anthropic"
)

// Summarize asks the model for a one-sentence summary of text.
func Summarize(ctx context.Context, p llm.Provider, model, text string) (string, error) {
	resp, err := p.Chat(ctx, &llm.Request{
		Model:    model,
		System:   "Summarize the user's text in one short sentence.",
		Messages: []llm.Message{llm.UserText(text)},
	})
	if err != nil {
		return "", err
	}
	return resp.Text(), nil
}

// StreamSummary writes the summary to w incrementally as it is generated.
func StreamSummary(ctx context.Context, p llm.Provider, model, text string, w io.Writer) error {
	events := p.ChatStream(ctx, &llm.Request{
		Model:    model,
		System:   "Summarize the user's text in one short sentence.",
		Messages: []llm.Message{llm.UserText(text)},
	})
	for chunk, err := range llm.StreamText(events) {
		if err != nil {
			return err
		}
		if _, err := io.WriteString(w, chunk); err != nil {
			return err
		}
	}
	return nil
}

func main() {
	ctx := context.Background()
	p, model := newProvider()

	summary, err := Summarize(ctx, p, model,
		"Go 1.18 added generics via type parameters after more than a decade of design iterations.")
	if err != nil {
		panic(err)
	}
	fmt.Println(summary)
}

// newProvider returns a real provider when API credentials are present in
// the environment, and the scripted llmtest fake otherwise.
func newProvider() (llm.Provider, string) {
	if os.Getenv("ANTHROPIC_API_KEY") != "" {
		p, err := anthropic.New() // reads ANTHROPIC_API_KEY
		if err != nil {
			panic(err)
		}
		return p, "claude-haiku-4-5"
	}
	fmt.Println("(offline demo — set ANTHROPIC_API_KEY to run against the real API)")
	p := llmtest.New()
	p.EnqueueResponse(&llm.Response{
		Provider: "llmtest",
		Model:    "demo-model",
		Parts:    []llm.Part{llm.Text("Go gained generics in 1.18.")},
	})
	return p, "demo-model"
}
