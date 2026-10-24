// Context-guard keeps a growing conversation inside the model's context
// window.
//
// llm.ContextGuard is middleware that measures each request before it is
// sent: exactly when the provider counts tokens (Anthropic, vLLM), else with
// a calibrated estimate. When the prompt passes a fraction of the window it
// calls OnNearFull, and with ClampMaxTokens it lowers MaxTokens so the reply
// still fits. go-llm ships no compaction policy, so here the hook aborts the
// call and the loop compacts the history itself (dropping the oldest turns
// for brevity; a real app would summarize them) and retries.
//
// With ANTHROPIC_API_KEY set it talks to Claude Haiku 4.5 and counts tokens
// exactly; without credentials it runs offline on the scripted llmtest
// provider. A small window override makes the guard fire within a few turns
// either way.
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/providers/anthropic"
)

// errCompact asks the loop to shrink the history and retry.
var errCompact = errors.New("context near full: compact and retry")

func main() {
	ctx := context.Background()
	base, model, fake := newProvider()

	p := llm.Wrap(base, llm.ContextGuard(llm.GuardConfig{
		Window:         600, // pretend the model has a tiny window
		MaxOccupancy:   0.7,
		Margin:         32,
		ClampMaxTokens: true,
		OnNearFull: func(_ context.Context, _ *llm.Request, report llm.GuardReport) error {
			fmt.Printf("  guard: %d tokens (exact=%v) is %.0f%% of %d\n",
				report.Tokens, report.Exact, report.Occupancy*100, report.Window)
			return errCompact
		},
	}))

	history := []llm.Message{}
	questions := []string{
		"Name a prime number.",
		"Name another, larger one.",
		"What is the sum of the two?",
		"Is that sum prime?",
		"Name one more prime.",
	}
	padding := strings.Repeat("Please keep answers short. ", 20) // grows the prompt quickly
	for i, question := range questions {
		history = append(history, llm.UserText(padding+question))
		for {
			req := &llm.Request{Model: model, Messages: history, MaxTokens: 400}
			// A pure pre-flight estimate, without the middleware.
			fmt.Printf("turn %d: estimate %d tokens\n", i+1, llm.EstimateRequestTokens(req))
			resp, err := p.Chat(ctx, req)
			// An exact count (Anthropic) that cannot fit at all fails with
			// ErrContextTooLong before the hook runs: compact for that too.
			if (errors.Is(err, errCompact) || errors.Is(err, llm.ErrContextTooLong)) && len(history) > 1 {
				history = compact(history)
				fmt.Printf("  compacted to %d messages\n", len(history))
				continue
			}
			if err != nil {
				panic(err)
			}
			fmt.Printf("  reply: %s\n", strings.TrimSpace(resp.Text()))
			if fake != nil {
				// ClampMaxTokens lowered the requested 400 so the reply fits.
				sent := fake.Requests()
				fmt.Printf("  sent MaxTokens %d\n", sent[len(sent)-1].MaxTokens)
			}
			history = append(history, llm.Message{Role: llm.RoleAssistant, Parts: resp.Parts})
			break
		}
	}
}

// compact drops the oldest user/assistant exchange, keeping the newest
// question. A real application would summarize instead.
func compact(history []llm.Message) []llm.Message {
	if len(history) <= 2 {
		return history[len(history)-1:]
	}
	return append([]llm.Message(nil), history[2:]...)
}

// newProvider returns Anthropic when ANTHROPIC_API_KEY is set, else the
// scripted fake (also returned, to show what it was sent).
func newProvider() (llm.Provider, string, *llmtest.Provider) {
	if os.Getenv("ANTHROPIC_API_KEY") != "" {
		p, err := anthropic.New() // reads ANTHROPIC_API_KEY
		if err != nil {
			panic(err)
		}
		return p, "claude-haiku-4-5", nil
	}
	fmt.Println("(offline demo — set ANTHROPIC_API_KEY to run against the real API)")
	p := llmtest.New()
	for _, reply := range []string{"7.", "13.", "20.", "No, 20 is not prime.", "29."} {
		p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text(reply)}})
	}
	return p, "demo-model", p
}
