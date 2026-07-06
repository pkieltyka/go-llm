package vllm_test

import (
	"context"
	"fmt"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/vllm"
)

// Example mirrors the README self-hosted snippet: host-first, keyless
// construction with typed vLLM extensions.
func Example() {
	ctx := context.Background()

	p, err := vllm.New("http://localhost:8000/v1") // keyless by default
	if err != nil {
		panic(err)
	}

	resp, err := p.Chat(ctx, &llm.Request{
		Model:    "Qwen/Qwen3.6-27B-FP8",
		Effort:   llm.EffortNone, // thinking-by-default models answer tersely
		Messages: []llm.Message{llm.UserText("hello")},
		ProviderOptions: vllm.Options{
			XArgs: map[string]any{"custom_engine_arg": "1"},
		},
	})
	if err != nil {
		fmt.Println("no vLLM server running:", err != nil)
		return
	}
	fmt.Println(resp.Text())
}

// ExampleProvider_Tokenize shows exact context accounting: the /tokenize
// extension returns the server-computed token count and max_model_len for a
// request BEFORE sending it, and TokenizeResult.ContextUsage bridges both
// into the unified llm.ContextUsage — ground truth instead of the
// estimate-based path (a prior response's Usage + a price-table window).
func ExampleProvider_Tokenize() {
	ctx := context.Background()

	p, err := vllm.New("http://localhost:8000/v1")
	if err != nil {
		panic(err)
	}

	req := &llm.Request{
		Model:    "Qwen/Qwen3.6-27B-FP8",
		Messages: []llm.Message{llm.UserText("hello")},
	}
	result, err := p.Tokenize(ctx, req) // same conversion + validation as Chat
	if err != nil {
		fmt.Println("no vLLM server running:", err != nil)
		return
	}
	usage := result.ContextUsage() // exact occupancy vs the model's window
	fmt.Printf("prompt occupies %d of %d tokens (%.2f%%), %d remaining\n",
		usage.UsedTokens, usage.Window, usage.UsedPercent, usage.Remaining)
}
