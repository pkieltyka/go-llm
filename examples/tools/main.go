// Tools runs a full tool-calling round trip: the model requests a tool call,
// the program executes it locally, hands the result back, and prints the
// model's final answer.
//
// With ANTHROPIC_API_KEY or OPENROUTER_API_KEY set, it runs against the real
// provider API. Without credentials, it falls back to a scripted llmtest
// provider that plays both turns offline.
package main

import (
	"context"
	"encoding/json"
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

	weatherTool := llm.Tool{
		Name:        "weather",
		Description: "Get the current weather for a city.",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"city": map[string]any{"type": "string"}},
			"required":   []string{"city"},
		},
	}

	h := llm.NewHistory()
	h.AddUserText("What is the weather in Toronto right now? Use the weather tool.")

	resp, err := p.Chat(ctx, &llm.Request{
		Model:    model,
		Messages: h.Messages(),
		Tools:    []llm.Tool{weatherTool},
	})
	if err != nil {
		panic(err)
	}

	// Tool loop: execute requested calls locally and hand the results back
	// until the model stops asking for tools and produces a final answer.
	for resp.StopReason == llm.StopReasonToolUse {
		h.AddResponse(resp)
		for _, call := range resp.ToolCalls() {
			fmt.Printf("tool call: %s(%s)\n", call.Name, call.Args)
			h.AddToolResults(llm.ToolResult(call.ID, lookupWeather(call.Args)))
		}
		resp, err = p.Chat(ctx, &llm.Request{
			Model:    model,
			Messages: h.Messages(),
			Tools:    []llm.Tool{weatherTool},
		})
		if err != nil {
			panic(err)
		}
	}
	fmt.Println(resp.Text())
}

// lookupWeather is the app-side implementation of the weather tool.
func lookupWeather(args json.RawMessage) string {
	var in struct {
		City string `json:"city"`
	}
	_ = json.Unmarshal(args, &in)
	return fmt.Sprintf(`{"city":%q,"temp_c":-4,"conditions":"light snow"}`, in.City)
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
// that first requests the weather tool, then answers with the result.
func offlineProvider() (llm.Provider, string) {
	fmt.Println("(offline demo — set ANTHROPIC_API_KEY or OPENROUTER_API_KEY to run against the real API)")
	p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityTools))
	p.EnqueueResponse(&llm.Response{
		Provider:   "llmtest",
		Model:      "demo-model",
		StopReason: llm.StopReasonToolUse,
		Parts: []llm.Part{
			llm.ToolCall("call_weather", "weather", json.RawMessage(`{"city":"Toronto"}`)),
		},
	})
	p.EnqueueResponse(&llm.Response{
		Provider:   "llmtest",
		Model:      "demo-model",
		StopReason: llm.StopReasonEndTurn,
		Parts:      []llm.Part{llm.Text("It is -4°C with light snow in Toronto.")},
	})
	return p, "demo-model"
}
