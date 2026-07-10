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
	"errors"
	"fmt"
	"os"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/providers/anthropic"
	"github.com/pkieltyka/go-llm/providers/openrouter"
	"github.com/pkieltyka/go-llm/schema"
)

type toolHandler struct {
	tool    llm.Tool
	execute func(context.Context, json.RawMessage) (string, error)
}

type toolResultSink interface {
	AddToolResults(...llm.ToolResultPart)
}

func main() {
	ctx := context.Background()
	p, model := newProvider()

	weather := toolHandler{tool: llm.Tool{
		Name:        "weather",
		Description: "Get the current weather for a city.",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"city": map[string]any{"type": "string"}},
			"required":   []string{"city"},
		},
	}, execute: lookupWeather}
	tools := map[string]toolHandler{weather.tool.Name: weather}

	h := llm.NewHistory()
	h.AddUserText("What is the weather in Toronto right now? Use the weather tool.")

	resp, err := p.Chat(ctx, &llm.Request{
		Model:    model,
		Messages: h.Messages(),
		Tools:    []llm.Tool{weather.tool},
	})
	if err != nil {
		panic(err)
	}

	// Tool loop: execute requested calls locally and hand the results back
	// until the model stops asking for tools and produces a final answer.
	for resp.StopReason == llm.StopReasonToolUse {
		h.AddResponse(resp)
		calls := resp.ToolCalls()
		for _, call := range calls {
			fmt.Printf("tool call: %s(%s)\n", call.Name, call.Args)
		}
		dispatchToolCalls(ctx, h, tools, calls)
		resp, err = p.Chat(ctx, &llm.Request{
			Model:    model,
			Messages: h.Messages(),
			Tools:    []llm.Tool{weather.tool},
		})
		if err != nil {
			panic(err)
		}
	}
	fmt.Println(resp.Text())
}

// dispatchToolCalls executes one assistant tool-use turn and appends every
// result as a single grouped tool message.
func dispatchToolCalls(ctx context.Context, sink toolResultSink, tools map[string]toolHandler, calls []llm.ToolCallPart) {
	results := make([]llm.ToolResultPart, 0, len(calls))
	for _, call := range calls {
		results = append(results, dispatchToolCall(ctx, tools, call))
	}
	if len(results) > 0 {
		sink.AddToolResults(results...)
	}
}

func dispatchToolCall(ctx context.Context, tools map[string]toolHandler, call llm.ToolCallPart) llm.ToolResultPart {
	handler, ok := tools[call.Name]
	if !ok {
		return toolError(call, fmt.Errorf("unknown tool %q", call.Name))
	}
	if err := schema.ValidateArgs(handler.tool, call.Args); err != nil {
		return toolError(call, fmt.Errorf("invalid arguments: %w", err))
	}
	result, err := handler.execute(ctx, call.Args)
	if err != nil {
		return toolError(call, fmt.Errorf("execution failed: %w", err))
	}
	part := llm.ToolResult(call.ID, result)
	part.Name = call.Name
	return part
}

func toolError(call llm.ToolCallPart, err error) llm.ToolResultPart {
	part := llm.ToolResult(call.ID, err.Error())
	part.Name = call.Name
	part.IsError = true
	return part
}

// lookupWeather is the app-side implementation of the weather tool.
func lookupWeather(_ context.Context, args json.RawMessage) (string, error) {
	var in struct {
		City string `json:"city"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", err
	}
	if in.City == "Atlantis" {
		return "", errors.New("weather station unavailable")
	}
	return fmt.Sprintf(`{"city":%q,"temp_c":-4,"conditions":"light snow"}`, in.City), nil
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
