package llm_test

import (
	"context"
	"encoding/json"
	"fmt"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

func ExampleProvider_chat() {
	ctx := context.Background()
	p := llmtest.New()
	p.EnqueueResponse(&llm.Response{
		Provider: "llmtest",
		Model:    "test-model",
		Parts:    []llm.Part{llm.Text("hello")},
	})

	resp, err := p.Chat(ctx, &llm.Request{
		Model:    "test-model",
		Messages: []llm.Message{llm.UserText("Say hello.")},
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(resp.Text())
	// Output: hello
}

func ExampleProvider_stream() {
	ctx := context.Background()
	p := llmtest.New()
	p.EnqueueStream(
		llm.MessageStart{Provider: "llmtest", Model: "test-model"},
		llm.TextDelta{Index: 0, Text: "hel"},
		llm.TextDelta{Index: 0, Text: "lo"},
		llm.MessageEnd{StopReason: llm.StopReasonEndTurn},
	)

	for text, err := range llm.StreamText(p.ChatStream(ctx, &llm.Request{
		Model:    "test-model",
		Messages: []llm.Message{llm.UserText("Say hello.")},
	})) {
		if err != nil {
			panic(err)
		}
		fmt.Print(text)
	}
	// Output: hello
}

func ExampleTool() {
	ctx := context.Background()
	p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityTools))
	p.EnqueueResponse(&llm.Response{
		Provider: "llmtest",
		Model:    "test-model",
		Parts: []llm.Part{
			llm.ToolCall("call_1", "lookup_weather", json.RawMessage(`{"city":"Toronto"}`)),
		},
		StopReason: llm.StopReasonToolUse,
	})

	resp, err := p.Chat(ctx, &llm.Request{
		Model:    "test-model",
		Messages: []llm.Message{llm.UserText("Weather in Toronto?")},
		Tools: []llm.Tool{{
			Name:        "lookup_weather",
			Description: "Look up current weather by city.",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
				"required":   []string{"city"},
			},
		}},
	})
	if err != nil {
		panic(err)
	}

	call := resp.ToolCalls()[0]
	fmt.Printf("%s %s\n", call.Name, call.Args)
	// Output: lookup_weather {"city":"Toronto"}
}

func ExampleParse() {
	type Answer struct {
		Answer string `json:"answer"`
	}

	ctx := context.Background()
	p := llmtest.New(llmtest.WithCapabilities(llm.CapabilityJSONSchema))
	p.EnqueueResponse(&llm.Response{
		Provider: "llmtest",
		Model:    "test-model",
		Parts:    []llm.Part{llm.Text(`{"answer":"blue"}`)},
	})

	out, _, err := llm.Parse[Answer](ctx, p, &llm.Request{
		Model:    "test-model",
		Messages: []llm.Message{llm.UserText("What color is the sky?")},
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(out.Answer)
	// Output: blue
}

func ExampleHistory() {
	h := llm.NewHistory()
	h.AddUserText("Hello")
	h.AddResponse(&llm.Response{
		Provider: "llmtest",
		Model:    "test-model",
		Parts:    []llm.Part{llm.Text("Hi there.")},
	})
	h.AddUserText("Continue.")

	fmt.Println(len(h.Messages()))
	// Output: 3
}

func ExampleWrap() {
	ctx := context.Background()
	p := llmtest.New()
	p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text("hello")}})

	withSuffix := llm.Middleware{
		Chat: func(next llm.ChatFunc) llm.ChatFunc {
			return func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				resp, err := next(ctx, req)
				if resp != nil {
					resp.Parts = append(resp.Parts, llm.Text("!"))
				}
				return resp, err
			}
		},
	}

	resp, err := llm.Wrap(p, withSuffix).Chat(ctx, &llm.Request{
		Model:    "test-model",
		Messages: []llm.Message{llm.UserText("Say hello.")},
	})
	if err != nil {
		panic(err)
	}

	fmt.Println(resp.Text())
	// Output: hello!
}

func ExampleUsageTracker() {
	ctx := context.Background()
	p := llmtest.New(llmtest.WithName("demo"))
	p.EnqueueResponse(&llm.Response{
		Provider: "demo",
		Model:    "test-model",
		Parts:    []llm.Part{llm.Text("ok")},
		Usage:    llm.Usage{InputTokens: 3, OutputTokens: 2, TotalTokens: 5},
	})

	tracker := llm.NewUsageTracker()
	_, err := llm.Wrap(p, tracker.Middleware()).Chat(ctx, &llm.Request{
		Model:    "test-model",
		Messages: []llm.Message{llm.UserText("Ping")},
	})
	if err != nil {
		panic(err)
	}

	stats := tracker.Stats()
	fmt.Printf("%d call, %d tokens\n", stats.Calls, stats.Usage.TotalTokens)
	// Output: 1 call, 5 tokens
}

func ExampleMarshalMessages() {
	h := llm.NewHistory()
	h.AddUserText("Hello")
	h.AddResponse(&llm.Response{
		Provider: "provider-a",
		Model:    "model-a",
		Parts:    []llm.Part{llm.Text("Hi.")},
	})

	data, err := llm.MarshalMessages(h.Messages())
	if err != nil {
		panic(err)
	}
	replayed, err := llm.UnmarshalMessages(data)
	if err != nil {
		panic(err)
	}

	fmt.Printf("%s/%s\n", replayed[1].Provider, replayed[1].Model)
	// Output: provider-a/model-a
}

func ExampleMarshalMessages_providerReplay() {
	ctx := context.Background()
	providerA := llmtest.New(llmtest.WithName("provider-a"))
	providerA.EnqueueResponse(&llm.Response{
		Provider: "provider-a",
		Model:    "model-a",
		Parts:    []llm.Part{llm.Text("Use the release checklist.")},
	})

	h := llm.NewHistory()
	h.AddUserText("How should I prepare?")
	first, err := providerA.Chat(ctx, &llm.Request{
		Model:    "model-a",
		Messages: h.Messages(),
	})
	if err != nil {
		panic(err)
	}
	h.AddResponse(first)

	data, err := llm.MarshalMessages(h.Messages())
	if err != nil {
		panic(err)
	}
	replayed, err := llm.UnmarshalMessages(data)
	if err != nil {
		panic(err)
	}

	providerB := llmtest.New(llmtest.WithName("provider-b"))
	providerB.EnqueueResponse(&llm.Response{
		Provider: "provider-b",
		Model:    "model-b",
		Parts:    []llm.Part{llm.Text("ship checklist")},
	})
	replayed = append(replayed, llm.UserText("Summarize that in two words."))
	resp, err := providerB.Chat(ctx, &llm.Request{
		Model:    "model-b",
		Messages: replayed,
	})
	if err != nil {
		panic(err)
	}

	requests := providerB.Requests()
	fmt.Printf("%s saw %d messages; replayed %s/%s; answered %q\n",
		providerB.Name(),
		len(requests[0].Messages),
		requests[0].Messages[1].Provider,
		requests[0].Messages[1].Model,
		resp.Text(),
	)
	// Output: provider-b saw 3 messages; replayed provider-a/model-a; answered "ship checklist"
}
