// Prompt-deltas changes the instructions and the offered tools in the middle
// of a conversation without rewriting the start of the prompt.
//
// A prompt cache matches a prefix, so editing Request.System or Request.Tools
// between turns invalidates the whole cache. Instead, a RoleSystem message
// with a SystemDeltaPart (a named section of the instructions) and a
// ToolsDeltaPart (tools added or removed) records the change where it
// happens. Models that accept such changes in place (per-model "behavior" in
// models.json) receive them as positional messages, keeping the cached prefix;
// for every other model the provider folds them into System and Tools, which
// llm.FoldPositional also does for your own code.
//
// With ANTHROPIC_API_KEY set it talks to Claude Haiku 5.5, which takes both
// changes in place, and prints the cache reads of each turn. Without
// credentials it falls back to the scripted llmtest provider and shows the
// request the provider received.
package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/providers/anthropic"
)

func main() {
	ctx := context.Background()
	p, model, fake := newProvider()

	lookupOrder := llm.Tool{
		Name:        "lookup_order",
		Description: "Look up the shipping status of an order by its ID.",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"order_id": map[string]any{"type": "string"}},
			"required":   []string{"order_id"},
		},
	}

	history := llm.NewHistory()
	history.AddUserText("Hi! What can you help me with?")
	req := &llm.Request{
		Model:       model,
		System:      "You are a concise support assistant. Answer in one sentence.",
		SystemCache: &llm.CacheHint{}, // cache the stable start of the prompt
		Messages:    history.Messages(),
		MaxTokens:   200,
	}
	turn(ctx, p, req, history, "turn 1")

	// The user moves on to orders: enable the order tools from here on,
	// without touching req.System or req.Tools.
	history.Add(llm.Message{Role: llm.RoleSystem, Parts: []llm.Part{
		llm.SystemDeltaPart{Section: "orders", Text: "Order tools are enabled: use lookup_order for any order question."},
		llm.ToolsDeltaPart{Added: []llm.Tool{lookupOrder}},
	}})
	history.AddUserText("Where is my order 42?")
	req.Messages = history.Messages()
	turn(ctx, p, req, history, "turn 2")

	if fake != nil {
		// The provider received the change as a positional system message;
		// System and Tools are unchanged, so the cached prefix still matches.
		sent := fake.Requests()[1]
		fmt.Printf("\nturn 2 request: System unchanged=%v, Tools=%d, positional system messages=%d\n",
			sent.System == req.System, len(sent.Tools), countSystemMessages(sent.Messages))
	}

	// What a model without in-place support is sent instead: the same state,
	// folded into System and Tools.
	folded := llm.FoldPositional(req, llm.PositionalSupport{})
	fmt.Printf("\nfolded for other models:\n  system: %q\n  tools:  %s\n",
		folded.System, toolNames(folded.Tools))
}

// turn sends req, prints the reply and its cache usage, and records the
// assistant turn.
func turn(ctx context.Context, p llm.Provider, req *llm.Request, history *llm.History, label string) {
	resp, err := p.Chat(ctx, req)
	if err != nil {
		panic(err)
	}
	reply := resp.Text()
	for _, call := range resp.ToolCalls() {
		reply += fmt.Sprintf(" [calls %s(%s)]", call.Name, call.Args)
	}
	fmt.Printf("%s: %s\n  cache read %d, cache write %d tokens\n",
		label, strings.TrimSpace(reply), resp.Usage.CacheReadTokens, resp.Usage.CacheWriteTokens)
	history.AddResponse(resp)
}

func countSystemMessages(messages []llm.Message) int {
	n := 0
	for _, msg := range messages {
		if msg.Role == llm.RoleSystem {
			n++
		}
	}
	return n
}

func toolNames(tools []llm.Tool) string {
	names := make([]string, len(tools))
	for i, tool := range tools {
		names[i] = tool.Name
	}
	return "[" + strings.Join(names, ", ") + "]"
}

// newProvider returns Anthropic when ANTHROPIC_API_KEY is set, else the
// scripted fake (also returned, to inspect what it received).
func newProvider() (llm.Provider, string, *llmtest.Provider) {
	if os.Getenv("ANTHROPIC_API_KEY") != "" {
		p, err := anthropic.New() // reads ANTHROPIC_API_KEY
		if err != nil {
			panic(err)
		}
		return p, "claude-haiku-5-5", nil
	}
	fmt.Println("(offline demo — set ANTHROPIC_API_KEY to run against the real API)")
	p := llmtest.New()
	p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text("I can answer questions about our products and services.")}})
	p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.ToolCallPart{ID: "call_1", Name: "lookup_order", Args: []byte(`{"order_id":"42"}`)}}})
	return p, "demo-model", p
}
