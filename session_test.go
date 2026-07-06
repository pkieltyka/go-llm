package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestSessionToolRoundTrip drives the full FS §10D session tool loop:
// chat → tool call → AddToolResults → Continue → final answer, asserting
// the session sends its configured tools/tool choice on every request and
// that Continue does NOT append a user turn.
func TestSessionToolRoundTrip(t *testing.T) {
	p := llmtest.New(llmtest.WithName("fake"), llmtest.WithCapabilities(llm.CapabilityTools))
	p.EnqueueResponse(&llm.Response{
		Parts:      []llm.Part{llm.ToolCall("call_1", "lookup", []byte(`{"q":"go"}`))},
		StopReason: llm.StopReasonToolUse,
	})
	p.EnqueueResponse(&llm.Response{
		Parts:      []llm.Part{llm.Text("go is a language")},
		StopReason: llm.StopReasonEndTurn,
	})

	lookup := llm.Tool{
		Name:        "lookup",
		Description: "Look up a short value.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`),
	}
	s := llm.NewSession(p, "model-a",
		llm.WithSessionTools(lookup),
		llm.WithSessionToolChoice(llm.ToolChoice{Mode: llm.ToolChoiceAuto}),
	)

	first, err := s.ChatText(context.Background(), "what is go?")
	if err != nil {
		t.Fatalf("ChatText returned error: %v", err)
	}
	calls := first.ToolCalls()
	if len(calls) != 1 || calls[0].Name != "lookup" {
		t.Fatalf("tool calls = %+v, want one lookup call", calls)
	}

	result := llm.ToolResult(calls[0].ID, `{"answer":"a language"}`)
	result.Name = calls[0].Name
	s.AddToolResults(result)

	final, err := s.Continue(context.Background())
	if err != nil {
		t.Fatalf("Continue returned error: %v", err)
	}
	if final.Text() != "go is a language" {
		t.Fatalf("final text = %q", final.Text())
	}

	requests := p.Requests()
	if len(requests) != 2 {
		t.Fatalf("recorded requests = %d, want 2", len(requests))
	}
	for i, req := range requests {
		if len(req.Tools) != 1 || req.Tools[0].Name != "lookup" {
			t.Fatalf("request %d tools = %+v, want session lookup tool", i, req.Tools)
		}
		if req.ToolChoice.Mode != llm.ToolChoiceAuto {
			t.Fatalf("request %d tool choice = %+v", i, req.ToolChoice)
		}
	}
	// Continue's request: user, assistant(tool call), tool results — and NO
	// extra user turn.
	roles := make([]llm.Role, 0, len(requests[1].Messages))
	for _, msg := range requests[1].Messages {
		roles = append(roles, msg.Role)
	}
	want := []llm.Role{llm.RoleUser, llm.RoleAssistant, llm.RoleTool}
	if len(roles) != len(want) {
		t.Fatalf("Continue request roles = %v, want %v", roles, want)
	}
	for i := range want {
		if roles[i] != want[i] {
			t.Fatalf("Continue request roles = %v, want %v", roles, want)
		}
	}

	// Full history after the loop: user, assistant, tool, assistant.
	msgs := s.Messages()
	if len(msgs) != 4 || msgs[3].Role != llm.RoleAssistant {
		t.Fatalf("session history = %d messages (last %q), want 4 ending assistant", len(msgs), msgs[len(msgs)-1].Role)
	}
}

// TestSessionContinueErrorLeavesHistoryUnchanged asserts Continue's rollback
// contract matches Chat's: after a failed call, nothing new is in history.
func TestSessionContinueErrorLeavesHistoryUnchanged(t *testing.T) {
	p := llmtest.New(llmtest.WithName("fake"))
	p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.ToolCall("call_1", "lookup", []byte(`{}`))}})
	p.EnqueueError(errors.New("boom"))

	s := llm.NewSession(p, "model-a")
	if _, err := s.ChatText(context.Background(), "hi"); err != nil {
		t.Fatalf("ChatText returned error: %v", err)
	}
	s.AddToolResults(llm.ToolResult("call_1", "result"))
	before := len(s.Messages())

	if _, err := s.Continue(context.Background()); err == nil {
		t.Fatalf("Continue returned nil error, want failure")
	}
	if got := len(s.Messages()); got != before {
		t.Fatalf("history length after failed Continue = %d, want %d", got, before)
	}
}

// TestSessionStreamAppliesMessageEndViaSharedCollect is the B2 regression
// test on the public surface: the session stream path uses the SAME event
// application as Collect (applyCollectEvent), so MessageEnd fields —
// usage here, Raw in the internal companion test in stream.go's tests —
// land on the session's collected response instead of diverging.
func TestSessionStreamAppliesMessageEndViaSharedCollect(t *testing.T) {
	p := llmtest.New(llmtest.WithName("fake"), llmtest.WithCapabilities(llm.CapabilityStreaming))
	p.EnqueueStream(
		llm.MessageStart{ID: "msg_1", Provider: "fake", Model: "model-a"},
		llm.TextDelta{Index: 0, Text: "hi"},
		llm.MessageEnd{
			StopReason: llm.StopReasonEndTurn,
			Usage:      llm.Usage{InputTokens: 3, OutputTokens: 5, TotalTokens: 8},
			Raw:        map[string]string{"upstream": "extras"},
		},
	)

	s := llm.NewSession(p, "model-a")
	for _, err := range s.ChatStream(context.Background(), llm.Text("hello")) {
		if err != nil {
			t.Fatalf("stream returned error: %v", err)
		}
	}
	if got := s.Usage(); got.TotalTokens != 8 || got.OutputTokens != 5 {
		t.Fatalf("session usage after stream = %+v, want MessageEnd usage", got)
	}
	msgs := s.Messages()
	if len(msgs) != 2 || msgs[1].Role != llm.RoleAssistant {
		t.Fatalf("session history = %+v, want appended assistant turn", msgs)
	}
}

func TestSessionTurnFlow(t *testing.T) {
	p := llmtest.New(llmtest.WithName("openai"))
	p.EnqueueResponse(&llm.Response{
		Parts: []llm.Part{llm.Text("hello")},
		Usage: llm.Usage{InputTokens: 10, CacheReadTokens: 5, CacheWriteTokens: 5, OutputTokens: 5, TotalTokens: 25},
	})
	s := llm.NewSession(p, "gpt-5.2", llm.WithSessionSystem("system"), llm.WithSessionID("session-1"), llm.WithSessionMaxTokens(100))

	resp, err := s.ChatText(context.Background(), "hi")
	if err != nil {
		t.Fatalf("ChatText returned error: %v", err)
	}
	if resp.Provider != "openai" || resp.Model != "gpt-5.2" {
		t.Fatalf("response provenance = %q/%q", resp.Provider, resp.Model)
	}
	requests := p.Requests()
	if len(requests) != 1 || requests[0].System != "system" || requests[0].SessionID != "session-1" || requests[0].MaxTokens != 100 {
		t.Fatalf("recorded request = %+v", requests)
	}
	if msgs := s.Messages(); len(msgs) != 2 || msgs[0].Role != llm.RoleUser || msgs[1].Role != llm.RoleAssistant {
		t.Fatalf("session messages = %+v", msgs)
	}
	s.AddToolResults(llm.ToolResultPart{ToolCallID: "call_1", Name: "lookup", Content: []llm.Part{llm.Text("result")}})
	if msgs := s.Messages(); len(msgs) != 3 || msgs[2].Role != llm.RoleTool {
		t.Fatalf("session tool messages = %+v", msgs)
	}
	if got := s.Usage().TotalTokens; got != 25 {
		t.Fatalf("session usage total = %d", got)
	}
	ctxUsage, ok := s.ContextUsage()
	if !ok || ctxUsage.UsedTokens != 25 {
		t.Fatalf("ContextUsage = %+v, %v", ctxUsage, ok)
	}

	unknown := llm.NewSession(p, "missing-model")
	if _, ok := unknown.ContextUsage(); ok {
		t.Fatalf("unknown model ContextUsage returned ok=true")
	}
}

func TestSessionChatStreamAppendsOnCompletion(t *testing.T) {
	p := llmtest.New(llmtest.WithName("openai"))
	p.EnqueueStream(
		llm.MessageStart{Provider: "openai", Model: "gpt-5.2"},
		llm.TextDelta{Index: 0, Text: "streamed"},
		llm.MessageEnd{Usage: llm.Usage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3}},
	)
	s := llm.NewSession(p, "gpt-5.2")
	stream := s.ChatStream(context.Background(), llm.Text("hi"))
	resp, err := llm.Collect(stream)
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if resp.Text() != "streamed" {
		t.Fatalf("stream text = %q", resp.Text())
	}
	if msgs := s.Messages(); len(msgs) != 2 || msgs[1].Role != llm.RoleAssistant || msgText(msgs[1]) != "streamed" {
		t.Fatalf("session messages = %+v", msgs)
	}

	resp, err = llm.Collect(stream)
	if resp != nil {
		t.Fatalf("second Collect response = %+v, want nil", resp)
	}
	if !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("second Collect error = %v, want ErrBadRequest", err)
	}
	if msgs := s.Messages(); len(msgs) != 2 || msgs[1].Role != llm.RoleAssistant || msgText(msgs[1]) != "streamed" {
		t.Fatalf("session messages after second range = %+v", msgs)
	}
}

func TestSessionChatRollsBackUserTurnOnError(t *testing.T) {
	p := llmtest.New(llmtest.WithName("openai"))
	p.EnqueueError(llm.ErrRateLimited)
	p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text("ok")}})
	s := llm.NewSession(p, "gpt-5.2")

	if _, err := s.ChatText(context.Background(), "first"); !errors.Is(err, llm.ErrRateLimited) {
		t.Fatalf("first ChatText error = %v, want ErrRateLimited", err)
	}
	if msgs := s.Messages(); len(msgs) != 0 {
		t.Fatalf("messages after failed chat = %+v, want empty", msgs)
	}

	if _, err := s.ChatText(context.Background(), "second"); err != nil {
		t.Fatalf("second ChatText returned error: %v", err)
	}
	requests := p.Requests()
	if len(requests) != 2 {
		t.Fatalf("requests len = %d, want 2", len(requests))
	}
	if got := msgText(requests[1].Messages[0]); got != "second" {
		t.Fatalf("retry request first message = %q, want second", got)
	}
}

func TestSessionChatStreamRollsBackOnEarlyBreak(t *testing.T) {
	p := llmtest.New(llmtest.WithName("openai"))
	p.EnqueueStream(
		llm.MessageStart{Provider: "openai", Model: "gpt-5.2"},
		llm.TextDelta{Index: 0, Text: "partial"},
		llm.MessageEnd{StopReason: llm.StopReasonEndTurn},
	)
	p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text("ok")}})
	s := llm.NewSession(p, "gpt-5.2")

	for range s.ChatStream(context.Background(), llm.Text("stream")) {
		break
	}
	if msgs := s.Messages(); len(msgs) != 0 {
		t.Fatalf("messages after early stream break = %+v, want empty", msgs)
	}

	if _, err := s.ChatText(context.Background(), "next"); err != nil {
		t.Fatalf("ChatText after early break returned error: %v", err)
	}
	requests := p.Requests()
	if len(requests) != 2 {
		t.Fatalf("requests len = %d, want 2", len(requests))
	}
	if got := msgText(requests[1].Messages[0]); got != "next" {
		t.Fatalf("request after early break first message = %q, want next", got)
	}
}
