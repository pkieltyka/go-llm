package llm_test

import (
	"context"
	"errors"
	"iter"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestMiddlewareBindProviderAware exercises the exported Middleware.Bind
// seam the way a third-party package would: binding captures the wrapped
// provider's identity/capabilities once at Wrap time, and the returned
// middleware's handlers are the ones composed.
func TestMiddlewareBindProviderAware(t *testing.T) {
	p := llmtest.New(llmtest.WithName("fake"), llmtest.WithCapabilities(llm.CapabilityTools))
	p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text("ok")}})

	var boundProvider string
	var boundCaps int
	var sawChat bool
	thirdParty := llm.Middleware{
		// A Bind-time Chat handler that must be REPLACED by the bound one.
		Chat: func(next llm.ChatFunc) llm.ChatFunc {
			return func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				t.Error("unbound Chat handler ran; Wrap must compose the bound middleware")
				return next(ctx, req)
			}
		},
		Bind: func(p llm.Provider) llm.Middleware {
			boundProvider = p.Name()
			boundCaps = len(p.Capabilities())
			return llm.Middleware{
				Chat: func(next llm.ChatFunc) llm.ChatFunc {
					return func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
						sawChat = true
						return next(ctx, req)
					}
				},
			}
		},
	}

	wrapped := llm.Wrap(p, thirdParty)
	if boundProvider != "fake" || boundCaps != 1 {
		t.Fatalf("bound identity = (%q, %d caps), want (fake, 1)", boundProvider, boundCaps)
	}
	if _, err := wrapped.Chat(context.Background(), &llm.Request{
		Model:    "model-a",
		Messages: []llm.Message{llm.UserText("hi")},
	}); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if !sawChat {
		t.Fatalf("bound Chat handler did not run")
	}
}

func TestWrapOrderingAndDelegation(t *testing.T) {
	p := llmtest.New(
		llmtest.WithName("fake"),
		llmtest.WithCapabilities(llm.CapabilityStreaming),
		llmtest.WithModels(llm.ModelInfo{ID: "model-a"}),
	)
	p.EnqueueResponse(&llm.Response{Provider: "fake", Model: "model-a", Parts: []llm.Part{llm.Text("ok")}})

	var order []string
	mw := func(name string) llm.Middleware {
		return llm.Middleware{
			Chat: func(next llm.ChatFunc) llm.ChatFunc {
				return func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
					order = append(order, name+" before")
					resp, err := next(ctx, req)
					order = append(order, name+" after")
					return resp, err
				}
			},
		}
	}

	wrapped := llm.Wrap(p, mw("a"), mw("b"))
	if wrapped.Name() != "fake" {
		t.Fatalf("Name = %q", wrapped.Name())
	}
	if caps := wrapped.Capabilities(); len(caps) != 1 || caps[0] != llm.CapabilityStreaming {
		t.Fatalf("Capabilities = %+v", caps)
	}
	if models, err := wrapped.Models(context.Background()); err != nil || len(models) != 1 || models[0].ID != "model-a" {
		t.Fatalf("Models = %+v, %v", models, err)
	}
	if _, err := wrapped.Chat(context.Background(), &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("hi")}}); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if strings.Join(order, ",") != "a before,b before,b after,a after" {
		t.Fatalf("middleware order = %v", order)
	}
}

func TestWrapComposesFactoriesAndBindOnce(t *testing.T) {
	p := llmtest.New(llmtest.WithName("fake"))
	p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text("one")}})
	p.EnqueueResponse(&llm.Response{Parts: []llm.Part{llm.Text("two")}})
	p.EnqueueStream(llm.MessageStart{}, llm.MessageEnd{})
	p.EnqueueStream(llm.MessageStart{}, llm.MessageEnd{})

	var binds, chatFactories, streamFactories int
	var chatCalls, streamCalls int
	middleware := llm.Middleware{Bind: func(llm.Provider) llm.Middleware {
		binds++
		return llm.Middleware{
			Chat: func(next llm.ChatFunc) llm.ChatFunc {
				chatFactories++
				return func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
					chatCalls++
					return next(ctx, req)
				}
			},
			Stream: func(next llm.StreamFunc) llm.StreamFunc {
				streamFactories++
				return func(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
					streamCalls++
					return next(ctx, req)
				}
			},
		}
	}}

	wrapped := llm.Wrap(p, middleware)
	if binds != 1 || chatFactories != 1 || streamFactories != 1 {
		t.Fatalf("construction counts = bind:%d chat:%d stream:%d, want 1 each", binds, chatFactories, streamFactories)
	}
	req := &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("hi")}}
	for i := 0; i < 2; i++ {
		if _, err := wrapped.Chat(context.Background(), req); err != nil {
			t.Fatalf("Chat %d returned error: %v", i, err)
		}
	}
	for i := 0; i < 2; i++ {
		if _, err := llm.Collect(wrapped.ChatStream(context.Background(), req)); err != nil {
			t.Fatalf("ChatStream %d returned error: %v", i, err)
		}
	}
	if binds != 1 || chatFactories != 1 || streamFactories != 1 {
		t.Fatalf("factories rebuilt after calls: bind:%d chat:%d stream:%d", binds, chatFactories, streamFactories)
	}
	if chatCalls != 2 || streamCalls != 2 {
		t.Fatalf("handler calls = chat:%d stream:%d, want 2 each", chatCalls, streamCalls)
	}
}

func TestRetryDroppedToolCalls(t *testing.T) {
	p := llmtest.New(llmtest.WithName("fake"))
	p.EnqueueResponse(&llm.Response{
		Provider: "fake",
		Model:    "model-a",
		Parts:    []llm.Part{llm.ToolCall("call_1", "lookup", []byte(`{"q":"go"}`))},
		DroppedToolCalls: []llm.DroppedToolCall{{
			Index:  1,
			Reason: "missing name and truncated args",
		}},
	})
	p.EnqueueResponse(&llm.Response{
		Provider: "fake",
		Model:    "model-a",
		Parts:    []llm.Part{llm.Text("ok")},
	})

	wrapped := llm.Wrap(p, llm.RetryDroppedToolCalls(1))
	resp, err := wrapped.Chat(context.Background(), &llm.Request{
		Model:    "model-a",
		Messages: []llm.Message{llm.UserText("call the tool")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if got := resp.Text(); got != "ok" {
		t.Fatalf("response text = %q, want ok", got)
	}

	requests := p.Requests()
	if len(requests) != 2 {
		t.Fatalf("requests len = %d, want 2", len(requests))
	}
	if len(requests[1].Messages) != 3 {
		t.Fatalf("retry messages len = %d, want original + assistant + correction", len(requests[1].Messages))
	}
	if requests[1].Messages[1].Role != llm.RoleAssistant || len(requests[1].Messages[1].Parts) != 1 {
		t.Fatalf("retry assistant message = %+v", requests[1].Messages[1])
	}
	if requests[1].Messages[2].Role != llm.RoleTool || len(requests[1].Messages[2].Parts) != 1 {
		t.Fatalf("retry correction message = %+v", requests[1].Messages[2])
	}
	result, ok := requests[1].Messages[2].Parts[0].(llm.ToolResultPart)
	if !ok || result.ToolCallID != "call_1" || result.Name != "lookup" || !result.IsError || !strings.Contains(result.Content[0].(llm.TextPart).Text, "malformed tool calls") {
		t.Fatalf("retry tool-result correction = %+v", requests[1].Messages[2].Parts[0])
	}
}

func TestRetryDroppedToolCallsReturnsPriorResponseOnRetryError(t *testing.T) {
	p := llmtest.New(llmtest.WithName("fake"))
	prior := &llm.Response{
		Provider: "fake",
		Model:    "model-a",
		Parts:    []llm.Part{llm.Text("usable"), llm.ToolCall("call_1", "lookup", []byte(`{"q":"go"}`))},
		DroppedToolCalls: []llm.DroppedToolCall{{
			Index:  2,
			Reason: "invalid tool arguments JSON",
		}},
	}
	p.EnqueueResponse(prior)
	p.EnqueueError(errors.New("retry failed"))

	wrapped := llm.Wrap(p, llm.RetryDroppedToolCalls(1))
	resp, err := wrapped.Chat(context.Background(), &llm.Request{
		Model:    "model-a",
		Messages: []llm.Message{llm.UserText("call the tool")},
	})
	// Both non-nil, like Collect's partial contract: the prior successful
	// response is salvageable AND the retry failure is observable.
	if err == nil || err.Error() != "retry failed" {
		t.Fatalf("Chat error = %v, want retry failed", err)
	}
	if resp == nil || resp.Text() != "usable" {
		t.Fatalf("response = %+v, want prior usable response", resp)
	}
}

func TestRetryDroppedToolCallsContextCancellationPropagates(t *testing.T) {
	p := llmtest.New(llmtest.WithName("fake"))
	prior := &llm.Response{
		Provider:         "fake",
		Model:            "model-a",
		Parts:            []llm.Part{llm.Text("usable")},
		DroppedToolCalls: []llm.DroppedToolCall{{Index: 0, Reason: "bad json"}},
	}
	p.EnqueueResponse(prior)

	ctx, cancel := context.WithCancel(context.Background())
	// RetryDroppedToolCalls outermost so every attempt passes through the
	// cancel middleware: the first (successful) attempt cancels ctx, the
	// retry attempt then observes the cancellation.
	wrapped := llm.Wrap(p, llm.RetryDroppedToolCalls(1), llm.Middleware{
		Chat: func(next llm.ChatFunc) llm.ChatFunc {
			return func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
				resp, err := next(ctx, req)
				cancel()
				return resp, err
			}
		},
	})
	// llmtest.Chat checks ctx.Err() first, so the retry attempt fails with
	// context.Canceled — which must propagate alongside the prior response.
	resp, err := wrapped.Chat(ctx, &llm.Request{
		Model:    "model-a",
		Messages: []llm.Message{llm.UserText("call the tool")},
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Chat error = %v, want context.Canceled", err)
	}
	if resp == nil || resp.Text() != "usable" {
		t.Fatalf("response = %+v, want prior usable response", resp)
	}
}

func TestRetryDroppedToolCallsUsageTrackerOrdering(t *testing.T) {
	p := llmtest.New(llmtest.WithName("fake"))
	p.EnqueueResponse(&llm.Response{
		Provider:         "fake",
		Model:            "model-a",
		Usage:            llm.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2},
		DroppedToolCalls: []llm.DroppedToolCall{{Index: 0, Reason: "bad json"}},
	})
	p.EnqueueResponse(&llm.Response{
		Provider: "fake",
		Model:    "model-a",
		Usage:    llm.Usage{InputTokens: 2, OutputTokens: 3, TotalTokens: 5},
		Parts:    []llm.Part{llm.Text("ok")},
	})

	tracker := llm.NewUsageTracker()
	wrapped := llm.Wrap(p, llm.RetryDroppedToolCalls(1), tracker.Middleware())
	if _, err := wrapped.Chat(context.Background(), &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("hi")}}); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}

	stats := tracker.Stats()
	if stats.Calls != 2 || stats.Usage.TotalTokens != 7 {
		t.Fatalf("stats = %+v, want 2 calls and 7 total tokens", stats)
	}
}
