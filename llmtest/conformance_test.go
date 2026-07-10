package llmtest_test

import (
	"context"
	"iter"
	"sync"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestRunConformance runs the exported conformance suite against llmtest's
// own Provider — self-conformance for the reference llm.Provider
// implementation. The scripted fake is made "unlimited" by a middleware
// that atomically enqueues one step per incoming call, so the suite can
// issue any number of Chat/ChatStream calls in any interleaving.
func TestRunConformance(t *testing.T) {
	llmtest.RunConformance(t, func(t *testing.T) llm.Provider {
		p := llmtest.New(
			llmtest.WithName("llmtest-conformance"),
			llmtest.WithCapabilities(llm.CapabilityStreaming, llm.CapabilityTools),
		)
		var mu sync.Mutex
		return llm.Wrap(p, llm.Middleware{
			Chat: func(next llm.ChatFunc) llm.ChatFunc {
				return func(ctx context.Context, req *llm.Request) (*llm.Response, error) {
					mu.Lock()
					defer mu.Unlock()
					p.EnqueueResponse(&llm.Response{
						ID:         "msg_conformance",
						Provider:   "llmtest-conformance",
						Model:      req.Model,
						Parts:      []llm.Part{llm.Text("pong")},
						StopReason: llm.StopReasonEndTurn,
						Usage:      llm.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2},
					})
					return next(ctx, req)
				}
			},
			Stream: func(next llm.StreamFunc) llm.StreamFunc {
				return func(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
					switch llmtest.ConformanceScenarioForModel(req.Model) {
					case llmtest.ConformanceCancel:
						return func(yield func(llm.Event, error) bool) {
							if !yield(llm.MessageStart{ID: "msg_conformance", Provider: "llmtest-conformance", Model: req.Model}, nil) {
								return
							}
							<-ctx.Done()
							yield(nil, ctx.Err())
						}
					case llmtest.ConformanceEmpty:
						return func(yield func(llm.Event, error) bool) {
							yield(nil, llm.ErrServer)
						}
					case llmtest.ConformanceTruncated:
						return func(yield func(llm.Event, error) bool) {
							if yield(llm.MessageStart{ID: "msg_conformance", Provider: "llmtest-conformance", Model: req.Model}, nil) {
								yield(nil, llm.ErrServer)
							}
						}
					}
					mu.Lock()
					defer mu.Unlock()
					// llmtest pops the scripted step when ChatStream is
					// invoked (inside next), so enqueue+pop stay atomic
					// under the mutex even with concurrent callers.
					p.EnqueueStream(
						llm.MessageStart{ID: "msg_conformance", Provider: "llmtest-conformance", Model: req.Model},
						llm.TextDelta{Index: 0, Text: "po"},
						llm.TextDelta{Index: 0, Text: "ng"},
						llm.MessageEnd{StopReason: llm.StopReasonEndTurn, StopReasonRaw: "end_turn", Usage: llm.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2}},
					)
					return next(ctx, req)
				}
			},
		})
	})
}
