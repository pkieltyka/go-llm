package llmtest

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

// Conformance-suite tuning knobs. The suite is deliberately conservative so
// it stays reliable under -race across every provider package.
const (
	// conformanceConcurrency is the number of goroutines the concurrent_use
	// subtest runs; each performs two Chat calls and drains two streams.
	conformanceConcurrency = 4
	// conformanceStreamTimeout bounds every stream drain: a conforming
	// stream must terminate (normally or with an error) well within it.
	conformanceStreamTimeout = 30 * time.Second
	// conformanceLeakSlack tolerates ambient goroutine churn (HTTP keepalive
	// pools, test servers) when checking that a canceled stream does not
	// leak goroutines.
	conformanceLeakSlack = 5
)

// RunConformance runs the executable form of the llm.Provider contract
// against providers built by newProvider. It is the checked complement of
// the prose contract on llm.Provider: single-use streams (a second range
// yields exactly one ErrBadRequest — never a silent empty stream), context
// cancellation mid-stream (the stream terminates and does not leak
// goroutines), goroutine-safe concurrent use, panic-freedom on odd but
// valid requests, and Collect's partial-response-on-error shape over the
// provider's own event streams.
//
// newProvider is called once per subtest and must return a provider able to
// answer any number of Chat and ChatStream calls, concurrently, without
// live credentials — typically wired to an httptest fixture server (or, for
// the llmtest fake itself, a self-replenishing script). Provider packages
// run this suite against their recorded fixture servers so the contract is
// machine-checked for every provider in the module.
func RunConformance(t *testing.T, newProvider func(t *testing.T) llm.Provider) {
	t.Helper()
	if newProvider == nil {
		t.Fatal("RunConformance requires a provider factory")
	}

	t.Run("stream_single_use", func(t *testing.T) {
		p := newProvider(t)
		stream := p.ChatStream(context.Background(), conformanceRequest())

		first := drainStream(t, stream)
		if first.panicked != nil {
			t.Fatalf("first stream range panicked: %v", first.panicked)
		}
		if first.err == nil && len(first.events) == 0 {
			t.Fatalf("first stream range yielded no events and no error")
		}

		second := drainStream(t, stream)
		if second.panicked != nil {
			t.Fatalf("second stream range panicked: %v", second.panicked)
		}
		if len(second.events) != 0 {
			t.Fatalf("second range of a consumed stream yielded %d events, want none: %+v", len(second.events), second.events)
		}
		if second.err == nil {
			t.Fatalf("second range of a consumed stream ended silently, want one ErrBadRequest")
		}
		if !errors.Is(second.err, llm.ErrBadRequest) {
			t.Fatalf("second range error = %v, want ErrBadRequest", second.err)
		}
		if second.yields != 1 {
			t.Fatalf("second range yielded %d times, want exactly one error yield", second.yields)
		}
	})

	t.Run("stream_context_cancel", func(t *testing.T) {
		p := newProvider(t)
		baseline := runtime.NumGoroutine()

		// Amplify: a provider leaking one or two goroutines per canceled
		// stream would hide inside the ambient slack after a single run;
		// eight runs multiply any per-stream leak well past it.
		const cancelRuns = 8
		for run := 0; run < cancelRuns; run++ {
			ctx, cancel := context.WithCancel(context.Background())
			result := drainStreamFunc(t, func() result {
				var res result
				canceled := false
				for event, err := range p.ChatStream(ctx, conformanceRequest()) {
					res.observe(event, err)
					if !canceled {
						canceled = true
						cancel()
					}
				}
				return res
			})
			cancel()
			if result.panicked != nil {
				t.Fatalf("canceled stream panicked (run %d): %v", run, result.panicked)
			}
			if result.eventsAfterError > 0 {
				t.Fatalf("stream yielded %d events after its terminal error (run %d)", result.eventsAfterError, run)
			}
		}

		// The canceled stream must not leave goroutines behind (transport
		// readers, adapter pumps). Poll: goroutine teardown is asynchronous.
		deadline := time.Now().Add(5 * time.Second)
		for {
			current := runtime.NumGoroutine()
			if current <= baseline+conformanceLeakSlack {
				break
			}
			if time.Now().After(deadline) {
				t.Fatalf("goroutine leak after canceled stream: %d goroutines, baseline %d (slack %d)", current, baseline, conformanceLeakSlack)
			}
			time.Sleep(50 * time.Millisecond)
		}
	})

	t.Run("concurrent_use", func(t *testing.T) {
		p := newProvider(t)
		var wg sync.WaitGroup
		for i := 0; i < conformanceConcurrency; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := 0; j < 2; j++ {
					func() {
						defer func() {
							if r := recover(); r != nil {
								t.Errorf("concurrent Chat panicked: %v", r)
							}
						}()
						resp, err := p.Chat(context.Background(), conformanceRequest())
						if err == nil && resp == nil {
							t.Errorf("concurrent Chat returned nil response and nil error")
						}
					}()
					func() {
						defer func() {
							if r := recover(); r != nil {
								t.Errorf("concurrent ChatStream panicked: %v", r)
							}
						}()
						var res result
						for event, err := range p.ChatStream(context.Background(), conformanceRequest()) {
							res.observe(event, err)
						}
						if res.eventsAfterError > 0 {
							t.Errorf("concurrent stream yielded events after its terminal error")
						}
					}()
				}
			}()
		}
		wg.Wait()
	})

	t.Run("odd_requests", func(t *testing.T) {
		p := newProvider(t)
		odd := []struct {
			name string
			req  *llm.Request
		}{
			{name: "empty_request", req: &llm.Request{}},
			{name: "model_only", req: &llm.Request{Model: "conformance-model"}},
			{name: "empty_parts_message", req: &llm.Request{Model: "conformance-model", Messages: []llm.Message{{Role: llm.RoleUser}}}},
			{name: "empty_text", req: &llm.Request{Model: "conformance-model", Messages: []llm.Message{llm.UserText("")}}},
			{name: "assistant_first", req: &llm.Request{Model: "conformance-model", Messages: []llm.Message{llm.AssistantParts(llm.Text("previous")), llm.UserText("continue")}}},
			{name: "unknown_role", req: &llm.Request{Model: "conformance-model", Messages: []llm.Message{{Role: "narrator", Parts: []llm.Part{llm.Text("hi")}}}}},
			{name: "tool_result_missing_id", req: &llm.Request{Model: "conformance-model", Messages: []llm.Message{
				llm.UserText("hi"),
				llm.AssistantParts(llm.ToolCall("call_1", "lookup", []byte(`{"q":"go"}`))),
				{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResultPart{Content: []llm.Part{llm.Text("result")}}}},
			}}},
		}
		for _, tt := range odd {
			t.Run(tt.name, func(t *testing.T) {
				func() {
					defer func() {
						if r := recover(); r != nil {
							t.Errorf("Chat panicked on odd request: %v", r)
						}
					}()
					resp, err := p.Chat(context.Background(), tt.req)
					if err == nil && resp == nil {
						t.Errorf("Chat returned nil response and nil error")
					}
				}()
				func() {
					defer func() {
						if r := recover(); r != nil {
							t.Errorf("ChatStream panicked on odd request: %v", r)
						}
					}()
					var res result
					for event, err := range p.ChatStream(context.Background(), tt.req) {
						res.observe(event, err)
					}
					if res.eventsAfterError > 0 {
						t.Errorf("stream yielded events after its terminal error")
					}
				}()
			})
		}
	})

	t.Run("collect_partial_on_error", func(t *testing.T) {
		p := newProvider(t)
		injected := errors.New("llmtest conformance: injected mid-stream failure")
		source := p.ChatStream(context.Background(), conformanceRequest())

		sawStart := false
		sawContent := false
		truncated := func(yield func(llm.Event, error) bool) {
			for event, err := range source {
				if err != nil {
					// The provider failed before the injection point; the
					// partial-shape assertion below degrades gracefully.
					yield(nil, err)
					return
				}
				if _, ok := event.(llm.MessageStart); ok {
					sawStart = true
				}
				if !yield(event, nil) {
					return
				}
				switch e := event.(type) {
				case llm.TextDelta:
					if e.Text != "" {
						sawContent = true
					}
				case llm.ToolCallStart:
					sawContent = true
				}
				if sawStart || sawContent {
					yield(nil, injected)
					return
				}
			}
		}

		resp, err := llm.Collect(truncated)
		if err == nil {
			t.Fatalf("Collect over the truncated stream returned nil error")
		}
		if !errors.Is(err, injected) {
			// The provider errored on its own before any content; nothing
			// more to assert about the partial shape.
			t.Skipf("stream failed before the injection point: %v", err)
		}
		if !sawStart && !sawContent {
			t.Fatalf("stream produced neither MessageStart nor content before the injected error")
		}
		if resp == nil {
			t.Fatalf("Collect returned nil partial response for a stream that started (Collect partial-on-error contract)")
		}
	})
}

// conformanceRequest is the plain request every conformance exercise sends.
func conformanceRequest() *llm.Request {
	return &llm.Request{
		Model:     "conformance-model",
		MaxTokens: 64,
		Messages:  []llm.Message{llm.UserText("conformance ping")},
	}
}

// result accumulates one stream drain's observations.
type result struct {
	events           []llm.Event
	err              error
	yields           int
	eventsAfterError int
	panicked         any
}

func (r *result) observe(event llm.Event, err error) {
	r.yields++
	if r.err != nil {
		r.eventsAfterError++
		return
	}
	if err != nil {
		r.err = err
		return
	}
	r.events = append(r.events, event)
}

// drainStream consumes seq on a watchdog goroutine so a non-terminating
// stream fails the test instead of hanging it.
func drainStream(t *testing.T, seq func(func(llm.Event, error) bool)) result {
	t.Helper()
	return drainStreamFunc(t, func() result {
		var res result
		for event, err := range seq {
			res.observe(event, err)
		}
		return res
	})
}

func drainStreamFunc(t *testing.T, drain func() result) result {
	t.Helper()
	results := make(chan result, 1)
	go func() {
		var res result
		defer func() {
			if r := recover(); r != nil {
				res.panicked = r
			}
			results <- res
		}()
		res = drain()
	}()
	select {
	case res := <-results:
		return res
	case <-time.After(conformanceStreamTimeout):
		t.Fatalf("stream did not terminate within %s", conformanceStreamTimeout)
		return result{}
	}
}
