package llmtest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
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
	// conformanceCancelTimeout bounds the cancellation-specific assertion.
	// The fixture does not release its stream until the request context ends.
	conformanceCancelTimeout = 5 * time.Second
	// conformanceBlockProbe proves the cancellation fixture remains blocked
	// after its first event without materially slowing the provider suites.
	conformanceBlockProbe = 20 * time.Millisecond
	// conformanceLeakSlack tolerates ambient goroutine churn (HTTP keepalive
	// pools, test servers) when checking that a canceled stream does not
	// leak goroutines.
	conformanceLeakSlack = 5
)

// ConformanceScenario identifies the deterministic fixture behavior requested
// by RunConformance. Provider fixture handlers should inspect each request with
// ConformanceScenarioFromRequest and implement all four scenarios.
type ConformanceScenario string

const (
	// ConformanceSuccess requests a complete successful response or stream.
	ConformanceSuccess ConformanceScenario = "success"
	// ConformanceCancel requests a stream that emits MessageStart, flushes it,
	// then blocks until the request context is canceled.
	ConformanceCancel ConformanceScenario = "cancel"
	// ConformanceEmpty requests a successful HTTP response with an empty body.
	ConformanceEmpty ConformanceScenario = "empty"
	// ConformanceTruncated requests only the provider's start event followed by
	// EOF, without a terminal event.
	ConformanceTruncated ConformanceScenario = "truncated"
)

const (
	conformanceSuccessModel   = "llmtest-conformance-success"
	conformanceCancelModel    = "llmtest-conformance-cancel"
	conformanceEmptyModel     = "llmtest-conformance-empty"
	conformanceTruncatedModel = "llmtest-conformance-truncated"
)

// ConformanceScenarioForModel maps RunConformance's model sentinels to their
// fixture scenario. Unknown models use the normal success behavior so the
// suite's odd-request checks can share the same fixture.
func ConformanceScenarioForModel(model string) ConformanceScenario {
	switch model {
	case conformanceCancelModel:
		return ConformanceCancel
	case conformanceEmptyModel:
		return ConformanceEmpty
	case conformanceTruncatedModel:
		return ConformanceTruncated
	default:
		return ConformanceSuccess
	}
}

// ConformanceScenarioFromRequest reads the JSON model field used by
// RunConformance and restores req.Body for the provider fixture or SDK.
func ConformanceScenarioFromRequest(req *http.Request) ConformanceScenario {
	if req == nil || req.Body == nil {
		return ConformanceSuccess
	}
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return ConformanceSuccess
	}
	req.Body = io.NopCloser(bytes.NewReader(body))
	var payload struct {
		Model string `json:"model"`
	}
	if json.Unmarshal(body, &payload) != nil {
		return ConformanceSuccess
	}
	return ConformanceScenarioForModel(payload.Model)
}

// RunConformance runs the executable form of the llm.Provider contract
// against providers built by newProvider. It is the checked complement of
// the prose contract on llm.Provider: single-use streams (a second range
// yields exactly one ErrBadRequest — never a silent empty stream), context
// cancellation mid-stream (the stream terminates with context.Canceled and
// does not leak goroutines), successful event grammar, empty/truncated EOF
// normalization, goroutine-safe independent concurrent streams,
// panic-freedom on odd but valid requests, early-break behavior, and
// Collect's partial-response-on-error shape over the provider's own events.
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
		stream := p.ChatStream(context.Background(), conformanceRequest(ConformanceSuccess))

		first := drainStream(t, stream)
		if err := successfulStreamError(first); err != nil {
			t.Fatalf("first stream range: %v", err)
		}

		second := drainStream(t, stream)
		if err := consumedStreamError(second); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("stream_early_break_is_consumed", func(t *testing.T) {
		p := newProvider(t)
		stream := p.ChatStream(context.Background(), conformanceRequest(ConformanceSuccess))
		var first llm.Event
		for event, err := range stream {
			if err != nil {
				t.Fatalf("first range yielded error before early break: %v", err)
			}
			first = event
			break
		}
		if _, ok := first.(llm.MessageStart); !ok {
			t.Fatalf("first event before early break = %T, want llm.MessageStart", first)
		}
		if err := consumedStreamError(drainStream(t, stream)); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("stream_context_cancel", func(t *testing.T) {
		p := newProvider(t)
		baseline := runtime.NumGoroutine()

		// More runs than the global-count slack make a one-goroutine-per-cancel
		// leak visible. cancellationProbe also directly requires each iterator
		// goroutine to terminate, avoiding reliance on this count alone.
		const cancelRuns = conformanceLeakSlack + 3
		for run := 0; run < cancelRuns; run++ {
			ctx, cancel := context.WithCancel(context.Background())
			_, err := cancellationProbe(
				p.ChatStream(ctx, conformanceRequest(ConformanceCancel)),
				cancel,
				conformanceBlockProbe,
				conformanceCancelTimeout,
			)
			if err != nil {
				t.Fatalf("canceled stream failed conformance (run %d): %v", run, err)
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

	for _, tt := range []struct {
		name           string
		scenario       ConformanceScenario
		wantEventCount int
	}{
		{name: "stream_empty_eof", scenario: ConformanceEmpty},
		{name: "stream_truncated_eof", scenario: ConformanceTruncated, wantEventCount: 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p := newProvider(t)
			res := drainStream(t, p.ChatStream(context.Background(), conformanceRequest(tt.scenario)))
			if res.panicked != nil {
				t.Fatalf("stream panicked: %v", res.panicked)
			}
			if !errors.Is(res.err, llm.ErrServer) {
				t.Fatalf("stream error = %v, want ErrServer", res.err)
			}
			if res.eventWithError != 0 {
				t.Fatalf("stream yielded %d event+error pairs, want none", res.eventWithError)
			}
			if res.nilEvents != 0 {
				t.Fatalf("stream yielded %d nil events without errors, want none", res.nilEvents)
			}
			if len(res.events) != tt.wantEventCount {
				t.Fatalf("stream events = %d, want %d: %+v", len(res.events), tt.wantEventCount, res.events)
			}
			if tt.wantEventCount == 1 {
				if _, ok := res.events[0].(llm.MessageStart); !ok {
					t.Fatalf("truncated stream first event = %T, want llm.MessageStart", res.events[0])
				}
			}
			if res.eventsAfterError != 0 {
				t.Fatalf("stream yielded %d events after its terminal error", res.eventsAfterError)
			}
		})
	}

	t.Run("concurrent_use", func(t *testing.T) {
		p := newProvider(t)
		var wg sync.WaitGroup
		errs := make(chan error, conformanceConcurrency*4)
		for i := 0; i < conformanceConcurrency; i++ {
			wg.Add(1)
			go func(worker int) {
				defer wg.Done()
				for j := 0; j < 2; j++ {
					ctx, cancel := context.WithTimeout(context.Background(), conformanceStreamTimeout)
					resp, err := p.Chat(ctx, conformanceRequest(ConformanceSuccess))
					cancel()
					if err != nil {
						errs <- fmt.Errorf("worker %d Chat %d: %w", worker, j, err)
					} else if resp == nil {
						errs <- fmt.Errorf("worker %d Chat %d returned nil response", worker, j)
					}

					ctx, cancel = context.WithTimeout(context.Background(), conformanceStreamTimeout)
					res := drainStreamWithoutWatchdog(p.ChatStream(ctx, conformanceRequest(ConformanceSuccess)))
					cancel()
					if err := successfulStreamError(res); err != nil {
						errs <- fmt.Errorf("worker %d stream %d: %w", worker, j, err)
					}
				}
			}(i)
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			t.Error(err)
		}
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
		source := p.ChatStream(context.Background(), conformanceRequest(ConformanceSuccess))

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
func conformanceRequest(scenario ConformanceScenario) *llm.Request {
	model := conformanceSuccessModel
	switch scenario {
	case ConformanceCancel:
		model = conformanceCancelModel
	case ConformanceEmpty:
		model = conformanceEmptyModel
	case ConformanceTruncated:
		model = conformanceTruncatedModel
	}
	return &llm.Request{
		Model:     model,
		MaxTokens: 64,
		Messages:  []llm.Message{llm.UserText("conformance ping")},
	}
}

func successfulStreamError(res result) error {
	if res.panicked != nil {
		return fmt.Errorf("panicked: %v", res.panicked)
	}
	if res.eventWithError != 0 {
		return fmt.Errorf("yielded %d event+error pairs", res.eventWithError)
	}
	if res.nilEvents != 0 {
		return fmt.Errorf("yielded %d nil events without errors", res.nilEvents)
	}
	if res.eventsAfterError != 0 {
		return fmt.Errorf("yielded %d pairs after terminal error", res.eventsAfterError)
	}
	if res.err != nil {
		return fmt.Errorf("returned error: %w", res.err)
	}
	if len(res.events) == 0 {
		return errors.New("yielded no events")
	}
	startCount := 0
	endCount := 0
	ended := false
	for i, event := range res.events {
		if _, ok := event.(llm.MessageStart); ok {
			startCount++
			if startCount > 1 {
				return fmt.Errorf("MessageStart count = %d at event %d, want exactly one", startCount, i)
			}
		}
		if _, ok := event.(llm.MessageEnd); ok {
			endCount++
			if endCount > 1 {
				return fmt.Errorf("MessageEnd count = %d at event %d, want exactly one", endCount, i)
			}
		}
		if ended {
			return fmt.Errorf("event %d (%T) yielded after MessageEnd", i, event)
		}
		if _, ok := event.(llm.MessageEnd); ok {
			ended = true
		}
	}
	if _, ok := res.events[0].(llm.MessageStart); !ok {
		return fmt.Errorf("first event = %T, want llm.MessageStart", res.events[0])
	}
	if startCount != 1 {
		return fmt.Errorf("MessageStart count = %d, want exactly one", startCount)
	}
	if endCount != 1 {
		return fmt.Errorf("MessageEnd count = %d, want exactly one", endCount)
	}
	return nil
}

func consumedStreamError(res result) error {
	if res.panicked != nil {
		return fmt.Errorf("second stream range panicked: %v", res.panicked)
	}
	if len(res.events) != 0 {
		return fmt.Errorf("second range yielded %d events, want none: %+v", len(res.events), res.events)
	}
	if res.eventWithError != 0 {
		return errors.New("second range yielded an event paired with its error")
	}
	if res.nilEvents != 0 || res.eventsAfterError != 0 {
		return errors.New("second range yielded malformed extra pairs")
	}
	if !errors.Is(res.err, llm.ErrBadRequest) {
		return fmt.Errorf("second range error = %v, want ErrBadRequest", res.err)
	}
	if res.yields != 1 {
		return fmt.Errorf("second range yielded %d times, want exactly one error yield", res.yields)
	}
	return nil
}

// result accumulates one stream drain's observations.
type result struct {
	events           []llm.Event
	err              error
	yields           int
	eventsAfterError int
	eventWithError   int
	nilEvents        int
	panicked         any
	sawError         bool
}

func (r *result) observe(event llm.Event, err error) {
	r.yields++
	if r.sawError {
		r.eventsAfterError++
		return
	}
	if err != nil {
		r.sawError = true
		r.err = err
		if event != nil {
			r.eventWithError++
		}
		return
	}
	if isNilEvent(event) {
		r.nilEvents++
		return
	}
	r.events = append(r.events, event)
}

func isNilEvent(event llm.Event) bool {
	if event == nil {
		return true
	}
	value := reflect.ValueOf(event)
	return value.Kind() == reflect.Pointer && value.IsNil()
}

type yieldedPair struct {
	event llm.Event
	err   error
}

// cancellationProbe proves the stream remains blocked after its first start
// event, then cancels and directly waits for the iterator goroutine to exit.
func cancellationProbe(
	seq func(func(llm.Event, error) bool),
	cancel context.CancelFunc,
	blockFor time.Duration,
	timeout time.Duration,
) (result, error) {
	if seq == nil {
		cancel()
		return result{}, errors.New("nil cancellation stream")
	}
	defer cancel()

	first := make(chan yieldedPair, 1)
	extra := make(chan yieldedPair, 1)
	done := make(chan result, 1)
	go func() {
		var res result
		defer func() {
			if recovered := recover(); recovered != nil {
				res.panicked = recovered
			}
			done <- res
		}()
		count := 0
		for event, err := range seq {
			count++
			pair := yieldedPair{event: event, err: err}
			switch count {
			case 1:
				first <- pair
			case 2:
				extra <- pair
			}
			res.observe(event, err)
		}
	}()

	select {
	case pair := <-first:
		if pair.err != nil || pair.event == nil {
			cancel()
			res, waitErr := waitCancellationResult(done, timeout)
			if waitErr != nil {
				return res, waitErr
			}
			return res, fmt.Errorf("first cancellation pair = (%T, %v), want (MessageStart, nil)", pair.event, pair.err)
		}
		if _, ok := pair.event.(llm.MessageStart); !ok {
			cancel()
			res, waitErr := waitCancellationResult(done, timeout)
			if waitErr != nil {
				return res, waitErr
			}
			return res, fmt.Errorf("first cancellation event = %T, want MessageStart", pair.event)
		}
	case res := <-done:
		return res, errors.New("cancellation stream terminated before its first event")
	case <-time.After(timeout):
		cancel()
		res, waitErr := waitCancellationResult(done, timeout)
		if waitErr != nil {
			return res, fmt.Errorf("cancellation stream did not yield its first event and %w", waitErr)
		}
		return res, errors.New("cancellation stream did not yield its first event promptly")
	}

	select {
	case pair := <-extra:
		cancel()
		res, waitErr := waitCancellationResult(done, timeout)
		if waitErr != nil {
			return res, waitErr
		}
		return res, fmt.Errorf("cancellation fixture yielded (%T, %v) before cancellation instead of blocking", pair.event, pair.err)
	case res := <-done:
		return res, errors.New("cancellation fixture terminated before cancellation instead of blocking")
	case <-time.After(blockFor):
	}

	cancel()
	res, err := waitCancellationResult(done, timeout)
	if err != nil {
		return res, err
	}
	if err := canceledStreamError(res); err != nil {
		return res, err
	}
	return res, nil
}

func waitCancellationResult(done <-chan result, timeout time.Duration) (result, error) {
	select {
	case res := <-done:
		return res, nil
	case <-time.After(timeout):
		return result{}, errors.New("canceled stream iterator did not terminate")
	}
}

func canceledStreamError(res result) error {
	if res.panicked != nil {
		return fmt.Errorf("canceled stream panicked: %v", res.panicked)
	}
	if res.eventWithError != 0 {
		return fmt.Errorf("canceled stream yielded %d event+error pairs", res.eventWithError)
	}
	if res.nilEvents != 0 {
		return fmt.Errorf("canceled stream yielded %d nil events without errors", res.nilEvents)
	}
	if res.eventsAfterError != 0 {
		return fmt.Errorf("canceled stream yielded %d pairs after its terminal error", res.eventsAfterError)
	}
	if len(res.events) != 1 {
		return fmt.Errorf("canceled stream events = %d, want one MessageStart: %+v", len(res.events), res.events)
	}
	if _, ok := res.events[0].(llm.MessageStart); !ok {
		return fmt.Errorf("canceled stream first event = %T, want MessageStart", res.events[0])
	}
	if !errors.Is(res.err, context.Canceled) {
		return fmt.Errorf("canceled stream error = %v, want context.Canceled", res.err)
	}
	if res.yields != 2 {
		return fmt.Errorf("canceled stream yielded %d pairs, want start then cancellation", res.yields)
	}
	return nil
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
	return drainStreamFuncWithin(t, conformanceStreamTimeout, drain)
}

func drainStreamFuncWithin(t *testing.T, timeout time.Duration, drain func() result) result {
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
	case <-time.After(timeout):
		t.Fatalf("stream did not terminate within %s", timeout)
		return result{}
	}
}

func drainStreamWithoutWatchdog(seq func(func(llm.Event, error) bool)) (res result) {
	defer func() {
		if r := recover(); r != nil {
			res.panicked = r
		}
	}()
	for event, err := range seq {
		res.observe(event, err)
	}
	return res
}
