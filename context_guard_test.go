package llm_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

func guardFixture(tokens int64, models ...llm.ModelInfo) *countingProvider {
	return &countingProvider{Provider: llmtest.New(llmtest.WithName("guard"), llmtest.WithModels(models...)), tokens: tokens}
}

func okResponse() *llm.Response {
	return &llm.Response{Parts: []llm.Part{llm.Text("ok")}, StopReason: llm.StopReasonEndTurn}
}

func TestContextGuardFailsExactOverflowBeforeSending(t *testing.T) {
	p := guardFixture(1000, llm.ModelInfo{ID: "m", ContextWindow: 1000})
	guarded := llm.Wrap(p, llm.ContextGuard(llm.GuardConfig{}))
	_, err := guarded.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}})
	if !errors.Is(err, llm.ErrContextTooLong) {
		t.Fatalf("exact overflow err = %v, want ErrContextTooLong", err)
	}
	if len(p.Requests()) != 0 {
		t.Fatal("an exact overflow must fail before the provider call")
	}
	for event, err := range guarded.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}) {
		if event != nil || !errors.Is(err, llm.ErrContextTooLong) {
			t.Fatalf("stream = (%v, %v), want a single ErrContextTooLong", event, err)
		}
	}
}

func TestContextGuardEstimateNeverFails(t *testing.T) {
	p := llmtest.New(llmtest.WithName("guard"))
	p.EnqueueResponse(okResponse())
	var report llm.GuardReport
	guarded := llm.Wrap(p, llm.ContextGuard(llm.GuardConfig{
		Window: 10,
		OnNearFull: func(_ context.Context, _ *llm.Request, r llm.GuardReport) error {
			report = r
			return nil
		},
	}))
	if _, err := guarded.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText(strings.Repeat("x", 400))}}); err != nil {
		t.Fatalf("estimated overflow must reach the server, got %v", err)
	}
	if report.Exact || report.Window != 10 || report.Occupancy < 1 {
		t.Fatalf("report = %+v, want an inexact over-window estimate", report)
	}
}

func TestContextGuardNearFullHook(t *testing.T) {
	p := guardFixture(800, llm.ModelInfo{ID: "m", ContextWindow: 1000})
	p.EnqueueResponse(okResponse())
	stop := errors.New("compact first")
	calls := 0
	hook := func(_ context.Context, req *llm.Request, r llm.GuardReport) error {
		calls++
		if r != (llm.GuardReport{Tokens: 800, Exact: true, Window: 1000, Occupancy: 0.8}) || req.Model != "m" {
			t.Errorf("hook report = %+v", r)
		}
		return stop
	}
	below := llm.Wrap(p, llm.ContextGuard(llm.GuardConfig{MaxOccupancy: 0.9, OnNearFull: hook}))
	if _, err := below.Chat(context.Background(), &llm.Request{Model: "m"}); err != nil || calls != 0 {
		t.Fatalf("below threshold: err=%v hook calls=%d", err, calls)
	}
	at := llm.Wrap(p, llm.ContextGuard(llm.GuardConfig{MaxOccupancy: 0.8, OnNearFull: hook}))
	if _, err := at.Chat(context.Background(), &llm.Request{Model: "m"}); !errors.Is(err, stop) || calls != 1 {
		t.Fatalf("at threshold: err=%v hook calls=%d, want the hook's error", err, calls)
	}
}

func TestContextGuardClampMaxTokens(t *testing.T) {
	ctx := context.Background()
	p := guardFixture(900, llm.ModelInfo{ID: "m", ContextWindow: 1000, MaxOutputTokens: 64})
	guarded := llm.Wrap(p, llm.ContextGuard(llm.GuardConfig{ClampMaxTokens: true, Margin: 10}))

	original := &llm.Request{Model: "m", MaxTokens: 500}
	p.EnqueueResponse(okResponse())
	if _, err := guarded.Chat(ctx, original); err != nil {
		t.Fatal(err)
	}
	if got := p.Requests()[0].MaxTokens; got != 64 {
		t.Fatalf("clamped MaxTokens = %d, want the model's 64 output limit", got)
	}
	if original.MaxTokens != 500 {
		t.Fatal("ContextGuard mutated the caller's request")
	}

	p.EnqueueResponse(okResponse())
	if _, err := guarded.Chat(ctx, &llm.Request{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	if got := p.Requests()[1].MaxTokens; got != 0 {
		t.Fatalf("unset MaxTokens = %d, want left unset (the clamp only lowers)", got)
	}

	tight := guardFixture(950, llm.ModelInfo{ID: "m", ContextWindow: 1000, MaxOutputTokens: 64})
	tight.EnqueueResponse(okResponse())
	if _, err := llm.Wrap(tight, llm.ContextGuard(llm.GuardConfig{ClampMaxTokens: true, Margin: 10})).Chat(ctx, &llm.Request{Model: "m", MaxTokens: 64}); err != nil {
		t.Fatal(err)
	}
	if got := tight.Requests()[0].MaxTokens; got != 40 {
		t.Fatalf("window-bound MaxTokens = %d, want 1000-950-10=40", got)
	}

	stream := guardFixture(900, llm.ModelInfo{ID: "m", ContextWindow: 1000})
	stream.EnqueueStream(llm.MessageStart{}, llm.MessageEnd{StopReason: llm.StopReasonEndTurn})
	seq := llm.Wrap(stream, llm.ContextGuard(llm.GuardConfig{ClampMaxTokens: true})).ChatStream(ctx, &llm.Request{Model: "m", MaxTokens: 500})
	if stream.calls.Load() != 0 {
		t.Fatal("ChatStream must not measure before the stream is iterated")
	}
	if _, err := llm.Collect(seq); err != nil {
		t.Fatal(err)
	}
	if got := stream.Requests()[0].MaxTokens; got != 100 || stream.calls.Load() != 1 {
		t.Fatalf("stream clamp = %d after %d counts, want 100 after 1", got, stream.calls.Load())
	}
}

type flakyModelsProvider struct {
	*countingProvider
	failures atomic.Int64
}

func (p *flakyModelsProvider) Models(ctx context.Context) ([]llm.ModelInfo, error) {
	if p.failures.Add(-1) >= 0 {
		return nil, llm.ErrServer
	}
	return p.countingProvider.Models(ctx)
}

func TestContextGuardRetriesTransientModelLookup(t *testing.T) {
	p := &flakyModelsProvider{countingProvider: guardFixture(5000, llm.ModelInfo{ID: "m", ContextWindow: 1000})}
	p.failures.Store(1)
	guarded := llm.Wrap(p, llm.ContextGuard(llm.GuardConfig{}))
	p.EnqueueResponse(okResponse())
	if _, err := guarded.Chat(context.Background(), &llm.Request{Model: "m"}); err != nil {
		t.Fatalf("first call (lookup failed, window unknown) must pass through, got %v", err)
	}
	if _, err := guarded.Chat(context.Background(), &llm.Request{Model: "m"}); !errors.Is(err, llm.ErrContextTooLong) {
		t.Fatalf("second call = %v, want the lookup retried and the exact overflow caught", err)
	}
}

func TestContextGuardMaxOccupancyAboveOneMeansFull(t *testing.T) {
	p := guardFixture(950, llm.ModelInfo{ID: "m", ContextWindow: 1000})
	p.EnqueueResponse(okResponse())
	fired := false
	guarded := llm.Wrap(p, llm.ContextGuard(llm.GuardConfig{MaxOccupancy: 95, OnNearFull: func(context.Context, *llm.Request, llm.GuardReport) error {
		fired = true
		return nil
	}}))
	if _, err := guarded.Chat(context.Background(), &llm.Request{Model: "m"}); err != nil || fired {
		t.Fatalf("MaxOccupancy 95 fired=%v err=%v; values above 1 must mean only a full window", fired, err)
	}
}

func TestContextGuardCancelledCountAborts(t *testing.T) {
	p := guardFixture(0, llm.ModelInfo{ID: "m", ContextWindow: 1000})
	p.err = context.Canceled
	ctx, cancel := context.WithCancel(context.Background())
	guarded := llm.Wrap(p, llm.ContextGuard(llm.GuardConfig{Window: 1000}))
	cancel()
	if _, err := guarded.Chat(ctx, &llm.Request{Model: "m"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled count = %v, want context.Canceled", err)
	}
	if len(p.Requests()) != 0 {
		t.Fatal("a cancelled guard check must not call the provider")
	}
}

func TestContextGuardConcurrentFirstUse(t *testing.T) {
	p := guardFixture(10, llm.ModelInfo{ID: "m", ContextWindow: 1000})
	guarded := llm.Wrap(p, llm.ContextGuard(llm.GuardConfig{}))
	var wg sync.WaitGroup
	for range 32 {
		p.EnqueueResponse(okResponse())
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := guarded.Chat(context.Background(), &llm.Request{Model: "m"}); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
}

func TestContextGuardPassesThroughUnknownWindow(t *testing.T) {
	p := guardFixture(1_000_000) // no models, no table entry for provider "guard"
	p.EnqueueResponse(okResponse())
	p.EnqueueResponse(okResponse())
	guarded := llm.Wrap(p, llm.ContextGuard(llm.GuardConfig{}))
	for range 2 {
		if _, err := guarded.Chat(context.Background(), &llm.Request{Model: "m"}); err != nil {
			t.Fatalf("unknown window must pass through, got %v", err)
		}
	}
	if p.calls.Load() != 0 {
		t.Fatalf("counter called %d times with no window to compare against", p.calls.Load())
	}
}

func TestContextGuardCounterErrorFallsBackToEstimate(t *testing.T) {
	p := guardFixture(0, llm.ModelInfo{ID: "m", ContextWindow: 1000})
	p.err = llm.ErrServer
	p.EnqueueResponse(okResponse())
	var report llm.GuardReport
	guarded := llm.Wrap(p, llm.ContextGuard(llm.GuardConfig{MaxOccupancy: 0.000001, OnNearFull: func(_ context.Context, _ *llm.Request, r llm.GuardReport) error {
		report = r
		return nil
	}}))
	if _, err := guarded.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hello")}}); err != nil {
		t.Fatalf("counter failure must not fail the call, got %v", err)
	}
	if report.Exact || report.Tokens == 0 {
		t.Fatalf("report = %+v, want an inexact estimate after the counter error", report)
	}
}
