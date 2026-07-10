package e2e

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

func TestAnthropicPromptCacheProbeRequiresObservedRead(t *testing.T) {
	responses := []*llm.Response{
		{Usage: llm.Usage{InputTokens: 100}},
		{Usage: llm.Usage{InputTokens: 10, CacheReadTokens: 90}},
	}
	calls := 0
	warmup, hit, err := probePromptCache(context.Background(), "anthropic", noDelayPromptCachePolicy(3, nil), func(context.Context) (*llm.Response, error) {
		response := responses[calls]
		calls++
		return response, nil
	})
	if err != nil {
		t.Fatalf("probePromptCache returned error: %v", err)
	}
	if warmup != responses[0] || hit != responses[1] || calls != 2 || hit.Usage.CacheReadTokens != 90 {
		t.Fatalf("warmup=%+v hit=%+v calls=%d", warmup, hit, calls)
	}
}

func TestPromptCacheProbeAcceptsWarmupCacheHit(t *testing.T) {
	warmup := &llm.Response{Usage: llm.Usage{InputTokens: 10, CacheReadTokens: 90}}
	calls := 0
	first, hit, err := probePromptCache(context.Background(), "openai-codex", noDelayPromptCachePolicy(3, nil), func(context.Context) (*llm.Response, error) {
		calls++
		return warmup, nil
	})
	if err != nil {
		t.Fatalf("probePromptCache returned error: %v", err)
	}
	if first != warmup || hit != warmup || calls != 1 {
		t.Fatalf("first=%+v hit=%+v calls=%d", first, hit, calls)
	}
}

func TestCodexPromptCacheProbeUsesBoundedDeterministicRetries(t *testing.T) {
	reads := []int64{0, 0, 0, 7} // warm-up followed by three probes.
	calls := 0
	var waits []time.Duration
	policy := noDelayPromptCachePolicy(3, &waits)
	_, hit, err := probePromptCache(context.Background(), "openai-codex", policy, func(context.Context) (*llm.Response, error) {
		response := &llm.Response{Usage: llm.Usage{InputTokens: 100, CacheReadTokens: reads[calls]}}
		calls++
		return response, nil
	})
	if err != nil {
		t.Fatalf("probePromptCache returned error: %v", err)
	}
	if calls != 4 || hit.Usage.CacheReadTokens != 7 {
		t.Fatalf("calls=%d hit=%+v", calls, hit)
	}
	wantWaits := []time.Duration{time.Second, 2 * time.Second}
	if len(waits) != len(wantWaits) || waits[0] != wantWaits[0] || waits[1] != wantWaits[1] {
		t.Fatalf("waits=%v, want %v", waits, wantWaits)
	}
}

func TestPromptCacheProbeFailsAfterBound(t *testing.T) {
	calls := 0
	_, _, err := probePromptCache(context.Background(), "openai-codex", noDelayPromptCachePolicy(3, nil), func(context.Context) (*llm.Response, error) {
		calls++
		return &llm.Response{Usage: llm.Usage{InputTokens: 100}}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "zero CacheReadTokens after 3 attempts") {
		t.Fatalf("error = %v", err)
	}
	if calls != 4 {
		t.Fatalf("calls = %d, want warm-up plus three probes", calls)
	}
}

func TestPromptCacheProbeBackoffHonorsContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	policy := defaultPromptCacheProbePolicy()
	policy.Backoff = func(int) time.Duration { return time.Hour }
	policy.Wait = func(ctx context.Context, _ time.Duration) error {
		cancel()
		return waitPromptCacheBackoff(ctx, time.Hour)
	}
	_, _, err := probePromptCache(ctx, "anthropic", policy, func(context.Context) (*llm.Response, error) {
		calls++
		return &llm.Response{Usage: llm.Usage{InputTokens: 100}}, nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if calls != 2 {
		t.Fatalf("calls = %d, want warm-up and first probe", calls)
	}
}

func noDelayPromptCachePolicy(attempts int, waits *[]time.Duration) promptCacheProbePolicy {
	return promptCacheProbePolicy{
		Attempts: attempts,
		Backoff: func(retry int) time.Duration {
			return time.Duration(retry) * time.Second
		},
		Wait: func(_ context.Context, delay time.Duration) error {
			if waits != nil {
				*waits = append(*waits, delay)
			}
			return nil
		},
	}
}
