package e2e

import (
	"context"
	"fmt"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

const promptCacheProbeAttempts = 3

type promptCacheProbePolicy struct {
	Attempts int
	Backoff  func(int) time.Duration
	Wait     func(context.Context, time.Duration) error
}

func defaultPromptCacheProbePolicy() promptCacheProbePolicy {
	return promptCacheProbePolicy{
		Attempts: promptCacheProbeAttempts,
		Backoff: func(retry int) time.Duration {
			return time.Duration(retry) * time.Second
		},
		Wait: waitPromptCacheBackoff,
	}
}

// probePromptCache performs one warm-up call followed by a bounded number of
// evidence probes. Advertising prompt-caching is considered verified only
// when a response reports a positive cache-read token count.
func probePromptCache(ctx context.Context, providerID string, policy promptCacheProbePolicy, call func(context.Context) (*llm.Response, error)) (*llm.Response, *llm.Response, error) {
	if policy.Attempts <= 0 || policy.Backoff == nil || policy.Wait == nil || call == nil {
		return nil, nil, fmt.Errorf("invalid %s prompt-cache probe configuration", providerID)
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	warmup, err := call(ctx)
	if err != nil {
		return nil, nil, fmt.Errorf("%s prompt-cache warm-up: %w", providerID, err)
	}
	if warmup == nil {
		return nil, nil, fmt.Errorf("%s prompt-cache warm-up returned a nil response", providerID)
	}

	var last *llm.Response
	for attempt := 1; attempt <= policy.Attempts; attempt++ {
		if attempt > 1 {
			if err := policy.Wait(ctx, policy.Backoff(attempt-1)); err != nil {
				return warmup, last, fmt.Errorf("%s prompt-cache backoff: %w", providerID, err)
			}
		}
		if err := ctx.Err(); err != nil {
			return warmup, last, err
		}
		last, err = call(ctx)
		if err != nil {
			return warmup, last, fmt.Errorf("%s prompt-cache probe %d/%d: %w", providerID, attempt, policy.Attempts, err)
		}
		if last == nil {
			return warmup, nil, fmt.Errorf("%s prompt-cache probe %d/%d returned a nil response", providerID, attempt, policy.Attempts)
		}
		if last.Usage.CacheReadTokens > 0 {
			return warmup, last, nil
		}
	}
	return warmup, last, fmt.Errorf("%s prompt-cache probes reported zero CacheReadTokens after %d attempts", providerID, policy.Attempts)
}

func waitPromptCacheBackoff(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
