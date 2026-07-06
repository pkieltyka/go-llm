package chatcompletions

import (
	"strings"
	"testing"
	"time"
)

func TestStreamRetryBackoffIsBoundedAndIncreasing(t *testing.T) {
	first := streamRetryBackoff(1)
	second := streamRetryBackoff(2)
	later := streamRetryBackoff(99)
	if first <= 0 {
		t.Fatalf("first backoff = %s, want positive", first)
	}
	if second <= first {
		t.Fatalf("second backoff = %s, want > first %s", second, first)
	}
	if later != 2*time.Second {
		t.Fatalf("capped backoff = %s, want 2s", later)
	}
}

func TestWithStreamEnabledPreservesGiantIntegers(t *testing.T) {
	// Splicing "stream": true must not route numbers through float64 —
	// integers beyond 2^53 have to survive verbatim.
	body := []byte(`{"model":"m","seed":9007199254740993,"max_tokens":123}`)
	out, err := withStreamEnabled(body)
	if err != nil {
		t.Fatalf("withStreamEnabled returned error: %v", err)
	}
	if !strings.Contains(string(out), `9007199254740993`) {
		t.Fatalf("giant integer mangled: %s", out)
	}
	if !strings.Contains(string(out), `"stream":true`) {
		t.Fatalf("stream flag missing: %s", out)
	}
}
