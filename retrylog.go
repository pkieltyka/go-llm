package llm

import (
	"log/slog"
	"net/http"
	"strconv"
	"time"
)

// NewRetryLogger wraps next and logs retryable provider responses. It observes
// SDK-managed retries at the same transport boundary as NewWireTap.
func NewRetryLogger(next http.RoundTripper, provider string, logger *slog.Logger) http.RoundTripper {
	if next == nil {
		next = http.DefaultTransport
	}
	if logger == nil {
		return next
	}
	return &retryLogTransport{next: next, provider: provider, logger: logger}
}

type retryLogTransport struct {
	next     http.RoundTripper
	provider string
	logger   *slog.Logger
}

func (t *retryLogTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.next.RoundTrip(req)
	if resp != nil && retryLogStatus(resp.StatusCode) {
		t.logger.WarnContext(req.Context(), "llm provider retryable response",
			"provider", t.provider,
			"status", resp.StatusCode,
			"retry_after", retryAfterHeader(resp),
			"retry_after_duration", RetryAfter(resp),
			"attempt", retryAttemptOrdinal(req),
		)
	}
	return resp, err
}

func retryLogStatus(status int) bool {
	switch status {
	case http.StatusTooManyRequests, http.StatusServiceUnavailable, 529:
		return true
	default:
		return false
	}
}

func retryAfterHeader(resp *http.Response) string {
	if resp == nil {
		return ""
	}
	if value := resp.Header.Get("Retry-After-Ms"); value != "" {
		return value
	}
	return resp.Header.Get("Retry-After")
}

// RetryAfter parses provider retry-after headers into a duration.
func RetryAfter(resp *http.Response) time.Duration {
	if resp == nil {
		return 0
	}
	if value := resp.Header.Get("Retry-After-Ms"); value != "" {
		if ms, err := strconv.ParseFloat(value, 64); err == nil {
			return time.Duration(ms * float64(time.Millisecond))
		}
	}
	value := resp.Header.Get("Retry-After")
	if value == "" {
		return 0
	}
	if seconds, err := strconv.ParseFloat(value, 64); err == nil {
		return time.Duration(seconds * float64(time.Second))
	}
	when, err := http.ParseTime(value)
	if err != nil {
		return 0
	}
	return time.Until(when)
}

func retryAttemptOrdinal(req *http.Request) int {
	if req == nil {
		return 1
	}
	count, err := strconv.Atoi(req.Header.Get("X-Stainless-Retry-Count"))
	if err != nil {
		return 1
	}
	return count + 1
}
