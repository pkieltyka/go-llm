package llm

import (
	"net/http"
	"testing"
	"time"
)

func TestRetryAfterRejectsNegativeAndPastValues(t *testing.T) {
	tests := []struct {
		name   string
		header http.Header
	}{
		{name: "negative seconds", header: http.Header{"Retry-After": []string{"-1"}}},
		{name: "negative milliseconds", header: http.Header{"Retry-After-Ms": []string{"-0.5"}}},
		{name: "past date", header: http.Header{"Retry-After": []string{time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC).Format(http.TimeFormat)}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := RetryAfter(&http.Response{Header: tt.header}); got != 0 {
				t.Fatalf("RetryAfter = %s, want 0", got)
			}
		})
	}
}

func TestRetryAfterParsesPositiveNumericValues(t *testing.T) {
	if got, ok := parseRetryDuration("1.5", time.Second); !ok || got != 1500*time.Millisecond {
		t.Fatalf("parseRetryDuration = %s, %v; want 1.5s, true", got, ok)
	}
	if got := RetryAfter(&http.Response{Header: http.Header{"Retry-After": []string{"1.5"}}}); got != 1500*time.Millisecond {
		t.Fatalf("seconds RetryAfter = %s, want 1.5s", got)
	}
	if got := RetryAfter(&http.Response{Header: http.Header{"Retry-After-Ms": []string{"250"}}}); got != 250*time.Millisecond {
		t.Fatalf("milliseconds RetryAfter = %s, want 250ms", got)
	}
	if got := RetryAfter(&http.Response{Header: http.Header{"Retry-After": []string{"1e-3"}}}); got != time.Millisecond {
		t.Fatalf("exponent RetryAfter = %s, want 1ms", got)
	}
}

func TestRetryAfterRejectsDurationOverflowExactly(t *testing.T) {
	tests := []struct {
		name  string
		value string
		want  time.Duration
	}{
		{name: "exact maximum", value: "9223372036.854775807", want: time.Duration(1<<63 - 1)},
		{name: "maximum with insignificant zero", value: "9223372036.8547758070", want: time.Duration(1<<63 - 1)},
		{name: "review rounded overflow", value: "9223372036.854776"},
		{name: "one nanosecond overflow", value: "9223372036.854775808"},
		{name: "sub-nanosecond overflow", value: "9223372036.8547758071"},
		{name: "below maximum truncates", value: "9223372036.8547758069", want: time.Duration(1<<63 - 2)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := RetryAfter(&http.Response{Header: http.Header{"Retry-After": []string{tt.value}}})
			if got != tt.want {
				t.Fatalf("RetryAfter(%q) = %d, want %d", tt.value, got, tt.want)
			}
			if got < 0 {
				t.Fatalf("RetryAfter(%q) returned negative duration %d", tt.value, got)
			}
		})
	}
}
