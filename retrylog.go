package llm

import (
	"log/slog"
	"net/http"
	"strconv"
	"strings"
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
		if duration, ok := parseRetryDuration(value, time.Millisecond); ok {
			return duration
		}
	}
	value := resp.Header.Get("Retry-After")
	if value == "" {
		return 0
	}
	if duration, ok := parseRetryDuration(value, time.Second); ok {
		return duration
	}
	when, err := http.ParseTime(value)
	if err != nil {
		return 0
	}
	duration := time.Until(when)
	if duration < 0 {
		return 0
	}
	return duration
}

func parseRetryDuration(value string, unit time.Duration) (time.Duration, bool) {
	if value == "" || value[0] == '-' {
		return 0, false
	}
	if value[0] == '+' {
		value = value[1:]
		if value == "" {
			return 0, false
		}
	}

	mantissa := value
	var exponent int64
	if index := strings.IndexAny(value, "eE"); index >= 0 {
		mantissa = value[:index]
		var err error
		exponent, err = strconv.ParseInt(value[index+1:], 10, 64)
		if err != nil {
			return 0, false
		}
	}

	digits := make([]byte, 0, len(mantissa))
	decimalSeen := false
	var fractionalDigits int64
	for i := 0; i < len(mantissa); i++ {
		character := mantissa[i]
		switch {
		case character == '.' && !decimalSeen:
			decimalSeen = true
		case character >= '0' && character <= '9':
			digits = append(digits, character)
			if decimalSeen {
				fractionalDigits++
			}
		default:
			return 0, false
		}
	}
	if len(digits) == 0 {
		return 0, false
	}
	firstNonZero := 0
	for firstNonZero < len(digits) && digits[firstNonZero] == '0' {
		firstNonZero++
	}
	if firstNonZero == len(digits) {
		return 0, true
	}
	digits = digits[firstNonZero:]

	var unitDigits int64
	switch unit {
	case time.Second:
		unitDigits = 9
	case time.Millisecond:
		unitDigits = 6
	default:
		return 0, false
	}
	baseIntegerDigits := int64(len(digits)) - fractionalDigits + unitDigits
	if exponent > 19-baseIntegerDigits {
		return 0, false
	}
	if exponent <= -baseIntegerDigits {
		return 0, true
	}
	integerDigits := baseIntegerDigits + exponent
	shift := exponent - fractionalDigits + unitDigits

	integer := append([]byte(nil), digits...)
	fractionalNonZero := false
	if shift >= 0 {
		for range int(shift) {
			integer = append(integer, '0')
		}
	} else {
		cut := len(digits) + int(shift)
		for _, digit := range digits[cut:] {
			if digit != '0' {
				fractionalNonZero = true
				break
			}
		}
		integer = integer[:cut]
	}
	if int64(len(integer)) != integerDigits {
		return 0, false
	}
	nanoseconds, err := strconv.ParseUint(string(integer), 10, 64)
	if err != nil {
		return 0, false
	}
	const maxDurationNanoseconds = uint64(1<<63 - 1)
	if nanoseconds > maxDurationNanoseconds || nanoseconds == maxDurationNanoseconds && fractionalNonZero {
		return 0, false
	}
	return time.Duration(nanoseconds), true
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
