package openrouter

import (
	"errors"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

// OpenRouter's explicit "402" mapping must keep winning at any status, and
// other status-less numeric codes classify through the shared fallback.
func TestDialectMapErrorStatusNumericCodes(t *testing.T) {
	cases := []struct {
		status  int
		code    string
		message string
		want    error
	}{
		{0, "402", "", llm.ErrInsufficientCredits},
		{500, "402", "", llm.ErrInsufficientCredits},
		{0, "429", "", llm.ErrRateLimited},
		{0, "429.5", "", llm.ErrServer},
	}
	for _, tc := range cases {
		if got := (dialect{}).MapErrorStatus(tc.status, tc.code, tc.message); !errors.Is(got, tc.want) {
			t.Fatalf("MapErrorStatus(%d, %q, %q) = %v, want %v", tc.status, tc.code, tc.message, got, tc.want)
		}
	}
}
