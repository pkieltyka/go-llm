package llm

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestProviderErrorWrapping(t *testing.T) {
	err := &ProviderError{
		Provider:   "openrouter",
		HTTPStatus: 429,
		Code:       "rate_limited",
		Message:    "slow down",
		RetryAfter: 2 * time.Second,
		Kind:       ErrRateLimited,
	}

	if !errors.Is(err, ErrRateLimited) {
		t.Fatalf("errors.Is(err, ErrRateLimited) = false")
	}

	var providerErr *ProviderError
	if !errors.As(err, &providerErr) {
		t.Fatalf("errors.As(err, *ProviderError) = false")
	}
	if providerErr.Provider != "openrouter" || providerErr.RetryAfter != 2*time.Second {
		t.Fatalf("provider error = %+v", providerErr)
	}
	if got := err.Error(); !strings.Contains(got, "llm/openrouter") || !strings.Contains(got, "429") || !strings.Contains(got, "rate_limited") {
		t.Fatalf("Error() = %q, want provider, status, and code", got)
	}
}
