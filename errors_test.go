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

func TestProviderErrorStringSkipsCodeEqualToStatus(t *testing.T) {
	dup := &ProviderError{Provider: "openrouter", HTTPStatus: 400, Code: "400", Message: "no such model", Kind: ErrBadRequest}
	if got, want := dup.Error(), "llm/openrouter: 400: no such model"; got != want {
		t.Fatalf("duplicated status not collapsed: got %q, want %q", got, want)
	}
	distinct := &ProviderError{Provider: "openai", HTTPStatus: 400, Code: "invalid_request_error", Message: "bad", Kind: ErrBadRequest}
	if got, want := distinct.Error(), "llm/openai: 400 invalid_request_error: bad"; got != want {
		t.Fatalf("distinct code lost: got %q, want %q", got, want)
	}
	noStatus := &ProviderError{Provider: "zai", Code: "1210", Message: "business error", Kind: ErrBadRequest}
	if got, want := noStatus.Error(), "llm/zai: 1210: business error"; got != want {
		t.Fatalf("code without status lost: got %q, want %q", got, want)
	}
}
