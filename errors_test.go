package llm

import (
	"errors"
	"fmt"
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

func TestProviderErrorSafeSummaryExcludesUntrustedFields(t *testing.T) {
	err := &ProviderError{
		Provider:   "openai",
		HTTPStatus: 400,
		Code:       "secret-code",
		Message:    "echoed prompt and bearer token",
		Metadata:   map[string]any{"token": "secret-metadata"},
		RawBody:    []byte("secret-body"),
		Kind:       ErrBadRequest,
	}
	got := err.SafeSummary()
	if want := "llm/openai: 400 (llm: bad request)"; got != want {
		t.Fatalf("SafeSummary() = %q, want %q", got, want)
	}
	for _, secret := range []string{"secret-code", "echoed prompt", "bearer token", "secret-metadata", "secret-body"} {
		if strings.Contains(got, secret) {
			t.Fatalf("SafeSummary() = %q contains %q", got, secret)
		}
	}
}

func TestProviderErrorSafeSummaryValidatesProviderLabel(t *testing.T) {
	for _, tt := range []struct {
		name     string
		provider string
		want     string
	}{
		{name: "known adapter", provider: "openai-codex", want: "llm/openai-codex: 500 (llm: server error)"},
		{name: "punctuation", provider: "openai\nsecret", want: "llm: 500 (llm: server error)"},
		{name: "unicode", provider: "openai-☃", want: "llm: 500 (llm: server error)"},
		{name: "too long", provider: strings.Repeat("a", 65), want: "llm: 500 (llm: server error)"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := &ProviderError{Provider: tt.provider, HTTPStatus: 500, Kind: ErrServer}
			if got := err.SafeSummary(); got != tt.want {
				t.Fatalf("SafeSummary() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestProviderErrorSafeSummaryRecognizesEverySentinel(t *testing.T) {
	for _, kind := range []error{
		ErrAuth,
		ErrPermission,
		ErrNotFound,
		ErrBadRequest,
		ErrRateLimited,
		ErrInsufficientCredits,
		ErrOverloaded,
		ErrServer,
		ErrTimeout,
		ErrContentFiltered,
		ErrContextTooLong,
		ErrUnsupported,
	} {
		if got, want := (&ProviderError{Kind: kind}).SafeSummary(), "llm: ("+kind.Error()+")"; got != want {
			t.Errorf("SafeSummary(%v) = %q, want %q", kind, got, want)
		}
	}
}

func TestSafeErrorFindsWrappedProviderErrorAndPreservesLocalErrors(t *testing.T) {
	providerErr := &ProviderError{Provider: "anthropic", Message: "sensitive", Kind: ErrOverloaded}
	wrapped := fmt.Errorf("request failed with echoed input: %w", providerErr)
	if got, want := SafeError(wrapped), "llm/anthropic: (llm: overloaded)"; got != want {
		t.Fatalf("SafeError(provider) = %q, want %q", got, want)
	}
	local := errors.New("local validation detail")
	if got := SafeError(local); got != local.Error() {
		t.Fatalf("SafeError(local) = %q, want %q", got, local.Error())
	}
}
