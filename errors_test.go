package llm

import (
	"context"
	"encoding/json"
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
	bare := &ProviderError{Provider: "openai", Message: "refused", Kind: ErrContentFiltered}
	if got, want := bare.Error(), "llm/openai: refused"; got != want {
		t.Fatalf("message only: got %q, want %q", got, want)
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
		{name: "known adapter", provider: "openrouter", want: "llm/openrouter: 500 (llm: server error)"},
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

func TestErrorClassCoversEveryExportedSentinel(t *testing.T) {
	for _, sentinel := range []error{
		ErrAuth, ErrPermission, ErrNotFound, ErrBadRequest, ErrRateLimited,
		ErrInsufficientCredits, ErrOverloaded, ErrServer, ErrTimeout,
		ErrContentFiltered, ErrContextTooLong, ErrUnsupported,
		ErrLoginExpired, ErrLoginStateMismatch, ErrLoginClosed,
		context.Canceled, context.DeadlineExceeded,
	} {
		wrapped := fmt.Errorf("outer: %w", &ProviderError{Provider: "p", Kind: sentinel})
		if got := ErrorClass(wrapped); got != sentinel {
			t.Errorf("ErrorClass(%v) = %v, want %v", wrapped, got, sentinel)
		}
		if info := SafeErrorInfo(wrapped); info.Class == "" {
			t.Errorf("SafeErrorInfo(%v).Class is empty", sentinel)
		}
	}
	if ErrorClass(nil) != nil || ErrorClass(errors.New("plain")) != nil {
		t.Fatal("nil and unclassified errors must have a nil class")
	}
}

func TestErrorClassesListsEveryClassInPrecedenceOrder(t *testing.T) {
	classes := ErrorClasses()
	if len(classes) != len(errorClasses) {
		t.Fatalf("ErrorClasses() has %d classes, want %d", len(classes), len(errorClasses))
	}
	for i, class := range errorClasses {
		if classes[i] != class.err {
			t.Fatalf("ErrorClasses()[%d] = %v, want %v", i, classes[i], class.err)
		}
	}
	if classes[0] != context.Canceled || classes[len(classes)-1] != ErrServer {
		t.Fatalf("ErrorClasses() order = %v", classes)
	}
	classes[0] = nil
	if ErrorClasses()[0] != context.Canceled {
		t.Fatal("ErrorClasses must return a copy")
	}
}

func TestErrorClassPrecedence(t *testing.T) {
	cases := []struct {
		err  error
		want error
	}{
		{fmt.Errorf("%w: %w", ErrBadRequest, ErrContextTooLong), ErrContextTooLong},
		{fmt.Errorf("%w: %w", ErrBadRequest, ErrUnsupported), ErrUnsupported},
		{&ProviderError{Kind: errors.Join(ErrServer, context.Canceled)}, context.Canceled},
		{&ProviderError{Kind: errors.Join(ErrServer, ErrTimeout)}, ErrTimeout},
	}
	for _, tc := range cases {
		if got := ErrorClass(tc.err); got != tc.want {
			t.Errorf("ErrorClass(%v) = %v, want %v", tc.err, got, tc.want)
		}
	}
}

func TestSafeErrorInfoExcludesProviderDetail(t *testing.T) {
	err := fmt.Errorf("call: %w", &ProviderError{
		Provider:   "openrouter",
		HTTPStatus: 429,
		Code:       "secret-code",
		Message:    "echoed prompt with bearer token",
		RetryAfter: 3 * time.Second,
		Metadata:   map[string]any{"token": "secret-metadata"},
		RawBody:    []byte("secret-body"),
		Kind:       ErrRateLimited,
	})
	info := SafeErrorInfo(err)
	want := ErrorInfo{Class: "rate_limited", Provider: "openrouter", HTTPStatus: 429, RetryAfter: 3 * time.Second}
	if info != want {
		t.Fatalf("SafeErrorInfo = %+v, want %+v", info, want)
	}
	encoded, marshalErr := json.Marshal(info)
	if marshalErr != nil {
		t.Fatal(marshalErr)
	}
	for _, secret := range []string{"secret-code", "echoed prompt", "bearer token", "secret-metadata", "secret-body"} {
		if strings.Contains(string(encoded), secret) {
			t.Fatalf("SafeErrorInfo JSON leaked %q: %s", secret, encoded)
		}
	}
	if got := SafeErrorInfo(&ProviderError{Provider: "bad label!", Kind: ErrServer}); got.Provider != "" {
		t.Fatalf("unsafe provider label leaked: %+v", got)
	}
	if got := SafeErrorInfo(nil); got != (ErrorInfo{}) {
		t.Fatalf("SafeErrorInfo(nil) = %+v, want zero", got)
	}
}

func TestTransient(t *testing.T) {
	transient := map[error]bool{
		ErrRateLimited: true, ErrOverloaded: true, ErrServer: true, ErrTimeout: true,
		ErrContextTooLong: false, ErrAuth: false, ErrPermission: false, ErrNotFound: false,
		ErrBadRequest: false, ErrInsufficientCredits: false, ErrContentFiltered: false,
		ErrUnsupported: false, ErrLoginExpired: false, context.Canceled: false,
		context.DeadlineExceeded: false,
	}
	for sentinel, want := range transient {
		if got := Transient(&ProviderError{Kind: sentinel}); got != want {
			t.Errorf("Transient(%v) = %v, want %v", sentinel, got, want)
		}
	}
	// A transient class wrapped alongside cancellation is not transient: the
	// caller chose to stop.
	if Transient(&ProviderError{Kind: errors.Join(ErrServer, context.Canceled)}) {
		t.Fatal("canceled server error must not be transient")
	}
	if Transient(nil) || Transient(errors.New("plain")) {
		t.Fatal("nil and unclassified errors must not be transient")
	}
}
