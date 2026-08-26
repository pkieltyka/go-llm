package openaicodex

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/provideroauth"
)

func TestOAuthRefreshRefusesRedirects(t *testing.T) {
	var trapHits atomic.Int32
	trap := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		trapHits.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer trap.Close()

	for _, status := range []int{http.StatusTemporaryRedirect, http.StatusPermanentRedirect} {
		t.Run(fmt.Sprintf("status_%d", status), func(t *testing.T) {
			redirector := httptest.NewServer(http.RedirectHandler(trap.URL, status))
			defer redirector.Close()

			client := &http.Client{}
			_, err := refreshCodexOAuth(context.Background(), client, redirector.URL,
				llm.AuthCredential{Type: "oauth", Refresh: "refresh-token"})
			if !errors.Is(err, provideroauth.ErrUnsafeRedirect) {
				t.Fatalf("refresh error = %v, want ErrUnsafeRedirect", err)
			}
			if client.CheckRedirect != nil {
				t.Fatal("caller client CheckRedirect was mutated")
			}
		})
	}
	if got := trapHits.Load(); got != 0 {
		t.Fatalf("redirect target hits = %d, want 0", got)
	}
}

func TestOAuthRefreshSanitizesRequestAndTransportFailures(t *testing.T) {
	const secret = "refresh-token-secret"
	tests := []struct {
		name string
		err  error
		want error
	}{
		{name: "server", err: errors.New("transport echoed " + secret), want: llm.ErrServer},
		{name: "auth", err: fmt.Errorf("transport echoed %s: %w", secret, llm.ErrAuth), want: llm.ErrAuth},
		{name: "rate limit", err: fmt.Errorf("transport echoed %s: %w", secret, llm.ErrRateLimited), want: llm.ErrRateLimited},
		{name: "timeout", err: fmt.Errorf("transport echoed %s: %w", secret, context.DeadlineExceeded), want: llm.ErrTimeout},
		{name: "cancel", err: fmt.Errorf("transport echoed %s: %w", secret, context.Canceled), want: context.Canceled},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client := &http.Client{Transport: loginRoundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, test.err
			})}
			_, err := refreshCodexOAuth(context.Background(), client, "https://example.test/token", llm.AuthCredential{Refresh: secret})
			if !errors.Is(err, test.want) {
				t.Fatalf("refresh error = %v, want %v", err, test.want)
			}
			if strings.Contains(err.Error(), secret) || strings.Contains(err.Error(), "transport echoed") {
				t.Fatalf("refresh error leaked custom transport details: %v", err)
			}
		})
	}

	_, err := refreshCodexOAuth(context.Background(), http.DefaultClient, "://invalid", llm.AuthCredential{Refresh: secret})
	if !errors.Is(err, llm.ErrBadRequest) || strings.Contains(err.Error(), secret) {
		t.Fatalf("request construction error = %v", err)
	}
}

func TestOAuthRefreshStrictBoundedResponse(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "malformed", body: `{"access_token":`},
		{name: "trailing", body: `{"access_token":"access"}{}`},
		{name: "oversized", body: strings.Repeat("x", maxCodexTokenBodyBytes+1)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(writer, test.body)
			}))
			defer server.Close()
			_, err := refreshCodexOAuth(context.Background(), server.Client(), server.URL, llm.AuthCredential{Refresh: "refresh"})
			if !errors.Is(err, llm.ErrAuth) {
				t.Fatalf("refresh error = %v, want ErrAuth", err)
			}
		})
	}
}
