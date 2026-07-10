package provideroauth

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestDoWithAuthRetryUsesAlreadyRefreshedCredential(t *testing.T) {
	var refreshCalls int
	source := mustNewSource(t,
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh"},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			refreshCalls++
			return llm.AuthCredential{Type: "oauth", Access: "new"}, nil
		},
		noOpPersistence,
	)

	var authHeaders []string
	req := mustRequest(t, "payload")
	resp, err := DoWithAuthRetry(req, func(r *http.Request) (*http.Response, error) {
		authHeaders = append(authHeaders, r.Header.Get("Authorization"))
		switch len(authHeaders) {
		case 1:
			if _, err := source.ForceRefreshIfCurrent(context.Background(), "old"); err != nil {
				t.Fatalf("simulated concurrent refresh returned error: %v", err)
			}
			return stringResponse(http.StatusUnauthorized, "expired"), nil
		case 2:
			body, _ := io.ReadAll(r.Body)
			if string(body) != "payload" {
				t.Fatalf("retried body = %q", body)
			}
			return stringResponse(http.StatusOK, "ok"), nil
		default:
			t.Fatalf("unexpected request count %d", len(authHeaders))
			return nil, nil
		}
	}, source, bearer)
	if err != nil {
		t.Fatalf("DoWithAuthRetry returned error: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if refreshCalls != 1 {
		t.Fatalf("refresh calls = %d, want 1", refreshCalls)
	}
	if got := strings.Join(authHeaders, ","); got != "Bearer old,Bearer new" {
		t.Fatalf("auth headers = %s", got)
	}
}

func TestDoWithAuthRetryReturnsRefreshError(t *testing.T) {
	refreshErr := errors.New("refresh failed")
	source := mustNewSource(t,
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh"},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			return llm.AuthCredential{}, refreshErr
		},
		noOpPersistence,
	)
	req := mustRequest(t, "")
	resp, err := DoWithAuthRetry(req, func(r *http.Request) (*http.Response, error) {
		return stringResponse(http.StatusUnauthorized, "expired"), nil
	}, source, bearer)
	if !errors.Is(err, refreshErr) {
		t.Fatalf("error = %v, want refresh error", err)
	}
	if resp != nil {
		t.Fatalf("response = %+v, want nil when refresh fails", resp)
	}
}

func bearer(req *http.Request, cred llm.AuthCredential) {
	req.Header.Set("Authorization", "Bearer "+cred.Access)
}

func mustRequest(t *testing.T, body string) *http.Request {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, "https://example.test", strings.NewReader(body))
	if err != nil {
		t.Fatalf("NewRequest returned error: %v", err)
	}
	return req
}

func stringResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     http.Header{},
	}
}
