package providerutil

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"
)

type retryRoundTripFunc func(*http.Request) (*http.Response, error)

func (fn retryRoundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

func noRetrySleep(context.Context, time.Duration) error { return nil }

func TestSafeRetryTransportRetriesOnlyExplicitRejections(t *testing.T) {
	for _, tt := range []struct {
		status    int
		wantCalls int
	}{
		{status: http.StatusTooManyRequests, wantCalls: 2},
		{status: http.StatusServiceUnavailable, wantCalls: 2},
		{status: 529, wantCalls: 2},
		{status: http.StatusRequestTimeout, wantCalls: 1},
		{status: http.StatusConflict, wantCalls: 1},
		{status: http.StatusInternalServerError, wantCalls: 1},
		{status: http.StatusBadGateway, wantCalls: 1},
		{status: http.StatusGatewayTimeout, wantCalls: 1},
	} {
		t.Run(http.StatusText(tt.status), func(t *testing.T) {
			calls := 0
			transport := &safeRetryTransport{
				maxRetries: 1,
				sleep:      noRetrySleep,
				next: retryRoundTripFunc(func(req *http.Request) (*http.Response, error) {
					calls++
					return &http.Response{
						StatusCode: tt.status,
						Header:     http.Header{},
						Body:       io.NopCloser(strings.NewReader("rejected")),
						Request:    req,
					}, nil
				}),
			}
			req, err := http.NewRequest(http.MethodPost, "https://provider.test/v1/chat", bytes.NewBufferString("payload"))
			if err != nil {
				t.Fatal(err)
			}
			resp, err := transport.RoundTrip(req)
			if err != nil {
				t.Fatalf("RoundTrip returned error: %v", err)
			}
			_ = resp.Body.Close()
			if calls != tt.wantCalls {
				t.Fatalf("calls = %d, want %d", calls, tt.wantCalls)
			}
		})
	}
}

func TestSafeRetryTransportDistinguishesPreSendAndAmbiguousErrors(t *testing.T) {
	temporaryDNS := &net.DNSError{Err: "temporary failure", Name: "provider.test", IsTemporary: true}
	notFoundDNS := &net.DNSError{Err: "no such host", Name: "missing.test", IsNotFound: true}
	for _, tt := range []struct {
		name      string
		firstErr  error
		wantCalls int
	}{
		{name: "temporary DNS", firstErr: temporaryDNS, wantCalls: 2},
		{name: "refused dial", firstErr: &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}, wantCalls: 2},
		{name: "connect timeout", firstErr: &net.OpError{Op: "dial", Net: "tcp", Err: timeoutError{}}, wantCalls: 2},
		{name: "proxy connect timeout", firstErr: &net.OpError{Op: "proxyconnect", Net: "tcp", Err: timeoutError{}}, wantCalls: 2},
		{name: "TLS handshake timeout", firstErr: errors.New("net/http: TLS handshake timeout"), wantCalls: 2},
		{name: "NXDOMAIN", firstErr: notFoundDNS, wantCalls: 1},
		{name: "EOF", firstErr: io.EOF, wantCalls: 1},
		{name: "unexpected EOF", firstErr: io.ErrUnexpectedEOF, wantCalls: 1},
		{name: "connection reset", firstErr: &net.OpError{Op: "read", Net: "tcp", Err: errors.New("connection reset by peer")}, wantCalls: 1},
		{name: "broken pipe", firstErr: &net.OpError{Op: "write", Net: "tcp", Err: errors.New("broken pipe")}, wantCalls: 1},
		{name: "read timeout", firstErr: &net.OpError{Op: "read", Net: "tcp", Err: timeoutError{}}, wantCalls: 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			transport := &safeRetryTransport{
				maxRetries: 1,
				sleep:      noRetrySleep,
				next: retryRoundTripFunc(func(req *http.Request) (*http.Response, error) {
					calls++
					if calls == 1 {
						return nil, tt.firstErr
					}
					return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("ok")), Request: req}, nil
				}),
			}
			req, err := http.NewRequest(http.MethodPost, "https://provider.test/v1/chat", bytes.NewBufferString("payload"))
			if err != nil {
				t.Fatal(err)
			}
			resp, gotErr := transport.RoundTrip(req)
			if resp != nil {
				_ = resp.Body.Close()
			}
			if calls != tt.wantCalls {
				t.Fatalf("calls = %d, want %d (error %v)", calls, tt.wantCalls, gotErr)
			}
			if tt.wantCalls == 2 && gotErr != nil {
				t.Fatalf("safe retry returned error: %v", gotErr)
			}
			if tt.wantCalls == 1 && !errors.Is(gotErr, tt.firstErr) {
				t.Fatalf("error = %v, want original %v", gotErr, tt.firstErr)
			}
		})
	}
}

func TestSafeRetryTransportReplaysBodyAndHonorsBounds(t *testing.T) {
	var bodies []string
	var retryCounts []string
	transport := &safeRetryTransport{
		maxRetries: 2,
		sleep:      noRetrySleep,
		next: retryRoundTripFunc(func(req *http.Request) (*http.Response, error) {
			body, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatalf("read body: %v", err)
			}
			bodies = append(bodies, string(body))
			retryCounts = append(retryCounts, req.Header.Get("X-Stainless-Retry-Count"))
			return &http.Response{StatusCode: http.StatusTooManyRequests, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("busy")), Request: req}, nil
		}),
	}
	req, err := http.NewRequest(http.MethodPost, "https://provider.test/v1/chat", bytes.NewBufferString("same-body"))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip returned error: %v", err)
	}
	_ = resp.Body.Close()
	if len(bodies) != 3 {
		t.Fatalf("attempts = %d, want 3", len(bodies))
	}
	for i, body := range bodies {
		if body != "same-body" {
			t.Fatalf("attempt %d body = %q, want same-body", i+1, body)
		}
	}
	if got := strings.Join(retryCounts, ","); got != ",1,2" {
		t.Fatalf("retry count headers = %q, want first empty then 1,2", got)
	}
}

func TestSafeRetryTransportDoesNotReplayRedirectedRequest(t *testing.T) {
	calls := 0
	transport := &safeRetryTransport{
		maxRetries: 2,
		sleep:      noRetrySleep,
		next: retryRoundTripFunc(func(*http.Request) (*http.Response, error) {
			calls++
			return nil, &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}
		}),
	}
	req, err := http.NewRequest(http.MethodPost, "https://redirect-target.test/v1/chat", bytes.NewBufferString("payload"))
	if err != nil {
		t.Fatal(err)
	}
	req.Response = &http.Response{StatusCode: http.StatusTemporaryRedirect}
	_, gotErr := transport.RoundTrip(req)
	if gotErr == nil {
		t.Fatal("RoundTrip returned nil error")
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want 1", calls)
	}
}

func TestSafeRetryTransportHonorsContextDuringBackoff(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	transport := &safeRetryTransport{
		maxRetries: 1,
		next: retryRoundTripFunc(func(req *http.Request) (*http.Response, error) {
			cancel()
			return &http.Response{StatusCode: http.StatusTooManyRequests, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("busy")), Request: req}, nil
		}),
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://provider.test/v1/chat", bytes.NewBufferString("payload"))
	if err != nil {
		t.Fatal(err)
	}
	_, gotErr := transport.RoundTrip(req)
	if !errors.Is(gotErr, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", gotErr)
	}
}

func TestRetryDelayHonorsAndCapsProviderValue(t *testing.T) {
	if got := retryDelay(1, 7*time.Second); got != 7*time.Second {
		t.Fatalf("Retry-After delay = %s, want 7s", got)
	}
	if got := retryDelay(1, time.Hour); got != safeRetryMaxDelay {
		t.Fatalf("capped Retry-After delay = %s, want %s", got, safeRetryMaxDelay)
	}
	if got := retryDelay(1, 0); got != safeRetryBaseDelay {
		t.Fatalf("first delay = %s, want %s", got, safeRetryBaseDelay)
	}
}

type timeoutError struct{}

func (timeoutError) Error() string   { return "timeout" }
func (timeoutError) Timeout() bool   { return true }
func (timeoutError) Temporary() bool { return true }
