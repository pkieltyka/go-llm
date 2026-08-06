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

func TestSafeRetryTransportResponseRetriesRequireOptIn(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		t.Run(map[bool]string{false: "disabled", true: "enabled"}[enabled], func(t *testing.T) {
			for _, status := range []int{http.StatusTooManyRequests, http.StatusServiceUnavailable, 529} {
				calls := 0
				transport := &safeRetryTransport{
					maxRetries:      1,
					responseRetries: enabled,
					sleep:           noRetrySleep,
					next: retryRoundTripFunc(func(req *http.Request) (*http.Response, error) {
						calls++
						return retryResponse(req, status, "rejected"), nil
					}),
				}
				resp, err := transport.RoundTrip(newRetryRequest(t))
				if err != nil {
					t.Fatalf("status %d: RoundTrip returned error: %v", status, err)
				}
				_ = resp.Body.Close()
				wantCalls := 1
				if enabled {
					wantCalls = 2
				}
				if calls != wantCalls {
					t.Fatalf("status %d: calls = %d, want %d", status, calls, wantCalls)
				}
			}
		})
	}
}

func TestSafeRetryTransportNeverRetriesOtherResponses(t *testing.T) {
	for _, status := range []int{
		http.StatusRequestTimeout,
		http.StatusConflict,
		http.StatusInternalServerError,
		http.StatusBadGateway,
		http.StatusGatewayTimeout,
	} {
		calls := 0
		transport := &safeRetryTransport{
			maxRetries:      1,
			responseRetries: true,
			sleep:           noRetrySleep,
			next: retryRoundTripFunc(func(req *http.Request) (*http.Response, error) {
				calls++
				return retryResponse(req, status, "rejected"), nil
			}),
		}
		resp, err := transport.RoundTrip(newRetryRequest(t))
		if err != nil {
			t.Fatalf("status %d: RoundTrip returned error: %v", status, err)
		}
		_ = resp.Body.Close()
		if calls != 1 {
			t.Fatalf("status %d: calls = %d, want 1", status, calls)
		}
	}
}

func TestSafeRetryTransportHonorsResponseRetryVeto(t *testing.T) {
	for _, tt := range []struct {
		name             string
		enabled          bool
		shouldRetryValue string
		wantCalls        int
	}{
		{name: "explicit false vetoes opt-in", enabled: true, shouldRetryValue: "false", wantCalls: 1},
		{name: "explicit true permits opt-in", enabled: true, shouldRetryValue: "true", wantCalls: 2},
		{name: "explicit true does not enable retries", enabled: false, shouldRetryValue: "true", wantCalls: 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			transport := &safeRetryTransport{
				maxRetries:      1,
				responseRetries: tt.enabled,
				sleep:           noRetrySleep,
				next: retryRoundTripFunc(func(req *http.Request) (*http.Response, error) {
					calls++
					resp := retryResponse(req, http.StatusTooManyRequests, "busy")
					resp.Header.Set("X-Should-Retry", tt.shouldRetryValue)
					return resp, nil
				}),
			}
			resp, err := transport.RoundTrip(newRetryRequest(t))
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

func TestSafeRetryTransportDistinguishesTypedPreSendAndAmbiguousErrors(t *testing.T) {
	temporaryDNS := &net.DNSError{Err: "temporary failure", Name: "provider.test", IsTemporary: true}
	notFoundDNS := &net.DNSError{Err: "no such host", Name: "missing.test", IsNotFound: true}
	for _, tt := range []struct {
		name      string
		firstErr  error
		wantCalls int
	}{
		{name: "temporary DNS", firstErr: temporaryDNS, wantCalls: 2},
		{name: "connect timeout", firstErr: &net.OpError{Op: "dial", Net: "tcp", Err: timeoutError{}}, wantCalls: 2},
		{name: "proxy connect timeout", firstErr: &net.OpError{Op: "proxyconnect", Net: "tcp", Err: timeoutError{}}, wantCalls: 2},
		{name: "string-only refused dial", firstErr: &net.OpError{Op: "dial", Net: "tcp", Err: errors.New("connection refused")}, wantCalls: 1},
		{name: "string-only TLS handshake timeout", firstErr: errors.New("net/http: TLS handshake timeout"), wantCalls: 1},
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
					return retryResponse(req, http.StatusOK, "ok"), nil
				}),
			}
			resp, gotErr := transport.RoundTrip(newRetryRequest(t))
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
		maxRetries:      2,
		responseRetries: true,
		sleep:           noRetrySleep,
		next: retryRoundTripFunc(func(req *http.Request) (*http.Response, error) {
			body, err := io.ReadAll(req.Body)
			if err != nil {
				t.Fatalf("read body: %v", err)
			}
			bodies = append(bodies, string(body))
			retryCounts = append(retryCounts, req.Header.Get("X-Stainless-Retry-Count"))
			return retryResponse(req, http.StatusTooManyRequests, "busy"), nil
		}),
	}
	resp, err := transport.RoundTrip(newRetryRequest(t))
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
	for _, response := range []bool{false, true} {
		t.Run(map[bool]string{false: "transport error", true: "response"}[response], func(t *testing.T) {
			calls := 0
			transport := &safeRetryTransport{
				maxRetries:      2,
				responseRetries: true,
				sleep:           noRetrySleep,
				next: retryRoundTripFunc(func(req *http.Request) (*http.Response, error) {
					calls++
					if response {
						return retryResponse(req, http.StatusTooManyRequests, "busy"), nil
					}
					return nil, &net.OpError{Op: "dial", Net: "tcp", Err: timeoutError{}}
				}),
			}
			req := newRetryRequest(t)
			req.Response = &http.Response{StatusCode: http.StatusTemporaryRedirect}
			resp, gotErr := transport.RoundTrip(req)
			if resp != nil {
				_ = resp.Body.Close()
			}
			if !response && gotErr == nil {
				t.Fatal("RoundTrip returned nil error")
			}
			if response && gotErr != nil {
				t.Fatalf("RoundTrip returned error: %v", gotErr)
			}
			if calls != 1 {
				t.Fatalf("calls = %d, want 1", calls)
			}
		})
	}
}

func TestSafeRetryTransportLeavesExcessiveRetryAfterResponseUntouched(t *testing.T) {
	body := &trackingReadCloser{Reader: strings.NewReader("still readable")}
	calls := 0
	transport := &safeRetryTransport{
		maxRetries:      1,
		responseRetries: true,
		sleep:           noRetrySleep,
		next: retryRoundTripFunc(func(req *http.Request) (*http.Response, error) {
			calls++
			resp := &http.Response{StatusCode: http.StatusTooManyRequests, Header: http.Header{}, Body: body, Request: req}
			resp.Header.Set("Retry-After", "31")
			return resp, nil
		}),
	}
	resp, err := transport.RoundTrip(newRetryRequest(t))
	if err != nil {
		t.Fatalf("RoundTrip returned error: %v", err)
	}
	if calls != 1 || body.closed {
		t.Fatalf("calls/closed = %d/%v, want 1/false", calls, body.closed)
	}
	got, err := io.ReadAll(resp.Body)
	if err != nil || string(got) != "still readable" {
		t.Fatalf("response body = %q, %v", got, err)
	}
	_ = resp.Body.Close()
}

func TestSafeRetryTransportHonorsContextDuringBackoff(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	transport := &safeRetryTransport{
		maxRetries:      1,
		responseRetries: true,
		next: retryRoundTripFunc(func(req *http.Request) (*http.Response, error) {
			cancel()
			return retryResponse(req, http.StatusTooManyRequests, "busy"), nil
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

func TestRetryDelayHonorsProviderValueAndBoundsExponentialDelay(t *testing.T) {
	if got := retryDelay(1, 7*time.Second); got != 7*time.Second {
		t.Fatalf("Retry-After delay = %s, want 7s", got)
	}
	if got := retryDelay(1, 0); got != safeRetryBaseDelay {
		t.Fatalf("first delay = %s, want %s", got, safeRetryBaseDelay)
	}
	if got := retryDelay(100, 0); got != safeRetryMaxDelay {
		t.Fatalf("bounded exponential delay = %s, want %s", got, safeRetryMaxDelay)
	}
}

func newRetryRequest(t *testing.T) *http.Request {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, "https://provider.test/v1/chat", bytes.NewBufferString("same-body"))
	if err != nil {
		t.Fatal(err)
	}
	return req
}

func retryResponse(req *http.Request, status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{},
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    req,
	}
}

type trackingReadCloser struct {
	io.Reader
	closed bool
}

func (r *trackingReadCloser) Close() error {
	r.closed = true
	return nil
}

type timeoutError struct{}

func (timeoutError) Error() string   { return "timeout" }
func (timeoutError) Timeout() bool   { return true }
func (timeoutError) Temporary() bool { return true }
