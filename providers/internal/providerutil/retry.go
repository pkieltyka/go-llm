package providerutil

import (
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

const (
	safeRetryBaseDelay = 500 * time.Millisecond
	safeRetryMaxDelay  = 30 * time.Second
	safeRetryDrainSize = 1 << 20
)

// SafeRetryHTTPClient returns a shallow copy of client whose transport retries
// a request only when replay is safe for a non-idempotent model invocation.
// The caller's client and transport are never mutated.
//
// Transport retries are limited to typed failures that prove no HTTP request
// bytes were sent: a temporary DNS lookup or a failed dial/proxy CONNECT.
// Ambiguous failures such as EOF, TLS handshake failures, connection reset,
// broken pipe, and read/write timeouts are returned immediately because the
// provider may already be processing and billing the request.
//
// When responseRetries is true, explicit 429/503/529 rejections may also be
// retried. Response retries are disabled by default by every built-in provider.
func SafeRetryHTTPClient(client *http.Client, maxRetries int, responseRetries bool) *http.Client {
	if client == nil {
		client = llm.DefaultHTTPClient()
	}
	if maxRetries <= 0 {
		return client
	}
	copied := *client
	transport := copied.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	copied.Transport = &safeRetryTransport{
		next:            transport,
		maxRetries:      maxRetries,
		responseRetries: responseRetries,
	}
	return &copied
}

type safeRetryTransport struct {
	next            http.RoundTripper
	maxRetries      int
	responseRetries bool
	sleep           func(context.Context, time.Duration) error
}

func (t *safeRetryTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req == nil {
		return nil, errors.New("provider retry: nil request")
	}
	next := t.next
	if next == nil {
		next = http.DefaultTransport
	}

	for attempt := 0; ; attempt++ {
		if err := req.Context().Err(); err != nil {
			return nil, err
		}
		attemptReq, err := retryAttemptRequest(req, attempt)
		if err != nil {
			return nil, err
		}
		resp, err := next.RoundTrip(attemptReq)
		if err != nil {
			if req.Context().Err() != nil {
				return nil, req.Context().Err()
			}
			if attempt >= t.maxRetries || req.Response != nil || !isProvablyPreSendError(err) || !requestReplayable(req) {
				return nil, err
			}
			if err := t.wait(req.Context(), retryDelay(attempt+1, 0)); err != nil {
				return nil, err
			}
			continue
		}

		if attempt >= t.maxRetries || req.Response != nil || !t.responseRetries ||
			!safeRetryStatus(resp.StatusCode) || responseRetryVeto(resp) || !requestReplayable(req) {
			return resp, nil
		}
		retryAfter := llm.RetryAfter(resp)
		if retryAfter > safeRetryMaxDelay {
			return resp, nil
		}
		delay := retryDelay(attempt+1, retryAfter)
		drainAndCloseForRetry(resp.Body)
		if err := t.wait(req.Context(), delay); err != nil {
			return nil, err
		}
	}
}

func retryAttemptRequest(req *http.Request, attempt int) (*http.Request, error) {
	attemptReq := req.Clone(req.Context())
	if attempt > 0 {
		attemptReq.Header.Set("X-Stainless-Retry-Count", strconv.Itoa(attempt))
	}
	if attempt == 0 {
		attemptReq.Body = req.Body
		return attemptReq, nil
	}
	if req.Body == nil {
		attemptReq.Body = nil
		return attemptReq, nil
	}
	if req.GetBody == nil {
		return nil, errors.New("provider retry: request body cannot be replayed")
	}
	body, err := req.GetBody()
	if err != nil {
		return nil, err
	}
	attemptReq.Body = body
	return attemptReq, nil
}

func requestReplayable(req *http.Request) bool {
	return req != nil && (req.Body == nil || req.GetBody != nil)
}

func safeRetryStatus(status int) bool {
	switch status {
	case http.StatusTooManyRequests, http.StatusServiceUnavailable, 529:
		return true
	default:
		return false
	}
}

func responseRetryVeto(resp *http.Response) bool {
	return resp != nil && strings.EqualFold(strings.TrimSpace(resp.Header.Get("X-Should-Retry")), "false")
}

func isProvablyPreSendError(err error) bool {
	if err == nil {
		return false
	}
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		return !dnsErr.IsNotFound && (dnsErr.IsTimeout || dnsErr.IsTemporary)
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) || isConnectionReset(err) {
		return false
	}

	if opErr := connectError(err); opErr != nil {
		if opErr.Timeout() {
			return true
		}
		for _, errno := range preSendDialErrors {
			if errors.Is(err, errno) {
				return true
			}
		}
	}
	return false
}

func connectError(err error) *net.OpError {
	var opErr *net.OpError
	if errors.As(err, &opErr) && (opErr.Op == "dial" || opErr.Op == "proxyconnect") {
		return opErr
	}
	return nil
}

func retryDelay(attempt int, retryAfter time.Duration) time.Duration {
	if retryAfter > 0 {
		return retryAfter
	}
	exponent := attempt - 1
	if exponent < 0 {
		exponent = 0
	}
	if exponent > 6 {
		exponent = 6
	}
	delay := safeRetryBaseDelay * time.Duration(1<<exponent)
	if delay > safeRetryMaxDelay {
		return safeRetryMaxDelay
	}
	return delay
}

func (t *safeRetryTransport) wait(ctx context.Context, delay time.Duration) error {
	if t.sleep != nil {
		return t.sleep(ctx, delay)
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func drainAndCloseForRetry(body io.ReadCloser) {
	if body == nil {
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(body, safeRetryDrainSize))
	_ = body.Close()
}
