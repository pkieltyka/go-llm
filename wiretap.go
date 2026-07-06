package llm

import (
	"bytes"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

const defaultWireCaptureBodyLimit = 8 << 20

var redactedHeaderNames = map[string]struct{}{
	"authorization":       {},
	"proxy-authorization": {},
	"x-api-key":           {},
	"api-key":             {},
	"cookie":              {},
	"set-cookie":          {},
	"chatgpt-account-id":  {},
}

// WireCapture contains one captured HTTP attempt. Secrets in headers are
// always redacted.
type WireCapture struct {
	Provider        string
	Method          string
	URL             string
	RequestHeaders  http.Header
	RequestBody     []byte
	Status          int
	ResponseHeaders http.Header
	ResponseBody    []byte
	StartedAt       time.Time
	Duration        time.Duration
	Err             error
}

// WireCaptureToLogger adapts wire captures into slog debug records.
func WireCaptureToLogger(l *slog.Logger) func(WireCapture) {
	if l == nil {
		return func(WireCapture) {}
	}
	return func(c WireCapture) {
		attrs := []any{
			"provider", c.Provider,
			"method", c.Method,
			"url", c.URL,
			"status", c.Status,
			"duration", c.Duration,
			"request_headers", c.RequestHeaders,
			"request_body", string(c.RequestBody),
			"response_headers", c.ResponseHeaders,
			"response_body", string(c.ResponseBody),
		}
		if c.Err != nil {
			attrs = append(attrs, "error", c.Err.Error())
		}
		l.Debug("llm wire capture", attrs...)
	}
}

// WireTapOption configures NewWireTap.
type WireTapOption func(*wireTapOptions)

type wireTapOptions struct {
	bodyLimit int
}

// WithWireTapBodyLimit sets the captured request and response body byte cap.
// Bodies larger than limit are truncated with a marker. A zero limit captures
// only the marker for non-empty bodies.
func WithWireTapBodyLimit(limit int) WireTapOption {
	return func(opts *wireTapOptions) {
		if limit >= 0 {
			opts.bodyLimit = limit
		}
	}
}

// NewWireTap wraps next with request/response capture and header redaction.
func NewWireTap(next http.RoundTripper, provider string, fn func(WireCapture), opts ...WireTapOption) http.RoundTripper {
	if next == nil {
		next = http.DefaultTransport
	}
	if fn == nil {
		return next
	}
	options := wireTapOptions{bodyLimit: defaultWireCaptureBodyLimit}
	for _, opt := range opts {
		opt(&options)
	}
	return &wireTapTransport{next: next, provider: provider, capture: fn, bodyLimit: options.bodyLimit}
}

type wireTapTransport struct {
	next      http.RoundTripper
	provider  string
	capture   func(WireCapture)
	bodyLimit int
}

func (t *wireTapTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	start := time.Now()
	capture := WireCapture{
		Provider:       t.provider,
		Method:         req.Method,
		URL:            req.URL.String(),
		RequestHeaders: redactHeaders(req.Header),
		StartedAt:      start,
	}

	if req.Body != nil {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			capture.Duration = time.Since(start)
			capture.Err = err
			t.capture(capture)
			return nil, err
		}
		_ = req.Body.Close()
		capture.RequestBody = capBody(body, t.bodyLimit)
		req = req.Clone(req.Context())
		req.Body = io.NopCloser(bytes.NewReader(body))
		req.ContentLength = int64(len(body))
	}

	resp, err := t.next.RoundTrip(req)
	if err != nil {
		capture.Duration = time.Since(start)
		capture.Err = err
		t.capture(capture)
		return nil, err
	}
	capture.Status = resp.StatusCode
	capture.ResponseHeaders = redactHeaders(resp.Header)
	if resp.Body == nil {
		capture.Duration = time.Since(start)
		t.capture(capture)
		return resp, nil
	}

	resp.Body = &captureBody{
		body:    resp.Body,
		limit:   t.bodyLimit,
		start:   start,
		capture: capture,
		fn:      t.capture,
	}
	return resp, nil
}

type captureBody struct {
	body    io.ReadCloser
	limit   int
	start   time.Time
	capture WireCapture
	fn      func(WireCapture)
	buf     bytes.Buffer
	closed  bool
	trunc   bool
}

func (b *captureBody) Read(p []byte) (int, error) {
	n, err := b.body.Read(p)
	if n > 0 {
		b.write(p[:n])
	}
	return n, err
}

func (b *captureBody) Close() error {
	err := b.body.Close()
	if b.closed {
		return err
	}
	b.closed = true
	b.capture.ResponseBody = append([]byte(nil), b.buf.Bytes()...)
	b.capture.Duration = time.Since(b.start)
	b.capture.Err = err
	b.fn(b.capture)
	return err
}

func (b *captureBody) write(p []byte) {
	if len(p) == 0 {
		return
	}
	if b.buf.Len() >= b.limit {
		b.markTruncated()
		return
	}
	remaining := b.limit - b.buf.Len()
	if len(p) > remaining {
		b.buf.Write(p[:remaining])
		b.markTruncated()
		return
	}
	b.buf.Write(p)
}

func (b *captureBody) markTruncated() {
	if b.trunc {
		return
	}
	b.trunc = true
	b.buf.WriteString("\n[truncated]")
}

func redactHeaders(headers http.Header) http.Header {
	out := make(http.Header, len(headers))
	for name, values := range headers {
		copied := append([]string(nil), values...)
		if _, ok := redactedHeaderNames[strings.ToLower(name)]; ok {
			for i := range copied {
				copied[i] = "[REDACTED]"
			}
		}
		out[name] = copied
	}
	return out
}

func capBody(body []byte, limit int) []byte {
	if limit < 0 {
		limit = defaultWireCaptureBodyLimit
	}
	if len(body) <= limit {
		return append([]byte(nil), body...)
	}
	out := append([]byte(nil), body[:limit]...)
	out = append(out, "\n[truncated]"...)
	return out
}
