package llm

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	defaultWireCaptureBodyLimit = 8 << 20
	wireCaptureTailLimit        = 4 << 10
)

var redactedHeaderNames = map[string]struct{}{
	"authorization":             {},
	"proxy-authorization":       {},
	"x-api-key":                 {},
	"api-key":                   {},
	"cookie":                    {},
	"etag":                      {},
	"set-cookie":                {},
	"chatgpt-account-id":        {},
	"anthropic-organization-id": {},
	"cf-ray":                    {},
	"nel":                       {},
	"openai-organization":       {},
	"openai-project":            {},
	"report-to":                 {},
	"request-id":                {},
	"traceresponse":             {},
	"x-generation-id":           {},
	"x-models-etag":             {},
	"x-oai-request-id":          {},
	"x-request-id":              {},
	"x-stainless-retry-count":   {},
}

var redactedHeaderPrefixes = []string{
	"anthropic-ratelimit-",
	"x-codex-",
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
	// ResponseIncomplete reports that the body was closed before EOF. It is
	// capture metadata only; WireTap never turns an early consumer close into
	// a transport error.
	ResponseIncomplete bool
}

// WireCaptureTracker exposes completion state for one WireTap transport.
// Recorders register trackers through WithWireCaptureObserver so even the
// first abandoned response remains visible when no capture callback fires.
type WireCaptureTracker struct {
	responseBodies *atomic.Int64
}

// OutstandingResponseBodies reports bodies that have not reached EOF or
// Close.
func (t WireCaptureTracker) OutstandingResponseBodies() int64 {
	if t.responseBodies == nil {
		return 0
	}
	return t.responseBodies.Load()
}

// WireCaptureObserver receives every WireTap tracker used by requests in its
// context. Implementations must be safe for concurrent calls.
type WireCaptureObserver interface {
	ObserveWireCaptureTracker(WireCaptureTracker)
}

type wireCaptureObserverContextKey struct{}

// WithWireCaptureObserver associates a tracker observer with ctx. Providers
// preserve request contexts across SDK and direct-stream transports.
func WithWireCaptureObserver(ctx context.Context, observer WireCaptureObserver) context.Context {
	if observer == nil {
		return ctx
	}
	return context.WithValue(ctx, wireCaptureObserverContextKey{}, observer)
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
		if c.ResponseIncomplete {
			attrs = append(attrs, "response_incomplete", true)
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
	responses atomic.Int64
}

func (t *wireTapTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	tracker := WireCaptureTracker{responseBodies: &t.responses}
	t.responses.Add(1)
	if observer, ok := req.Context().Value(wireCaptureObserverContextKey{}).(WireCaptureObserver); ok {
		observer.ObserveWireCaptureTracker(tracker)
	}
	start := time.Now()
	capture := WireCapture{
		Provider:       t.provider,
		Method:         req.Method,
		URL:            req.URL.String(),
		RequestHeaders: redactHeaders(req.Header),
		StartedAt:      start,
	}

	var requestCapture *requestBodyCapture
	if req.Body != nil {
		requestCapture = newRequestBodyCapture(t.bodyLimit)
		req = req.Clone(req.Context())
		req.Body = requestCapture.wrap(req.Body)
		if req.GetBody != nil {
			getBody := req.GetBody
			req.GetBody = func() (io.ReadCloser, error) {
				body, err := getBody()
				if err != nil {
					return nil, err
				}
				return requestCapture.wrap(body), nil
			}
		}
	}

	resp, err := t.next.RoundTrip(req)
	if err != nil {
		if requestCapture != nil {
			requestCapture.closeAll()
			capture.RequestBody = requestCapture.snapshot()
		}
		capture.Duration = time.Since(start)
		capture.Err = err
		t.capture(capture)
		t.responses.Add(-1)
		return nil, err
	}
	capture.Status = resp.StatusCode
	capture.ResponseHeaders = redactHeaders(resp.Header)
	if resp.Body == nil {
		if requestCapture != nil {
			capture.RequestBody = requestCapture.snapshot()
		}
		capture.Duration = time.Since(start)
		t.capture(capture)
		t.responses.Add(-1)
		return resp, nil
	}

	resp.Body = &captureBody{
		body:           resp.Body,
		limit:          t.bodyLimit,
		expectedLength: resp.ContentLength,
		start:          start,
		capture:        capture,
		fn:             t.capture,
		outstanding:    &t.responses,
		request:        requestCapture,
	}
	return resp, nil
}

// requestBodyCapture records only bytes consumed by the wrapped transport.
// A later GetBody generation replaces an earlier partial attempt so retries
// cannot concatenate the same payload in one capture.
type requestBodyCapture struct {
	mu          sync.Mutex
	limit       int
	next        uint64
	active      uint64
	activeSet   bool
	buf         boundedCapture
	outstanding map[*requestCaptureBody]struct{}
}

func newRequestBodyCapture(limit int) *requestBodyCapture {
	return &requestBodyCapture{
		limit:       limit,
		buf:         boundedCapture{limit: limit},
		outstanding: make(map[*requestCaptureBody]struct{}),
	}
}

func (c *requestBodyCapture) wrap(body io.ReadCloser) io.ReadCloser {
	c.mu.Lock()
	generation := c.next
	c.next++
	wrapped := &requestCaptureBody{body: body, capture: c, generation: generation}
	c.outstanding[wrapped] = struct{}{}
	c.mu.Unlock()
	return wrapped
}

func (c *requestBodyCapture) begin(generation uint64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.activeSet && generation <= c.active {
		return
	}
	c.active = generation
	c.activeSet = true
	c.buf = boundedCapture{limit: c.limit}
}

func (c *requestBodyCapture) write(generation uint64, p []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.activeSet || generation != c.active {
		return
	}
	c.buf.write(p)
}

func (c *requestBodyCapture) closed(body *requestCaptureBody) {
	c.mu.Lock()
	delete(c.outstanding, body)
	c.mu.Unlock()
}

func (c *requestBodyCapture) closeAll() {
	c.mu.Lock()
	bodies := make([]*requestCaptureBody, 0, len(c.outstanding))
	for body := range c.outstanding {
		bodies = append(bodies, body)
	}
	c.mu.Unlock()
	for _, body := range bodies {
		_ = body.Close()
	}
}

func (c *requestBodyCapture) snapshot() []byte {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.buf.snapshot()
}

type requestCaptureBody struct {
	body       io.ReadCloser
	capture    *requestBodyCapture
	generation uint64
	closeOnce  sync.Once
	closeErr   error
}

func (b *requestCaptureBody) Read(p []byte) (int, error) {
	b.capture.begin(b.generation)
	n, err := b.body.Read(p)
	if n > 0 {
		b.capture.write(b.generation, p[:n])
	}
	return n, err
}

func (b *requestCaptureBody) Close() error {
	b.closeOnce.Do(func() {
		b.closeErr = b.body.Close()
		b.capture.closed(b)
	})
	return b.closeErr
}

type boundedCapture struct {
	limit int
	buf   bytes.Buffer
	trunc bool
}

func (b *boundedCapture) write(p []byte) {
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

func (b *boundedCapture) markTruncated() {
	if b.trunc {
		return
	}
	b.trunc = true
	b.buf.WriteString("\n[truncated]")
}

func (b *boundedCapture) snapshot() []byte {
	return append([]byte(nil), b.buf.Bytes()...)
}

type captureBody struct {
	body           io.ReadCloser
	limit          int
	expectedLength int64
	start          time.Time
	capture        WireCapture
	fn             func(WireCapture)
	outstanding    *atomic.Int64
	request        *requestBodyCapture
	buf            bytes.Buffer
	tail           []byte
	read           int64
	finalizeOnce   sync.Once
	trunc          bool
}

func (b *captureBody) Read(p []byte) (int, error) {
	n, err := b.body.Read(p)
	if n > 0 {
		b.write(p[:n])
		b.read += int64(n)
	}
	if err != nil {
		if err == io.EOF {
			b.finalize(true, nil)
		} else {
			b.finalize(false, err)
		}
	}
	return n, err
}

func (b *captureBody) Close() error {
	err := b.body.Close()
	complete := b.expectedLength >= 0 && b.read >= b.expectedLength
	if !complete {
		complete = hasTerminalSSEDone(b.capture.ResponseHeaders, b.tail)
	}
	b.finalize(complete, err)
	return err
}

func (b *captureBody) finalize(complete bool, err error) {
	b.finalizeOnce.Do(func() {
		if b.request != nil {
			b.capture.RequestBody = b.request.snapshot()
		}
		b.capture.ResponseBody = append([]byte(nil), b.buf.Bytes()...)
		b.capture.Duration = time.Since(b.start)
		b.capture.Err = err
		b.capture.ResponseIncomplete = !complete
		b.fn(b.capture)
		// Publishing the capture happens-before zero outstanding becomes
		// observable to a recorder snapshot.
		b.outstanding.Add(-1)
	})
}

func hasTerminalSSEDone(headers http.Header, body []byte) bool {
	if !strings.Contains(strings.ToLower(headers.Get("Content-Type")), "text/event-stream") {
		return false
	}
	normalized := strings.ReplaceAll(string(body), "\r\n", "\n")
	normalized = strings.ReplaceAll(normalized, "\r", "\n")
	events := strings.Split(normalized, "\n\n")
	if len(events) < 2 {
		return false
	}
	terminal := false
	for _, event := range events[:len(events)-1] {
		data := make([]string, 0, 2)
		for _, line := range strings.Split(event, "\n") {
			if value, ok := wireSSEDataLineValue(line); ok {
				data = append(data, value)
			}
		}
		if len(data) > 0 {
			terminal = strings.Join(data, "\n") == "[DONE]"
		}
	}
	for _, line := range strings.Split(events[len(events)-1], "\n") {
		if _, ok := wireSSEDataLineValue(line); ok {
			return false
		}
	}
	return terminal
}

func wireSSEDataLineValue(line string) (string, bool) {
	if line == "data" {
		return "", true
	}
	if !strings.HasPrefix(line, "data:") {
		return "", false
	}
	value := strings.TrimPrefix(line, "data:")
	if strings.HasPrefix(value, " ") {
		value = strings.TrimPrefix(value, " ")
	}
	return value, true
}

func (b *captureBody) write(p []byte) {
	if len(p) == 0 {
		return
	}
	b.writeTail(p)
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

func (b *captureBody) writeTail(p []byte) {
	if len(p) >= wireCaptureTailLimit {
		b.tail = append(b.tail[:0], p[len(p)-wireCaptureTailLimit:]...)
		return
	}
	overflow := len(b.tail) + len(p) - wireCaptureTailLimit
	if overflow > 0 {
		copy(b.tail, b.tail[overflow:])
		b.tail = b.tail[:len(b.tail)-overflow]
	}
	b.tail = append(b.tail, p...)
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
		if isRedactedHeaderName(name) {
			for i := range copied {
				copied[i] = "[REDACTED]"
			}
		}
		out[name] = copied
	}
	return out
}

func isRedactedHeaderName(name string) bool {
	name = strings.ToLower(name)
	if _, ok := redactedHeaderNames[name]; ok {
		return true
	}
	for _, prefix := range redactedHeaderPrefixes {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}
