package llm_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
	"github.com/pkieltyka/go-llm/llmtest"
)

type trackedWireBody struct {
	reader io.Reader
	reads  atomic.Int32
	closes atomic.Int32
}

func newTrackedWireBody(body string) *trackedWireBody {
	return &trackedWireBody{reader: strings.NewReader(body)}
}

func (b *trackedWireBody) Read(p []byte) (int, error) {
	b.reads.Add(1)
	return b.reader.Read(p)
}

func (b *trackedWireBody) Close() error {
	b.closes.Add(1)
	return nil
}

type wireCaptureTrackerObserver struct {
	mu       sync.Mutex
	trackers []llm.WireCaptureTracker
}

func (o *wireCaptureTrackerObserver) ObserveWireCaptureTracker(tracker llm.WireCaptureTracker) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.trackers = append(o.trackers, tracker)
}

func (o *wireCaptureTrackerObserver) first(t *testing.T) llm.WireCaptureTracker {
	t.Helper()
	o.mu.Lock()
	defer o.mu.Unlock()
	if len(o.trackers) == 0 {
		t.Fatal("no WireTap tracker was observed")
	}
	return o.trackers[0]
}

func TestUsageTrackerAggregatesChatAndStream(t *testing.T) {
	p := llmtest.New(llmtest.WithName("fake"))
	for i := 0; i < 16; i++ {
		p.EnqueueResponse(&llm.Response{
			Provider: "fake",
			Model:    "model-a",
			Usage:    llm.Usage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3},
		})
	}
	p.EnqueueStream(
		llm.MessageStart{Provider: "fake", Model: "model-a"},
		llm.TextDelta{Index: 0, Text: "hello"},
		llm.MessageEnd{Usage: llm.Usage{InputTokens: 3, OutputTokens: 4, TotalTokens: 7}},
	)

	tracker := llm.NewUsageTracker()
	wrapped := llm.Wrap(p, tracker.Middleware())
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := wrapped.Chat(context.Background(), &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("hi")}})
			if err != nil {
				t.Errorf("Chat returned error: %v", err)
			}
		}()
	}
	wg.Wait()
	if _, err := llm.Collect(wrapped.ChatStream(context.Background(), &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("hi")}})); err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}

	stats := tracker.Stats()
	if stats.Calls != 17 || stats.Usage.InputTokens != 19 || stats.Usage.OutputTokens != 36 {
		t.Fatalf("stats = %+v", stats)
	}
	modelStats := stats.ByProviderModel["fake/model-a"]
	if modelStats.Calls != 17 || modelStats.Usage.TotalTokens != 55 {
		t.Fatalf("model stats = %+v", modelStats)
	}
}

// TestUsageTrackerCostSourcePassthrough asserts CostSource provenance
// survives aggregation: all-native sums stay native; mixing in an estimated
// cost downgrades the sum to estimated (a total is only billing-grade when
// every part of it is).
func TestUsageTrackerCostSourcePassthrough(t *testing.T) {
	nativeCost := 0.5
	estimatedCost := 0.25
	p := llmtest.New(llmtest.WithName("fake"))
	p.EnqueueResponse(&llm.Response{
		Provider: "fake",
		Model:    "model-a",
		Usage:    llm.Usage{TotalTokens: 3, CostUSD: &nativeCost, CostSource: llm.CostSourceNative},
	})
	p.EnqueueResponse(&llm.Response{
		Provider: "fake",
		Model:    "model-a",
		Usage:    llm.Usage{TotalTokens: 3, CostUSD: &nativeCost, CostSource: llm.CostSourceNative},
	})
	p.EnqueueResponse(&llm.Response{
		Provider: "fake",
		Model:    "model-a",
		Usage:    llm.Usage{TotalTokens: 3, CostUSD: &estimatedCost, CostSource: llm.CostSourceEstimated},
	})

	tracker := llm.NewUsageTracker()
	wrapped := llm.Wrap(p, tracker.Middleware())
	req := &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("hi")}}

	if _, err := wrapped.Chat(context.Background(), req); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	stats := tracker.Stats()
	if stats.Usage.CostSource != llm.CostSourceNative || stats.Usage.CostUSD == nil || *stats.Usage.CostUSD != nativeCost {
		t.Fatalf("single native stats = %+v (%q)", stats.Usage.CostUSD, stats.Usage.CostSource)
	}

	if _, err := wrapped.Chat(context.Background(), req); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	stats = tracker.Stats()
	if stats.Usage.CostSource != llm.CostSourceNative || *stats.Usage.CostUSD != 2*nativeCost {
		t.Fatalf("all-native sum = %v (%q), want native", *stats.Usage.CostUSD, stats.Usage.CostSource)
	}

	if _, err := wrapped.Chat(context.Background(), req); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	stats = tracker.Stats()
	if stats.Usage.CostSource != llm.CostSourceEstimated || *stats.Usage.CostUSD != 2*nativeCost+estimatedCost {
		t.Fatalf("mixed sum = %v (%q), want estimated", *stats.Usage.CostUSD, stats.Usage.CostSource)
	}
}

// A token-bearing call with unknown cost demotes an otherwise-native dollar
// total: the sum no longer covers every call, so it is not billing-grade.
func TestUsageTrackerNilCostComponentDemotesNativeSum(t *testing.T) {
	p := llmtest.New(llmtest.WithName("fake"))
	nativeCost := 0.5
	p.EnqueueResponse(&llm.Response{
		Provider: "fake", Model: "model-a",
		Parts: []llm.Part{llm.Text("one")},
		Usage: llm.Usage{TotalTokens: 3, CostUSD: &nativeCost, CostSource: llm.CostSourceNative},
	})
	p.EnqueueResponse(&llm.Response{
		Provider: "fake", Model: "model-a",
		Parts: []llm.Part{llm.Text("two")},
		Usage: llm.Usage{TotalTokens: 7}, // tokens, no cost
	})
	tracker := llm.NewUsageTracker()
	wrapped := llm.Wrap(p, tracker.Middleware())
	req := &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("hi")}}

	for i := 0; i < 2; i++ {
		if _, err := wrapped.Chat(context.Background(), req); err != nil {
			t.Fatalf("Chat %d returned error: %v", i, err)
		}
	}
	stats := tracker.Stats()
	if stats.Usage.CostSource != llm.CostSourceEstimated {
		t.Fatalf("CostSource = %q, want estimated after nil-cost component", stats.Usage.CostSource)
	}
	if stats.Usage.CostUSD == nil || *stats.Usage.CostUSD != nativeCost {
		t.Fatalf("CostUSD = %v, want %v", stats.Usage.CostUSD, nativeCost)
	}
	if stats.Usage.TotalTokens != 10 {
		t.Fatalf("TotalTokens = %d, want 10", stats.Usage.TotalTokens)
	}
}

func TestUsageTrackerCostProvenanceIsCommutative(t *testing.T) {
	withCost := func(tokens int64, cost float64, source string) llm.Usage {
		return llm.Usage{TotalTokens: tokens, CostUSD: &cost, CostSource: source}
	}
	missing := llm.Usage{TotalTokens: 7}
	native := withCost(3, 0.5, llm.CostSourceNative)
	estimated := withCost(5, 0.25, llm.CostSourceEstimated)

	tests := []struct {
		name       string
		components []llm.Usage
		wantCost   float64
		wantSource string
	}{
		{name: "native then missing", components: []llm.Usage{native, missing}, wantCost: 0.5, wantSource: llm.CostSourceEstimated},
		{name: "missing then native", components: []llm.Usage{missing, native}, wantCost: 0.5, wantSource: llm.CostSourceEstimated},
		{name: "native then estimated", components: []llm.Usage{native, estimated}, wantCost: 0.75, wantSource: llm.CostSourceEstimated},
		{name: "estimated then native", components: []llm.Usage{estimated, native}, wantCost: 0.75, wantSource: llm.CostSourceEstimated},
		{name: "estimated then missing", components: []llm.Usage{estimated, missing}, wantCost: 0.25, wantSource: llm.CostSourceEstimated},
		{name: "missing then estimated", components: []llm.Usage{missing, estimated}, wantCost: 0.25, wantSource: llm.CostSourceEstimated},
		{name: "all native", components: []llm.Usage{native, withCost(2, 0.1, llm.CostSourceNative)}, wantCost: 0.6, wantSource: llm.CostSourceNative},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := llmtest.New(llmtest.WithName("fake"))
			for _, usage := range tt.components {
				p.EnqueueResponse(&llm.Response{Provider: "fake", Model: "model-a", Usage: usage})
			}
			tracker := llm.NewUsageTracker()
			wrapped := llm.Wrap(p, tracker.Middleware())
			req := &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("hi")}}
			for range tt.components {
				if _, err := wrapped.Chat(context.Background(), req); err != nil {
					t.Fatalf("Chat returned error: %v", err)
				}
			}
			usage := tracker.Stats().Usage
			if usage.CostUSD == nil || *usage.CostUSD != tt.wantCost || usage.CostSource != tt.wantSource {
				t.Fatalf("usage cost = %v (%q), want %v (%q)", usage.CostUSD, usage.CostSource, tt.wantCost, tt.wantSource)
			}
		})
	}
}

func TestUsageTrackerBucketsFailedChatByProvider(t *testing.T) {
	p := llmtest.New(llmtest.WithName("fake"))
	p.EnqueueError(llm.ErrRateLimited)
	tracker := llm.NewUsageTracker()
	wrapped := llm.Wrap(p, tracker.Middleware())

	_, err := wrapped.Chat(context.Background(), &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("hi")}})
	if !errors.Is(err, llm.ErrRateLimited) {
		t.Fatalf("Chat error = %v, want ErrRateLimited", err)
	}

	stats := tracker.Stats()
	if stats.Errors != 1 {
		t.Fatalf("stats errors = %d, want 1", stats.Errors)
	}
	if _, ok := stats.ByProviderModel["/model-a"]; ok {
		t.Fatalf("failed call was bucketed without provider: %+v", stats.ByProviderModel)
	}
	modelStats := stats.ByProviderModel["fake/model-a"]
	if modelStats.Calls != 1 || modelStats.Errors != 1 {
		t.Fatalf("fake/model-a stats = %+v", modelStats)
	}
}

func TestNewWireTapRedactsAndCapturesBody(t *testing.T) {
	var captures []llm.WireCapture
	rt := testutil.RoundTripFunc(func(req *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		if string(body) != `{"prompt":"hi"}` {
			t.Fatalf("request body seen by transport = %q", body)
		}
		return &http.Response{
			StatusCode: 200,
			Header: http.Header{
				"Set-Cookie":                   []string{"secret=1"},
				"Etag":                         []string{"private-etag"},
				"Anthropic-Ratelimit-Requests": []string{"private-limit"},
				"X-Safe-Response":              []string{"visible"},
			},
			Body: io.NopCloser(strings.NewReader("data: hello\n\n")),
		}, nil
	})

	client := &http.Client{Transport: llm.NewWireTap(rt, "fake", func(c llm.WireCapture) {
		captures = append(captures, c)
	})}
	req, err := http.NewRequest(http.MethodPost, "https://example.test/chat", strings.NewReader(`{"prompt":"hi"}`))
	if err != nil {
		t.Fatalf("NewRequest returned error: %v", err)
	}
	req.Header.Set("Authorization", "Bearer secret")
	req.Header.Set("X-Api-Key", "secret")
	req.Header.Set("Anthropic-Organization-Id", "org-secret")
	req.Header.Set("OpenAI-Project", "project-secret")
	req.Header.Set("X-Codex-Account-Id", "account-secret")
	req.Header.Set("X-Safe-Request", "visible")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do returned error: %v", err)
	}
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll returned error: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}
	if string(respBody) != "data: hello\n\n" {
		t.Fatalf("response body = %q", respBody)
	}
	if len(captures) != 1 {
		t.Fatalf("captures len = %d, want 1", len(captures))
	}
	capture := captures[0]
	for _, name := range []string{"Authorization", "X-Api-Key", "Anthropic-Organization-Id", "OpenAI-Project", "X-Codex-Account-Id"} {
		if got := capture.RequestHeaders.Get(name); got != "[REDACTED]" {
			t.Fatalf("%s capture = %q, want redacted", name, got)
		}
	}
	for _, name := range []string{"Set-Cookie", "ETag", "Anthropic-RateLimit-Requests"} {
		if got := capture.ResponseHeaders.Get(name); got != "[REDACTED]" {
			t.Fatalf("%s capture = %q, want redacted", name, got)
		}
	}
	if got := capture.RequestHeaders.Get("X-Safe-Request"); got != "visible" {
		t.Fatalf("safe request header = %q, want visible", got)
	}
	if got := capture.ResponseHeaders.Get("X-Safe-Response"); got != "visible" {
		t.Fatalf("safe response header = %q, want visible", got)
	}
	if string(capture.RequestBody) != `{"prompt":"hi"}` || string(capture.ResponseBody) != "data: hello\n\n" {
		t.Fatalf("capture bodies = request %q response %q", capture.RequestBody, capture.ResponseBody)
	}
}

func TestWireTapRequestCaptureFollowsTransportConsumption(t *testing.T) {
	tests := []struct {
		name     string
		read     int
		wantBody string
	}{
		{name: "none", read: 0, wantBody: ""},
		{name: "partial", read: 4, wantBody: "requ"},
		{name: "full", read: -1, wantBody: "request-body"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := newTrackedWireBody("request-body")
			var capture llm.WireCapture
			rt := testutil.RoundTripFunc(func(req *http.Request) (*http.Response, error) {
				if got := body.reads.Load(); got != 0 {
					t.Fatalf("body was read %d times before transport consumption", got)
				}
				if tt.read < 0 {
					if _, err := io.Copy(io.Discard, req.Body); err != nil {
						return nil, err
					}
				} else if tt.read > 0 {
					buf := make([]byte, tt.read)
					if _, err := io.ReadFull(req.Body, buf); err != nil {
						return nil, err
					}
				}
				if err := req.Body.Close(); err != nil {
					return nil, err
				}
				return &http.Response{StatusCode: http.StatusNoContent, ContentLength: 0, Body: http.NoBody}, nil
			})
			transport := llm.NewWireTap(rt, "fake", func(c llm.WireCapture) { capture = c })
			req, err := http.NewRequest(http.MethodPost, "https://example.test/chat", body)
			if err != nil {
				t.Fatalf("NewRequest returned error: %v", err)
			}
			resp, err := transport.RoundTrip(req)
			if err != nil {
				t.Fatalf("RoundTrip returned error: %v", err)
			}
			if err := resp.Body.Close(); err != nil {
				t.Fatalf("response Close returned error: %v", err)
			}
			if got := string(capture.RequestBody); got != tt.wantBody {
				t.Fatalf("captured request body = %q, want %q", got, tt.wantBody)
			}
			if got := body.closes.Load(); got != 1 {
				t.Fatalf("request body close count = %d, want 1", got)
			}
		})
	}
}

func TestWireTapClosesRequestBodyOnTransportError(t *testing.T) {
	wantErr := errors.New("transport failed")
	body := newTrackedWireBody("request-body")
	var capture llm.WireCapture
	rt := testutil.RoundTripFunc(func(req *http.Request) (*http.Response, error) {
		buf := make([]byte, 4)
		if _, err := io.ReadFull(req.Body, buf); err != nil {
			return nil, err
		}
		return nil, wantErr
	})
	transport := llm.NewWireTap(rt, "fake", func(c llm.WireCapture) { capture = c })
	req, err := http.NewRequest(http.MethodPost, "https://example.test/chat", body)
	if err != nil {
		t.Fatalf("NewRequest returned error: %v", err)
	}
	resp, err := transport.RoundTrip(req)
	if resp != nil || !errors.Is(err, wantErr) {
		t.Fatalf("RoundTrip = (%v, %v), want nil and transport error", resp, err)
	}
	if got := body.closes.Load(); got != 1 {
		t.Fatalf("request body close count = %d, want 1", got)
	}
	if got := string(capture.RequestBody); got != "requ" {
		t.Fatalf("captured request body = %q, want %q", got, "requ")
	}
	if !errors.Is(capture.Err, wantErr) {
		t.Fatalf("capture error = %v, want transport error", capture.Err)
	}
}

func TestWireTapPreservesGetBodyAndCapturesLatestRetry(t *testing.T) {
	initial := newTrackedWireBody("request-body")
	var retryBodies []*trackedWireBody
	var capture llm.WireCapture
	req, err := http.NewRequest(http.MethodPost, "https://example.test/chat", initial)
	if err != nil {
		t.Fatalf("NewRequest returned error: %v", err)
	}
	req.ContentLength = 12
	req.GetBody = func() (io.ReadCloser, error) {
		body := newTrackedWireBody("request-body")
		retryBodies = append(retryBodies, body)
		return body, nil
	}

	rt := testutil.RoundTripFunc(func(got *http.Request) (*http.Response, error) {
		if got.GetBody == nil {
			t.Fatal("GetBody was removed")
		}
		partial := make([]byte, 3)
		if _, err := io.ReadFull(got.Body, partial); err != nil {
			return nil, err
		}
		retry, err := got.GetBody()
		if err != nil {
			return nil, err
		}
		if _, err := io.Copy(io.Discard, retry); err != nil {
			return nil, err
		}
		if err := retry.Close(); err != nil {
			return nil, err
		}
		if err := got.Body.Close(); err != nil {
			return nil, err
		}
		return &http.Response{StatusCode: http.StatusNoContent, ContentLength: 0, Body: http.NoBody}, nil
	})
	transport := llm.NewWireTap(rt, "fake", func(c llm.WireCapture) { capture = c })
	resp, err := transport.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip returned error: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("response Close returned error: %v", err)
	}
	if got := string(capture.RequestBody); got != "request-body" {
		t.Fatalf("captured retry body = %q, want full retry body", got)
	}
	if req.ContentLength != 12 || req.GetBody == nil {
		t.Fatalf("original request was mutated: ContentLength=%d GetBody=%v", req.ContentLength, req.GetBody != nil)
	}
	if initial.closes.Load() != 1 || len(retryBodies) != 1 || retryBodies[0].closes.Load() != 1 {
		t.Fatalf("body close counts = initial %d retries %+v", initial.closes.Load(), retryBodies)
	}
	originalReplay, err := req.GetBody()
	if err != nil {
		t.Fatalf("original GetBody returned error: %v", err)
	}
	if _, ok := originalReplay.(*trackedWireBody); !ok {
		t.Fatalf("original GetBody was replaced with %T", originalReplay)
	}
	_ = originalReplay.Close()
}

func TestWireTapUnusedGetBodyDoesNotReplaceConsumedAttempt(t *testing.T) {
	wantErr := errors.New("retry abandoned")
	initial := newTrackedWireBody("request-body")
	retry := newTrackedWireBody("request-body")
	var capture llm.WireCapture
	req, err := http.NewRequest(http.MethodPost, "https://example.test/chat", initial)
	if err != nil {
		t.Fatalf("NewRequest returned error: %v", err)
	}
	req.GetBody = func() (io.ReadCloser, error) { return retry, nil }

	rt := testutil.RoundTripFunc(func(got *http.Request) (*http.Response, error) {
		buf := make([]byte, 4)
		if _, err := io.ReadFull(got.Body, buf); err != nil {
			return nil, err
		}
		if _, err := got.GetBody(); err != nil {
			return nil, err
		}
		return nil, wantErr
	})
	transport := llm.NewWireTap(rt, "fake", func(c llm.WireCapture) { capture = c })
	resp, err := transport.RoundTrip(req)
	if resp != nil || !errors.Is(err, wantErr) {
		t.Fatalf("RoundTrip = (%v, %v), want nil and retry error", resp, err)
	}
	if got := string(capture.RequestBody); got != "requ" {
		t.Fatalf("captured request body = %q, want consumed initial prefix", got)
	}
	if initial.closes.Load() != 1 || retry.closes.Load() != 1 {
		t.Fatalf("body close counts = initial %d retry %d, want 1 each", initial.closes.Load(), retry.closes.Load())
	}
}

func TestWireTapPreservesRequestContentLength(t *testing.T) {
	for _, contentLength := range []int64{0, 12, -1} {
		t.Run(fmt.Sprintf("length_%d", contentLength), func(t *testing.T) {
			body := newTrackedWireBody("request-body")
			var capture llm.WireCapture
			rt := testutil.RoundTripFunc(func(req *http.Request) (*http.Response, error) {
				if req.ContentLength != contentLength {
					t.Fatalf("transport ContentLength = %d, want %d", req.ContentLength, contentLength)
				}
				if _, err := io.Copy(io.Discard, req.Body); err != nil {
					return nil, err
				}
				if err := req.Body.Close(); err != nil {
					return nil, err
				}
				return &http.Response{StatusCode: http.StatusNoContent, ContentLength: 0, Body: http.NoBody}, nil
			})
			transport := llm.NewWireTap(rt, "fake", func(c llm.WireCapture) { capture = c })
			req, err := http.NewRequest(http.MethodPost, "https://example.test/chat", body)
			if err != nil {
				t.Fatalf("NewRequest returned error: %v", err)
			}
			req.ContentLength = contentLength
			resp, err := transport.RoundTrip(req)
			if err != nil {
				t.Fatalf("RoundTrip returned error: %v", err)
			}
			if err := resp.Body.Close(); err != nil {
				t.Fatalf("response Close returned error: %v", err)
			}
			if req.ContentLength != contentLength {
				t.Fatalf("original ContentLength = %d, want %d", req.ContentLength, contentLength)
			}
			if got := string(capture.RequestBody); got != "request-body" {
				t.Fatalf("captured request body = %q", got)
			}
		})
	}
}

func TestWireTapRequestBodyLimitBoundary(t *testing.T) {
	for _, tt := range []struct {
		name  string
		limit int
		body  string
		want  string
	}{
		{name: "exact", limit: 4, body: "abcd", want: "abcd"},
		{name: "truncated", limit: 4, body: "abcde", want: "abcd\n[truncated]"},
		{name: "zero", limit: 0, body: "a", want: "\n[truncated]"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var capture llm.WireCapture
			rt := testutil.RoundTripFunc(func(req *http.Request) (*http.Response, error) {
				if _, err := io.Copy(io.Discard, req.Body); err != nil {
					return nil, err
				}
				if err := req.Body.Close(); err != nil {
					return nil, err
				}
				return &http.Response{StatusCode: http.StatusNoContent, ContentLength: 0, Body: http.NoBody}, nil
			})
			transport := llm.NewWireTap(rt, "fake", func(c llm.WireCapture) { capture = c }, llm.WithWireTapBodyLimit(tt.limit))
			req, err := http.NewRequest(http.MethodPost, "https://example.test/chat", strings.NewReader(tt.body))
			if err != nil {
				t.Fatalf("NewRequest returned error: %v", err)
			}
			resp, err := transport.RoundTrip(req)
			if err != nil {
				t.Fatalf("RoundTrip returned error: %v", err)
			}
			if err := resp.Body.Close(); err != nil {
				t.Fatalf("response Close returned error: %v", err)
			}
			if got := string(capture.RequestBody); got != tt.want {
				t.Fatalf("captured request body = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestWireTapConcurrentRequestCapture(t *testing.T) {
	const requests = 32
	var mu sync.Mutex
	captures := make(map[string]string, requests)
	rt := testutil.RoundTripFunc(func(req *http.Request) (*http.Response, error) {
		if _, err := io.Copy(io.Discard, req.Body); err != nil {
			return nil, err
		}
		if err := req.Body.Close(); err != nil {
			return nil, err
		}
		return &http.Response{StatusCode: http.StatusNoContent, ContentLength: 0, Body: http.NoBody}, nil
	})
	transport := llm.NewWireTap(rt, "fake", func(c llm.WireCapture) {
		mu.Lock()
		captures[c.URL] = string(c.RequestBody)
		mu.Unlock()
	})

	var wg sync.WaitGroup
	for i := 0; i < requests; i++ {
		i := i
		wg.Add(1)
		go func() {
			defer wg.Done()
			body := fmt.Sprintf("request-%d", i)
			url := fmt.Sprintf("https://example.test/%d", i)
			req, err := http.NewRequest(http.MethodPost, url, strings.NewReader(body))
			if err != nil {
				t.Errorf("NewRequest returned error: %v", err)
				return
			}
			resp, err := transport.RoundTrip(req)
			if err != nil {
				t.Errorf("RoundTrip returned error: %v", err)
				return
			}
			if err := resp.Body.Close(); err != nil {
				t.Errorf("response Close returned error: %v", err)
			}
		}()
	}
	wg.Wait()

	mu.Lock()
	defer mu.Unlock()
	if len(captures) != requests {
		t.Fatalf("capture count = %d, want %d", len(captures), requests)
	}
	for i := 0; i < requests; i++ {
		url := fmt.Sprintf("https://example.test/%d", i)
		if got, want := captures[url], fmt.Sprintf("request-%d", i); got != want {
			t.Fatalf("capture %s = %q, want %q", url, got, want)
		}
	}
}

func TestWireTapFinalizesResponseCaptureAtEOF(t *testing.T) {
	var captures []llm.WireCapture
	rt := testutil.RoundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusOK,
			ContentLength: -1,
			Body:          io.NopCloser(strings.NewReader("complete response")),
		}, nil
	})
	client := &http.Client{Transport: llm.NewWireTap(rt, "fake", func(c llm.WireCapture) {
		captures = append(captures, c)
	})}

	resp, err := client.Get("https://example.test/stream")
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	if _, err := io.ReadAll(resp.Body); err != nil {
		t.Fatalf("ReadAll returned error: %v", err)
	}
	if len(captures) != 1 {
		t.Fatalf("captures after EOF = %d, want 1", len(captures))
	}
	if captures[0].ResponseIncomplete || captures[0].Err != nil {
		t.Fatalf("EOF capture = %+v", captures[0])
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}
	if len(captures) != 1 {
		t.Fatalf("captures after Close = %d, want no duplicate", len(captures))
	}
}

func TestWireTapEarlyCloseIsIncompleteWithoutConsumerError(t *testing.T) {
	var captures []llm.WireCapture
	rt := testutil.RoundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusOK,
			ContentLength: -1,
			Body:          io.NopCloser(strings.NewReader("stream response")),
		}, nil
	})
	client := &http.Client{Transport: llm.NewWireTap(rt, "fake", func(c llm.WireCapture) {
		captures = append(captures, c)
	})}

	resp, err := client.Get("https://example.test/stream")
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	buf := make([]byte, 1)
	if n, err := resp.Body.Read(buf); err != nil || n != 1 {
		t.Fatalf("Read = %d, %v", n, err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("early Close returned error: %v", err)
	}
	if len(captures) != 1 {
		t.Fatalf("captures = %d, want 1", len(captures))
	}
	if !captures[0].ResponseIncomplete || captures[0].Err != nil {
		t.Fatalf("early-close capture = %+v", captures[0])
	}
	if got := string(captures[0].ResponseBody); got != "s" {
		t.Fatalf("captured response prefix = %q, want s", got)
	}
}

func TestWireTapCappedSSEDoneTailIsCompleteOnClose(t *testing.T) {
	var captures []llm.WireCapture
	body := strings.Repeat("data: {\"delta\":\"0123456789\"}\n\n", 8) + "data: [DONE]\n\n"
	rt := testutil.RoundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusOK,
			ContentLength: -1,
			Header:        http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:          io.NopCloser(strings.NewReader(body)),
		}, nil
	})
	client := &http.Client{Transport: llm.NewWireTap(rt, "fake", func(c llm.WireCapture) {
		captures = append(captures, c)
	}, llm.WithWireTapBodyLimit(24))}

	resp, err := client.Get("https://example.test/stream")
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	read := make([]byte, len(body))
	if _, err := io.ReadFull(resp.Body, read); err != nil {
		t.Fatalf("ReadFull returned error: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}
	if len(captures) != 1 {
		t.Fatalf("captures = %d, want 1", len(captures))
	}
	if captures[0].ResponseIncomplete || captures[0].Err != nil {
		t.Fatalf("capped terminal [DONE] capture = %+v", captures[0])
	}
	if strings.Contains(string(captures[0].ResponseBody), "[DONE]") || !strings.HasSuffix(string(captures[0].ResponseBody), "\n[truncated]") {
		t.Fatalf("captured body did not remain a capped prefix: %q", captures[0].ResponseBody)
	}
}

func TestWireTapRejectsMalformedSSEDoneOnClose(t *testing.T) {
	for name, body := range map[string]string{
		"leading_whitespace": " data: [DONE]\n\n",
		"unterminated_event": "data: [DONE]\n",
	} {
		t.Run(name, func(t *testing.T) {
			var captures []llm.WireCapture
			rt := testutil.RoundTripFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode:    http.StatusOK,
					ContentLength: -1,
					Header:        http.Header{"Content-Type": []string{"text/event-stream"}},
					Body:          io.NopCloser(strings.NewReader(body)),
				}, nil
			})
			client := &http.Client{Transport: llm.NewWireTap(rt, "fake", func(c llm.WireCapture) {
				captures = append(captures, c)
			})}
			resp, err := client.Get("https://example.test/stream")
			if err != nil {
				t.Fatalf("Get returned error: %v", err)
			}
			read := make([]byte, len(body))
			if _, err := io.ReadFull(resp.Body, read); err != nil {
				t.Fatalf("ReadFull returned error: %v", err)
			}
			if err := resp.Body.Close(); err != nil {
				t.Fatalf("Close returned error: %v", err)
			}
			if len(captures) != 1 || !captures[0].ResponseIncomplete || captures[0].Err != nil {
				t.Fatalf("malformed terminal capture = %+v", captures)
			}
		})
	}
}

func TestWireTapZeroContentLengthIsCompleteOnClose(t *testing.T) {
	var captures []llm.WireCapture
	rt := testutil.RoundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusNoContent,
			ContentLength: 0,
			Body:          io.NopCloser(strings.NewReader("")),
		}, nil
	})
	client := &http.Client{Transport: llm.NewWireTap(rt, "fake", func(c llm.WireCapture) {
		captures = append(captures, c)
	})}
	resp, err := client.Get("https://example.test/empty")
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}
	if len(captures) != 1 || captures[0].ResponseIncomplete || captures[0].Err != nil {
		t.Fatalf("zero-length capture = %+v", captures)
	}
}

func TestWireTapTracksOutstandingResponseBodies(t *testing.T) {
	observer := &wireCaptureTrackerObserver{}
	rt := testutil.RoundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusOK,
			ContentLength: -1,
			Body:          io.NopCloser(strings.NewReader("response")),
		}, nil
	})
	client := &http.Client{Transport: llm.NewWireTap(rt, "fake", func(llm.WireCapture) {})}
	ctx := llm.WithWireCaptureObserver(context.Background(), observer)

	firstReq, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://example.test/first", nil)
	if err != nil {
		t.Fatalf("first NewRequestWithContext returned error: %v", err)
	}
	first, err := client.Do(firstReq)
	if err != nil {
		t.Fatalf("first Get returned error: %v", err)
	}
	if _, err := io.ReadAll(first.Body); err != nil {
		t.Fatalf("first ReadAll returned error: %v", err)
	}
	if err := first.Body.Close(); err != nil {
		t.Fatalf("first Close returned error: %v", err)
	}
	secondReq, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://example.test/second", nil)
	if err != nil {
		t.Fatalf("second NewRequestWithContext returned error: %v", err)
	}
	second, err := client.Do(secondReq)
	if err != nil {
		t.Fatalf("second Get returned error: %v", err)
	}
	tracker := observer.first(t)
	if got := tracker.OutstandingResponseBodies(); got != 1 {
		t.Fatalf("outstanding response bodies = %d, want 1", got)
	}
	if err := second.Body.Close(); err != nil {
		t.Fatalf("second Close returned error: %v", err)
	}
	if got := tracker.OutstandingResponseBodies(); got != 0 {
		t.Fatalf("outstanding response bodies after Close = %d, want 0", got)
	}
}

func TestWireTapPublishesCaptureBeforeOutstandingReachesZero(t *testing.T) {
	observer := &wireCaptureTrackerObserver{}
	callbackStarted := make(chan struct{})
	releaseCallback := make(chan struct{})
	readDone := make(chan error, 1)
	rt := testutil.RoundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusOK,
			ContentLength: -1,
			Body:          io.NopCloser(strings.NewReader("complete")),
		}, nil
	})
	client := &http.Client{Transport: llm.NewWireTap(rt, "fake", func(llm.WireCapture) {
		close(callbackStarted)
		<-releaseCallback
	})}
	ctx := llm.WithWireCaptureObserver(context.Background(), observer)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://example.test/stream", nil)
	if err != nil {
		t.Fatalf("NewRequestWithContext returned error: %v", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do returned error: %v", err)
	}
	go func() {
		_, readErr := io.ReadAll(resp.Body)
		readDone <- readErr
	}()
	<-callbackStarted
	tracker := observer.first(t)
	if got := tracker.OutstandingResponseBodies(); got != 1 {
		t.Fatalf("outstanding during capture publication = %d, want 1", got)
	}
	close(releaseCallback)
	if err := <-readDone; err != nil {
		t.Fatalf("ReadAll returned error: %v", err)
	}
	if got := tracker.OutstandingResponseBodies(); got != 0 {
		t.Fatalf("outstanding after capture publication = %d, want 0", got)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}
}

func TestNewWireTapMarksExactLimitResponseTruncation(t *testing.T) {
	var capture llm.WireCapture
	body := strings.Repeat("a", 8<<20) + "b"
	rt := testutil.RoundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(strings.NewReader(body)),
		}, nil
	})

	client := &http.Client{Transport: llm.NewWireTap(rt, "fake", func(c llm.WireCapture) {
		capture = c
	})}
	resp, err := client.Get("https://example.test/stream")
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	if _, err := io.Copy(io.Discard, resp.Body); err != nil {
		t.Fatalf("Copy returned error: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}
	if !strings.HasSuffix(string(capture.ResponseBody), "\n[truncated]") {
		t.Fatalf("captured response missing truncation marker, suffix %q", string(capture.ResponseBody[len(capture.ResponseBody)-20:]))
	}
}

func TestNewWireTapBodyLimitOption(t *testing.T) {
	var capture llm.WireCapture
	rt := testutil.RoundTripFunc(func(req *http.Request) (*http.Response, error) {
		if _, err := io.Copy(io.Discard, req.Body); err != nil {
			return nil, err
		}
		return &http.Response{
			StatusCode: 200,
			Body:       io.NopCloser(strings.NewReader("response-body")),
		}, nil
	})

	client := &http.Client{Transport: llm.NewWireTap(rt, "fake", func(c llm.WireCapture) {
		capture = c
	}, llm.WithWireTapBodyLimit(4))}
	resp, err := client.Post("https://example.test/chat", "application/json", strings.NewReader("request-body"))
	if err != nil {
		t.Fatalf("Post returned error: %v", err)
	}
	if _, err := io.Copy(io.Discard, resp.Body); err != nil {
		t.Fatalf("Copy returned error: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}
	if got, want := string(capture.RequestBody), "requ\n[truncated]"; got != want {
		t.Fatalf("request body capture = %q, want %q", got, want)
	}
	if got, want := string(capture.ResponseBody), "resp\n[truncated]"; got != want {
		t.Fatalf("response body capture = %q, want %q", got, want)
	}
}

func TestWireCaptureToLogger(t *testing.T) {
	var b bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&b, &slog.HandlerOptions{Level: slog.LevelDebug}))
	llm.WireCaptureToLogger(logger)(llm.WireCapture{
		Provider:     "fake",
		Method:       http.MethodPost,
		URL:          "https://example.test",
		Status:       200,
		RequestBody:  []byte("request"),
		ResponseBody: []byte("response"),
	})
	got := b.String()
	for _, want := range []string{"llm wire capture", "provider=fake", "request", "response"} {
		if !strings.Contains(got, want) {
			t.Fatalf("log output %q missing %q", got, want)
		}
	}
}

func TestNewRetryLoggerWarnsOnRetryableResponses(t *testing.T) {
	var b bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&b, &slog.HandlerOptions{Level: slog.LevelWarn}))
	rt := testutil.RoundTripFunc(func(req *http.Request) (*http.Response, error) {
		if got := req.Header.Get("X-Stainless-Retry-Count"); got != "1" {
			t.Fatalf("retry count header = %q, want 1", got)
		}
		return &http.Response{
			StatusCode: http.StatusTooManyRequests,
			Header:     http.Header{"Retry-After": []string{"2"}},
			Body:       io.NopCloser(strings.NewReader("rate limited")),
		}, nil
	})

	client := &http.Client{Transport: llm.NewRetryLogger(rt, "fake", logger)}
	req, err := http.NewRequest(http.MethodPost, "https://example.test/chat", nil)
	if err != nil {
		t.Fatalf("NewRequest returned error: %v", err)
	}
	req.Header.Set("X-Stainless-Retry-Count", "1")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do returned error: %v", err)
	}
	_ = resp.Body.Close()

	got := b.String()
	for _, want := range []string{"level=WARN", "llm provider retryable response", "provider=fake", "status=429", "retry_after=2", "attempt=2"} {
		if !strings.Contains(got, want) {
			t.Fatalf("log output %q missing %q", got, want)
		}
	}
}

func TestUsageTrackerStreamLatencyTelemetry(t *testing.T) {
	p := llmtest.New(llmtest.WithName("fake"))
	// Blocking call: every stream telemetry field must stay zero.
	p.EnqueueResponse(&llm.Response{Provider: "fake", Model: "model-a", Usage: llm.Usage{TotalTokens: 1}})
	// Text stream: MessageStart and first-content samples.
	p.EnqueueStream(
		llm.MessageStart{Provider: "fake", Model: "model-a"},
		llm.TextDelta{Index: 0, Text: "hello"},
		llm.MessageEnd{Usage: llm.Usage{TotalTokens: 2}},
	)
	// Tool-call stream: ToolCallStart counts as first content.
	p.EnqueueStream(
		llm.MessageStart{Provider: "fake", Model: "model-a"},
		llm.ToolCallStart{Index: 0, ID: "call_1", Name: "lookup"},
		llm.ToolCallEnd{Index: 0},
		llm.MessageEnd{Usage: llm.Usage{TotalTokens: 2}},
	)
	// Empty deltas are not content: MessageStart sample only.
	p.EnqueueStream(
		llm.MessageStart{Provider: "fake", Model: "model-a"},
		llm.TextDelta{Index: 0, Text: ""},
		llm.MessageEnd{Usage: llm.Usage{TotalTokens: 2}},
	)
	// Error-only stream: a stream call with no samples at all.
	p.EnqueueError(errors.New("boom"))

	tracker := llm.NewUsageTracker()
	wrapped := llm.Wrap(p, tracker.Middleware())
	ctx := context.Background()
	req := func() *llm.Request {
		return &llm.Request{Model: "model-a", Messages: []llm.Message{llm.UserText("hi")}}
	}

	if _, err := wrapped.Chat(ctx, req()); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	afterChat := tracker.Stats()
	if afterChat.StreamCalls != 0 || afterChat.MessageStartSamples != 0 || afterChat.TotalTimeToMessageStart != 0 ||
		afterChat.FirstContentSamples != 0 || afterChat.TotalTimeToFirstContent != 0 {
		t.Fatalf("blocking call produced stream telemetry: %+v", afterChat)
	}

	for i := 0; i < 3; i++ {
		if _, err := llm.Collect(wrapped.ChatStream(ctx, req())); err != nil {
			t.Fatalf("Collect %d returned error: %v", i, err)
		}
	}
	if _, err := llm.Collect(wrapped.ChatStream(ctx, req())); err == nil {
		t.Fatal("error stream did not error")
	}

	stats := tracker.Stats()
	if stats.StreamCalls != 4 {
		t.Fatalf("StreamCalls = %d, want 4", stats.StreamCalls)
	}
	if stats.MessageStartSamples != 3 || stats.TotalTimeToMessageStart <= 0 {
		t.Fatalf("MessageStart samples/total = %d/%v, want 3/>0", stats.MessageStartSamples, stats.TotalTimeToMessageStart)
	}
	if stats.FirstContentSamples != 2 || stats.TotalTimeToFirstContent <= 0 {
		t.Fatalf("FirstContent samples/total = %d/%v, want 2/>0", stats.FirstContentSamples, stats.TotalTimeToFirstContent)
	}
	bucket := stats.ByProviderModel["fake/model-a"]
	if bucket.StreamCalls != 4 || bucket.MessageStartSamples != 3 || bucket.FirstContentSamples != 2 {
		t.Fatalf("bucket telemetry = %+v", bucket)
	}
}
