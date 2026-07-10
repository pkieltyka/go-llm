package llm_test

import (
	"bytes"
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
	"github.com/pkieltyka/go-llm/llmtest"
)

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
			Header:     http.Header{"Set-Cookie": []string{"secret=1"}},
			Body:       io.NopCloser(strings.NewReader("data: hello\n\n")),
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
	for _, name := range []string{"Authorization", "X-Api-Key"} {
		if got := capture.RequestHeaders.Get(name); got != "[REDACTED]" {
			t.Fatalf("%s capture = %q, want redacted", name, got)
		}
	}
	if got := capture.ResponseHeaders.Get("Set-Cookie"); got != "[REDACTED]" {
		t.Fatalf("Set-Cookie capture = %q, want redacted", got)
	}
	if string(capture.RequestBody) != `{"prompt":"hi"}` || string(capture.ResponseBody) != "data: hello\n\n" {
		t.Fatalf("capture bodies = request %q response %q", capture.RequestBody, capture.ResponseBody)
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
