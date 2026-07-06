package chatcompletions

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// WithTimeout must govern DoJSONURL like every other provider call — the
// tokenizer extension family reaches the wire through it (prior gap: the
// three vllm tokenize methods silently ignored the option).
func TestDoJSONURLHonorsWithTimeout(t *testing.T) {
	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	// Cleanups run LIFO: release the handler BEFORE server.Close, which
	// blocks until active handlers return (the r.Context() escape hatch is
	// not reliable here — with an unread POST body the server does not
	// cancel the request context on client disconnect, which deadlocked
	// this test's teardown for the full binary timeout).
	t.Cleanup(server.Close)
	t.Cleanup(func() { close(release) })

	p, err := New(server.URL+"/v1", WithTimeout(50*time.Millisecond), WithMaxRetries(0))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	start := time.Now()
	var out map[string]any
	err = p.DoJSONURL(context.Background(), http.MethodPost, server.URL+"/tokenize", map[string]any{"prompt": "x"}, &out)
	if err == nil {
		t.Fatal("DoJSONURL returned nil error against a stalled server")
	}
	if !errors.Is(err, context.DeadlineExceeded) && !isTimeoutErr(err) {
		t.Fatalf("DoJSONURL error = %v, want deadline-driven timeout", err)
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("DoJSONURL took %v; WithTimeout(50ms) not applied", elapsed)
	}
}

func isTimeoutErr(err error) bool {
	var netErr interface{ Timeout() bool }
	return errors.As(err, &netErr) && netErr.Timeout()
}
