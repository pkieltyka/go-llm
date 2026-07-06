package main

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"
)

// Regression for the SIGINT hang: a stdin read blocked on a held-open,
// never-EOF pipe must return promptly once the signal context is canceled
// instead of blocking in io.ReadAll until SIGKILL.
func TestReadAllContextReturnsOnCancelDuringBlockedRead(t *testing.T) {
	r, w := io.Pipe() // no writes, no close: Read blocks forever
	defer w.Close()
	defer r.Close()

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	type result struct {
		text string
		err  error
	}
	done := make(chan result, 1)
	go func() {
		text, err := readAllContext(ctx, r)
		done <- result{text, err}
	}()

	select {
	case res := <-done:
		if !errors.Is(res.err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", res.err)
		}
		if res.text != "" {
			t.Fatalf("text = %q, want empty on canceled read", res.text)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("readAllContext still blocked 5s after context cancellation")
	}
}

func TestReadAllContextReadsToEOF(t *testing.T) {
	text, err := readAllContext(context.Background(), strings.NewReader("piped input"))
	if err != nil {
		t.Fatal(err)
	}
	if text != "piped input" {
		t.Fatalf("text = %q", text)
	}
}
