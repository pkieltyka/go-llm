package providerutil

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestStreamContractSuccess(t *testing.T) {
	stream := StreamContract("test", eventStream(
		llm.MessageStart{Provider: "test", Model: "model"},
		llm.TextDelta{Index: 0, Text: "ok"},
		llm.MessageEnd{StopReason: llm.StopReasonEndTurn},
	))
	resp, err := llm.Collect(stream)
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if got := resp.Text(); got != "ok" {
		t.Fatalf("response text = %q, want ok", got)
	}
}

func TestStreamContractRejectsEmptyAndTruncatedEOF(t *testing.T) {
	tests := []struct {
		name string
		seq  iter.Seq2[llm.Event, error]
	}{
		{name: "empty", seq: eventStream()},
		{name: "truncated", seq: eventStream(
			llm.MessageStart{Provider: "test", Model: "model"},
			llm.TextDelta{Index: 0, Text: "partial"},
		)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := llm.Collect(StreamContract("test", tt.seq))
			if !errors.Is(err, llm.ErrServer) {
				t.Fatalf("error = %v, want ErrServer", err)
			}
			var providerErr *llm.ProviderError
			if !errors.As(err, &providerErr) || providerErr.Provider != "test" {
				t.Fatalf("error = %#v, want test ProviderError", err)
			}
		})
	}
}

func TestStreamContractEarlyBreakDoesNotReportTruncation(t *testing.T) {
	upstreamStopped := false
	seq := func(yield func(llm.Event, error) bool) {
		if !yield(llm.MessageStart{Provider: "test", Model: "model"}, nil) {
			upstreamStopped = true
			return
		}
		yield(llm.TextDelta{Index: 0, Text: "unreachable"}, nil)
	}
	count := 0
	for _, err := range StreamContract("test", seq) {
		if err != nil {
			t.Fatalf("early break yielded error: %v", err)
		}
		count++
		break
	}
	if count != 1 || !upstreamStopped {
		t.Fatalf("events = %d, upstream stopped = %v", count, upstreamStopped)
	}
}

func TestStreamContractIgnoresErrorAfterMessageEnd(t *testing.T) {
	postEndCalled := false
	seq := func(yield func(llm.Event, error) bool) {
		yield(llm.MessageStart{Provider: "test", Model: "model"}, nil)
		yield(llm.MessageEnd{StopReason: llm.StopReasonEndTurn}, nil)
		postEndCalled = true
		yield(nil, errors.New("post-end failure"))
	}
	resp, err := llm.Collect(StreamContract("test", seq))
	if err != nil {
		t.Fatalf("Collect returned post-end error: %v", err)
	}
	if resp == nil || resp.StopReason != llm.StopReasonEndTurn || !postEndCalled {
		t.Fatalf("response/post-end producer = %#v/%v", resp, postEndCalled)
	}
}

func TestStreamContractIgnoresEventAfterMessageEnd(t *testing.T) {
	seq := func(yield func(llm.Event, error) bool) {
		yield(llm.MessageStart{Provider: "test", Model: "model"}, nil)
		yield(llm.MessageEnd{StopReason: llm.StopReasonEndTurn}, nil)
		yield(llm.TextDelta{Index: 0, Text: "late"}, nil)
	}
	resp, err := llm.Collect(StreamContract("test", seq))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if resp.Text() != "" {
		t.Fatalf("post-end text = %q, want ignored", resp.Text())
	}
}

func TestNormalizeRemoteError(t *testing.T) {
	providerErr := &llm.ProviderError{Provider: "test", Kind: llm.ErrRateLimited, Message: "slow"}
	for _, tt := range []struct {
		name string
		err  error
		want error
	}{
		{name: "canceled", err: context.Canceled, want: context.Canceled},
		{name: "deadline", err: context.DeadlineExceeded, want: context.DeadlineExceeded},
		{name: "provider", err: providerErr, want: llm.ErrRateLimited},
		{name: "decode", err: fmt.Errorf("decode failed"), want: llm.ErrServer},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := NormalizeRemoteError("test", tt.err)
			if !errors.Is(got, tt.want) {
				t.Fatalf("NormalizeRemoteError = %v, want %v", got, tt.want)
			}
			if tt.name == "decode" {
				var normalized *llm.ProviderError
				if !errors.As(got, &normalized) {
					t.Fatalf("error = %T, want ProviderError", got)
				}
			}
		})
	}
}

func TestNormalizeRemoteErrorPreservesOriginalCause(t *testing.T) {
	cause := errors.New("persist credential")
	err := NormalizeRemoteError("test", cause)

	if !errors.Is(err, llm.ErrServer) {
		t.Fatalf("NormalizeRemoteError() = %v, want ErrServer", err)
	}
	if !errors.Is(err, cause) {
		t.Fatalf("NormalizeRemoteError() = %v, want original cause", err)
	}
}

func eventStream(events ...llm.Event) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		for _, event := range events {
			if !yield(event, nil) {
				return
			}
		}
	}
}
