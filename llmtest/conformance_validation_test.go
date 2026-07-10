package llmtest

import (
	"context"
	"errors"
	"iter"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

type maliciousConformanceProvider struct {
	stream func(context.Context) iter.Seq2[llm.Event, error]
}

func (*maliciousConformanceProvider) Name() string { return "malicious" }

func (*maliciousConformanceProvider) Capabilities() []llm.Capability {
	return []llm.Capability{llm.CapabilityStreaming}
}

func (*maliciousConformanceProvider) Models(context.Context) ([]llm.ModelInfo, error) {
	return nil, nil
}

func (*maliciousConformanceProvider) Chat(context.Context, *llm.Request) (*llm.Response, error) {
	return &llm.Response{}, nil
}

func (p *maliciousConformanceProvider) ChatStream(ctx context.Context, _ *llm.Request) iter.Seq2[llm.Event, error] {
	return p.stream(ctx)
}

func TestSuccessfulStreamGrammarRejectsMaliciousProviders(t *testing.T) {
	injected := errors.New("injected")
	tests := []struct {
		name string
		want string
		seq  iter.Seq2[llm.Event, error]
	}{
		{
			name: "duplicate_message_start",
			want: "MessageStart count",
			seq: streamPairs(
				yieldedPair{event: llm.MessageStart{}},
				yieldedPair{event: llm.TextDelta{Index: 0, Text: "a"}},
				yieldedPair{event: llm.MessageStart{}},
				yieldedPair{event: llm.MessageEnd{}},
			),
		},
		{
			name: "nil_event",
			want: "nil events",
			seq: streamPairs(
				yieldedPair{event: llm.MessageStart{}},
				yieldedPair{},
				yieldedPair{event: llm.MessageEnd{}},
			),
		},
		{
			name: "typed_nil_event",
			want: "nil events",
			seq: streamPairs(
				yieldedPair{event: llm.MessageStart{}},
				yieldedPair{event: (*llm.TextDelta)(nil)},
				yieldedPair{event: llm.MessageEnd{}},
			),
		},
		{
			name: "event_after_message_end",
			want: "after MessageEnd",
			seq: streamPairs(
				yieldedPair{event: llm.MessageStart{}},
				yieldedPair{event: llm.MessageEnd{}},
				yieldedPair{event: llm.TextDelta{Index: 0, Text: "late"}},
			),
		},
		{
			name: "multiple_message_end",
			want: "MessageEnd count",
			seq: streamPairs(
				yieldedPair{event: llm.MessageStart{}},
				yieldedPair{event: llm.MessageEnd{}},
				yieldedPair{event: llm.MessageEnd{}},
			),
		},
		{
			name: "missing_message_end",
			want: "MessageEnd count",
			seq: streamPairs(
				yieldedPair{event: llm.MessageStart{}},
				yieldedPair{event: llm.TextDelta{Index: 0, Text: "unfinished"}},
			),
		},
		{
			name: "event_paired_with_error",
			want: "event+error pairs",
			seq: streamPairs(
				yieldedPair{event: llm.MessageStart{}},
				yieldedPair{event: llm.TextDelta{Index: 0, Text: "bad"}, err: injected},
			),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &maliciousConformanceProvider{
				stream: func(context.Context) iter.Seq2[llm.Event, error] { return tt.seq },
			}
			res := drainStreamWithoutWatchdog(provider.ChatStream(context.Background(), &llm.Request{}))
			err := successfulStreamError(res)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("successfulStreamError = %v, want rejection containing %q", err, tt.want)
			}
		})
	}
}

func TestCancellationProbeRejectsMalformedProviders(t *testing.T) {
	tests := []struct {
		name string
		want string
		seq  func(context.Context) iter.Seq2[llm.Event, error]
	}{
		{
			name: "event_paired_with_cancellation",
			want: "event+error pairs",
			seq: func(ctx context.Context) iter.Seq2[llm.Event, error] {
				return func(yield func(llm.Event, error) bool) {
					if !yield(llm.MessageStart{}, nil) {
						return
					}
					<-ctx.Done()
					yield(llm.TextDelta{Index: 0, Text: "paired"}, ctx.Err())
				}
			},
		},
		{
			name: "typed_nil_event_paired_with_cancellation",
			want: "event+error pairs",
			seq: func(ctx context.Context) iter.Seq2[llm.Event, error] {
				return func(yield func(llm.Event, error) bool) {
					if !yield(llm.MessageStart{}, nil) {
						return
					}
					<-ctx.Done()
					yield((*llm.TextDelta)(nil), ctx.Err())
				}
			},
		},
		{
			name: "event_after_cancellation_error",
			want: "after its terminal error",
			seq: func(ctx context.Context) iter.Seq2[llm.Event, error] {
				return func(yield func(llm.Event, error) bool) {
					if !yield(llm.MessageStart{}, nil) {
						return
					}
					<-ctx.Done()
					yield(nil, ctx.Err())
					yield(llm.TextDelta{Index: 0, Text: "late"}, nil)
				}
			},
		},
		{
			name: "does_not_block_before_cancellation",
			want: "before",
			seq: func(context.Context) iter.Seq2[llm.Event, error] {
				return streamPairs(
					yieldedPair{event: llm.MessageStart{}},
					yieldedPair{event: llm.MessageEnd{}},
				)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			provider := &maliciousConformanceProvider{stream: tt.seq}
			_, err := cancellationProbe(provider.ChatStream(ctx, &llm.Request{}), cancel, 5*time.Millisecond, 250*time.Millisecond)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("cancellationProbe = %v, want rejection containing %q", err, tt.want)
			}
		})
	}
}

func TestCancellationProbeRequiresIteratorCompletion(t *testing.T) {
	release := make(chan struct{})
	finished := make(chan struct{})
	provider := &maliciousConformanceProvider{
		stream: func(ctx context.Context) iter.Seq2[llm.Event, error] {
			return func(yield func(llm.Event, error) bool) {
				defer close(finished)
				if !yield(llm.MessageStart{}, nil) {
					return
				}
				<-ctx.Done()
				<-release
				yield(nil, ctx.Err())
			}
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	_, err := cancellationProbe(provider.ChatStream(ctx, &llm.Request{}), cancel, 5*time.Millisecond, 50*time.Millisecond)
	close(release)
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("cleanable malicious iterator did not exit after release")
	}
	if err == nil || !strings.Contains(err.Error(), "did not terminate") {
		t.Fatalf("cancellationProbe = %v, want iterator non-termination rejection", err)
	}
}

func TestCancellationProbeAcceptsBlockedContextAwareProvider(t *testing.T) {
	provider := &maliciousConformanceProvider{
		stream: func(ctx context.Context) iter.Seq2[llm.Event, error] {
			return func(yield func(llm.Event, error) bool) {
				if !yield(llm.MessageStart{}, nil) {
					return
				}
				<-ctx.Done()
				yield(nil, ctx.Err())
			}
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	if _, err := cancellationProbe(provider.ChatStream(ctx, &llm.Request{}), cancel, 5*time.Millisecond, 250*time.Millisecond); err != nil {
		t.Fatalf("cancellationProbe returned error: %v", err)
	}
}

func streamPairs(pairs ...yieldedPair) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		for _, pair := range pairs {
			if !yield(pair.event, pair.err) {
				return
			}
		}
	}
}
