package providerutil

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"net"

	llm "github.com/pkieltyka/go-llm"
)

// NormalizeRemoteError preserves cancellation and provider errors while
// bringing unknown remote transport and decode failures into the shared
// two-layer error contract.
func NormalizeRemoteError(provider string, err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	var providerErr *llm.ProviderError
	if errors.As(err, &providerErr) {
		return err
	}

	kind := llm.ErrServer
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		kind = llm.ErrTimeout
	}
	return &llm.ProviderError{
		Provider: provider,
		Message:  err.Error(),
		Kind:     errors.Join(kind, err),
	}
}

// StreamContract enforces the provider stream grammar for a remote stream.
// A fully exhausted successful stream has exactly one MessageStart as its
// first event and exactly one MessageEnd as its last event. Consumer early
// exit is not upstream exhaustion and therefore never produces a synthetic
// truncation error.
func StreamContract(provider string, seq iter.Seq2[llm.Event, error]) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		if seq == nil {
			yield(nil, streamContractError(provider, "nil stream"))
			return
		}

		var started, ended, failed, downstreamStopped bool
		seq(func(event llm.Event, err error) bool {
			if failed || downstreamStopped || ended {
				return false
			}
			if err != nil {
				failed = true
				yield(nil, NormalizeRemoteError(provider, err))
				return false
			}

			normalized := DerefEvent(event)
			if normalized == nil {
				failed = true
				yield(nil, streamContractError(provider, "nil stream event"))
				return false
			}
			_, isStart := normalized.(llm.MessageStart)
			_, isEnd := normalized.(llm.MessageEnd)
			switch {
			case !started && !isStart:
				failed = true
				yield(nil, streamContractError(provider, fmt.Sprintf("first event is %T, want llm.MessageStart", normalized)))
				return false
			case started && isStart:
				failed = true
				yield(nil, streamContractError(provider, "duplicate MessageStart"))
				return false
			}

			started = true
			ended = isEnd
			if !yield(event, nil) {
				downstreamStopped = true
				return false
			}
			if ended {
				return false
			}
			return true
		})

		if failed || downstreamStopped {
			return
		}
		if !started {
			yield(nil, streamContractError(provider, "stream ended before MessageStart"))
			return
		}
		if !ended {
			yield(nil, streamContractError(provider, "stream ended before MessageEnd"))
		}
	}
}

func streamContractError(provider, message string) error {
	return &llm.ProviderError{
		Provider: provider,
		Message:  message,
		Kind:     llm.ErrServer,
	}
}
