package e2e

import (
	"context"
	"encoding/json"
	"fmt"

	llm "github.com/pkieltyka/go-llm"
)

// probeOpenRouterReasoningEvidence retries only successful responses that
// lack OpenRouter's structured reasoning_details evidence. OpenRouter may
// route identical requests to different upstreams, some of which return the
// model's thinking as ordinary content even when reasoning is requested.
// Transport and provider errors remain fail-fast to avoid retrying requests
// whose billing state is unknown.
func probeOpenRouterReasoningEvidence(ctx context.Context, attempts int, call func(context.Context) (*llm.Response, error)) (*llm.Response, error) {
	if attempts <= 0 || call == nil {
		return nil, fmt.Errorf("invalid openrouter reasoning probe configuration")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	var last *llm.Response
	for attempt := 1; attempt <= attempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return last, err
		}
		resp, err := call(ctx)
		if err != nil {
			return resp, fmt.Errorf("openrouter reasoning probe %d/%d: %w", attempt, attempts, err)
		}
		if resp == nil {
			return nil, fmt.Errorf("openrouter reasoning probe %d/%d returned a nil response", attempt, attempts)
		}
		last = resp
		if hasOpenRouterReasoningRaw(resp) {
			return resp, nil
		}
	}

	return last, fmt.Errorf(
		"openrouter reasoning probes returned no structured reasoning after %d attempts (text_len=%d stop_reason=%s reasoning_tokens=%d)",
		attempts, len(last.Text()), last.StopReason, last.Usage.ReasoningTokens,
	)
}

func hasOpenRouterReasoningRaw(resp *llm.Response) bool {
	if resp == nil {
		return false
	}
	for _, part := range resp.Parts {
		// Parts are value types (adapters never emit pointer parts).
		if reasoning, ok := part.(llm.ReasoningPart); ok && reasoning.Provider == "openrouter" && len(reasoning.Raw) > 0 && json.Valid(reasoning.Raw) {
			return true
		}
	}
	return false
}
