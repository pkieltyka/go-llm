package llm

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"sync"
)

// GuardConfig configures ContextGuard.
type GuardConfig struct {
	// MaxOccupancy is the prompt fraction of the context window
	// (0 < MaxOccupancy <= 1) at or above which OnNearFull runs. Zero or a
	// negative value means 0.9; a value above 1 counts as 1 (fire only when
	// the measured prompt fills the window). It is a fraction, not a
	// percentage.
	MaxOccupancy float64
	// Margin is the number of tokens kept free beyond the prompt, both for
	// the exact-overflow check and for ClampMaxTokens. Negative values count
	// as zero.
	Margin int
	// ClampMaxTokens lowers an explicit Request.MaxTokens so prompt + output
	// + Margin fits the window, and to the model's MaxOutputTokens when that
	// is known and smaller. It only ever lowers: an unset MaxTokens stays
	// unset, because only the provider knows its effective default (raising
	// it can change provider behavior, such as Anthropic requiring streaming
	// for large non-streaming budgets). Opt-in: go-llm never clamps silently.
	ClampMaxTokens bool
	// OnNearFull, when set, runs before the call once occupancy reaches
	// MaxOccupancy. Returning an error aborts the call with that error;
	// returning nil lets it proceed. It must not mutate req: the guard does
	// not re-measure after the hook. To compact, return an error, rebuild the
	// request, and retry; go-llm ships no compaction policy. This is the
	// go-agent compaction trigger.
	OnNearFull func(ctx context.Context, req *Request, report GuardReport) error
	// Window overrides the context window in tokens. Zero resolves it per
	// model through ResolveModelInfo on the bound provider (ClampMaxTokens
	// also resolves it for MaxOutputTokens). Successful lookups and unknown
	// models are cached; a transient lookup failure is retried on the next
	// request. A model whose window is unknown passes through unguarded.
	Window int64
}

// GuardReport describes one guarded request.
type GuardReport struct {
	// Tokens is the measured prompt size and Exact whether a provider
	// TokenCounter computed it (false marks an EstimateRequestTokens
	// estimate, including a fallback after a counter error).
	Tokens int64
	Exact  bool
	// Window is the model context window in tokens.
	Window int64
	// Occupancy is Tokens / Window.
	Occupancy float64
}

// ContextGuard returns middleware that measures each request before it is
// sent and enforces GuardConfig. Measurement uses the bound provider's
// TokenCounter when it has one (see CountTokens; a counter error falls back
// to the estimate) and EstimateRequestTokens otherwise.
//
// Per request, in order:
//   - an EXACT count with prompt + Margin at or above the window fails before
//     the network call with an error wrapping ErrContextTooLong; an estimate
//     never fails a request, because only the server knows;
//   - at or above MaxOccupancy, OnNearFull runs, and its error aborts;
//   - with ClampMaxTokens, MaxTokens is lowered (on a copy of the request)
//     to fit, when the count says it can fit.
//
// The guard adds a counting call per request (for providers with a
// TokenCounter), visible as its own wire capture; it is never part of Chat
// or ChatStream implicitly. The caller's Request is never mutated. For
// ChatStream the check runs when the stream is iterated, like the provider
// call it guards.
//
// Ordering: the guard measures the request it receives, while CountTokens
// asks the underlying provider directly. Place ContextGuard last among the
// Wrap arguments (innermost) so that middleware which rewrites requests
// runs before it and the measured request is the one sent.
func ContextGuard(cfg GuardConfig) Middleware {
	return Middleware{Bind: func(p Provider) Middleware {
		g := &contextGuard{cfg: cfg, provider: p, models: map[string]ModelInfo{}}
		if g.cfg.MaxOccupancy <= 0 {
			g.cfg.MaxOccupancy = 0.9
		} else if g.cfg.MaxOccupancy > 1 {
			g.cfg.MaxOccupancy = 1
		}
		if g.cfg.Margin < 0 {
			g.cfg.Margin = 0
		}
		return Middleware{
			Chat: func(next ChatFunc) ChatFunc {
				return func(ctx context.Context, req *Request) (*Response, error) {
					guarded, err := g.check(ctx, req)
					if err != nil {
						return nil, err
					}
					return next(ctx, guarded)
				}
			},
			Stream: func(next StreamFunc) StreamFunc {
				return func(ctx context.Context, req *Request) iter.Seq2[Event, error] {
					return func(yield func(Event, error) bool) {
						guarded, err := g.check(ctx, req)
						if err != nil {
							yield(nil, err)
							return
						}
						for event, err := range next(ctx, guarded) {
							if !yield(event, err) {
								return
							}
						}
					}
				}
			},
		}
	}}
}

type contextGuard struct {
	cfg      GuardConfig
	provider Provider

	mu     sync.Mutex
	models map[string]ModelInfo
}

func (g *contextGuard) check(ctx context.Context, req *Request) (*Request, error) {
	if req == nil {
		return req, nil
	}
	var info ModelInfo
	if g.cfg.Window <= 0 || g.cfg.ClampMaxTokens {
		info = g.modelInfo(ctx, req.Model)
	}
	window := g.cfg.Window
	if window <= 0 {
		window = int64(info.ContextWindow)
	}
	if window <= 0 {
		return req, nil
	}
	count, err := CountTokens(ctx, g.provider, req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		count = TokenCount{Tokens: EstimateRequestTokens(req)}
	}
	report := GuardReport{
		Tokens:    count.Tokens,
		Exact:     count.Exact,
		Window:    window,
		Occupancy: float64(count.Tokens) / float64(window),
	}
	if count.Exact && count.Tokens+int64(g.cfg.Margin) >= window {
		return nil, fmt.Errorf("%w: context guard: prompt is %d tokens (exact) for a %d-token window with margin %d", ErrContextTooLong, count.Tokens, window, g.cfg.Margin)
	}
	if report.Occupancy >= g.cfg.MaxOccupancy && g.cfg.OnNearFull != nil {
		if err := g.cfg.OnNearFull(ctx, req, report); err != nil {
			return nil, err
		}
	}
	if !g.cfg.ClampMaxTokens {
		return req, nil
	}
	requested := req.MaxTokens
	if requested <= 0 {
		return req, nil
	}
	if info.MaxOutputTokens > 0 && requested > info.MaxOutputTokens {
		requested = info.MaxOutputTokens
	}
	fit, err := FitMaxTokens(count.Tokens, window, requested, g.cfg.Margin)
	if err != nil {
		// An estimate that leaves no room is not grounds to fail or to
		// shrink the budget to nothing; the server decides. The model's
		// output limit is known independently, so it still applies.
		fit = requested
	}
	if fit == req.MaxTokens {
		return req, nil
	}
	clamped := *req
	clamped.MaxTokens = fit
	return &clamped, nil
}

func (g *contextGuard) modelInfo(ctx context.Context, model string) ModelInfo {
	g.mu.Lock()
	info, cached := g.models[model]
	g.mu.Unlock()
	if cached {
		return info
	}
	info, err := ResolveModelInfo(ctx, g.provider, model)
	if err != nil && !errors.Is(err, ErrNotFound) {
		// Transient failures (network, rate limit, cancellation) are retried
		// on the next request rather than disabling the guard for good.
		return ModelInfo{}
	}
	g.mu.Lock()
	g.models[model] = info
	g.mu.Unlock()
	return info
}
