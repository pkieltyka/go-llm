package llm

import (
	"context"
	"iter"
)

// ChatFunc is the function shape middleware uses to decorate Chat.
type ChatFunc func(ctx context.Context, req *Request) (*Response, error)

// StreamFunc is the function shape middleware uses to decorate ChatStream.
type StreamFunc func(ctx context.Context, req *Request) iter.Seq2[Event, error]

// Middleware optionally decorates Chat and ChatStream.
type Middleware struct {
	Chat   func(next ChatFunc) ChatFunc
	Stream func(next StreamFunc) StreamFunc
	// Bind makes middleware provider-aware: when set, Wrap calls it once
	// with the provider being wrapped and composes the returned Middleware
	// instead (its own Bind is ignored). Use it when handlers need the
	// wrapped provider's identity or capabilities — UsageTracker.Middleware
	// binds this way to label its buckets, and third-party middleware can
	// do the same.
	Bind func(p Provider) Middleware
}

// Wrap returns p decorated by middleware. The first middleware argument is
// outermost. Name, Capabilities, and Models delegate to p unchanged. Wrap also
// binds provider-aware middleware (Middleware.Bind) to p once before
// composing handlers.
func Wrap(p Provider, mw ...Middleware) Provider {
	if p == nil || len(mw) == 0 {
		return p
	}
	bound := make([]Middleware, len(mw))
	for i, middleware := range mw {
		if middleware.Bind != nil {
			middleware = middleware.Bind(p)
		}
		bound[i] = middleware
	}
	return &wrappedProvider{provider: p, middleware: bound}
}

type wrappedProvider struct {
	provider   Provider
	middleware []Middleware
}

func (p *wrappedProvider) Name() string {
	return p.provider.Name()
}

func (p *wrappedProvider) Capabilities() []Capability {
	return p.provider.Capabilities()
}

func (p *wrappedProvider) Models(ctx context.Context) ([]ModelInfo, error) {
	return p.provider.Models(ctx)
}

func (p *wrappedProvider) Chat(ctx context.Context, req *Request) (*Response, error) {
	next := p.provider.Chat
	for i := len(p.middleware) - 1; i >= 0; i-- {
		if p.middleware[i].Chat != nil {
			next = p.middleware[i].Chat(next)
		}
	}
	return next(ctx, req)
}

func (p *wrappedProvider) ChatStream(ctx context.Context, req *Request) iter.Seq2[Event, error] {
	next := p.provider.ChatStream
	for i := len(p.middleware) - 1; i >= 0; i-- {
		if p.middleware[i].Stream != nil {
			next = p.middleware[i].Stream(next)
		}
	}
	return next(ctx, req)
}
