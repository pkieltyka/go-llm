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
	chat := ChatFunc(p.Chat)
	stream := StreamFunc(p.ChatStream)
	for i := len(bound) - 1; i >= 0; i-- {
		if bound[i].Chat != nil {
			chat = bound[i].Chat(chat)
		}
		if bound[i].Stream != nil {
			stream = bound[i].Stream(stream)
		}
	}
	return &wrappedProvider{provider: p, chat: chat, stream: stream}
}

type wrappedProvider struct {
	provider Provider
	chat     ChatFunc
	stream   StreamFunc
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
	return p.chat(ctx, req)
}

func (p *wrappedProvider) ChatStream(ctx context.Context, req *Request) iter.Seq2[Event, error] {
	return p.stream(ctx, req)
}
