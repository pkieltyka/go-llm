package llm

import (
	"context"
	"iter"
)

// Provider is the common interface implemented by all LLM provider clients.
//
// Implementations must be safe for concurrent use. Streams returned by
// ChatStream are single-use iterators.
type Provider interface {
	Name() string
	Capabilities() []Capability
	Models(ctx context.Context) ([]ModelInfo, error)
	Chat(ctx context.Context, req *Request) (*Response, error)
	ChatStream(ctx context.Context, req *Request) iter.Seq2[Event, error]
}

// ModelInfo describes a model exposed by a provider.
type ModelInfo struct {
	ID              string
	CanonicalID     string
	DisplayName     string
	ContextWindow   int
	MaxOutputTokens int
	Pricing         *ModelPricing
	Raw             any
}

// ModelPricing stores per-million-token prices in USD.
type ModelPricing struct {
	InputPerMTok      float64
	OutputPerMTok     float64
	CacheReadPerMTok  float64
	CacheWritePerMTok float64
}
