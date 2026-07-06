package llm

import (
	"context"
	"iter"
)

// Ptr returns a pointer to v. It shortens setting optional pointer-typed
// request fields: req.Temperature = llm.Ptr(0.2).
func Ptr[T any](v T) *T {
	return &v
}

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
	ID string
	// CanonicalID is the upstream provider/model identity for aggregator
	// aliases. It is empty when unknown or identical to the row's provider/ID.
	CanonicalID     string
	DisplayName     string
	ContextWindow   int
	MaxOutputTokens int
	Pricing         *ModelPricing
	Raw             any
}

// ModelPricing stores per-million-token prices in USD.
type ModelPricing struct {
	InputPerMTok      float64 `json:"input_per_mtok"`
	OutputPerMTok     float64 `json:"output_per_mtok"`
	CacheReadPerMTok  float64 `json:"cache_read_per_mtok"`
	CacheWritePerMTok float64 `json:"cache_write_per_mtok"`
}
