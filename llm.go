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
	// SupportedEfforts enumerates the reasoning Effort levels the model
	// supports, ordered weakest → strongest. A nil slice means the provider
	// omitted the ladder or it is unknown; a non-nil empty slice means the
	// provider supplied a ladder containing no known unified efforts. The
	// metadata is advisory: request forwarding and server-side validation are
	// unchanged, and preflight never rejects an effort based on it.
	SupportedEfforts []Effort
	// DefaultEffort is the provider-advertised default reasoning effort.
	// Empty means unknown. It is advisory and never changes request forwarding.
	DefaultEffort Effort
	// ReasoningRequired reports a positive provider claim that reasoning cannot
	// be disabled for this model. False means false or unknown; it never causes
	// client-side request rejection or rewriting.
	ReasoningRequired bool
	// Capabilities lists capabilities positively advertised for this model.
	// The metadata is advisory: empty means unknown, and request validation
	// continues to use Provider.Capabilities().
	Capabilities []Capability
	Raw          any
}

// ModelPricingTier replaces the base rates for an entire request when its
// prompt occupancy strictly exceeds InputTokensAbove.
type ModelPricingTier struct {
	InputTokensAbove  int64   `json:"input_tokens_above"`
	InputPerMTok      float64 `json:"input_per_mtok"`
	OutputPerMTok     float64 `json:"output_per_mtok"`
	CacheReadPerMTok  float64 `json:"cache_read_per_mtok"`
	CacheWritePerMTok float64 `json:"cache_write_per_mtok"`
}

// ModelPricing stores per-million-token prices in USD. Tiers, when present,
// apply one complete rate set to the entire request based on prompt occupancy.
type ModelPricing struct {
	InputPerMTok      float64            `json:"input_per_mtok"`
	OutputPerMTok     float64            `json:"output_per_mtok"`
	CacheReadPerMTok  float64            `json:"cache_read_per_mtok"`
	CacheWritePerMTok float64            `json:"cache_write_per_mtok"`
	Tiers             []ModelPricingTier `json:"tiers,omitempty"`
}
