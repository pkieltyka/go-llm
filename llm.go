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
	// Availability distinguishes an explicitly reported zero/free component
	// from an unknown component. Nil preserves the legacy contract: input and
	// output rates are known, while cache rates are known when non-zero.
	Availability *ModelPricingAvailability `json:"availability,omitempty"`
}

// ModelPricingAvailability records which independently optional base rates a
// provider reported as valid. A non-nil value is authoritative for all four
// components, including explicit zero/free rates.
type ModelPricingAvailability struct {
	InputPerMTok      bool `json:"input_per_mtok,omitempty"`
	OutputPerMTok     bool `json:"output_per_mtok,omitempty"`
	CacheReadPerMTok  bool `json:"cache_read_per_mtok,omitempty"`
	CacheWritePerMTok bool `json:"cache_write_per_mtok,omitempty"`
}

// HasInputPrice reports whether InputPerMTok is known, including when it is
// explicitly zero/free.
func (p *ModelPricing) HasInputPrice() bool {
	return p != nil && (p.Availability == nil || p.Availability.InputPerMTok)
}

// HasOutputPrice reports whether OutputPerMTok is known, including when it is
// explicitly zero/free.
func (p *ModelPricing) HasOutputPrice() bool {
	return p != nil && (p.Availability == nil || p.Availability.OutputPerMTok)
}

// HasCacheReadPrice reports whether CacheReadPerMTok is known, including when
// it is explicitly zero/free.
func (p *ModelPricing) HasCacheReadPrice() bool {
	return p != nil && ((p.Availability == nil && p.CacheReadPerMTok != 0) ||
		(p.Availability != nil && p.Availability.CacheReadPerMTok))
}

// HasCacheWritePrice reports whether CacheWritePerMTok is known, including
// when it is explicitly zero/free.
func (p *ModelPricing) HasCacheWritePrice() bool {
	return p != nil && ((p.Availability == nil && p.CacheWritePerMTok != 0) ||
		(p.Availability != nil && p.Availability.CacheWritePerMTok))
}
