// Package chatcompletions implements a reusable go-llm provider over the
// OpenAI Chat Completions wire shape — the de-facto standard surface spoken
// by OpenAI-compatible servers (vLLM, Ollama, llama.cpp, Groq, Together,
// LM Studio, ...).
//
// New connects to any such server by base URL. The API key is OPTIONAL:
// self-hosted servers are commonly keyless, and no environment fallback is
// consulted. Server quirks are declared as data via WithCompat; preset
// packages (providers/openrouter, providers/vllm, providers/ollama) build on
// this engine and pre-configure the quirks for you.
//
//	p, err := chatcompletions.New("http://localhost:8000/v1",
//		chatcompletions.WithName("myserver"),          // provider name in responses/logs
//		chatcompletions.WithAPIKey("sk-..."),          // only if the server requires one
//	)
//
// The full Dialect interface (NewWithDialect) remains available for presets
// that need behavioral hooks; it is an advanced, stability-exempt surface.
package chatcompletions

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

// DefaultProviderName is the provider name used by New when WithName is not
// supplied.
const DefaultProviderName = "chatcompletions"

// defaultCapabilities is the generous baseline declared by New: the engine
// can BUILD requests for all of these, and OpenAI-compatible servers accept
// the spellings (unknown fields are commonly ignored). Narrow the set with
// WithCapabilities when the target server should fail fast instead.
var defaultCapabilities = []llm.Capability{
	llm.CapabilityStreaming,
	llm.CapabilityTools,
	llm.CapabilityToolChoiceRequired,
	llm.CapabilityToolStreaming,
	llm.CapabilityParallelTools,
	llm.CapabilityStrictTools,
	llm.CapabilityJSONSchema,
	llm.CapabilityJSONMode,
	llm.CapabilityReasoning,
	llm.CapabilityImageInput,
	llm.CapabilityStopSequences,
	llm.CapabilityModelsListing,
}

// Option configures New.
type Option func(*options)

type options struct {
	name         string
	capabilities []llm.Capability
	compat       Compat
	config       Config
}

// WithName sets the provider name reported by Name(), Response.Provider,
// logs, and error metadata. Default: "chatcompletions".
func WithName(name string) Option {
	return func(o *options) { o.name = name }
}

// WithCompat declares the server's quirks. The zero Compat targets a plain
// OpenAI-compatible server.
func WithCompat(compat Compat) Option {
	return func(o *options) { o.compat = compat }
}

// WithCapabilities replaces the default capability set. Request validation
// rejects features outside the declared set before any network call.
func WithCapabilities(caps ...llm.Capability) Option {
	return func(o *options) { o.capabilities = append([]llm.Capability(nil), caps...) }
}

// WithAPIKey sets a static API key, sent as an Authorization bearer token.
// Keys are optional for New: without one, requests carry no Authorization
// header (no environment fallback is consulted).
func WithAPIKey(key string) Option {
	return func(o *options) {
		o.config.APIKey = key
		o.config.APIKeyFunc = nil
	}
}

// WithAPIKeyFunc sets a per-request key resolver. It wins over WithAPIKey.
func WithAPIKeyFunc(fn func(context.Context) (string, error)) Option {
	return func(o *options) { o.config.APIKeyFunc = fn }
}

// WithHTTPClient replaces the provider's default HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(o *options) { o.config.HTTPClient = client }
}

// WithMaxRetries bounds retry attempts. Blocking calls delegate the count to
// the OpenAI SDK's retry layer; streaming calls use the adapter's direct SSE
// transport, which applies the same bound to billing-safe pre-stream
// rejections only (retry decisions are made on the response status line
// before any stream bytes are consumed — ARCH §3.3/§3.4).
func WithMaxRetries(n int) Option {
	return func(o *options) { o.config.MaxRetries = &n }
}

// WithTimeout applies a context deadline to provider calls.
func WithTimeout(timeout time.Duration) Option {
	return func(o *options) { o.config.Timeout = timeout }
}

// WithPriceTable enables cost estimation for the server's models. Self-hosted
// servers report no native cost; without a table, Usage.CostUSD stays nil.
func WithPriceTable(table llm.PriceTable) Option {
	return func(o *options) { o.config.PriceTable = table }
}

// WithLogger enables provider-level operational logging.
func WithLogger(logger *slog.Logger) Option {
	return func(o *options) { o.config.Logger = logger }
}

// WithWireCapture enables redacted wire capture.
func WithWireCapture(fn func(llm.WireCapture)) Option {
	return func(o *options) { o.config.WireCapture = fn }
}

// New constructs a provider for any OpenAI-compatible chat-completions
// server at baseURL (e.g. "http://localhost:8000/v1"). The API key is
// optional — see WithAPIKey — and quirks are declared via WithCompat.
func New(baseURL string, opts ...Option) (*Provider, error) {
	o := options{name: DefaultProviderName}
	for _, opt := range opts {
		opt(&o)
	}
	caps := o.capabilities
	if caps == nil {
		caps = defaultCapabilities
	}
	cfg := o.config
	cfg.BaseURL = baseURL
	cfg.KeyOptional = true
	cfg.Dialect = genericDialect{name: o.name, capabilities: caps, compat: o.compat}
	return NewWithDialect(cfg)
}
