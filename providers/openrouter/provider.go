package openrouter

import (
	"context"
	"iter"

	sdk "github.com/openai/openai-go/v3"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
	"github.com/pkieltyka/go-llm/providers/internal/systemone"
)

var capabilities = []llm.Capability{
	llm.CapabilityStreaming,
	llm.CapabilityTools,
	llm.CapabilityToolChoiceRequired,
	llm.CapabilityToolStreaming,
	llm.CapabilityParallelTools,
	llm.CapabilityStrictTools,
	llm.CapabilityJSONSchema,
	llm.CapabilityReasoning,
	llm.CapabilityImageInput,
	llm.CapabilityStopSequences,
	llm.CapabilityPromptCaching,
	llm.CapabilitySessionAffinity,
	llm.CapabilityCostReporting,
	llm.CapabilityModelsListing,
	llm.CapabilityClassify,
	llm.Capability("openrouter/routing"),
	llm.Capability("openrouter/plugins"),
}

// Provider wraps the shared OpenAI-compatible adapter for OpenRouter. It
// also classifies with OpenRouter's classifier (decision) models through
// the System One protocol (FS §13C).
type Provider struct {
	inner      *chatcompletions.Provider
	classifier *systemone.Engine
}

var _ llm.Classifier = (*Provider)(nil)

// New constructs an OpenRouter provider.
func New(opts ...Option) (*Provider, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := cfg.resolveAPIKey(); err != nil {
		return nil, err
	}
	compatCfg, err := cfg.chatcompletionsConfig()
	if err != nil {
		return nil, err
	}
	inner, err := chatcompletions.NewWithDialect(compatCfg)
	if err != nil {
		return nil, err
	}
	classifier, err := cfg.classifier(compatCfg)
	if err != nil {
		return nil, err
	}
	return &Provider{inner: inner, classifier: classifier}, nil
}

// Client exposes the underlying OpenAI SDK client.
func (p *Provider) Client() *sdk.Client {
	if p == nil || p.inner == nil {
		return nil
	}
	return p.inner.Client()
}

func (p *Provider) Name() string { return providerName }

func (p *Provider) Capabilities() []llm.Capability {
	return append([]llm.Capability(nil), capabilities...)
}

func (p *Provider) Models(ctx context.Context) ([]llm.ModelInfo, error) {
	return p.inner.Models(ctx)
}

func (p *Provider) Chat(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	return p.inner.Chat(ctx, req)
}

func (p *Provider) ChatStream(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return p.inner.ChatStream(ctx, req)
}

// Classify answers req with an OpenRouter classifier model (for example
// "typesafe/jev-1.14", "cloudflare/clef-flash" or
// "openai/gpt-6-luna-decisions") over POST /api/v1/systemone, using the same
// key, base URL and HTTP client as chat. Images go as image_url parts in the
// state; OpenRouter reports the cost natively.
func (p *Provider) Classify(ctx context.Context, req *llm.ClassifyRequest) (*llm.ClassifyResult, error) {
	return p.classifier.Classify(ctx, req)
}

func (p *Provider) buildParams(req *llm.Request, stream bool) (sdk.ChatCompletionNewParams, error) {
	return p.inner.BuildParams(req, stream)
}
