package openrouter

import (
	"context"
	"iter"

	sdk "github.com/openai/openai-go/v3"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
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
	llm.Capability("openrouter/routing"),
	llm.Capability("openrouter/plugins"),
}

// Provider wraps the shared OpenAI-compatible adapter for OpenRouter.
type Provider struct {
	inner *chatcompletions.Provider
}

// New constructs an OpenRouter provider.
func New(opts ...Option) (*Provider, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	compatCfg, err := cfg.chatcompletionsConfig()
	if err != nil {
		return nil, err
	}
	inner, err := chatcompletions.NewWithDialect(compatCfg)
	if err != nil {
		return nil, err
	}
	return &Provider{inner: inner}, nil
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

func (p *Provider) buildParams(req *llm.Request, stream bool) (sdk.ChatCompletionNewParams, error) {
	return p.inner.BuildParams(req, stream)
}
