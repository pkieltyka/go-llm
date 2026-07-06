package openaicodex

import (
	"context"
	"iter"
	"time"

	sdk "github.com/openai/openai-go/v3"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
	"github.com/pkieltyka/go-llm/providers/internal/responsesapi"
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
	llm.CapabilityPDFInput,
	llm.CapabilityPromptCaching,
	llm.CapabilitySessionAffinity,
	llm.CapabilityModelsListing,
}

// staticModels is the curated Codex subscription list (the backend has no
// public models endpoint). Source: pi's generated catalog
// (packages/ai/src/providers/openai-codex.models.ts), verified 2026-07-03.
// Note gpt-5.3-codex-spark really does publish MaxOutputTokens equal to its
// ContextWindow (128000/128000) in that catalog — not a typo.
var staticModels = []llm.ModelInfo{
	{ID: "gpt-5.3-codex-spark", DisplayName: "GPT-5.3 Codex Spark", ContextWindow: 128000, MaxOutputTokens: 128000},
	{ID: "gpt-5.4-mini", DisplayName: "GPT-5.4 mini", ContextWindow: 272000, MaxOutputTokens: 128000},
	{ID: "gpt-5.4", DisplayName: "GPT-5.4", ContextWindow: 272000, MaxOutputTokens: 128000},
	{ID: "gpt-5.5", DisplayName: "GPT-5.5", ContextWindow: 272000, MaxOutputTokens: 128000},
}

// New constructs an OpenAI Codex subscription provider.
//
// The codex backend accepts a narrower request surface than the public
// Responses API, so three sampling/budget knobs are silently DROPPED from
// outgoing requests instead of failing every call:
//   - Request.MaxTokens (max_output_tokens) — the subscription backend
//     controls the output budget itself.
//   - Request.TopP (top_p) — not part of the codex request surface.
//   - Request.Temperature — live-verified 2026-07-03: the backend returns
//     400 "Unsupported parameter: temperature" for the pinned gpt-5.x
//     models.
func New(opts ...Option) (*Provider, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	source := newCodexOAuthSource(cfg)
	client := sdk.NewClient(cfg.sdkOptions(source)...)
	return &Provider{
		client:     &client,
		transport:  cfg.directTransport(source),
		priceTable: cfg.priceTable,
		logger:     cfg.logger,
		timeout:    cfg.timeout,
	}, nil
}

// Client exposes the underlying OpenAI SDK client.
func (p *Provider) Client() *sdk.Client {
	if p == nil {
		return nil
	}
	return p.client
}

// Name returns the provider identifier.
func (p *Provider) Name() string { return providerName }

// Capabilities returns Codex Responses provider-level capabilities.
func (p *Provider) Capabilities() []llm.Capability {
	return append([]llm.Capability(nil), capabilities...)
}

// Models returns the curated Codex subscription model list.
func (p *Provider) Models(ctx context.Context) ([]llm.ModelInfo, error) {
	models := make([]llm.ModelInfo, len(staticModels))
	for i, model := range staticModels {
		models[i] = model
		if p != nil && p.priceTable != nil {
			models[i].Pricing = priceForModel(p.priceTable, model.ID)
		} else if info, ok := llm.LookupModelInfo(providerName, model.ID); ok && info.Pricing != nil {
			pricing := *info.Pricing
			models[i].Pricing = &pricing
		}
	}
	return models, nil
}

func priceForModel(table llm.PriceTable, model string) *llm.ModelPricing {
	pricing, ok := table.Lookup(providerName, model)
	if !ok {
		return nil
	}
	return &pricing
}

// Chat performs a blocking Codex Responses request.
func (p *Provider) Chat(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	start := time.Now()
	ctx, cancel := p.contextWithTimeout(ctx)
	defer cancel()

	params, err := p.adapter().BuildParams(req, false)
	if err != nil {
		p.logFailure(ctx, req, start, err)
		return nil, err
	}

	response, err := llm.Collect(p.codexEvents(ctx, params))
	if err != nil {
		p.logFailure(ctx, req, start, err)
		return nil, err
	}
	p.logSuccess(ctx, response, start)
	return response, nil
}

// ChatStream streams Codex Responses events as normalized go-llm events.
func (p *Provider) ChatStream(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return providerutil.SingleUse(providerName, func(yield func(llm.Event, error) bool) {
		start := time.Now()
		ctx, cancel := p.contextWithTimeout(ctx)
		defer cancel()

		params, err := p.adapter().BuildParams(req, true)
		if err != nil {
			p.logFailure(ctx, req, start, err)
			yield(nil, err)
			return
		}

		model := ""
		for event, err := range p.codexEvents(ctx, params) {
			if err != nil {
				p.logFailure(ctx, req, start, err)
				yield(nil, err)
				return
			}
			if startEvent, ok := event.(llm.MessageStart); ok {
				model = startEvent.Model
			}
			if end, ok := event.(llm.MessageEnd); ok {
				p.logStreamEnd(ctx, req, end, model, start)
			}
			if !yield(event, nil) {
				return
			}
		}
		if err := ctx.Err(); err != nil {
			p.logFailure(ctx, req, start, err)
			yield(nil, err)
		}
	})
}

// adapter returns the shared Responses mapping configured for this provider.
//
// Decision (ARCH §3.2A): reasoning-item `phase` metadata round-trips via
// ReasoningPart.Raw (the raw reasoning item is replayed verbatim);
// assistant-message-level phase is dropped by the part model and not needed
// for the currently pinned codex models — revisit when adopting
// gpt-5.3-codex+ models that require phase on assistant messages.
func (p *Provider) adapter() responsesapi.Adapter {
	var priceTable llm.PriceTable
	if p != nil {
		priceTable = p.priceTable
	}
	return responsesapi.Adapter{
		ProviderName: providerName,
		Capabilities: capabilities,
		PriceTable:   priceTable,
	}
}

func (p *Provider) contextWithTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	return providerutil.ContextWithTimeout(ctx, p.timeout)
}

func (p *Provider) logSuccess(ctx context.Context, resp *llm.Response, start time.Time) {
	providerutil.LogSuccess(ctx, p.logger, providerName, resp, start)
}

func (p *Provider) logStreamEnd(ctx context.Context, req *llm.Request, end llm.MessageEnd, model string, start time.Time) {
	providerutil.LogStreamEnd(ctx, p.logger, providerName, req, end, model, start)
}

func (p *Provider) logFailure(ctx context.Context, req *llm.Request, start time.Time, err error) {
	providerutil.LogFailure(ctx, p.logger, providerName, req, start, err)
}
