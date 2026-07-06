package openai

import (
	"context"
	"iter"
	"time"

	"github.com/openai/openai-go/v3/responses"
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

// Name returns the provider identifier.
func (p *Provider) Name() string { return providerName }

// Capabilities returns OpenAI Responses provider-level capabilities.
func (p *Provider) Capabilities() []llm.Capability {
	return append([]llm.Capability(nil), capabilities...)
}

// adapter returns the shared Responses mapping configured for this provider.
func (p *Provider) adapter() responsesapi.Adapter {
	var priceTable llm.PriceTable
	if p != nil {
		priceTable = p.priceTable
	}
	return responsesapi.Adapter{
		ProviderName: providerName,
		Capabilities: capabilities,
		PriceTable:   priceTable,
		ApplyOptions: func(req *llm.Request, params *responses.ResponseNewParams) error {
			options, err := requestOptions(req)
			if err != nil {
				return err
			}
			applyOptions(options, params)
			return nil
		},
	}
}

// mapError normalizes OpenAI SDK errors; error mapping does not depend on
// provider configuration.
func mapError(err error) error {
	return responsesapi.Adapter{ProviderName: providerName}.MapError(err)
}

// Models lists OpenAI models available to the configured account.
func (p *Provider) Models(ctx context.Context) ([]llm.ModelInfo, error) {
	ctx, cancel := p.contextWithTimeout(ctx)
	defer cancel()

	pager := p.client.Models.ListAutoPaging(ctx)
	var models []llm.ModelInfo
	for pager.Next() {
		model := pager.Current()
		info := llm.ModelInfo{
			ID:  model.ID,
			Raw: model,
		}
		if embedded, ok := llm.LookupModelInfo(providerName, model.ID); ok {
			info.DisplayName = embedded.DisplayName
			info.ContextWindow = embedded.ContextWindow
			info.MaxOutputTokens = embedded.MaxOutputTokens
			if embedded.Pricing != nil {
				pricing := *embedded.Pricing
				info.Pricing = &pricing
			}
		}
		models = append(models, info)
	}
	if err := pager.Err(); err != nil {
		return nil, mapError(err)
	}
	return models, nil
}

// Chat performs a blocking OpenAI Responses request.
func (p *Provider) Chat(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	start := time.Now()
	ctx, cancel := p.contextWithTimeout(ctx)
	defer cancel()

	params, err := p.adapter().BuildParams(req, false)
	if err != nil {
		p.logFailure(ctx, req, start, err)
		return nil, err
	}

	response, err := p.client.Responses.New(ctx, params)
	if err != nil {
		err = mapError(err)
		p.logFailure(ctx, req, start, err)
		return nil, err
	}
	resp, err := p.adapter().MapResponse(response)
	if err != nil {
		p.logFailure(ctx, req, start, err)
		return nil, err
	}
	p.logSuccess(ctx, resp, start)
	return resp, nil
}

// ChatStream streams OpenAI Responses events as normalized go-llm events.
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

		stream := p.client.Responses.NewStreaming(ctx, params)
		defer stream.Close()

		state := p.adapter().NewStreamState()
		for stream.Next() {
			events, err := state.MapEvent(stream.Current())
			if err != nil {
				err = mapError(err)
				p.logFailure(ctx, req, start, err)
				yield(nil, err)
				return
			}
			for _, event := range events {
				if end, ok := event.(llm.MessageEnd); ok {
					p.logStreamEnd(ctx, req, end, state.Model(), start)
				}
				if !yield(event, nil) {
					return
				}
			}
		}
		if err := stream.Err(); err != nil {
			err = mapError(err)
			p.logFailure(ctx, req, start, err)
			yield(nil, err)
		}
	})
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
