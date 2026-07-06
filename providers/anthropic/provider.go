package anthropic

import (
	"context"
	"iter"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
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
	llm.CapabilityStopSequences,
	llm.CapabilityPromptCaching,
	llm.CapabilityModelsListing,
}

// Name returns the provider identifier.
func (p *Provider) Name() string { return providerName }

// Capabilities returns Anthropic's provider-level capabilities.
func (p *Provider) Capabilities() []llm.Capability {
	return append([]llm.Capability(nil), capabilities...)
}

// Models lists Anthropic models available to the configured account.
func (p *Provider) Models(ctx context.Context) ([]llm.ModelInfo, error) {
	ctx, cancel := p.contextWithTimeout(ctx)
	defer cancel()

	pager := p.client.Models.ListAutoPaging(ctx, sdk.ModelListParams{})
	var models []llm.ModelInfo
	for pager.Next() {
		model := pager.Current()
		models = append(models, llm.ModelInfo{
			ID:              model.ID,
			DisplayName:     model.DisplayName,
			ContextWindow:   int(model.MaxInputTokens),
			MaxOutputTokens: int(model.MaxTokens),
			Raw:             model,
		})
		if info, ok := llm.LookupModelInfo(providerName, model.ID); ok && info.Pricing != nil {
			pricing := *info.Pricing
			models[len(models)-1].Pricing = &pricing
		}
	}
	if err := pager.Err(); err != nil {
		return nil, mapError(err)
	}
	return models, nil
}

// Chat performs a blocking Anthropic Messages request.
func (p *Provider) Chat(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	start := time.Now()
	ctx, cancel := p.contextWithTimeout(ctx)
	defer cancel()

	params, requestOptions, err := p.buildParams(req)
	if err != nil {
		p.logFailure(ctx, req, start, err)
		return nil, err
	}

	msg, err := p.client.Messages.New(ctx, params, requestOptions...)
	if err != nil {
		err = mapError(err)
		p.logFailure(ctx, req, start, err)
		return nil, err
	}
	resp, err := p.mapMessage(msg)
	if err != nil {
		p.logFailure(ctx, req, start, err)
		return nil, err
	}
	p.logSuccess(ctx, resp, start)
	return resp, nil
}

// ChatStream streams Anthropic Messages events as normalized go-llm events.
func (p *Provider) ChatStream(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return providerutil.SingleUse(providerName, func(yield func(llm.Event, error) bool) {
		start := time.Now()
		ctx, cancel := p.contextWithTimeout(ctx)
		defer cancel()

		params, requestOptions, err := p.buildStreamParams(req)
		if err != nil {
			p.logFailure(ctx, req, start, err)
			yield(nil, err)
			return
		}

		stream := p.client.Messages.NewStreaming(ctx, params, requestOptions...)
		defer stream.Close()

		state := newStreamState(p)
		for stream.Next() {
			events, err := state.mapEvent(stream.Current())
			if err != nil {
				err = mapError(err)
				p.logFailure(ctx, req, start, err)
				yield(nil, err)
				return
			}
			for _, event := range events {
				if end, ok := event.(llm.MessageEnd); ok {
					p.logStreamEnd(ctx, req, end, state.model, start)
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
