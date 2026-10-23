package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"slices"
	"time"

	sdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
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
	llm.CapabilityPositionalInstructions,
	llm.CapabilityPositionalTools,
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
			return applyOptions(options, params)
		},
	}
}

// mapError normalizes OpenAI SDK errors; error mapping does not depend on
// provider configuration.
func mapError(err error) error {
	return responsesapi.Adapter{ProviderName: providerName}.MapError(err)
}

// Models lists the models available to the configured credential: the API's
// {"data": [...]} list for API keys, or the {"models": [...]} catalog that
// Sign in with ChatGPT returns, which also reports context windows, reasoning
// ladders, and image input.
func (p *Provider) Models(ctx context.Context) ([]llm.ModelInfo, error) {
	ctx, cancel := p.contextWithTimeout(ctx)
	defer cancel()

	var body []byte
	if err := p.client.Get(ctx, "models", nil, &body); err != nil {
		return nil, mapError(err)
	}
	var page struct {
		Data   []sdk.Model       `json:"data"`
		Models []json.RawMessage `json:"models"`
	}
	if err := json.Unmarshal(body, &page); err != nil {
		return nil, fmt.Errorf("%w: OpenAI models response was not valid JSON", llm.ErrServer)
	}
	models := make([]llm.ModelInfo, 0, len(page.Data)+len(page.Models))
	for _, model := range page.Data {
		info := llm.ModelInfo{ID: model.ID, Raw: model}
		if embedded, ok := llm.LookupModelInfo(providerName, model.ID); ok {
			info.DisplayName = embedded.DisplayName
			info.ContextWindow = embedded.ContextWindow
			info.MaxOutputTokens = embedded.MaxOutputTokens
			if embedded.Pricing != nil {
				info.Pricing = embedded.Pricing.Clone()
			}
			info.SupportedEfforts = embedded.SupportedEfforts
			info.Behavior = embedded.Behavior
		}
		models = append(models, info)
	}
	for _, raw := range page.Models {
		if info, ok := chatGPTModelInfo(raw); ok {
			models = append(models, info)
		}
	}
	return models, nil
}

// chatGPTModelInfo maps one listed row of the Sign in with ChatGPT catalog.
// Hidden rows (internal review and reserve models) are skipped.
func chatGPTModelInfo(raw json.RawMessage) (llm.ModelInfo, bool) {
	var row struct {
		Slug                     string   `json:"slug"`
		DisplayName              string   `json:"display_name"`
		Visibility               string   `json:"visibility"`
		ContextWindow            int      `json:"context_window"`
		InputModalities          []string `json:"input_modalities"`
		SupportsParallelTools    bool     `json:"supports_parallel_tool_calls"`
		DefaultReasoningLevel    string   `json:"default_reasoning_level"`
		SupportedReasoningLevels []struct {
			Effort string `json:"effort"`
		} `json:"supported_reasoning_levels"`
	}
	if err := json.Unmarshal(raw, &row); err != nil || row.Slug == "" || row.Visibility != "list" {
		return llm.ModelInfo{}, false
	}
	info := llm.ModelInfo{
		ID:            row.Slug,
		DisplayName:   row.DisplayName,
		ContextWindow: row.ContextWindow,
		Raw:           raw,
	}
	if embedded, ok := llm.LookupModelInfo(providerName, row.Slug); ok {
		info.Behavior = embedded.Behavior
	}
	if row.SupportedReasoningLevels != nil {
		info.SupportedEfforts = []llm.Effort{}
		for _, level := range row.SupportedReasoningLevels {
			if effort := llm.Effort(level.Effort); knownEffort(effort) {
				info.SupportedEfforts = append(info.SupportedEfforts, effort)
			}
		}
		info.Capabilities = append(info.Capabilities, llm.CapabilityReasoning)
	}
	if effort := llm.Effort(row.DefaultReasoningLevel); knownEffort(effort) {
		info.DefaultEffort = effort
	}
	if row.SupportsParallelTools {
		info.Capabilities = append(info.Capabilities, llm.CapabilityParallelTools)
	}
	if slices.Contains(row.InputModalities, "image") {
		info.Capabilities = append(info.Capabilities, llm.CapabilityImageInput)
	}
	return info, true
}

func knownEffort(effort llm.Effort) bool {
	switch effort {
	case llm.EffortNone, llm.EffortMinimal, llm.EffortLow, llm.EffortMedium, llm.EffortHigh, llm.EffortXHigh, llm.EffortMax:
		return true
	}
	return false
}

// Chat performs a blocking OpenAI Responses request. With a stored Sign in
// with ChatGPT credential it drains a stream instead, since that path
// accepts only streaming requests.
func (p *Provider) Chat(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	start := time.Now()
	// The timeout covers the credential read too: a file store may wait
	// out another process's write under its lock.
	ctx, cancel := p.contextWithTimeout(ctx)
	defer cancel()
	ctx, signIn, err := p.signInWithChatGPT(ctx)
	if err != nil {
		p.logFailure(ctx, req, start, err)
		return nil, err
	}
	if signIn {
		return llm.Collect(func(yield func(llm.Event, error) bool) { p.stream(ctx, req, true, yield) })
	}

	params, err := p.buildParams(req, false, false)
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
		ctx, cancel := p.contextWithTimeout(ctx)
		defer cancel()
		ctx, signIn, err := p.signInWithChatGPT(ctx)
		if err != nil {
			p.logFailure(ctx, req, time.Now(), err)
			yield(nil, err)
			return
		}
		p.stream(ctx, req, signIn, yield)
	})
}

// stream runs one streaming request; the caller has applied the provider
// timeout to ctx.
func (p *Provider) stream(ctx context.Context, req *llm.Request, signIn bool, yield func(llm.Event, error) bool) {
	start := time.Now()

	params, err := p.buildParams(req, true, signIn)
	if err != nil {
		p.logFailure(ctx, req, start, err)
		yield(nil, err)
		return
	}

	streamCtx, cancelStream := context.WithCancel(ctx)
	defer cancelStream()
	stream := p.client.Responses.NewStreaming(streamCtx, params)
	defer stream.Close()

	state := p.adapter().NewStreamState(string(params.Model))
	remote := providerutil.StreamContract(providerName, func(remoteYield func(llm.Event, error) bool) {
		for stream.Next() {
			events, err := state.MapEvent(stream.Current())
			for _, event := range events {
				if !remoteYield(event, nil) {
					return
				}
			}
			if err != nil {
				remoteYield(nil, err)
				return
			}
		}
		for _, event := range state.Finish() {
			if !remoteYield(event, nil) {
				return
			}
		}
		if err := stream.Err(); err != nil {
			remoteYield(nil, p.adapter().MapError(err))
			return
		}
		if err := ctx.Err(); err != nil {
			remoteYield(nil, err)
		}
	})
	for event, err := range remote {
		if err != nil {
			p.logFailure(ctx, req, start, err)
			yield(nil, err)
			return
		}
		end, ended := providerutil.DerefEvent(event).(llm.MessageEnd)
		if ended {
			p.logStreamEnd(ctx, req, end, state.Model(), start)
		}
		more := yield(event, nil)
		if ended {
			providerutil.DrainStream(stream.Next, cancelStream)
		}
		if !more {
			return
		}
	}
}

// signInWithChatGPT reads the stored credential once per call and reports
// whether it is a Sign in with ChatGPT token. The returned context carries
// the read to the auth middleware, so the request shape and the token come
// from the same credential.
func (p *Provider) signInWithChatGPT(ctx context.Context) (context.Context, bool, error) {
	if p.credentials == nil {
		return ctx, false, nil
	}
	cred, ok, err := readCredential(ctx, p.credentials)
	if err != nil {
		return ctx, false, err
	}
	return context.WithValue(ctx, storedCredentialKey{}, storedCredential{cred: cred, ok: ok}), ok && cred.Type == "oauth", nil
}

// buildParams maps req for the Responses API. Sign in with ChatGPT rejects
// sampling, output-limit, and prompt-cache retention fields, so they are
// omitted on that path, as pi does.
func (p *Provider) buildParams(req *llm.Request, stream, signIn bool) (responses.ResponseNewParams, error) {
	params, err := p.adapter().BuildParams(req, stream)
	if err != nil || !signIn {
		return params, err
	}
	params.Temperature = param.Opt[float64]{}
	params.TopP = param.Opt[float64]{}
	params.MaxOutputTokens = param.Opt[int64]{}
	params.PromptCacheRetention = ""
	params.PromptCacheOptions = responses.ResponseNewParamsPromptCacheOptions{}
	return params, nil
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
