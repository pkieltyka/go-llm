package anthropic

import (
	"context"
	"encoding/json"

	sdk "github.com/anthropics/anthropic-sdk-go"
	sdkparam "github.com/anthropics/anthropic-sdk-go/packages/param"
	llm "github.com/pkieltyka/go-llm"
)

// countTokensFields are the Messages request fields that the count_tokens
// endpoint accepts; generation-only fields (max_tokens, sampling, stop
// sequences, metadata, service tier) are not sent.
var countTokensFields = []string{
	"model", "messages", "system", "tools", "tool_choice", "thinking", "output_config",
}

var _ llm.TokenCounter = (*Provider)(nil)

// CountTokens implements llm.TokenCounter with Anthropic's POST
// /v1/messages/count_tokens endpoint. The request is validated and converted
// exactly like a Chat call (messages, system, tools, tool choice, effort and
// thinking, provider options and beta headers), then reduced to the fields
// the endpoint accepts, so the count reflects what Chat would send. The
// result is the provider's exact input-token count. It is never called
// implicitly by Chat or ChatStream.
func (p *Provider) CountTokens(ctx context.Context, req *llm.Request) (llm.TokenCount, error) {
	ctx, cancel := p.contextWithTimeout(ctx)
	defer cancel()

	params, requestOptions, err := p.buildParams(req)
	if err != nil {
		return llm.TokenCount{}, err
	}
	wire, err := json.Marshal(params)
	if err != nil {
		return llm.TokenCount{}, err
	}
	var full map[string]json.RawMessage
	if err := json.Unmarshal(wire, &full); err != nil {
		return llm.TokenCount{}, err
	}
	body := make(map[string]json.RawMessage, len(countTokensFields))
	for _, field := range countTokensFields {
		if value, ok := full[field]; ok {
			body[field] = value
		}
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return llm.TokenCount{}, err
	}
	result, err := p.client.Messages.CountTokens(ctx, sdkparam.Override[sdk.MessageCountTokensParams](json.RawMessage(raw)), requestOptions...)
	if err != nil {
		return llm.TokenCount{}, mapError(err)
	}
	return llm.TokenCount{Tokens: result.InputTokens, Exact: true}, nil
}
