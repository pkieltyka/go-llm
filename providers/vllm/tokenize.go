package vllm

import (
	"context"
	"encoding/json"
	"net/http"

	llm "github.com/pkieltyka/go-llm"
)

// TokenizeResult is the response of vLLM's POST /tokenize extension: the
// exact token accounting for a request after server-side chat-template
// rendering (messages, tools, generation prompt).
type TokenizeResult struct {
	// Count is the exact prompt token count the request would consume.
	Count int `json:"count"`
	// MaxModelLen is the server's context window for the model.
	MaxModelLen int `json:"max_model_len"`
	// Tokens are the rendered prompt's token ids.
	Tokens []int `json:"tokens"`
}

// ContextUsage bridges the exact count into the unified context-accounting
// type: UsedTokens is the server-computed Count and Window is MaxModelLen —
// ground truth, unlike the estimate-based path that derives occupancy from a
// previous response's Usage plus a price-table context window.
func (r TokenizeResult) ContextUsage() llm.ContextUsage {
	return llm.Usage{InputTokens: int64(r.Count)}.ContextUsage(int64(r.MaxModelLen))
}

// Tokenize is a vLLM extension (not part of the llm.Provider interface): it
// submits the request through the same message/tool conversion as Chat and
// asks the server's POST /tokenize endpoint (at the server root, not under
// /v1) for the exact prompt token count and max_model_len. The request is
// validated and converted exactly like a Chat call — same messages, tools,
// chat_template_kwargs (including Options.EnableThinking), and provider
// option rules — so Count matches what a real request would consume.
//
// The server root is derived by trimming a trailing "/v1" from the base URL
// (http://host:8000/v1 → http://host:8000, proxy prefixes preserved); a
// proxy that forwards only /v1/* cannot reach the tokenize family.
//
// Request.Effort is mirrored the way vLLM's chat handler applies it
// server-side: a non-empty Effort injects chat_template_kwargs.enable_thinking
// (false for EffortNone, true otherwise; live-verified token-count parity on
// a 0.23/0.24-family host). An explicit enable_thinking in
// Options.ChatTemplateKwargs or Options.EnableThinking wins.
func (p *Provider) Tokenize(ctx context.Context, req *llm.Request) (TokenizeResult, error) {
	params, err := p.inner.BuildParams(req, false)
	if err != nil {
		return TokenizeResult{}, err
	}
	wire, err := json.Marshal(params)
	if err != nil {
		return TokenizeResult{}, err
	}
	var chatBody struct {
		Model              json.RawMessage `json:"model"`
		Messages           json.RawMessage `json:"messages"`
		Tools              json.RawMessage `json:"tools"`
		ChatTemplateKwargs map[string]any  `json:"chat_template_kwargs"`
	}
	if err := json.Unmarshal(wire, &chatBody); err != nil {
		return TokenizeResult{}, err
	}
	kwargs := chatBody.ChatTemplateKwargs
	if req.Effort != "" {
		if _, explicit := kwargs["enable_thinking"]; !explicit {
			if kwargs == nil {
				kwargs = map[string]any{}
			}
			kwargs["enable_thinking"] = req.Effort != llm.EffortNone
		}
	}
	body := map[string]any{
		"model":                 chatBody.Model,
		"messages":              chatBody.Messages,
		"add_generation_prompt": true,
	}
	if len(chatBody.Tools) > 0 && string(chatBody.Tools) != "null" {
		body["tools"] = chatBody.Tools
	}
	if len(kwargs) > 0 {
		body["chat_template_kwargs"] = kwargs
	}
	var result TokenizeResult
	if err := p.inner.DoJSONURL(ctx, http.MethodPost, p.serverRoot+"/tokenize", body, &result); err != nil {
		return TokenizeResult{}, err
	}
	return result, nil
}

// Detokenize is a vLLM extension: POST /detokenize renders token ids back to
// the prompt string through the served model's tokenizer.
func (p *Provider) Detokenize(ctx context.Context, tokens []int) (string, error) {
	if tokens == nil {
		tokens = []int{}
	}
	var out struct {
		Prompt string `json:"prompt"`
	}
	if err := p.inner.DoJSONURL(ctx, http.MethodPost, p.serverRoot+"/detokenize", map[string]any{"tokens": tokens}, &out); err != nil {
		return "", err
	}
	return out.Prompt, nil
}

// TokenizerInfo is a vLLM extension: GET /tokenizer_info returns the
// tokenizer class, chat template, and tokenizer config. The payload is raw
// because the schema varies by server version (the upstream model is
// extra="allow" passthrough of the tokenizer config). The endpoint is gated
// behind a server flag on current builds (--enable-tokenizer-info-endpoint);
// hosts without it return llm.ErrNotFound — observed on a 0.23/0.24-family
// host whose /tokenize and /detokenize worked.
func (p *Provider) TokenizerInfo(ctx context.Context) (json.RawMessage, error) {
	var out json.RawMessage
	if err := p.inner.DoJSONURL(ctx, http.MethodGet, p.serverRoot+"/tokenizer_info", nil, &out); err != nil {
		return nil, err
	}
	return out, nil
}
