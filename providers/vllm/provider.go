package vllm

import (
	"context"
	"fmt"
	"iter"
	"strings"

	sdk "github.com/openai/openai-go/v3"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
)

// capabilities is honest to what the preset maps; several remain
// server-flag-dependent at runtime (see the package documentation): tool
// calling with "auto" needs --enable-auto-tool-choice + --tool-call-parser,
// reasoning needs --reasoning-parser, and image input needs a served
// multimodal model.
var capabilities = []llm.Capability{
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

// Provider wraps the shared chat-completions engine with the vLLM dialect.
type Provider struct {
	inner *chatcompletions.Provider
	// serverRoot is the base URL with a trailing /v1 segment removed — the
	// tokenizer extension endpoints (/tokenize, /detokenize, /tokenizer_info)
	// live at the server root, outside the OpenAI /v1 prefix (probe-verified:
	// /v1/tokenize is 404).
	serverRoot string
}

// New constructs a vLLM provider for the server at baseURL — vLLM's OpenAI
// surface root, conventionally "http://host:8000/v1". The API key is
// optional (see WithAPIKey).
func New(baseURL string, opts ...Option) (*Provider, error) {
	if baseURL == "" {
		return nil, fmt.Errorf("%w: vllm requires a base URL (e.g. http://localhost:8000/v1)", llm.ErrBadRequest)
	}
	cfg := config{}
	for _, opt := range opts {
		opt(&cfg)
	}
	inner, err := chatcompletions.NewWithDialect(cfg.chatcompletionsConfig(baseURL))
	if err != nil {
		return nil, err
	}
	root := strings.TrimSuffix(strings.TrimRight(baseURL, "/"), "/v1")
	return &Provider{inner: inner, serverRoot: root}, nil
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
