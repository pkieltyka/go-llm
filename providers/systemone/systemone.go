// Package systemone talks to classifier (decision) models over TypeSafe's
// System One protocol (FS §13C): TypeSafe itself (NewTypeSafe), Vercel AI
// Gateway (NewVercelAIGateway), OpenCode Zen (NewOpenCode), and any other
// System One server, such as llama.cpp 0.6.0 and later serving a decision
// model (New). OpenRouter's classifier models are reached through
// providers/openrouter, which classifies beside chat, and Cloudflare Workers
// AI's through providers/cloudflare.
//
// A Provider here only classifies: Chat and ChatStream fail with
// ErrUnsupported. Use it through llm.Classify or llm.NewClassifierModel.
package systemone

import (
	"context"
	"fmt"
	"iter"
	"log/slog"
	"net/http"
	"os"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
	engine "github.com/pkieltyka/go-llm/providers/internal/systemone"
)

// Default base URLs and environment variables of the System One services.
const (
	TypeSafeBaseURL        = "https://api.typesafe.ai/v1"
	VercelAIGatewayBaseURL = "https://ai-gateway.vercel.sh/typesafe/v1"
	OpenCodeBaseURL        = "https://opencode.ai/inference/v1"

	typeSafeKeyEnv = "TYPESAFE_API_KEY"
	vercelKeyEnv   = "AI_GATEWAY_API_KEY"
	openCodeKeyEnv = "OPENCODE_API_KEY"
)

// ImageForm is how a System One service takes images.
type ImageForm = engine.ImageForm

// Image forms (see the internal engine for the wire details).
const (
	// ImageAssets is TypeSafe's assets map (PNG and JPEG).
	ImageAssets = engine.ImageAssets
	// ImageDataURLs is a list of data URLs beside the state (Clef).
	ImageDataURLs = engine.ImageDataURLs
	// ImageStateURLs places image_url parts before the state (OpenRouter).
	ImageStateURLs = engine.ImageStateURLs
)

// Provider classifies through one System One service.
type Provider struct {
	name   string
	images bool
	engine *engine.Engine
}

var _ llm.Classifier = (*Provider)(nil)

// Option configures a Provider.
type Option func(*config)

type config struct {
	name            string
	baseURL         string
	apiKey          string
	apiKeySet       bool
	apiKeyFunc      func(context.Context) (string, error)
	credentials     llm.CredentialStore
	headers         http.Header
	imageForm       ImageForm
	imageFormSet    bool
	imageOptIn      bool
	textOnly        bool
	httpClient      *http.Client
	maxRetries      *int
	responseRetries bool
	timeout         time.Duration
	priceTable      llm.PriceTable
	logger          *slog.Logger
	wireCapture     func(llm.WireCapture)
}

// WithName sets the provider name (default "systemone"); it is also the
// auth-file entry WithCredentials reads.
func WithName(name string) Option { return func(c *config) { c.name = name } }

// WithBaseURL overrides a preset's base URL (a proxy or regional endpoint),
// the root that serves /systemone.
func WithBaseURL(url string) Option { return func(c *config) { c.baseURL = url } }

// WithAPIKey sets a static bearer key.
func WithAPIKey(key string) Option {
	return func(c *config) {
		c.apiKey, c.apiKeySet, c.apiKeyFunc = key, true, nil
	}
}

// WithAPIKeyFunc sets a per-request key resolver. It wins over WithAPIKey.
func WithAPIKeyFunc(fn func(context.Context) (string, error)) Option {
	return func(c *config) { c.apiKeyFunc = fn }
}

// WithCredentials reads the key on every request from store's entry named
// after the provider, which must be an "api_key" credential; with nothing
// stored, the preset's environment variable is used (New reads none).
func WithCredentials(store llm.CredentialStore) Option {
	return func(c *config) { c.credentials = store }
}

// WithImageForm sets how the service takes images. New defaults to
// ImageDataURLs, llama.cpp's images list; a server that takes TypeSafe's
// assets needs ImageAssets (and WithImageOptIn for TypeSafe's own API).
func WithImageForm(form ImageForm) Option {
	return func(c *config) { c.imageForm, c.imageFormSet = form, true }
}

// WithImageOptIn sends TypeSafe's TypeSafe-Danger-Zone-Images header with
// requests that carry assets images, as TypeSafe's own API requires.
func WithImageOptIn(enabled bool) Option { return func(c *config) { c.imageOptIn = enabled } }

// WithHeaders adds headers to every request.
func WithHeaders(headers http.Header) Option {
	return func(c *config) { c.headers = headers.Clone() }
}

// WithHTTPClient replaces the default HTTP client.
func WithHTTPClient(client *http.Client) Option { return func(c *config) { c.httpClient = client } }

// WithMaxRetries bounds safe transport retries (and response retries with
// WithResponseRetries). Default: 2.
func WithMaxRetries(n int) Option { return func(c *config) { c.maxRetries = &n } }

// WithResponseRetries enables retries of explicit 429/503/529 responses.
func WithResponseRetries(enabled bool) Option { return func(c *config) { c.responseRetries = enabled } }

// WithTimeout applies a deadline to each call.
func WithTimeout(timeout time.Duration) Option { return func(c *config) { c.timeout = timeout } }

// WithPriceTable overrides the embedded price table for cost estimates.
func WithPriceTable(table llm.PriceTable) Option { return func(c *config) { c.priceTable = table } }

// WithLogger enables provider-level operational logging.
func WithLogger(logger *slog.Logger) Option { return func(c *config) { c.logger = logger } }

// WithWireCapture enables redacted wire capture.
func WithWireCapture(fn func(llm.WireCapture)) Option { return func(c *config) { c.wireCapture = fn } }

// New returns a provider for the System One server at baseURL, the root
// that serves /systemone (for llama.cpp, "http://host:8080/v1"). Images go
// as data URLs in an images list, as llama.cpp takes them (WithImageForm
// changes it). The key is optional and never read from the environment.
func New(baseURL string, opts ...Option) (*Provider, error) {
	if baseURL == "" {
		return nil, fmt.Errorf("%w: systemone.New needs a base URL (for example http://localhost:8080/v1)", llm.ErrBadRequest)
	}
	cfg := config{name: "systemone", baseURL: baseURL, imageForm: ImageDataURLs, imageFormSet: true}
	return build(cfg, "", "", false, opts)
}

// NewTypeSafe returns a provider for TypeSafe's own API (Jev). The key is
// TYPESAFE_API_KEY unless set; images go as assets behind TypeSafe's opt-in
// header.
func NewTypeSafe(opts ...Option) (*Provider, error) {
	cfg := config{name: "typesafe", baseURL: TypeSafeBaseURL, imageForm: ImageAssets, imageFormSet: true, imageOptIn: true}
	return build(cfg, typeSafeKeyEnv, os.Getenv(typeSafeKeyEnv), true, opts)
}

// NewVercelAIGateway returns a provider for Vercel AI Gateway's classifier
// models (Jev and others it lists as evaluation models). The key is
// AI_GATEWAY_API_KEY unless set. Its classifier models take text only.
func NewVercelAIGateway(opts ...Option) (*Provider, error) {
	cfg := config{name: "vercel-ai-gateway", baseURL: VercelAIGatewayBaseURL, textOnly: true}
	return build(cfg, vercelKeyEnv, os.Getenv(vercelKeyEnv), true, opts)
}

// NewOpenCode returns a provider for OpenCode Zen's classifier models
// (jev-1.13, jev-1.13-free). The key is OPENCODE_API_KEY unless set. Its
// classifier models take text only.
func NewOpenCode(opts ...Option) (*Provider, error) {
	cfg := config{name: "opencode", baseURL: OpenCodeBaseURL, textOnly: true}
	return build(cfg, openCodeKeyEnv, os.Getenv(openCodeKeyEnv), true, opts)
}

// build finishes a provider; ambient is the preset's environment key, read
// by the caller at construction.
func build(cfg config, keyEnv, ambient string, keyRequired bool, opts []Option) (*Provider, error) {
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	if cfg.name == "" {
		return nil, fmt.Errorf("%w: empty system one provider name", llm.ErrBadRequest)
	}
	keyFunc, err := resolveKey(cfg, ambient, keyEnv, keyRequired)
	if err != nil {
		return nil, err
	}
	maxRetries := 2
	if cfg.maxRetries != nil {
		maxRetries = *cfg.maxRetries
	}
	form := cfg.imageForm
	e, err := engine.New(engine.Config{
		Provider:        cfg.name,
		BaseURL:         cfg.baseURL,
		APIKey:          keyFunc,
		Headers:         cfg.headers,
		ImageForm:       func(string) ImageForm { return form },
		ImageOptIn:      cfg.imageOptIn,
		TextOnly:        cfg.textOnly,
		HTTPClient:      cfg.httpClient,
		MaxRetries:      maxRetries,
		ResponseRetries: cfg.responseRetries,
		Timeout:         cfg.timeout,
		PriceTable:      cfg.priceTable,
		Logger:          cfg.logger,
		WireCapture:     cfg.wireCapture,
	})
	if err != nil {
		return nil, err
	}
	return &Provider{name: cfg.name, images: !cfg.textOnly, engine: e}, nil
}

func resolveKey(cfg config, ambient, keyEnv string, required bool) (func(context.Context) (string, error), error) {
	if cfg.credentials != nil {
		if cfg.apiKeySet || cfg.apiKeyFunc != nil {
			return nil, fmt.Errorf("%w: %s WithCredentials cannot be combined with WithAPIKey or WithAPIKeyFunc", llm.ErrBadRequest, cfg.name)
		}
		if required {
			return providerutil.StoredAPIKey(cfg.credentials, cfg.name, ambient), nil
		}
		store, name := cfg.credentials, cfg.name
		return func(ctx context.Context) (string, error) {
			cred, ok, err := store.Read(ctx, name)
			if err != nil {
				return "", fmt.Errorf("%w: read %s credential: %w", llm.ErrAuth, name, err)
			}
			if !ok || (cred.Key == "" && (cred.Type == "" || cred.Type == "api_key")) {
				return "", nil // keyless server
			}
			return providerutil.APIKeyFromCredential(name, cred, ok, "")
		}, nil
	}
	if cfg.apiKeyFunc != nil {
		return cfg.apiKeyFunc, nil
	}
	key := cfg.apiKey
	if !cfg.apiKeySet {
		key = ambient
	}
	if key == "" && required {
		return nil, fmt.Errorf("%w: missing %s API key; set WithAPIKey or %s", llm.ErrAuth, cfg.name, keyEnv)
	}
	return func(context.Context) (string, error) { return key, nil }, nil
}

// Name returns the provider name.
func (p *Provider) Name() string { return p.name }

// Capabilities lists CapabilityClassify, and CapabilityImageInput when the
// service takes images.
func (p *Provider) Capabilities() []llm.Capability {
	if p.images {
		return []llm.Capability{llm.CapabilityClassify, llm.CapabilityImageInput}
	}
	return []llm.Capability{llm.CapabilityClassify}
}

// Classify answers req's questions.
func (p *Provider) Classify(ctx context.Context, req *llm.ClassifyRequest) (*llm.ClassifyResult, error) {
	return p.engine.Classify(ctx, req)
}

// Models is not offered: System One services have no common listing. The
// embedded table (llm.LookupModelInfo) knows the catalogued models.
func (p *Provider) Models(context.Context) ([]llm.ModelInfo, error) {
	return nil, fmt.Errorf("%w: %s does not list models", llm.ErrUnsupported, p.name)
}

// Chat is not offered: this provider only classifies.
func (p *Provider) Chat(context.Context, *llm.Request) (*llm.Response, error) {
	return nil, fmt.Errorf("%w: %s serves classifier models only; use llm.Classify", llm.ErrUnsupported, p.name)
}

// ChatStream is not offered: this provider only classifies.
func (p *Provider) ChatStream(context.Context, *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		yield(nil, fmt.Errorf("%w: %s serves classifier models only; use llm.Classify", llm.ErrUnsupported, p.name))
	}
}
