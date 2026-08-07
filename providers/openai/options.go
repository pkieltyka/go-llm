package openai

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	sdk "github.com/openai/openai-go/v3"
	sdkoption "github.com/openai/openai-go/v3/option"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

const (
	providerName         = "openai"
	apiKeyEnv            = "OPENAI_API_KEY"
	defaultOpenAIBaseURL = "https://api.openai.com/v1/"
	customHeadersEnv     = providerutil.CustomHeadersEnv
	organizationHeader   = "OpenAI-Organization"
	projectHeader        = "OpenAI-Project"
)

// Option configures an OpenAI provider.
type Option func(*config)

type apiKeyFunc func(context.Context) (string, error)

type config struct {
	apiKey          string
	apiKeyFunc      apiKeyFunc
	baseURL         string
	httpClient      *http.Client
	maxRetries      *int
	responseRetries bool
	timeout         time.Duration
	priceTable      llm.PriceTable
	logger          *slog.Logger
	wireCapture     func(llm.WireCapture)
	organization    string
	project         string
}

func defaultConfig() config {
	return config{
		apiKey:     os.Getenv(apiKeyEnv),
		httpClient: llm.DefaultHTTPClient(),
	}
}

// WithAPIKey sets a static OpenAI API key. Empty values disable env fallback.
func WithAPIKey(key string) Option {
	return func(c *config) {
		c.apiKey = key
		c.apiKeyFunc = nil
	}
}

// WithAPIKeyFunc sets a per-request key resolver. It wins over WithAPIKey.
func WithAPIKeyFunc(fn func(context.Context) (string, error)) Option {
	return func(c *config) {
		c.apiKeyFunc = fn
	}
}

// WithBaseURL overrides the OpenAI API base URL.
func WithBaseURL(url string) Option {
	return func(c *config) { c.baseURL = url }
}

// WithHTTPClient replaces the provider's default HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) { c.httpClient = client }
}

// WithMaxRetries bounds automatic transport retries and, when enabled by
// WithResponseRetries, response retries. Default: 2 additional attempts.
func WithMaxRetries(n int) Option {
	return func(c *config) { c.maxRetries = &n }
}

// WithResponseRetries enables or disables retries of explicit 429/503/529
// responses. They are disabled by default because model requests are not
// idempotent. Typed failures proven to occur before request bytes were sent
// may still be retried within the WithMaxRetries bound.
func WithResponseRetries(enabled bool) Option {
	return func(c *config) { c.responseRetries = enabled }
}

// WithTimeout applies a context deadline to provider calls.
func WithTimeout(timeout time.Duration) Option {
	return func(c *config) { c.timeout = timeout }
}

// WithPriceTable overrides embedded cost estimates.
func WithPriceTable(table llm.PriceTable) Option {
	return func(c *config) { c.priceTable = table }
}

// WithLogger enables provider-level operational logging.
func WithLogger(logger *slog.Logger) Option {
	return func(c *config) { c.logger = logger }
}

// WithWireCapture enables redacted wire capture.
func WithWireCapture(fn func(llm.WireCapture)) Option {
	return func(c *config) { c.wireCapture = fn }
}

// WithOrganization sets the OpenAI organization header explicitly.
func WithOrganization(organization string) Option {
	return func(c *config) { c.organization = organization }
}

// WithProject sets the OpenAI project header explicitly.
func WithProject(project string) Option {
	return func(c *config) { c.project = project }
}

func (c config) validate() error {
	if c.apiKeyFunc == nil && c.apiKey == "" {
		return fmt.Errorf("%w: missing OpenAI API key; set WithAPIKey or %s", llm.ErrAuth, apiKeyEnv)
	}
	if c.httpClient == nil {
		return fmt.Errorf("%w: nil HTTP client", llm.ErrBadRequest)
	}
	if c.maxRetries != nil && *c.maxRetries < 0 {
		return fmt.Errorf("%w: max retries must be >= 0", llm.ErrBadRequest)
	}
	return nil
}

func (c config) sdkOptions() []sdkoption.RequestOption {
	maxRetries := 2
	if c.maxRetries != nil {
		maxRetries = *c.maxRetries
	}
	observed := providerutil.ObservedHTTPClient(c.httpClient, providerName, c.logger, c.wireCapture)
	opts := []sdkoption.RequestOption{
		sdkoption.WithHTTPClient(providerutil.SafeRetryHTTPClient(observed, maxRetries, c.responseRetries)),
		// The shared transport owns retry classification. Leaving SDK retries on
		// would replay ambiguous no-response errors and stack a second loop.
		sdkoption.WithMaxRetries(0),
		sdkoption.WithBaseURL(defaultOpenAIBaseURL),
		sdkoption.WithAdminAPIKey(""),
		sdkoption.WithHeaderDel(organizationHeader),
		sdkoption.WithHeaderDel(projectHeader),
	}
	opts = append(opts, providerutil.AmbientCustomHeaderDeleteOptions()...)
	if c.baseURL != "" {
		opts = append(opts, sdkoption.WithBaseURL(c.baseURL))
	}
	if c.organization != "" {
		opts = append(opts, sdkoption.WithOrganization(c.organization))
	}
	if c.project != "" {
		opts = append(opts, sdkoption.WithProject(c.project))
	}
	if c.apiKeyFunc != nil {
		opts = append(opts,
			sdkoption.WithAPIKey("dynamic"),
			sdkoption.WithMiddleware(func(req *http.Request, next sdkoption.MiddlewareNext) (*http.Response, error) {
				key, err := c.apiKeyFunc(req.Context())
				if err != nil {
					return nil, err
				}
				req.Header.Set("Authorization", "Bearer "+key)
				return next(req)
			}),
		)
	} else {
		opts = append(opts, sdkoption.WithAPIKey(c.apiKey))
	}
	return opts
}

// Provider is the OpenAI Responses API implementation of llm.Provider. It
// wraps openai-go's Responses surface directly and, by default, keeps every
// request stateless (store: false + encrypted reasoning round-tripping).
type Provider struct {
	client     *sdk.Client
	priceTable llm.PriceTable
	logger     *slog.Logger
	timeout    time.Duration
}

// New constructs an OpenAI Responses provider.
func New(opts ...Option) (*Provider, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	client := sdk.NewClient(cfg.sdkOptions()...)
	return &Provider{
		client:     &client,
		priceTable: cfg.priceTable,
		logger:     cfg.logger,
		timeout:    cfg.timeout,
	}, nil
}

// Client exposes the underlying OpenAI SDK client as an advanced escape hatch.
// Its vendor-typed signature is not part of the stable ordinary provider API;
// callers using it accept source changes from openai-go upgrades.
func (p *Provider) Client() *sdk.Client {
	if p == nil {
		return nil
	}
	return p.client
}
