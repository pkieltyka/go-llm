package anthropic

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	sdkoption "github.com/anthropics/anthropic-sdk-go/option"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

const (
	providerName     = "anthropic"
	apiKeyEnv        = "ANTHROPIC_API_KEY"
	defaultMaxTokens = 16384
)

// Options carries Anthropic-specific request extensions.
type Options struct {
	BetaHeaders            []string
	ServiceTier            string
	Container              string
	MetadataUserID         string
	TopK                   *int64
	DisableParallelToolUse *bool
}

// ForProvider identifies these options as Anthropic-specific.
func (Options) ForProvider() string { return providerName }

// Option configures an Anthropic provider.
type Option func(*config)

type apiKeyFunc func(context.Context) (string, error)

type config struct {
	apiKey           string
	apiKeyFunc       apiKeyFunc
	baseURL          string
	httpClient       *http.Client
	maxRetries       *int
	responseRetries  bool
	timeout          time.Duration
	priceTable       llm.PriceTable
	logger           *slog.Logger
	wireCapture      func(llm.WireCapture)
	defaultMaxTokens int
}

func defaultConfig() config {
	return config{
		apiKey:           os.Getenv(apiKeyEnv),
		httpClient:       llm.DefaultHTTPClient(),
		defaultMaxTokens: defaultMaxTokens,
	}
}

// WithAPIKey sets a static Anthropic API key. Empty values disable env fallback.
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

// WithBaseURL overrides the Anthropic API base URL.
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

// WithDefaultMaxTokens changes the Anthropic-required MaxTokens default.
func WithDefaultMaxTokens(n int) Option {
	return func(c *config) { c.defaultMaxTokens = n }
}

func (c config) validate() error {
	if c.apiKeyFunc == nil && c.apiKey == "" {
		return fmt.Errorf("%w: missing Anthropic API key; set WithAPIKey or %s", llm.ErrAuth, apiKeyEnv)
	}
	if c.httpClient == nil {
		return fmt.Errorf("%w: nil HTTP client", llm.ErrBadRequest)
	}
	if c.maxRetries != nil && *c.maxRetries < 0 {
		return fmt.Errorf("%w: max retries must be >= 0", llm.ErrBadRequest)
	}
	if c.defaultMaxTokens < 0 {
		return fmt.Errorf("%w: default max tokens must be >= 0", llm.ErrBadRequest)
	}
	return nil
}

func (c config) sdkOptions() []sdkoption.RequestOption {
	maxRetries := 2
	if c.maxRetries != nil {
		maxRetries = *c.maxRetries
	}
	observed := providerutil.ObservedHTTPClient(c.httpClient, providerName, c.logger, c.wireCapture)
	client := providerutil.SafeRetryHTTPClient(observed, maxRetries, c.responseRetries)

	opts := []sdkoption.RequestOption{
		sdkoption.WithoutEnvironmentDefaults(),
		sdkoption.WithHTTPClient(client),
		// The shared transport owns retry classification. Leaving SDK retries on
		// would replay ambiguous no-response errors and stack a second loop.
		sdkoption.WithMaxRetries(0),
	}
	if c.baseURL != "" {
		opts = append(opts, sdkoption.WithBaseURL(c.baseURL))
	}
	if c.apiKeyFunc != nil {
		opts = append(opts, sdkoption.WithMiddleware(func(req *http.Request, next sdkoption.MiddlewareNext) (*http.Response, error) {
			key, err := c.apiKeyFunc(req.Context())
			if err != nil {
				return nil, err
			}
			req.Header.Set("X-Api-Key", key)
			return next(req)
		}))
	} else {
		opts = append(opts, sdkoption.WithAPIKey(c.apiKey))
	}
	return opts
}

// Provider is the Anthropic Messages API implementation of llm.Provider. It
// wraps anthropic-sdk-go directly using API-key authentication.
type Provider struct {
	client           *sdk.Client
	defaultMaxTokens int
	priceTable       llm.PriceTable
	logger           *slog.Logger
	timeout          time.Duration
}

// New constructs an Anthropic provider.
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
		client:           &client,
		defaultMaxTokens: cfg.defaultMaxTokens,
		priceTable:       cfg.priceTable,
		logger:           cfg.logger,
		timeout:          cfg.timeout,
	}, nil
}

// Client exposes the underlying Anthropic SDK client.
func (p *Provider) Client() *sdk.Client {
	if p == nil {
		return nil
	}
	return p.client
}
