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
	apiKeySet        bool
	apiKeyFunc       apiKeyFunc
	credentials      llm.CredentialStore
	oauthTokenURL    string
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
		c.apiKeySet = true
		c.apiKeyFunc = nil
	}
}

// WithCredentials resolves typed API-key or OAuth credentials from the exact
// "anthropic" entry. An owned entry never falls back to another identity.
// WithCredentials is mutually exclusive with WithAPIKey and WithAPIKeyFunc.
func WithCredentials(store llm.CredentialStore) Option {
	return func(c *config) { c.credentials = store }
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

// WithWireCapture enables redacted wire capture. OAuth captures contain only
// request/status/timing metadata and sanitized errors; payloads and headers
// are withheld because remote responses can echo credentials.
func WithWireCapture(fn func(llm.WireCapture)) Option {
	return func(c *config) { c.wireCapture = fn }
}

// WithDefaultMaxTokens changes the Anthropic-required MaxTokens default.
func WithDefaultMaxTokens(n int) Option {
	return func(c *config) { c.defaultMaxTokens = n }
}

func (c config) validate() error {
	if c.credentials != nil && (c.apiKeySet || c.apiKeyFunc != nil) {
		return fmt.Errorf("%w: Anthropic WithCredentials cannot be combined with WithAPIKey or WithAPIKeyFunc", llm.ErrBadRequest)
	}
	if c.apiKeyFunc == nil && c.apiKey == "" && c.credentials == nil {
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

func (c config) sdkOptions(p *Provider) []sdkoption.RequestOption {
	maxRetries := 2
	if c.maxRetries != nil {
		maxRetries = *c.maxRetries
	}
	httpClient := c.httpClient
	if c.credentials != nil {
		copied := *httpClient
		next := copied.Transport
		if next == nil {
			next = http.DefaultTransport
		}
		copied.Transport = &operationCaptureTransport{
			next:    llm.NewRetryLogger(&ownedAuthTransport{provider: p, next: next}, providerName, c.logger),
			capture: c.wireCapture,
		}
		originalRedirect := copied.CheckRedirect
		copied.CheckRedirect = func(req *http.Request, via []*http.Request) error {
			if op, ok := req.Context().Value(operationKey{}).(*preparedOperation); ok && op.provider == p && op.oauth {
				return http.ErrUseLastResponse
			}
			if originalRedirect != nil {
				return originalRedirect(req, via)
			}
			if len(via) >= 10 {
				return fmt.Errorf("stopped after 10 redirects")
			}
			return nil
		}
		httpClient = &copied
	}
	observed := httpClient
	if c.credentials == nil {
		observed = providerutil.ObservedHTTPClient(httpClient, providerName, c.logger, c.wireCapture)
	}
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
	if c.credentials != nil {
		opts = append(opts, sdkoption.WithAPIKey(""), sdkoption.WithMiddleware(p.authMiddleware))
	} else if c.apiKeyFunc != nil {
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
// wraps anthropic-sdk-go with typed API-key or OAuth authentication.
type Provider struct {
	client           *sdk.Client
	credentials      llm.CredentialStore
	ambient          string
	baseURL          string
	oauthTokenURL    string
	httpClient       *http.Client
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
	p := &Provider{
		credentials: cfg.credentials, ambient: cfg.apiKey, baseURL: cfg.baseURL,
		oauthTokenURL: cfg.oauthTokenURL, httpClient: cfg.httpClient,
		defaultMaxTokens: cfg.defaultMaxTokens, priceTable: cfg.priceTable,
		logger: cfg.logger, timeout: cfg.timeout,
	}
	client := sdk.NewClient(cfg.sdkOptions(p)...)
	p.client = &client
	return p, nil
}

// Client exposes the underlying Anthropic SDK client. Raw SDK operations support
// API keys only; stored OAuth requires Chat, ChatStream, Models or CountTokens
// so provider-owned request preparation cannot be bypassed.
func (p *Provider) Client() *sdk.Client {
	if p == nil {
		return nil
	}
	return p.client
}
