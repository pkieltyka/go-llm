package anthropic

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	sdkoption "github.com/anthropics/anthropic-sdk-go/option"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/provideroauth"
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
	timeout          time.Duration
	priceTable       llm.PriceTable
	logger           *slog.Logger
	wireCapture      func(llm.WireCapture)
	defaultMaxTokens int
	oauth            bool
	oauthCred        llm.AuthCredential
	oauthOnRefresh   func(llm.AuthCredential)
	oauthTokenURL    string
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
		c.oauth = false
	}
}

// WithAPIKeyFunc sets a per-request key resolver. It wins over WithAPIKey.
func WithAPIKeyFunc(fn func(context.Context) (string, error)) Option {
	return func(c *config) {
		c.apiKeyFunc = fn
		c.oauth = false
	}
}

// WithOAuth enables Claude subscription OAuth credentials.
func WithOAuth(cred llm.AuthCredential, onRefresh func(llm.AuthCredential)) Option {
	return func(c *config) {
		c.oauth = true
		c.oauthCred = cred
		c.oauthOnRefresh = onRefresh
		c.apiKey = ""
		c.apiKeyFunc = nil
	}
}

// WithBaseURL overrides the Anthropic API base URL.
func WithBaseURL(url string) Option {
	return func(c *config) { c.baseURL = url }
}

// WithHTTPClient replaces the shared default HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) { c.httpClient = client }
}

// WithMaxRetries delegates retry count to the Anthropic SDK.
func WithMaxRetries(n int) Option {
	return func(c *config) { c.maxRetries = &n }
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

func withOAuthTokenURL(url string) Option {
	return func(c *config) { c.oauthTokenURL = url }
}

func (c config) validate() error {
	if c.oauth {
		if c.oauthCred.Access == "" && c.oauthCred.Refresh == "" {
			return fmt.Errorf("%w: missing Anthropic OAuth credential", llm.ErrAuth)
		}
	} else if c.apiKeyFunc == nil && c.apiKey == "" {
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
	client := providerutil.ObservedHTTPClient(c.httpClient, providerName, c.logger, c.wireCapture)

	opts := []sdkoption.RequestOption{
		sdkoption.WithoutEnvironmentDefaults(),
		sdkoption.WithHTTPClient(client),
	}
	if c.baseURL != "" {
		opts = append(opts, sdkoption.WithBaseURL(c.baseURL))
	}
	if c.oauth {
		source := newAnthropicOAuthSource(c)
		opts = append(opts,
			sdkoption.WithAuthToken("oauth"),
			sdkoption.WithHeaderDel("X-Api-Key"),
			sdkoption.WithMiddleware(func(req *http.Request, next sdkoption.MiddlewareNext) (*http.Response, error) {
				return provideroauth.DoWithAuthRetry(req, provideroauth.MiddlewareNext(next), source, applyAnthropicOAuthHeaders)
			}),
		)
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
	if c.maxRetries != nil {
		opts = append(opts, sdkoption.WithMaxRetries(*c.maxRetries))
	}
	return opts
}

// applyAnthropicOAuthHeaders applies the full Claude Code identity set
// required in OAuth mode (FS §17C / ARCH §3.1): subscription tokens are only
// served to Claude-Code-identified traffic, so alongside bearer auth the
// request must carry the claude-code + oauth betas, a claude-cli user-agent,
// and x-app: cli — mirroring pi's api/anthropic-messages.ts OAuth client.
// These headers are set ONLY on the OAuth path; api-key requests are
// untouched.
func applyAnthropicOAuthHeaders(req *http.Request, cred llm.AuthCredential) {
	req.Header.Del("X-Api-Key")
	req.Header.Set("Authorization", "Bearer "+cred.Access)
	req.Header.Set("User-Agent", anthropicOAuthUserAgent)
	req.Header.Set("X-App", "cli")
	req.Header.Set("anthropic-beta", oauthBetaHeaderValue(req.Header.Values("anthropic-beta")))
}

// oauthBetaHeaderValue builds the single comma-joined anthropic-beta value pi
// sends in OAuth mode: the Claude Code identity betas first, then any
// caller-requested beta features (Options.BetaHeaders), deduplicated.
func oauthBetaHeaderValue(existing []string) string {
	betas := []string{anthropicClaudeCodeBeta, anthropicOAuthBeta}
	seen := map[string]struct{}{anthropicClaudeCodeBeta: {}, anthropicOAuthBeta: {}}
	for _, value := range existing {
		for beta := range strings.SplitSeq(value, ",") {
			beta = strings.TrimSpace(beta)
			if beta == "" {
				continue
			}
			if _, ok := seen[beta]; ok {
				continue
			}
			seen[beta] = struct{}{}
			betas = append(betas, beta)
		}
	}
	return strings.Join(betas, ",")
}

// Provider is the Anthropic Messages API implementation of llm.Provider. It
// wraps anthropic-sdk-go directly and supports both api-key auth and Claude
// subscription OAuth (WithOAuth), which adds the Claude Code identity
// headers and system block required by subscription tokens.
type Provider struct {
	client           *sdk.Client
	defaultMaxTokens int
	priceTable       llm.PriceTable
	logger           *slog.Logger
	timeout          time.Duration
	oauth            bool
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
		oauth:            cfg.oauth,
	}, nil
}

// Client exposes the underlying Anthropic SDK client.
func (p *Provider) Client() *sdk.Client {
	if p == nil {
		return nil
	}
	return p.client
}
