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
	sdkparam "github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
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

// Options carries OpenAI Responses-specific request extensions.
type Options struct {
	Store                *bool
	PreviousResponseID   string
	ConversationID       string
	Conversation         responses.ResponseNewParamsConversationUnion
	Include              []responses.ResponseIncludable
	Background           *bool
	HostedTools          []responses.ToolUnionParam
	Verbosity            responses.ResponseTextConfigVerbosity
	Metadata             shared.Metadata
	ServiceTier          responses.ResponseNewParamsServiceTier
	SafetyIdentifier     string
	PromptCacheRetention responses.ResponseNewParamsPromptCacheRetention
}

// ForProvider identifies these options as OpenAI-specific.
func (Options) ForProvider() string { return providerName }

// Option configures an OpenAI provider.
type Option func(*config)

type apiKeyFunc func(context.Context) (string, error)

type config struct {
	apiKey       string
	apiKeyFunc   apiKeyFunc
	baseURL      string
	httpClient   *http.Client
	maxRetries   *int
	timeout      time.Duration
	priceTable   llm.PriceTable
	logger       *slog.Logger
	wireCapture  func(llm.WireCapture)
	organization string
	project      string
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

// WithHTTPClient replaces the shared default HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) { c.httpClient = client }
}

// WithMaxRetries delegates retry count to the OpenAI SDK.
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

// WithOrganization sets the OpenAI organization header explicitly.
func WithOrganization(organization string) Option {
	return func(c *config) { c.organization = organization }
}

// WithProject sets the OpenAI project header explicitly.
func WithProject(project string) Option {
	return func(c *config) { c.project = project }
}

func requestOptions(req *llm.Request) (*Options, error) {
	options, ok, err := providerutil.OptionsOf[Options](req)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, nil
	}
	return &options, nil
}

func applyOptions(options *Options, params *responses.ResponseNewParams) {
	if options == nil {
		return
	}
	if options.Store != nil {
		params.Store = sdk.Bool(*options.Store)
	}
	if options.PreviousResponseID != "" {
		params.PreviousResponseID = sdk.String(options.PreviousResponseID)
	}
	if !sdkparam.IsOmitted(options.Conversation) {
		params.Conversation = options.Conversation
	} else if options.ConversationID != "" {
		params.Conversation.OfString = sdkparam.NewOpt(options.ConversationID)
	}
	if len(options.Include) > 0 {
		params.Include = mergeIncludes(params.Include, options.Include)
	}
	if options.Background != nil {
		params.Background = sdk.Bool(*options.Background)
	}
	if len(options.HostedTools) > 0 {
		params.Tools = append(params.Tools, options.HostedTools...)
	}
	if options.Verbosity != "" {
		params.Text.Verbosity = options.Verbosity
	}
	if len(options.Metadata) > 0 {
		params.Metadata = options.Metadata
	}
	if options.ServiceTier != "" {
		params.ServiceTier = options.ServiceTier
	}
	if options.SafetyIdentifier != "" {
		params.SafetyIdentifier = sdk.String(options.SafetyIdentifier)
	}
	if options.PromptCacheRetention != "" {
		params.PromptCacheRetention = options.PromptCacheRetention
	}
}

func mergeIncludes(base, extra []responses.ResponseIncludable) []responses.ResponseIncludable {
	out := append([]responses.ResponseIncludable(nil), base...)
	seen := make(map[responses.ResponseIncludable]struct{}, len(out)+len(extra))
	for _, include := range out {
		seen[include] = struct{}{}
	}
	for _, include := range extra {
		if include == "" {
			continue
		}
		if _, ok := seen[include]; ok {
			continue
		}
		seen[include] = struct{}{}
		out = append(out, include)
	}
	return out
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
	opts := []sdkoption.RequestOption{
		sdkoption.WithHTTPClient(providerutil.ObservedHTTPClient(c.httpClient, providerName, c.logger, c.wireCapture)),
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
	if c.maxRetries != nil {
		opts = append(opts, sdkoption.WithMaxRetries(*c.maxRetries))
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

// Client exposes the underlying OpenAI SDK client.
func (p *Provider) Client() *sdk.Client {
	if p == nil {
		return nil
	}
	return p.client
}
