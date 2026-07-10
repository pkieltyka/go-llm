package openrouter

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
)

const (
	providerName             = "openrouter"
	apiKeyEnv                = "OPENROUTER_API_KEY"
	defaultOpenRouterBaseURL = "https://openrouter.ai/api/v1"
)

// Options carries OpenRouter-specific request extensions.
type Options struct {
	Models            []string
	Provider          any
	Plugins           []any
	Prediction        any
	Reasoning         map[string]any
	TopK              *int
	MinP              *float64
	TopA              *float64
	RepetitionPenalty *float64
	HTTPReferer       string
	XTitle            string
}

// ForProvider identifies these options as OpenRouter-specific.
func (Options) ForProvider() string { return providerName }

// Option configures an OpenRouter provider.
type Option func(*config)

type config struct {
	apiKey      string
	apiKeySet   bool
	apiKeyFunc  func(context.Context) (string, error)
	baseURL     string
	httpClient  *http.Client
	maxRetries  *int
	timeout     time.Duration
	priceTable  llm.PriceTable
	logger      *slog.Logger
	wireCapture func(llm.WireCapture)
	httpReferer string
	xTitle      string
}

func defaultConfig() config {
	return config{
		httpClient: llm.DefaultHTTPClient(),
	}
}

func (c *config) resolveAPIKey() {
	if c.apiKeySet || c.apiKeyFunc != nil {
		return
	}
	c.apiKey = os.Getenv(apiKeyEnv)
	c.apiKeySet = true
}

// WithAPIKey sets a static OpenRouter API key. Empty values disable env fallback.
func WithAPIKey(key string) Option {
	return func(c *config) {
		c.apiKey = key
		c.apiKeySet = true
		c.apiKeyFunc = nil
	}
}

// WithAPIKeyFunc sets a per-request key resolver. It wins over WithAPIKey.
func WithAPIKeyFunc(fn func(context.Context) (string, error)) Option {
	return func(c *config) { c.apiKeyFunc = fn }
}

// WithBaseURL overrides the OpenRouter API base URL.
func WithBaseURL(url string) Option {
	return func(c *config) { c.baseURL = url }
}

// WithHTTPClient replaces the shared default HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) { c.httpClient = client }
}

// WithMaxRetries bounds retry attempts. Blocking calls delegate the count to
// the OpenAI SDK's retry layer; streaming calls use the adapter's direct SSE
// transport, which applies the same bound to billing-safe pre-stream
// rejections only (retry decisions are made on the response status line
// before any stream bytes are consumed — ARCH §3.3/§3.4).
func WithMaxRetries(n int) Option {
	return func(c *config) { c.maxRetries = &n }
}

// WithTimeout applies a context deadline to provider calls.
func WithTimeout(timeout time.Duration) Option {
	return func(c *config) { c.timeout = timeout }
}

// WithPriceTable overrides embedded cost estimates when OpenRouter does not
// report native cost.
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

// WithAttribution sets default OpenRouter attribution headers on every
// request. Per-request values in Options.HTTPReferer/Options.XTitle override
// them on both Chat and ChatStream.
func WithAttribution(httpReferer, xTitle string) Option {
	return func(c *config) {
		c.httpReferer = httpReferer
		c.xTitle = xTitle
	}
}

func (c config) chatcompletionsConfig() (chatcompletions.Config, error) {
	if c.httpClient == nil {
		return chatcompletions.Config{}, fmt.Errorf("%w: nil HTTP client", llm.ErrBadRequest)
	}
	headers := http.Header{}
	if c.httpReferer != "" {
		headers.Set("HTTP-Referer", c.httpReferer)
	}
	if c.xTitle != "" {
		headers.Set("X-Title", c.xTitle)
	}
	return chatcompletions.Config{
		Dialect:     dialect{headers: headers},
		APIKey:      c.apiKey,
		APIKeyFunc:  c.apiKeyFunc,
		BaseURL:     c.baseURL,
		HTTPClient:  c.httpClient,
		MaxRetries:  c.maxRetries,
		Timeout:     c.timeout,
		PriceTable:  c.priceTable,
		Logger:      c.logger,
		WireCapture: c.wireCapture,
	}, nil
}
