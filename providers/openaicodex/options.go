package openaicodex

import (
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	sdk "github.com/openai/openai-go/v3"
	sdkoption "github.com/openai/openai-go/v3/option"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/provideroauth"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

const (
	providerName          = "openai-codex"
	defaultCodexBaseURL   = "https://chatgpt.com/backend-api/codex"
	defaultOriginator     = "codex_cli_rs"
	organizationHeader    = "OpenAI-Organization"
	projectHeader         = "OpenAI-Project"
	accountIDHeader       = "chatgpt-account-id"
	originatorHeader      = "originator"
	defaultCodexUserAgent = "codex-cli/0.1"
)

// Option configures an OpenAI Codex subscription provider.
type Option func(*config)

type config struct {
	oauthCred     llm.AuthCredential
	persistence   llm.OAuthPersistenceFunc
	baseURL       string
	httpClient    *http.Client
	maxRetries    *int
	timeout       time.Duration
	priceTable    llm.PriceTable
	logger        *slog.Logger
	wireCapture   func(llm.WireCapture)
	tokenURL      string
	originator    string
	customHeaders http.Header
}

func defaultConfig() config {
	return config{
		httpClient:    llm.DefaultHTTPClient(),
		originator:    defaultOriginator,
		customHeaders: providerutil.AmbientCustomHeaders(),
	}
}

// WithOAuth sets the ChatGPT subscription OAuth credential. A credential with
// a refresh token requires non-nil persist; access-only credentials may pass
// nil. persist must honor its context and return only after durable storage.
// An explicit no-op opts into in-memory-only rotation and risks a stale stored
// refresh token after restart.
func WithOAuth(cred llm.AuthCredential, persist llm.OAuthPersistenceFunc) Option {
	return func(c *config) {
		c.oauthCred = cred
		c.persistence = persist
	}
}

// WithBaseURL overrides the Codex backend base URL.
func WithBaseURL(url string) Option {
	return func(c *config) { c.baseURL = url }
}

// WithHTTPClient replaces the shared default HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) { c.httpClient = client }
}

// WithMaxRetries bounds automatic retries. It applies both to SDK-managed
// endpoints and to the direct Codex streaming transport, which retries only
// billing-safe rejections (429/503/529 — the request was never accepted)
// before any stream bytes are read, honoring Retry-After with exponential
// backoff otherwise. Default: 2 additional attempts.
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

// WithOriginator overrides the Codex originator header.
//
// The originator identifies the client that minted the OAuth token flow. The
// default, "codex_cli_rs", matches tokens minted by the official Codex CLI.
// Tokens minted through other clients (e.g. pi's login flow) may carry a
// different originator claim, and the backend can cross-check the header
// against the token's mint flow — if the backend rejects requests with an
// originator mismatch, set this to the value used by the CLI that minted the
// credential.
func WithOriginator(originator string) Option {
	return func(c *config) {
		if strings.TrimSpace(originator) != "" {
			c.originator = originator
		}
	}
}

func withOAuthTokenURL(url string) Option {
	return func(c *config) { c.tokenURL = url }
}

func (c config) validate() error {
	if c.oauthCred.Access == "" && c.oauthCred.Refresh == "" {
		return fmt.Errorf("%w: missing OpenAI Codex OAuth credential", llm.ErrAuth)
	}
	if err := provideroauth.ValidatePersistence(c.oauthCred, c.persistence); err != nil {
		return err
	}
	if c.httpClient == nil {
		return fmt.Errorf("%w: nil HTTP client", llm.ErrBadRequest)
	}
	if c.maxRetries != nil && *c.maxRetries < 0 {
		return fmt.Errorf("%w: max retries must be >= 0", llm.ErrBadRequest)
	}
	return nil
}

func (c config) sdkOptions(source *provideroauth.Source) []sdkoption.RequestOption {
	opts := []sdkoption.RequestOption{
		sdkoption.WithHTTPClient(c.observedHTTPClient()),
		sdkoption.WithBaseURL(codexBaseURL(c.baseURL)),
		sdkoption.WithAdminAPIKey(""),
		sdkoption.WithHeaderDel(organizationHeader),
		sdkoption.WithHeaderDel(projectHeader),
	}
	opts = append(opts, providerutil.AmbientCustomHeaderDeleteOptions()...)
	opts = append(opts,
		sdkoption.WithAPIKey("oauth"),
		sdkoption.WithHeader("User-Agent", defaultCodexUserAgent),
		sdkoption.WithMiddleware(func(req *http.Request, next sdkoption.MiddlewareNext) (*http.Response, error) {
			return provideroauth.DoWithAuthRetry(req, provideroauth.MiddlewareNext(next), source, c.applyOAuthHeaders)
		}),
	)
	if c.maxRetries != nil {
		opts = append(opts, sdkoption.WithMaxRetries(*c.maxRetries))
	}
	return opts
}

func (c config) observedHTTPClient() *http.Client {
	return providerutil.ObservedHTTPClient(c.httpClient, providerName, c.logger, c.wireCapture)
}

func (c config) applyOAuthHeaders(req *http.Request, cred llm.AuthCredential) {
	applyOAuthHeaders(c.originatorValue())(req, cred)
}

func (c config) directTransport(source *provideroauth.Source) codexTransport {
	maxRetries := defaultTransportMaxRetries
	if c.maxRetries != nil {
		maxRetries = *c.maxRetries
	}
	return codexTransport{
		endpoint:     codexResponsesEndpoint(c.baseURL),
		httpClient:   c.observedHTTPClient(),
		source:       source,
		originator:   c.originatorValue(),
		headerFunc:   c.applyCodexHeaders,
		authFunc:     applyOAuthHeaders(c.originatorValue()),
		providerName: providerName,
		maxRetries:   maxRetries,
	}
}

func (c config) applyCodexHeaders(req *http.Request) {
	for name, values := range c.customHeaders {
		for _, value := range values {
			req.Header.Add(name, value)
		}
	}
	req.Header.Del(organizationHeader)
	req.Header.Del(projectHeader)
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("OpenAI-Beta", "responses=experimental")
	req.Header.Set("User-Agent", defaultCodexUserAgent)
}

func (c config) originatorValue() string {
	originator := strings.TrimSpace(c.originator)
	if originator == "" {
		return defaultOriginator
	}
	return originator
}

func applyOAuthHeaders(originator string) provideroauth.ApplyHeadersFunc {
	if strings.TrimSpace(originator) == "" {
		originator = defaultOriginator
	}
	return func(req *http.Request, cred llm.AuthCredential) {
		req.Header.Set("Authorization", "Bearer "+cred.Access)
		if accountID := codexAccountID(cred); accountID != "" {
			req.Header.Set(accountIDHeader, accountID)
		}
		req.Header.Set(originatorHeader, originator)
	}
}

func codexResponsesEndpoint(baseURL string) string {
	base := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if base == "" {
		base = defaultCodexBaseURL
	}
	if strings.HasSuffix(base, "/responses") {
		return base
	}
	return base + "/responses"
}

func codexBaseURL(baseURL string) string {
	base := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if base == "" {
		return defaultCodexBaseURL
	}
	if strings.HasSuffix(base, "/responses") {
		return strings.TrimSuffix(base, "/responses")
	}
	return base
}

// Provider is the OpenAI Codex subscription implementation of llm.Provider.
// It speaks the Responses wire shape against chatgpt.com/backend-api/codex
// using ChatGPT Plus/Pro OAuth credentials. See New for the request knobs
// the codex backend does not accept.
type Provider struct {
	client     *sdk.Client
	transport  codexTransport
	priceTable llm.PriceTable
	logger     *slog.Logger
	timeout    time.Duration
}
