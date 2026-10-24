// Package cloudflare classifies with Cloudflare Workers AI's classifier
// (decision) models: Cloudflare's Clef, Clef Flash and Clef Omni, and
// TypeSafe's Jev hosted there (FS §13C).
//
// Workers AI wraps TypeSafe's System One protocol in its own envelope:
// POST https://api.cloudflare.com/client/v4/accounts/{account}/ai/run with
// {model, input: <System One request>}. Cloudflare-hosted models answer
// {success, result: {answers, usage}}; third-party models (Jev) add a run
// record, {success, result: {state: "Completed", result: {answers,
// usage}}}. Clef takes images as data URLs, Jev as System One assets.
//
// A Provider here only classifies: Chat and ChatStream fail with
// ErrUnsupported. Use it through llm.Classify or llm.NewClassifierModel.
package cloudflare

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
	"github.com/pkieltyka/go-llm/providers/internal/systemone"
)

const (
	// providerName is pi's provider ID, so an auth.json shared with pi
	// finds the same "cloudflare-workers-ai" entry.
	providerName = "cloudflare-workers-ai"
	// BaseURL is the Workers AI REST root; {account} is replaced by the
	// account ID.
	BaseURL      = "https://api.cloudflare.com/client/v4/accounts/{account}/ai"
	apiKeyEnv    = "CLOUDFLARE_API_KEY"
	accountIDEnv = "CLOUDFLARE_ACCOUNT_ID"
)

// Provider classifies through Cloudflare Workers AI.
type Provider struct {
	engine *systemone.Engine
}

var _ llm.Classifier = (*Provider)(nil)

// Option configures a Provider.
type Option func(*config)

type config struct {
	baseURL         string
	apiKey          string
	accountID       string
	credentials     llm.CredentialStore
	httpClient      *http.Client
	maxRetries      *int
	responseRetries bool
	timeout         time.Duration
	priceTable      llm.PriceTable
	logger          *slog.Logger
	wireCapture     func(llm.WireCapture)
}

// WithAPIKey sets the Cloudflare API token.
func WithAPIKey(key string) Option { return func(c *config) { c.apiKey = key } }

// WithAccountID sets the Cloudflare account ID.
func WithAccountID(id string) Option { return func(c *config) { c.accountID = id } }

// WithCredentials resolves the API token and the account ID on every
// request, as pi does, per field: the options first, then store's
// "cloudflare-workers-ai" entry ("api_key" credential; account in
// env.CLOUDFLARE_ACCOUNT_ID), then CLOUDFLARE_API_KEY and
// CLOUDFLARE_ACCOUNT_ID.
func WithCredentials(store llm.CredentialStore) Option {
	return func(c *config) { c.credentials = store }
}

// WithBaseURL overrides the REST root (a proxy); "{account}" in it is
// replaced by the account ID, and a root without it needs no account ID.
func WithBaseURL(url string) Option { return func(c *config) { c.baseURL = url } }

// WithHTTPClient replaces the default HTTP client.
func WithHTTPClient(client *http.Client) Option { return func(c *config) { c.httpClient = client } }

// WithMaxRetries bounds safe transport retries. Default: 2.
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

// New returns a Workers AI classifier provider. Without WithCredentials the
// token and account come from the options, then CLOUDFLARE_API_KEY and
// CLOUDFLARE_ACCOUNT_ID, and both must be known.
func New(opts ...Option) (*Provider, error) {
	cfg := config{baseURL: BaseURL}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	ambientKey, ambientAccount := os.Getenv(apiKeyEnv), os.Getenv(accountIDEnv)
	resolve := cfg.resolver(ambientKey, ambientAccount, strings.Contains(cfg.baseURL, "{account}"))
	if cfg.credentials == nil {
		if _, _, err := resolve(context.Background()); err != nil {
			return nil, err
		}
	}
	maxRetries := 2
	if cfg.maxRetries != nil {
		maxRetries = *cfg.maxRetries
	}
	base := cfg.baseURL
	e, err := systemone.New(systemone.Config{
		Provider: providerName,
		// One credential read per request yields both the account and the
		// key, so a rotated entry never pairs one's key with another's
		// account.
		Endpoint: func(ctx context.Context) (string, string, error) {
			key, account, err := resolve(ctx)
			if err != nil {
				return "", "", err
			}
			return strings.TrimRight(strings.ReplaceAll(base, "{account}", account), "/") + "/run", key, nil
		},
		Payload: func(model string, request map[string]any) any {
			return map[string]any{"model": model, "input": request}
		},
		Output: output,
		ImageForm: func(model string) systemone.ImageForm {
			if isClef(model) {
				return systemone.ImageDataURLs
			}
			return systemone.ImageAssets
		},
		Preflight:       preflight,
		ErrorKind:       errorKind,
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
	return &Provider{engine: e}, nil
}

// resolver returns the per-request token and account resolution; the
// account is required only when the base URL has an {account} placeholder.
func (c config) resolver(ambientKey, ambientAccount string, needAccount bool) func(context.Context) (string, string, error) {
	return func(ctx context.Context) (string, string, error) {
		key, account := c.apiKey, c.accountID
		if c.credentials != nil && (key == "" || account == "") {
			cred, ok, err := c.credentials.Read(ctx, providerName)
			if err != nil {
				return "", "", fmt.Errorf("%w: read %s credential: %w", llm.ErrAuth, providerName, err)
			}
			if ok {
				if cred.Type != "" && cred.Type != "api_key" {
					return "", "", fmt.Errorf("%w: stored %s credential has type %q; Workers AI takes an API token", llm.ErrAuth, providerName, cred.Type)
				}
				if key == "" {
					key = cred.Key
				}
				if account == "" {
					account = cred.Env[accountIDEnv]
				}
			}
		}
		if key == "" {
			key = ambientKey
		}
		if account == "" {
			account = ambientAccount
		}
		switch {
		case key == "":
			return "", "", fmt.Errorf("%w: missing Cloudflare API token; set WithAPIKey, %s, or run 'llm-cli login cloudflare-workers-ai'", llm.ErrAuth, apiKeyEnv)
		case account == "" && needAccount:
			return "", "", fmt.Errorf("%w: missing Cloudflare account ID; set WithAccountID, %s, or run 'llm-cli login cloudflare-workers-ai'", llm.ErrAuth, accountIDEnv)
		case needAccount && !accountIDPattern.MatchString(account):
			return "", "", fmt.Errorf("%w: the Cloudflare account ID must be letters, digits, '-' or '_' (it is a URL path segment)", llm.ErrAuth)
		}
		return key, account, nil
	}
}

// accountIDPattern bounds the account ID to one URL path segment
// (Cloudflare's IDs are 32 hex digits).
var accountIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

func isClef(model string) bool { return strings.HasPrefix(model, "@cf/cloudflare/clef") }

// Clef's documented request limits, checked before I/O.
const (
	maxClefImages    = 4
	maxClefQuestions = 64
)

var clefQuestionName = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,100}$`)

func preflight(req *llm.ClassifyRequest) error {
	if !isClef(req.Model) {
		return nil
	}
	if len(req.Images) > maxClefImages {
		return fmt.Errorf("%w: Clef accepts at most %d images, got %d", llm.ErrBadRequest, maxClefImages, len(req.Images))
	}
	if len(req.Questions) > maxClefQuestions {
		return fmt.Errorf("%w: Clef accepts at most %d questions, got %d", llm.ErrBadRequest, maxClefQuestions, len(req.Questions))
	}
	for _, q := range req.Questions {
		if !clefQuestionName.MatchString(q.Name) {
			return fmt.Errorf("%w: Clef question names are 1-100 letters, digits, '_', '.' or '-'; got %q", llm.ErrBadRequest, q.Name)
		}
	}
	return nil
}

// errorKind classifies Workers AI's native error codes: 5007 (no such
// model), 3006 (request too large) and 3007 (timeout).
func errorKind(_ int, code, _ string) error {
	switch code {
	case "5007":
		return llm.ErrNotFound
	case "3006":
		return llm.ErrContextTooLong
	case "3007":
		return llm.ErrTimeout
	}
	return nil
}

// output unwraps Cloudflare's envelope to the System One output.
func output(status int, body []byte) (json.RawMessage, error) {
	var envelope struct {
		Success *bool `json:"success"`
		Errors  []struct {
			Code    any    `json:"code"` // documented as integers; strings tolerated
			Message string `json:"message"`
		} `json:"errors"`
		Result json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, providerutil.ResponseError(providerName, fmt.Errorf("unreadable response: %w", err))
	}
	if envelope.Success != nil && !*envelope.Success {
		var code string
		messages := make([]string, 0, len(envelope.Errors))
		for _, e := range envelope.Errors {
			if code == "" {
				code = systemone.CodeString(e.Code)
			}
			if e.Message != "" {
				messages = append(messages, e.Message)
			}
		}
		kind := errorKind(status, code, "")
		if kind == nil {
			kind = llm.ErrServer
		}
		message := "Workers AI request failed"
		if len(messages) > 0 {
			message += ": " + strings.Join(messages, "; ")
		}
		return nil, &llm.ProviderError{Provider: providerName, HTTPStatus: status, Code: code, Message: message, RawBody: body, Kind: kind}
	}
	var result struct {
		Answers json.RawMessage `json:"answers"`
		State   string          `json:"state"`
		Result  json.RawMessage `json:"result"`
	}
	if err := json.Unmarshal(envelope.Result, &result); err != nil {
		return nil, providerutil.ResponseError(providerName, errors.New("unexpected response"))
	}
	if len(result.Answers) > 0 {
		return envelope.Result, nil // Cloudflare-hosted models answer directly
	}
	if result.State != "Completed" {
		return nil, providerutil.ResponseError(providerName, fmt.Errorf("run did not complete (state %q)", result.State))
	}
	return result.Result, nil
}

// Name returns "cloudflare-workers-ai", pi's provider ID.
func (p *Provider) Name() string { return providerName }

// Capabilities lists CapabilityClassify and CapabilityImageInput.
func (p *Provider) Capabilities() []llm.Capability {
	return []llm.Capability{llm.CapabilityClassify, llm.CapabilityImageInput}
}

// Classify answers req's questions.
func (p *Provider) Classify(ctx context.Context, req *llm.ClassifyRequest) (*llm.ClassifyResult, error) {
	return p.engine.Classify(ctx, req)
}

// Models is not offered: Workers AI has no unauthenticated classifier
// catalog. The embedded table (llm.LookupModelInfo) knows the catalogued
// models.
func (p *Provider) Models(context.Context) ([]llm.ModelInfo, error) {
	return nil, fmt.Errorf("%w: %s does not list models", llm.ErrUnsupported, providerName)
}

// Chat is not offered: this provider only classifies.
func (p *Provider) Chat(context.Context, *llm.Request) (*llm.Response, error) {
	return nil, fmt.Errorf("%w: %s serves classifier models only; use llm.Classify", llm.ErrUnsupported, providerName)
}

// ChatStream is not offered: this provider only classifies.
func (p *Provider) ChatStream(context.Context, *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		yield(nil, fmt.Errorf("%w: %s serves classifier models only; use llm.Classify", llm.ErrUnsupported, providerName))
	}
}
