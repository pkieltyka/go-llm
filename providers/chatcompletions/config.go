package chatcompletions

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

// Dialect supplies provider-specific behavior for a chat-completions-shaped
// provider. Behavioral hooks live here; quirks that are pure data are
// declared on the Compat struct instead.
//
// Dialect is an ADVANCED surface: it exists for preset packages
// (providers/openrouter, providers/vllm) that need code-level hooks. Most
// OpenAI-compatible servers are reachable with New plus a declarative Compat
// alone. The interface is exempt from the package's pre-1.0 stability
// intentions — prefer Compat, which only grows.
type Dialect interface {
	Name() string
	DefaultBaseURL() string
	// APIKeyEnv names the environment variable consulted when no API key is
	// configured. An empty name disables the env fallback.
	APIKeyEnv() string
	Capabilities() []llm.Capability
	// Compat declares the dialect's data-expressible quirks.
	Compat() Compat
	// ApplyRequest injects dialect-specific fields into the outgoing params.
	// extras is the mutable map the adapter attaches as JSON extra fields
	// after ApplyRequest returns; the adapter pre-populates it with the
	// unified fields it owns (currently the Compat-mapped Effort fields),
	// which dialects may merge with but must let win on conflict (FS §14).
	ApplyRequest(req *llm.Request, params *sdk.ChatCompletionNewParams, extras JSONObject) error
	MapStopReason(raw string) llm.StopReason
	MapErrorStatus(status int, code, message string) error
	// ExtractParts maps a choice message to parts. Returning (nil, nil, nil)
	// defers to the adapter's default chat-completions mapping (reasoning +
	// reasoning_details, content, refusal, tool calls with malformed-call
	// drops), which tags ReasoningPart.Provider with Dialect.Name().
	ExtractParts(raw JSONObject, message RawMessage) ([]llm.Part, []llm.DroppedToolCall, error)
	ExtractExtras(raw JSONObject, choice RawChoice) any
	MapUsage(model string, raw RawUsage, table llm.PriceTable) llm.Usage
	Models(ctx context.Context, p *Provider) ([]llm.ModelInfo, error)
}

// Compat declares the data-expressible quirks of a chat-completions-shaped
// provider (ARCH §3.3). New accepts it directly via WithCompat; anything
// requiring code stays a Dialect hook. SSE comment keep-alives (OpenRouter's
// ": OPENROUTER PROCESSING") need no flag: the adapter's SSE reader always
// skips comment lines per the SSE spec.
type Compat struct {
	// StreamIncludeUsage sets stream_options.include_usage on streaming
	// requests so the final chunk carries usage. The adapter emits MessageEnd
	// only at stream end ([DONE] or EOF), so a trailing usage-only chunk —
	// OpenRouter delivers usage+cost on a final empty-choices chunk — is
	// never dropped.
	StreamIncludeUsage bool

	// MapEffort translates the unified Effort into top-level wire request
	// fields (nil result: send nothing). When MapEffort itself is nil the
	// adapter sends {"reasoning_effort": "<level>"} — the spelling shared by
	// OpenAI's Chat Completions surface and vLLM. Dialects with an object
	// spelling return it whole, e.g. OpenRouter's
	// {"reasoning": {"effort": "<level>"}}.
	MapEffort func(llm.Effort) map[string]any

	// ReasoningReplayField, when non-empty, names the assistant-message field
	// used to replay same-provider plain-text reasoning (ReasoningPart.Text
	// with a matching Provider and no Raw payload) back to the server — e.g.
	// "reasoning" for modern vLLM, "reasoning_content" for pre-rename
	// servers. Raw reasoning_details payloads always replay as
	// reasoning_details regardless of this field. Empty (the default) drops
	// plain-text reasoning on replay, matching servers whose chat templates
	// discard prior thinking anyway.
	ReasoningReplayField string

	// SniffMidStreamErrors treats choice-less SSE data events that carry an
	// error payload — nested {"error":{...}} or legacy flat
	// {"object":"error",...} — as normalized in-stream errors. vLLM emits
	// these after HTTP 200 when generation fails mid-stream; without the
	// sniff such events would be silently absorbed as extras.
	SniffMidStreamErrors bool

	// NormalizeToolUseStop upgrades an end-turn stop to StopReasonToolUse
	// when the response carries tool calls, applying FS §5's semantic
	// definition (the model wants tool results) for servers that end
	// NAMED-function forced tool calls with finish_reason "stop" — observed
	// on vLLM, where tool_choice:"required" reports "tool_calls" correctly.
	// Truncation and error finishes are never
	// upgraded, and StopReasonRaw keeps the wire value.
	NormalizeToolUseStop bool

	// DefaultHeaders are applied to every request (e.g. OpenRouter
	// attribution headers set at construction). Per-request headers from the
	// dialect win key-by-key on both the blocking and streaming paths.
	DefaultHeaders http.Header
}

// Config contains provider construction settings for NewWithDialect.
type Config struct {
	Dialect    Dialect
	APIKey     string
	APIKeyFunc func(context.Context) (string, error)
	// KeyOptional permits construction without any API key (self-hosted
	// servers are commonly keyless). Requests are then sent without an
	// Authorization header.
	KeyOptional bool
	BaseURL     string
	HTTPClient  *http.Client
	MaxRetries  *int
	Timeout     time.Duration
	PriceTable  llm.PriceTable
	Logger      *slog.Logger
	WireCapture func(llm.WireCapture)
}

// NewWithDialect constructs a provider from a full Dialect implementation.
// It is the advanced entry point used by preset packages; most callers want
// New.
func NewWithDialect(cfg Config) (*Provider, error) {
	if cfg.Dialect == nil {
		return nil, fmt.Errorf("%w: nil chatcompletions dialect", llm.ErrBadRequest)
	}
	if cfg.APIKey == "" && cfg.APIKeyFunc == nil {
		if env := cfg.Dialect.APIKeyEnv(); env != "" {
			cfg.APIKey = os.Getenv(env)
		}
	}
	if cfg.APIKey == "" && cfg.APIKeyFunc == nil && !cfg.KeyOptional {
		return nil, fmt.Errorf("%w: missing %s API key; set WithAPIKey or %s", llm.ErrAuth, cfg.Dialect.Name(), cfg.Dialect.APIKeyEnv())
	}
	if cfg.HTTPClient == nil {
		cfg.HTTPClient = llm.DefaultHTTPClient()
	}
	if cfg.MaxRetries != nil && *cfg.MaxRetries < 0 {
		return nil, fmt.Errorf("%w: max retries must be >= 0", llm.ErrBadRequest)
	}
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = cfg.Dialect.DefaultBaseURL()
	}
	if baseURL == "" {
		return nil, fmt.Errorf("%w: %s requires a base URL", llm.ErrBadRequest, cfg.Dialect.Name())
	}
	compat := cfg.Dialect.Compat()
	client := sdk.NewClient(sdkOptions(cfg, baseURL)...)
	return &Provider{
		dialect:    cfg.Dialect,
		compat:     compat,
		client:     &client,
		httpClient: providerutil.ObservedHTTPClient(cfg.HTTPClient, cfg.Dialect.Name(), cfg.Logger, cfg.WireCapture),
		apiKey:     cfg.APIKey,
		apiKeyFunc: cfg.APIKeyFunc,
		baseURL:    baseURL,
		maxRetries: compatMaxRetries(cfg.MaxRetries),
		timeout:    cfg.Timeout,
		priceTable: cfg.PriceTable,
		logger:     cfg.Logger,
		headers:    cloneHeader(compat.DefaultHeaders),
	}, nil
}

func compatMaxRetries(configured *int) int {
	if configured == nil {
		return 2
	}
	return *configured
}

func sdkOptions(cfg Config, baseURL string) []sdkoption.RequestOption {
	opts := []sdkoption.RequestOption{
		sdkoption.WithHTTPClient(providerutil.ObservedHTTPClient(cfg.HTTPClient, cfg.Dialect.Name(), cfg.Logger, cfg.WireCapture)),
		sdkoption.WithBaseURL(baseURL),
		sdkoption.WithAdminAPIKey(""),
		// The SDK also injects OpenAI-Organization / OpenAI-Project from
		// OPENAI_ORG_ID / OPENAI_PROJECT_ID — identity headers that must not
		// leak to arbitrary third-party base URLs (self-hosted vLLM, etc.).
		sdkoption.WithHeaderDel("OpenAI-Organization"),
		sdkoption.WithHeaderDel("OpenAI-Project"),
	}
	opts = append(opts, providerutil.AmbientCustomHeaderDeleteOptions()...)
	// Default and per-request headers are applied per call via requestOptions
	// (Chat) and requestHeaders (streaming) so per-request values win on both
	// paths; no header middleware here — SDK middleware runs after request
	// options and would re-assert construction defaults over per-request ones.
	if cfg.APIKeyFunc != nil {
		opts = append(opts,
			sdkoption.WithAPIKey("dynamic"),
			sdkoption.WithMiddleware(func(req *http.Request, next sdkoption.MiddlewareNext) (*http.Response, error) {
				key, err := cfg.APIKeyFunc(req.Context())
				if err != nil {
					return nil, err
				}
				req.Header.Set("Authorization", "Bearer "+key)
				return next(req)
			}),
		)
	} else {
		// An empty key sends no Authorization header at all (keyless
		// self-hosted servers): the SDK omits the header for empty keys.
		opts = append(opts, sdkoption.WithAPIKey(cfg.APIKey))
	}
	if cfg.MaxRetries != nil {
		opts = append(opts, sdkoption.WithMaxRetries(*cfg.MaxRetries))
	}
	return opts
}

func applyHeaders(dst http.Header, src http.Header) {
	if len(src) == 0 {
		return
	}
	if dst == nil {
		return
	}
	for key, values := range src {
		dst.Del(key)
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

func cloneHeader(in http.Header) http.Header {
	if len(in) == 0 {
		return nil
	}
	out := make(http.Header, len(in))
	for key, values := range in {
		out[key] = append([]string(nil), values...)
	}
	return out
}
