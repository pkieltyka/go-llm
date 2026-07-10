package vllm

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
)

const providerName = "vllm"

// Options carries vLLM-specific request extensions, passed through
// Request.ProviderOptions. All fields are top-level body extensions from
// vLLM's chat-completions protocol; unset fields are omitted from the wire.
type Options struct {
	// TopK, MinP, RepetitionPenalty are vLLM sampling extras.
	TopK              *int
	MinP              *float64
	RepetitionPenalty *float64
	// StopTokenIDs stops generation on exact token ids (stop_token_ids).
	StopTokenIDs []int
	// EnableThinking toggles chat_template_kwargs.enable_thinking — the
	// Qwen3-style thinking switch (Granite/DeepSeek templates use their own
	// keys via ChatTemplateKwargs). It merges into ChatTemplateKwargs and
	// wins on conflict.
	EnableThinking *bool
	// ChatTemplateKwargs passes arbitrary chat-template variables
	// (chat_template_kwargs).
	ChatTemplateKwargs map[string]any
	// StructuredOutputs selects a native constrained-decoding mode (the
	// v0.12+ structured_outputs request param). See the type documentation
	// for modes, conflict rules, and the thinking interaction.
	StructuredOutputs *StructuredOutputs
	// XArgs is vLLM's generic engine passthrough (vllm_xargs). Values must
	// be strings, numbers, or lists per the vLLM protocol.
	XArgs map[string]any
}

// StructuredOutputs selects one of vLLM's native constrained-decoding modes,
// sent as the top-level structured_outputs request param (v0.12+; also
// present on the 0.23/0.24 line). Exactly one of Regex, Choice, Grammar, or
// StructuralTag must be set. JSON-schema output deliberately does NOT live
// here: use the unified llm.Request.ResponseFormat, which the preset sends
// as the era-portable response_format json_schema spelling.
//
// Conflicts fail loud at request build (FS §14: one constraint system per
// request):
//   - set together with Request.ResponseFormat → llm.ErrBadRequest
//   - set on a WithLegacyEra provider → llm.ErrUnsupported: the param only
//     exists on v0.12+ servers, and the pre-v0.12 guided_* spelling is not
//     emitted as a fallback because modern servers silently ignore unknown
//     fields (a probed guided_json request free-forms with no error)
//   - zero or multiple mode fields → llm.ErrBadRequest (the server 400s on
//     multiple modes; the client fails first with a clearer message)
//
// Live finding (vLLM 0.23/0.24-family, qwen3 reasoning parser): constraint
// modes misbehave INTERMITTENTLY while thinking is active — a Choice
// constraint can return concatenated choice members ("greengreen",
// "greenblue") with thinking on (reproduced 5/5 at temperature 1.0, only
// sometimes at temperature 0), and exact output with it off — so disable
// thinking (llm.EffortNone or EnableThinking=false) when using these modes
// on reasoning-parser hosts.
type StructuredOutputs struct {
	// Regex constrains output to a regular expression (Rust-regex syntax on
	// the xgrammar/guidance backends; lm-format-enforcer uses Python re).
	Regex string
	// Choice constrains output to exactly one of the listed strings.
	Choice []string
	// Grammar constrains output with a context-free EBNF grammar.
	Grammar string
	// StructuralTag is the raw structural-tag constraint JSON (schemas
	// confined within tags); raw because its shape varies across vLLM
	// versions.
	StructuralTag json.RawMessage
	// WhitespacePattern overrides the constrained decoder's whitespace
	// pattern. vLLM documents it for JSON-schema decoding; it is carried
	// here for wire parity with StructuredOutputsParams.
	WhitespacePattern string
}

// ForProvider identifies these options as vLLM-specific.
func (Options) ForProvider() string { return providerName }

// Option configures a vLLM provider.
type Option func(*config)

type config struct {
	apiKey      string
	apiKeyFunc  func(context.Context) (string, error)
	httpClient  *http.Client
	maxRetries  *int
	timeout     time.Duration
	priceTable  llm.PriceTable
	logger      *slog.Logger
	wireCapture func(llm.WireCapture)
	legacyEra   bool
}

// WithAPIKey sets a static API key (vLLM's --api-key bearer token). Keys are
// optional: keyless servers need no auth and no environment fallback is
// consulted.
func WithAPIKey(key string) Option {
	return func(c *config) {
		c.apiKey = key
		c.apiKeyFunc = nil
	}
}

// WithAPIKeyFunc sets a per-request key resolver. It wins over WithAPIKey.
func WithAPIKeyFunc(fn func(context.Context) (string, error)) Option {
	return func(c *config) { c.apiKeyFunc = fn }
}

// WithLegacyEra targets pre-v0.12 vLLM servers: assistant reasoning replays
// under the legacy `reasoning_content` field instead of `reasoning`.
// Response parsing reads both spellings in either era, and structured output
// uses the era-portable `response_format: json_schema` spelling throughout.
func WithLegacyEra() Option {
	return func(c *config) { c.legacyEra = true }
}

// WithHTTPClient replaces the provider's default HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) { c.httpClient = client }
}

// WithMaxRetries bounds retry attempts. Blocking calls delegate the count to
// the OpenAI SDK's retry layer; streaming calls use the engine's direct SSE
// transport, which applies the same bound to billing-safe pre-stream
// rejections only (ARCH §3.3/§3.4).
func WithMaxRetries(n int) Option {
	return func(c *config) { c.maxRetries = &n }
}

// WithTimeout applies a context deadline to provider calls.
func WithTimeout(timeout time.Duration) Option {
	return func(c *config) { c.timeout = timeout }
}

// WithPriceTable enables cost estimation for served models. Self-hosted
// servers report no native cost; without a table, Usage.CostUSD stays nil.
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

func (c config) chatcompletionsConfig(baseURL string) chatcompletions.Config {
	return chatcompletions.Config{
		Dialect:     dialect{legacyEra: c.legacyEra},
		APIKey:      c.apiKey,
		APIKeyFunc:  c.apiKeyFunc,
		KeyOptional: true,
		BaseURL:     baseURL,
		HTTPClient:  c.httpClient,
		MaxRetries:  c.maxRetries,
		Timeout:     c.timeout,
		PriceTable:  c.priceTable,
		Logger:      c.logger,
		WireCapture: c.wireCapture,
	}
}
