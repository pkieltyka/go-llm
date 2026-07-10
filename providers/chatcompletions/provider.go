package chatcompletions

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"iter"
	"log/slog"
	"net/http"
	"strings"
	"time"

	sdk "github.com/openai/openai-go/v3"
	sdkoption "github.com/openai/openai-go/v3/option"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// Provider is a shared implementation for OpenAI-compatible chat providers.
type Provider struct {
	dialect    Dialect
	compat     Compat
	client     *sdk.Client
	httpClient *http.Client
	apiKey     string
	apiKeyFunc func(context.Context) (string, error)
	baseURL    string
	maxRetries int
	timeout    time.Duration
	priceTable llm.PriceTable
	logger     *slog.Logger
	headers    http.Header
}

// Client exposes the underlying OpenAI SDK client.
func (p *Provider) Client() *sdk.Client {
	if p == nil {
		return nil
	}
	return p.client
}

// Name returns the provider identifier.
func (p *Provider) Name() string { return p.dialect.Name() }

// Capabilities returns provider-level capabilities.
func (p *Provider) Capabilities() []llm.Capability {
	return append([]llm.Capability(nil), p.dialect.Capabilities()...)
}

// Models lists provider models.
func (p *Provider) Models(ctx context.Context) ([]llm.ModelInfo, error) {
	ctx, cancel := p.contextWithTimeout(ctx)
	defer cancel()
	return p.dialect.Models(ctx, p)
}

// Chat performs a blocking chat completion.
func (p *Provider) Chat(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	start := time.Now()
	ctx, cancel := p.contextWithTimeout(ctx)
	defer cancel()

	params, err := p.BuildParams(req, false)
	if err != nil {
		p.logFailure(ctx, req, start, err)
		return nil, err
	}
	resp, err := p.client.Chat.Completions.New(ctx, params, p.requestOptions(req)...)
	if err != nil {
		err = p.mapError(err)
		p.logFailure(ctx, req, start, err)
		return nil, err
	}
	mapped, err := p.mapResponse(resp)
	if err != nil {
		p.logFailure(ctx, req, start, err)
		return nil, err
	}
	p.logSuccess(ctx, mapped, start)
	return mapped, nil
}

// ChatStream streams a chat completion as normalized events.
func (p *Provider) ChatStream(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return providerutil.SingleUse(p.Name(), func(yield func(llm.Event, error) bool) {
		start := time.Now()
		ctx, cancel := p.contextWithTimeout(ctx)
		defer cancel()

		params, err := p.BuildParams(req, true)
		if err != nil {
			p.logFailure(ctx, req, start, err)
			yield(nil, err)
			return
		}
		remote := providerutil.StreamContract(p.Name(), p.streamEvents(ctx, req, params))
		for event, err := range remote {
			if err != nil {
				p.logFailure(ctx, req, start, err)
				yield(nil, err)
				return
			}
			if end, ok := event.(llm.MessageEnd); ok {
				providerutil.LogStreamEnd(ctx, p.logger, p.Name(), req, end, "", start)
			}
			if !yield(event, nil) {
				return
			}
		}
	})
}

type requestHeaderer interface {
	RequestHeaders(*llm.Request) http.Header
}

func (p *Provider) requestHeaders(req *llm.Request) http.Header {
	headers := cloneHeader(p.headers)
	if headerer, ok := p.dialect.(requestHeaderer); ok {
		requestHeaders := headerer.RequestHeaders(req)
		if len(requestHeaders) > 0 && headers == nil {
			headers = http.Header{}
		}
		applyHeaders(headers, requestHeaders)
	}
	return headers
}

func (p *Provider) requestOptions(req *llm.Request) []sdkoption.RequestOption {
	headers := p.requestHeaders(req)
	if len(headers) == 0 {
		return nil
	}
	opts := make([]sdkoption.RequestOption, 0, len(headers))
	for key, values := range headers {
		for i, value := range values {
			if i == 0 {
				opts = append(opts, sdkoption.WithHeader(key, value))
			} else {
				opts = append(opts, sdkoption.WithHeaderAdd(key, value))
			}
		}
	}
	return opts
}

func (p *Provider) contextWithTimeout(ctx context.Context) (context.Context, context.CancelFunc) {
	return providerutil.ContextWithTimeout(ctx, p.timeout)
}

// DoJSON performs an authenticated JSON request against the provider base URL.
func (p *Provider) DoJSON(ctx context.Context, method, path string, body any, out any) error {
	return p.DoJSONURL(ctx, method, strings.TrimRight(p.baseURL, "/")+path, body, out)
}

// DoJSONURL is DoJSON against an absolute URL — same auth, headers, and
// error mapping. Presets use it for server extension endpoints that live
// outside the API base path (e.g. vLLM parks its /tokenize family at the
// server root while the OpenAI surface hangs off /v1). WithTimeout applies
// here like every other provider call (an already-tighter caller deadline
// still governs).
func (p *Provider) DoJSONURL(ctx context.Context, method, url string, body any, out any) error {
	ctx, cancel := p.contextWithTimeout(ctx)
	defer cancel()
	var reader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return err
		}
		reader = bytes.NewReader(data)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, reader)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	key := p.apiKey
	if p.apiKeyFunc != nil {
		key, err = p.apiKeyFunc(ctx)
		if err != nil {
			return err
		}
	}
	if key != "" {
		// Keyless providers (self-hosted servers) send no Authorization header.
		req.Header.Set("Authorization", "Bearer "+key)
	}
	applyHeaders(req.Header, p.headers)
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return p.mapError(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return p.mapHTTPResponseError(resp)
	}
	if out == nil {
		return nil
	}
	return providerutil.NormalizeRemoteError(p.Name(), json.NewDecoder(resp.Body).Decode(out))
}

func (p *Provider) logSuccess(ctx context.Context, resp *llm.Response, start time.Time) {
	providerutil.LogSuccess(ctx, p.logger, p.Name(), resp, start)
}

func (p *Provider) logFailure(ctx context.Context, req *llm.Request, start time.Time, err error) {
	providerutil.LogFailure(ctx, p.logger, p.Name(), req, start, err)
}
