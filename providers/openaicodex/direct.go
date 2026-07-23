package openaicodex

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
	"time"

	"github.com/openai/openai-go/v3/packages/ssestream"
	"github.com/openai/openai-go/v3/responses"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/provideroauth"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

const (
	// defaultTransportMaxRetries matches the OpenAI SDK's default of two
	// additional attempts. WithMaxRetries overrides it.
	defaultTransportMaxRetries = 2
	transportRetryBaseDelay    = 500 * time.Millisecond
	transportRetryMaxDelay     = 30 * time.Second
)

type codexTransport struct {
	endpoint     string
	httpClient   *http.Client
	source       *provideroauth.Source
	originator   string
	headerFunc   func(*http.Request)
	authFunc     provideroauth.ApplyHeadersFunc
	providerName string
	maxRetries   int
	// sleep overrides retry backoff waiting in tests; nil uses a real timer.
	sleep func(context.Context, time.Duration) error
}

// postStream issues the streaming request with bounded, billing-safe retries.
// Only 429/503/529 rejections are retried: those statuses mean the request
// was never accepted, so a retry cannot double-bill. The retry decision is
// made purely on the response status line — a 2xx response is returned
// untouched before any stream bytes are consumed, and is never retried.
func (t codexTransport) postStream(ctx context.Context, body []byte, lite bool) (*http.Response, error) {
	if strings.TrimSpace(t.endpoint) == "" {
		return nil, fmt.Errorf("%w: missing OpenAI Codex endpoint", llm.ErrBadRequest)
	}
	if t.source == nil {
		return nil, fmt.Errorf("%w: missing OpenAI Codex OAuth source", llm.ErrAuth)
	}
	for attempt := 0; ; attempt++ {
		resp, err := t.doAttempt(ctx, body, lite)
		if err != nil {
			return nil, err
		}
		if resp == nil || attempt >= t.maxRetries || !retryableCodexStatus(resp.StatusCode) {
			return resp, nil
		}
		delay := llm.RetryAfter(resp)
		drainAndClose(resp.Body)
		if delay <= 0 {
			delay = transportRetryBaseDelay << attempt
		}
		if delay > transportRetryMaxDelay {
			delay = transportRetryMaxDelay
		}
		if err := t.wait(ctx, delay); err != nil {
			return nil, err
		}
	}
}

func (t codexTransport) doAttempt(ctx context.Context, body []byte, lite bool) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	t.applyHeaders(req)
	// Applied after applyHeaders so test header overrides compose with it:
	// the backend serves the gpt-5.6 family only when the Lite header is set.
	if lite {
		req.Header.Set(codexResponsesLiteHeader, "true")
	}
	client := t.httpClient
	if client == nil {
		client = llm.DefaultHTTPClient()
	}
	return provideroauth.DoWithAuthRetry(req, client.Do, t.source, t.applyAuth)
}

// retryableCodexStatus reports whether a status is safe to retry: the codex
// backend rejected the request before doing any work (rate limit or
// overload), so no tokens were billed and no stream was started.
func retryableCodexStatus(status int) bool {
	switch status {
	case http.StatusTooManyRequests, http.StatusServiceUnavailable, 529:
		return true
	default:
		return false
	}
}

func (t codexTransport) wait(ctx context.Context, d time.Duration) error {
	if t.sleep != nil {
		return t.sleep(ctx, d)
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// drainAndClose releases the connection for reuse before a retry. Unlike the
// unlimited twin in providers/internal/provideroauth (which drains small
// token-endpoint error envelopes), this one caps the drain at 1MB: the bodies
// discarded here are rejected streaming responses that could be arbitrarily
// large. The difference is intentional.
func drainAndClose(body io.ReadCloser) {
	if body == nil {
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(body, 1<<20))
	_ = body.Close()
}

func (t codexTransport) applyHeaders(req *http.Request) {
	if t.headerFunc != nil {
		t.headerFunc(req)
		return
	}
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", defaultCodexUserAgent)
}

func (t codexTransport) applyAuth(req *http.Request, cred llm.AuthCredential) {
	if t.authFunc != nil {
		t.authFunc(req, cred)
		return
	}
	applyOAuthHeaders(t.originator)(req, cred)
}

// codexStreamingBody rewrites Responses params for the codex backend, which
// accepts a narrower parameter surface than the public Responses API.
//
// Dropped fields (keep this list in sync with the godoc on New):
//   - max_output_tokens — the subscription backend controls the output
//     budget itself; pi's openai-codex-responses.ts never sends it.
//   - top_p — not part of pi's codex request surface.
//   - temperature — pi's request builder forwards it when a caller sets it,
//     but the live backend hard-rejects it (400 "Unsupported parameter:
//     temperature" observed 2026-07-03 against the pinned gpt-5.x models),
//     so it is dropped rather than failing every tuned request.
func codexStreamingBody(params responses.ResponseNewParams) ([]byte, error) {
	raw, err := json.Marshal(params)
	if err != nil {
		return nil, err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, err
	}
	delete(fields, "max_output_tokens")
	delete(fields, "top_p")
	delete(fields, "temperature")
	if isGPT56Model(string(params.Model)) {
		if err := applyResponsesLite(fields); err != nil {
			return nil, err
		}
	}
	fields["stream"] = json.RawMessage("true")
	return json.Marshal(fields)
}

func (p *Provider) codexEvents(ctx context.Context, params responses.ResponseNewParams) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		requestModel := string(params.Model)
		body, err := codexStreamingBody(params)
		if err != nil {
			yield(nil, err)
			return
		}
		remote := providerutil.StreamContract(providerName, func(remoteYield func(llm.Event, error) bool) {
			resp, err := p.transport.postStream(ctx, body, isGPT56Model(requestModel))
			if err != nil {
				remoteYield(nil, p.adapter().MapError(err))
				return
			}
			if resp == nil {
				remoteYield(nil, &llm.ProviderError{Provider: providerName, Message: "nil HTTP response", Kind: llm.ErrServer})
				return
			}
			defer resp.Body.Close()
			if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
				remoteYield(nil, p.adapter().MapHTTPResponseError(resp))
				return
			}

			decoder := ssestream.NewDecoder(resp)
			if decoder == nil {
				remoteYield(nil, &llm.ProviderError{Provider: providerName, Message: "nil SSE decoder", Kind: llm.ErrServer})
				return
			}
			defer decoder.Close()

			state := p.adapter().NewStreamState(requestModel)
			for decoder.Next() {
				evt := decoder.Event()
				data := bytes.TrimSpace(evt.Data)
				if bytes.Equal(data, []byte("[DONE]")) {
					continue
				}
				var streamEvent responses.ResponseStreamEventUnion
				if err := json.Unmarshal(data, &streamEvent); err != nil {
					for _, event := range state.Finish() {
						event = normalizeCodexStreamEvent(event, requestModel)
						if !remoteYield(event, nil) {
							return
						}
					}
					remoteYield(nil, fmt.Errorf("malformed openai-codex stream event: %w", err))
					return
				}
				if streamEvent.Type == "" && evt.Type != "" {
					streamEvent.Type = evt.Type
				}
				events, err := state.MapEvent(streamEvent)
				for _, event := range events {
					event = normalizeCodexStreamEvent(event, requestModel)
					if !remoteYield(event, nil) {
						return
					}
				}
				if err != nil {
					remoteYield(nil, err)
					return
				}
			}
			for _, event := range state.Finish() {
				event = normalizeCodexStreamEvent(event, requestModel)
				if !remoteYield(event, nil) {
					return
				}
			}
			if err := decoder.Err(); err != nil {
				remoteYield(nil, p.adapter().MapError(err))
				return
			}
			if err := ctx.Err(); err != nil {
				remoteYield(nil, err)
			}
		})
		remote(yield)
	}
}

// normalizeCodexStreamEvent keeps the server-reported model identity on
// MessageStart, matching the openai provider. The codex backend normally
// reports the model (sometimes a dated snapshot of the requested alias);
// only when an event omits it entirely do we fall back to the request model
// so Response.Model is never empty.
func normalizeCodexStreamEvent(event llm.Event, requestModel string) llm.Event {
	if requestModel == "" {
		return event
	}
	if e, ok := providerutil.DerefEvent(event).(llm.MessageStart); ok && e.Model == "" {
		e.Model = requestModel
		return e
	}
	return event
}
