// Package providerutil hosts small helpers shared by all provider packages
// and engines (anthropic, openai, openaicodex, openrouter, responsesapi,
// chatcompletions): part/event deref normalization, the unified error-kind
// classifier, observability wiring, strict-schema checks, and OpenAI SDK
// ambient-header hygiene.
package providerutil

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	sdkoption "github.com/openai/openai-go/v3/option"
	llm "github.com/pkieltyka/go-llm"
)

// CustomHeadersEnv is the OpenAI SDK ambient env var that injects arbitrary
// request headers.
const CustomHeadersEnv = "OPENAI_CUSTOM_HEADERS"

// AmbientCustomHeaderDeleteOptions removes ambient Authorization before a
// provider applies its own authentication. Other OPENAI_CUSTOM_HEADERS values
// remain available to callers.
func AmbientCustomHeaderDeleteOptions() []sdkoption.RequestOption {
	return []sdkoption.RequestOption{sdkoption.WithHeaderDel("Authorization")}
}

// AmbientCustomHeaders returns non-authentication headers supplied through
// OPENAI_CUSTOM_HEADERS. Direct transports use it to match SDK-backed paths
// without inheriting an ambient bearer credential.
func AmbientCustomHeaders() http.Header {
	headers := http.Header{}
	for _, line := range strings.Split(os.Getenv(CustomHeadersEnv), "\n") {
		name, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		name = http.CanonicalHeaderKey(strings.TrimSpace(name))
		if name == "" || strings.EqualFold(name, "Authorization") {
			continue
		}
		headers.Set(name, strings.TrimSpace(value))
	}
	if len(headers) == 0 {
		return nil
	}
	return headers
}

// ObservedHTTPClient wraps client's transport with retry Warn logging and
// redacted wire capture when either is configured; otherwise it returns the
// client unchanged.
func ObservedHTTPClient(client *http.Client, providerName string, logger *slog.Logger, capture func(llm.WireCapture)) *http.Client {
	if client == nil {
		client = llm.DefaultHTTPClient()
	}
	if logger == nil && capture == nil {
		return client
	}
	copied := *client
	transport := copied.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	if logger != nil {
		transport = llm.NewRetryLogger(transport, providerName, logger)
	}
	copied.Transport = llm.NewWireTap(transport, providerName, capture)
	return &copied
}

// LogSuccess logs a completed blocking provider call at Debug.
func LogSuccess(ctx context.Context, logger *slog.Logger, providerName string, resp *llm.Response, start time.Time) {
	if logger == nil || resp == nil {
		return
	}
	logger.DebugContext(ctx, "llm provider call",
		slog.String("provider", providerName),
		slog.String("model", resp.Model),
		slog.Duration("duration", time.Since(start)),
		slog.String("stop_reason", string(resp.StopReason)),
		slog.Int64("input_tokens", resp.Usage.InputTokens),
		slog.Int64("output_tokens", resp.Usage.OutputTokens),
	)
}

// LogStreamEnd logs a completed streaming provider call at Debug.
func LogStreamEnd(ctx context.Context, logger *slog.Logger, providerName string, req *llm.Request, end llm.MessageEnd, model string, start time.Time) {
	if logger == nil {
		return
	}
	if model == "" && req != nil {
		model = req.Model
	}
	logger.DebugContext(ctx, "llm provider stream",
		slog.String("provider", providerName),
		slog.String("model", model),
		slog.Duration("duration", time.Since(start)),
		slog.String("stop_reason", string(end.StopReason)),
		slog.Int64("input_tokens", end.Usage.InputTokens),
		slog.Int64("output_tokens", end.Usage.OutputTokens),
	)
}

// LogFailure logs a failed provider call at Error.
func LogFailure(ctx context.Context, logger *slog.Logger, providerName string, req *llm.Request, start time.Time, err error) {
	if logger == nil || err == nil {
		return
	}
	model := ""
	if req != nil {
		model = req.Model
	}
	logger.ErrorContext(ctx, "llm provider call failed",
		slog.String("provider", providerName),
		slog.String("model", model),
		slog.Duration("duration", time.Since(start)),
		slog.String("error", err.Error()),
	)
}
