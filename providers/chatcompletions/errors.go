package chatcompletions

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	sdk "github.com/openai/openai-go/v3"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// mapError normalizes SDK/transport errors.
func (p *Provider) mapError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	var apiErr *sdk.Error
	if errors.As(err, &apiErr) {
		raw := []byte(apiErr.RawJSON())
		code := apiErr.Code
		if code == "" {
			code = apiErr.Type
		}
		_, _, metadata := parseProviderError(raw)
		return &llm.ProviderError{
			Provider:   p.Name(),
			HTTPStatus: apiErr.StatusCode,
			Code:       code,
			Message:    apiErr.Message,
			RetryAfter: llm.RetryAfter(apiErr.Response),
			Metadata:   metadata,
			RawBody:    raw,
			Kind:       p.dialect.MapErrorStatus(apiErr.StatusCode, code, apiErr.Message),
		}
	}
	return providerutil.NormalizeRemoteError(p.Name(), err)
}

func (p *Provider) mapHTTPResponseError(resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	code, message, metadata := parseProviderError(body)
	if message == "" {
		message = http.StatusText(resp.StatusCode)
	}
	providerErr := &llm.ProviderError{
		Provider:   p.Name(),
		HTTPStatus: resp.StatusCode,
		Code:       code,
		Message:    message,
		RetryAfter: llm.RetryAfter(resp),
		Metadata:   metadata,
		RawBody:    body,
		Kind:       p.dialect.MapErrorStatus(resp.StatusCode, code, message),
	}
	return providerErr
}

func (p *Provider) mapChunkError(err *rawError, raw []byte) error {
	if err == nil {
		return nil
	}
	code := stringifyCode(err.Code)
	return &llm.ProviderError{
		Provider: p.Name(),
		Code:     code,
		Message:  err.Message,
		Metadata: err.Metadata,
		RawBody:  raw,
		Kind:     p.dialect.MapErrorStatus(0, code, err.Message),
	}
}

func parseProviderError(body []byte) (string, string, map[string]any) {
	var wrapped struct {
		Error rawError `json:"error"`
	}
	if err := json.Unmarshal(body, &wrapped); err == nil && (wrapped.Error.Message != "" || wrapped.Error.Code != nil) {
		return stringifyCode(wrapped.Error.Code), wrapped.Error.Message, wrapped.Error.Metadata
	}
	var direct rawError
	if err := json.Unmarshal(body, &direct); err == nil && (direct.Message != "" || direct.Code != nil) {
		return stringifyCode(direct.Code), direct.Message, direct.Metadata
	}
	return "", strings.TrimSpace(string(body)), nil
}

func stringifyCode(code any) string {
	switch value := code.(type) {
	case nil:
		return ""
	case string:
		return value
	case float64:
		return strings.TrimSuffix(strings.TrimSuffix(fmt.Sprintf("%.0f", value), ".0"), ".")
	default:
		return fmt.Sprint(value)
	}
}

// DefaultErrorKind is the dialect-default error classification: the unified
// providerutil classifier with no extra family hook (FS §16 canonical status
// mapping plus the shared scoped code/message heuristics). Dialects layer
// their own vocabulary on top (e.g. OpenRouter's numeric "402" stream code).
func DefaultErrorKind(status int, code, message string) error {
	return providerutil.ErrorKind(status, code, message)
}
