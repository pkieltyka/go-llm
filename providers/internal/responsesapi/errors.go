package responsesapi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"strings"

	sdk "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// MapError normalizes OpenAI SDK errors for this Responses provider.
func (a Adapter) MapError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}

	var apiErr *sdk.Error
	if !errors.As(err, &apiErr) {
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			return &llm.ProviderError{
				Provider: a.ProviderName,
				Message:  err.Error(),
				Kind:     llm.ErrTimeout,
			}
		}
		return err
	}

	raw := []byte(apiErr.RawJSON())
	code := apiErr.Code
	if code == "" {
		code = apiErr.Type
	}
	providerErr := &llm.ProviderError{
		Provider:   a.ProviderName,
		HTTPStatus: apiErr.StatusCode,
		Code:       code,
		Message:    apiErr.Message,
		RetryAfter: llm.RetryAfter(apiErr.Response),
		RawBody:    raw,
		Kind:       errorKind(apiErr.StatusCode, code, apiErr.Type, apiErr.Message),
	}
	if apiErr.Param != "" || apiErr.Type != "" {
		providerErr.Metadata = map[string]any{}
		if apiErr.Param != "" {
			providerErr.Metadata["param"] = apiErr.Param
		}
		if apiErr.Type != "" {
			providerErr.Metadata["type"] = apiErr.Type
		}
	}
	return providerErr
}

// MapResponseError normalizes a failed Responses terminal event.
func (a Adapter) MapResponseError(err responses.ResponseError) error {
	raw := json.RawMessage(err.RawJSON())
	return &llm.ProviderError{
		Provider: a.ProviderName,
		Code:     string(err.Code),
		Message:  err.Message,
		RawBody:  raw,
		Kind:     errorKind(0, string(err.Code), "", err.Message),
	}
}

// MapStreamError normalizes a Responses stream error event.
func (a Adapter) MapStreamError(code, message, param string) error {
	providerErr := &llm.ProviderError{
		Provider: a.ProviderName,
		Code:     code,
		Message:  message,
		Kind:     errorKind(0, code, "", message),
	}
	if param != "" {
		providerErr.Metadata = map[string]any{"param": param}
	}
	return providerErr
}

// MapHTTPResponseError normalizes a non-2xx direct Responses HTTP response.
func (a Adapter) MapHTTPResponseError(resp *http.Response) error {
	if resp == nil {
		return &llm.ProviderError{
			Provider: a.ProviderName,
			Message:  "nil HTTP response",
			Kind:     llm.ErrServer,
		}
	}
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	code, typ, message, param := parseErrorBody(body)
	if message == "" {
		message = http.StatusText(resp.StatusCode)
	}
	providerErr := &llm.ProviderError{
		Provider:   a.ProviderName,
		HTTPStatus: resp.StatusCode,
		Code:       code,
		Message:    message,
		RetryAfter: llm.RetryAfter(resp),
		RawBody:    body,
		Kind:       errorKind(resp.StatusCode, code, typ, message),
	}
	if param != "" || typ != "" {
		providerErr.Metadata = map[string]any{}
		if param != "" {
			providerErr.Metadata["param"] = param
		}
		if typ != "" {
			providerErr.Metadata["type"] = typ
		}
	}
	return providerErr
}

func parseErrorBody(body []byte) (code, typ, message, param string) {
	var wrapped struct {
		Error struct {
			Code    string `json:"code"`
			Type    string `json:"type"`
			Message string `json:"message"`
			Param   string `json:"param"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &wrapped); err == nil && (wrapped.Error.Message != "" || wrapped.Error.Code != "") {
		return wrapped.Error.Code, wrapped.Error.Type, wrapped.Error.Message, wrapped.Error.Param
	}
	var direct struct {
		Code    string `json:"code"`
		Type    string `json:"type"`
		Message string `json:"message"`
		Param   string `json:"param"`
	}
	if err := json.Unmarshal(body, &direct); err == nil && (direct.Message != "" || direct.Code != "") {
		return direct.Code, direct.Type, direct.Message, direct.Param
	}
	return "", "", strings.TrimSpace(string(body)), ""
}

// errorKind delegates to the unified providerutil classifier (FS §16), with
// a Responses-family hook that classifies from the error envelope's "type"
// field. The old bare-"context" code substring match is intentionally gone:
// the shared table keeps only the scoped context-overflow heuristics.
func errorKind(status int, code, typ, message string) error {
	lowerType := strings.ToLower(typ)
	typeHook := func(int, string, string) error {
		switch {
		case strings.Contains(lowerType, "authentication"):
			return llm.ErrAuth
		case strings.Contains(lowerType, "rate_limit"):
			return llm.ErrRateLimited
		case strings.Contains(lowerType, "insufficient_quota"):
			return llm.ErrInsufficientCredits
		default:
			return nil
		}
	}
	if code == "" && strings.Contains(lowerType, "invalid_request") {
		// Give code-less errors typed invalid_request_error the same weak
		// bad-request fallback the classifier applies to codes.
		code = "invalid_request"
	}
	return providerutil.ErrorKind(status, code, message, typeHook)
}
