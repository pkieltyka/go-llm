package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

func mapError(err error) error {
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
				Provider: providerName,
				Message:  err.Error(),
				Kind:     llm.ErrTimeout,
			}
		}
		return err
	}

	raw := []byte(apiErr.RawJSON())
	code := string(apiErr.Type())
	message := errorMessage(raw)
	providerErr := &llm.ProviderError{
		Provider:   providerName,
		HTTPStatus: apiErr.StatusCode,
		Code:       code,
		Message:    message,
		RetryAfter: llm.RetryAfter(apiErr.Response),
		RawBody:    raw,
		Kind:       errorKind(apiErr.StatusCode, code, message),
	}
	return providerErr
}

func errorMessage(raw []byte) string {
	var envelope struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &envelope) == nil {
		return envelope.Error.Message
	}
	return ""
}

func errorKind(status int, code, message string) error {
	switch code {
	case string(sdk.ErrorTypeAuthenticationError):
		return llm.ErrAuth
	case string(sdk.ErrorTypePermissionError):
		return llm.ErrPermission
	case string(sdk.ErrorTypeNotFoundError):
		return llm.ErrNotFound
	case string(sdk.ErrorTypeRateLimitError):
		return llm.ErrRateLimited
	case string(sdk.ErrorTypeTimeoutError):
		return llm.ErrTimeout
	case string(sdk.ErrorTypeOverloadedError):
		return llm.ErrOverloaded
	case string(sdk.ErrorTypeBillingError):
		return llm.ErrInsufficientCredits
	case string(sdk.ErrorTypeInvalidRequestError):
		if isContextOverflowMessage(message) {
			return llm.ErrContextTooLong
		}
		return llm.ErrBadRequest
	case string(sdk.ErrorTypeAPIError):
		return llm.ErrServer
	}

	// Status fallback: the unified providerutil classifier's canonical
	// mapping (FS §16) — identical sentinels across all provider engines
	// for identical statuses (402→credits, 408→timeout, 503/529→overloaded).
	return providerutil.StatusErrorKind(status)
}

func isContextOverflowMessage(message string) bool {
	lower := strings.ToLower(message)
	return strings.Contains(lower, "context") ||
		strings.Contains(lower, "prompt is too long") ||
		(strings.Contains(lower, "tokens >") && strings.Contains(lower, "maximum"))
}
