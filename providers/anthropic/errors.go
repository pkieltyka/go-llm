package anthropic

import (
	"context"
	"encoding/json"
	"errors"

	sdk "github.com/anthropics/anthropic-sdk-go"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

func mapError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) {
		return context.Canceled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return context.DeadlineExceeded
	}

	var apiErr *sdk.Error
	if !errors.As(err, &apiErr) {
		return providerutil.NormalizeRemoteError(providerName, err)
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
		if providerutil.IsContextOverflow(code, message) {
			return llm.ErrContextTooLong
		}
		return llm.ErrBadRequest
	case string(sdk.ErrorTypeAPIError):
		return llm.ErrServer
	}

	// Overflow types the switch does not name (request_too_large on HTTP 413
	// or in-stream, model_context_window_exceeded) classify before the
	// status fallback, so a status-less in-stream event is not ErrServer.
	if providerutil.IsContextOverflow(code, message) {
		return llm.ErrContextTooLong
	}

	// Status fallback: the unified providerutil classifier's canonical
	// mapping (FS §16) — identical sentinels across all provider engines
	// for identical statuses (402→credits, 408→timeout, 503/529→overloaded).
	return providerutil.StatusErrorKind(status)
}
