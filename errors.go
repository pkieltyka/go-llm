package llm

import (
	"errors"
	"fmt"
	"time"
)

var (
	ErrAuth                = errors.New("llm: authentication failed")
	ErrPermission          = errors.New("llm: permission denied")
	ErrNotFound            = errors.New("llm: not found")
	ErrBadRequest          = errors.New("llm: bad request")
	ErrRateLimited         = errors.New("llm: rate limited")
	ErrInsufficientCredits = errors.New("llm: insufficient credits")
	ErrOverloaded          = errors.New("llm: overloaded")
	ErrServer              = errors.New("llm: server error")
	ErrTimeout             = errors.New("llm: timeout")
	ErrContentFiltered     = errors.New("llm: content filtered")
	ErrContextTooLong      = errors.New("llm: context too long")
	ErrUnsupported         = errors.New("llm: unsupported")
)

// ProviderError carries normalized and provider-specific error details.
type ProviderError struct {
	Provider   string
	HTTPStatus int
	Code       string
	Message    string
	RetryAfter time.Duration
	Metadata   map[string]any
	RawBody    []byte
	Kind       error
}

func (e *ProviderError) Error() string {
	if e == nil {
		return "<nil>"
	}
	prefix := "llm"
	if e.Provider != "" {
		prefix += "/" + e.Provider
	}
	status := ""
	if e.HTTPStatus != 0 {
		status = fmt.Sprintf(" %d", e.HTTPStatus)
	}
	code := ""
	if e.Code != "" {
		code = " " + e.Code
	}
	if e.Message == "" {
		return prefix + ":" + status + code
	}
	return prefix + ":" + status + code + ": " + e.Message
}

func (e *ProviderError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Kind
}
