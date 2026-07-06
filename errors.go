package llm

import (
	"errors"
	"fmt"
	"strconv"
	"time"
)

// Normalized sentinel errors classifying provider failures (FS §16).
// Match with errors.Is: adapters wrap the matching sentinel as
// ProviderError.Kind, and ProviderError.Unwrap returns Kind, so
// errors.Is(err, ErrRateLimited) works through the whole taxonomy.
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

// Error formats as "llm/<provider>: <status> <code>: <message>", omitting
// empty fields.
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
	// Skip Code in display when it merely repeats the HTTP status (some
	// providers put the numeric status in the error body's code field),
	// avoiding "llm/openrouter: 400 400: ...". Code stays populated for
	// programmatic use either way.
	if e.Code != "" && (e.HTTPStatus == 0 || e.Code != strconv.Itoa(e.HTTPStatus)) {
		code = " " + e.Code
	}
	if e.Message == "" {
		return prefix + ":" + status + code
	}
	return prefix + ":" + status + code + ": " + e.Message
}

// Unwrap returns the normalized sentinel stored in Kind, so errors.Is
// matches ProviderError against the sentinel taxonomy (architecture §2.6).
func (e *ProviderError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Kind
}
