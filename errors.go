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
	// ErrLoginExpired reports that an interactive login exceeded its bounded
	// lifetime and must be restarted.
	ErrLoginExpired = errors.New("llm: login expired")
	// ErrLoginStateMismatch reports a rejected OAuth response whose CSRF state
	// did not match the pending login.
	ErrLoginStateMismatch = errors.New("llm: login state mismatch")
	// ErrLoginClosed reports use of a cancelled or already-consumed login flow.
	ErrLoginClosed = errors.New("llm: login flow closed")
)

// ProviderError carries normalized and provider-specific error details.
// Message, Metadata, and RawBody originate outside the process and are
// untrusted: providers may echo prompt content, credentials, or arbitrary
// payloads in them. Use SafeSummary or SafeError for operational logging.
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

// SafeSummary returns a provider error description suitable for logs and
// metrics. It deliberately excludes Code, Message, Metadata, and RawBody
// because all four may contain provider-controlled request data. Provider is
// included only when it is a short adapter identifier made of ASCII letters,
// digits, '.', '_', or '-'.
func (e *ProviderError) SafeSummary() string {
	if e == nil {
		return "<nil>"
	}
	prefix := "llm"
	if provider := safeProviderLabel(e.Provider); provider != "" {
		prefix += "/" + provider
	}
	if e.HTTPStatus != 0 {
		prefix += fmt.Sprintf(": %d", e.HTTPStatus)
	} else {
		prefix += ":"
	}
	if kind := safeProviderErrorKind(e.Kind); kind != "" {
		return prefix + " (" + kind + ")"
	}
	return prefix + " (provider error)"
}

func safeProviderLabel(provider string) string {
	if provider == "" || len(provider) > 64 {
		return ""
	}
	for i := 0; i < len(provider); i++ {
		c := provider[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-' {
			continue
		}
		return ""
	}
	return provider
}

// SafeError formats ProviderError values without their untrusted detail
// fields. Errors that do not wrap ProviderError are returned verbatim; callers
// must therefore use it only where non-provider errors are trusted local
// validation, configuration, or programming errors.
func SafeError(err error) string {
	if err == nil {
		return "<nil>"
	}
	var providerErr *ProviderError
	if errors.As(err, &providerErr) {
		return providerErr.SafeSummary()
	}
	return err.Error()
}

func safeProviderErrorKind(kind error) string {
	for _, candidate := range []error{
		ErrAuth,
		ErrPermission,
		ErrNotFound,
		ErrBadRequest,
		ErrRateLimited,
		ErrInsufficientCredits,
		ErrOverloaded,
		ErrServer,
		ErrTimeout,
		ErrContentFiltered,
		ErrContextTooLong,
		ErrUnsupported,
	} {
		if errors.Is(kind, candidate) {
			return candidate.Error()
		}
	}
	return ""
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
