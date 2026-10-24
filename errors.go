package llm

import (
	"context"
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

// errorClasses is the ErrorClass precedence order: cancellation first (the
// caller's own decision), then specific provider conditions before the
// generic ErrBadRequest/ErrServer buckets they may be wrapped alongside.
var errorClasses = []struct {
	err  error
	name string
}{
	{context.Canceled, "canceled"},
	{context.DeadlineExceeded, "deadline_exceeded"},
	{ErrContextTooLong, "context_too_long"},
	{ErrContentFiltered, "content_filtered"},
	{ErrUnsupported, "unsupported"},
	{ErrInsufficientCredits, "insufficient_credits"},
	{ErrRateLimited, "rate_limited"},
	{ErrOverloaded, "overloaded"},
	{ErrTimeout, "timeout"},
	{ErrAuth, "auth"},
	{ErrPermission, "permission"},
	{ErrNotFound, "not_found"},
	{ErrLoginExpired, "login_expired"},
	{ErrLoginStateMismatch, "login_state_mismatch"},
	{ErrLoginClosed, "login_closed"},
	{ErrBadRequest, "bad_request"},
	{ErrServer, "server"},
}

// ErrorClass returns the normalized class of err: one of the package
// sentinels (ErrAuth … ErrLoginClosed) or context.Canceled /
// context.DeadlineExceeded, matched with errors.Is through the whole chain.
// It returns nil for a nil or unclassified error. When a chain matches
// several classes, context.Canceled wins, then context.DeadlineExceeded,
// then specific conditions
// (ErrContextTooLong, ErrContentFiltered, ...) win over the generic
// ErrBadRequest and ErrServer.
func ErrorClass(err error) error {
	if err == nil {
		return nil
	}
	for _, class := range errorClasses {
		if errors.Is(err, class.err) {
			return class.err
		}
	}
	return nil
}

// ErrorClasses returns every class ErrorClass can return, in its precedence
// order: context.Canceled, context.DeadlineExceeded, then the package
// sentinels, ending with the generic ErrBadRequest and ErrServer. A wrapper
// that snapshots an error can test each with errors.Is to keep every
// classification the original matched, including new sentinels as go-llm
// adds them. The returned slice is a copy.
func ErrorClasses() []error {
	classes := make([]error, len(errorClasses))
	for i, class := range errorClasses {
		classes[i] = class.err
	}
	return classes
}

// ErrorInfo is a serializable, log-safe description of an error, suitable
// for events, metrics labels, and durable records. It never carries
// provider-controlled detail (ProviderError Code, Message, Metadata, or
// RawBody).
type ErrorInfo struct {
	// Class is the stable lowercase name of ErrorClass(err): "canceled",
	// "deadline_exceeded", "context_too_long", "content_filtered",
	// "unsupported", "insufficient_credits", "rate_limited", "overloaded",
	// "timeout", "auth", "permission", "not_found", "login_expired",
	// "login_state_mismatch", "login_closed", "bad_request", or "server".
	// Empty when the error is unclassified.
	Class string `json:"class,omitempty"`
	// Provider is the bounded adapter identity from a wrapped ProviderError,
	// following SafeSummary's label rules; empty otherwise.
	Provider string `json:"provider,omitempty"`
	// HTTPStatus is the wrapped ProviderError's HTTP status, or zero.
	HTTPStatus int `json:"http_status,omitempty"`
	// RetryAfter is the wrapped ProviderError's provider-requested delay, or
	// zero. It serializes as integer nanoseconds.
	RetryAfter time.Duration `json:"retry_after,omitempty"`
}

// SafeErrorInfo describes err without any provider-controlled detail. A nil
// error returns the zero ErrorInfo.
func SafeErrorInfo(err error) ErrorInfo {
	if err == nil {
		return ErrorInfo{}
	}
	var info ErrorInfo
	class := ErrorClass(err)
	for _, candidate := range errorClasses {
		if candidate.err == class {
			info.Class = candidate.name
			break
		}
	}
	var providerErr *ProviderError
	if errors.As(err, &providerErr) && providerErr != nil {
		info.Provider = safeProviderLabel(providerErr.Provider)
		info.HTTPStatus = providerErr.HTTPStatus
		info.RetryAfter = providerErr.RetryAfter
	}
	return info
}

// Transient reports whether err's class is conventionally worth retrying as a
// new, unchanged attempt: ErrRateLimited, ErrOverloaded, ErrServer, or
// ErrTimeout. It is false for everything else, including ErrContextTooLong
// (the request must change first), authentication, credits, content
// filtering, bad requests, and cancellation or deadline expiry. An expired
// provider WithTimeout surfaces as context.DeadlineExceeded and is therefore
// not transient here; a caller that wants to retry its own per-call timeout
// decides that explicitly.
//
// Transient is classification only. It never sleeps or retries, and attempt
// budgets and backoff remain the caller's policy. Honor
// ProviderError.RetryAfter (see SafeErrorInfo) when it is set. go-llm's own
// transport retries have already run by the time an error reaches the caller.
func Transient(err error) bool {
	switch ErrorClass(err) {
	case ErrRateLimited, ErrOverloaded, ErrServer, ErrTimeout:
		return true
	default:
		return false
	}
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
	if status == "" && code == "" {
		return prefix + ": " + e.Message
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
