package providerutil

import (
	"net/http"
	"strconv"
	"strings"

	llm "github.com/pkieltyka/go-llm"
)

// This file is the single error-kind classifier shared by every provider
// engine (responsesapi, chatcompletions, and anthropic's status fallback),
// so errors.Is(err, sentinel) behaves identically across providers for
// identical failures (FS §16).

// ErrorKindHook classifies provider-family-specific error codes/messages
// ahead of the shared classification. Returning nil falls through.
type ErrorKindHook func(status int, code, message string) error

// ErrorKind classifies a provider error into a normalized sentinel: family
// hooks first, then the shared OpenAI-style code/message heuristics, then
// the canonical status mapping (StatusErrorKind). Status-less errors
// (in-stream error events, status 0) whose code is itself an integer HTTP
// status in 400–599 — servers commonly mirror the status into `code` after
// a 200 stream begins, e.g. {"error":{"code":"429"}} — classify through the
// canonical status table; non-integral codes ("429.5") never do. Weak code
// matches (invalid_request → ErrBadRequest, server_error → ErrServer) also
// apply only to status-less errors, so an HTTP status always classifies per
// the canonical table.
func ErrorKind(status int, code, message string, hooks ...ErrorKindHook) error {
	for _, hook := range hooks {
		if kind := hook(status, code, message); kind != nil {
			return kind
		}
	}
	if kind := heuristicErrorKind(code, message); kind != nil {
		return kind
	}
	if status >= 400 {
		return StatusErrorKind(status)
	}
	if status == 0 {
		if n, err := strconv.Atoi(strings.TrimSpace(code)); err == nil && n >= 400 && n < 600 {
			return StatusErrorKind(n)
		}
	}
	if kind := weakCodeKind(code); kind != nil {
		return kind
	}
	return fallbackStatusKind(status)
}

// StatusErrorKind maps an HTTP status onto the canonical sentinel (FS §16):
// 401→ErrAuth, 402→ErrInsufficientCredits, 403→ErrPermission,
// 404→ErrNotFound, 408→ErrTimeout, 429→ErrRateLimited, 503 and 529→
// ErrOverloaded, any other 5xx→ErrServer, any other 4xx→ErrBadRequest.
// Statuses outside 4xx/5xx (including 0) classify as ErrServer.
func StatusErrorKind(status int) error {
	if kind := canonicalStatusKind(status); kind != nil {
		return kind
	}
	return fallbackStatusKind(status)
}

func canonicalStatusKind(status int) error {
	switch status {
	case http.StatusUnauthorized:
		return llm.ErrAuth
	case http.StatusPaymentRequired:
		return llm.ErrInsufficientCredits
	case http.StatusForbidden:
		return llm.ErrPermission
	case http.StatusNotFound:
		return llm.ErrNotFound
	case http.StatusRequestTimeout:
		return llm.ErrTimeout
	case http.StatusTooManyRequests:
		return llm.ErrRateLimited
	case http.StatusServiceUnavailable, 529:
		return llm.ErrOverloaded
	default:
		return nil
	}
}

func fallbackStatusKind(status int) error {
	switch {
	case status >= 500:
		return llm.ErrServer
	case status >= 400:
		return llm.ErrBadRequest
	default:
		return llm.ErrServer
	}
}

// heuristicErrorKind holds the shared OpenAI-family code/message
// classification (the union of the previously drifting responsesapi and
// chatcompletions tables). Context-overflow detection is scoped to the
// shapes providers actually emit — a bare "context" code substring match
// would misclassify unrelated errors that merely mention the word.
func heuristicErrorKind(code, message string) error {
	lowerCode := strings.ToLower(code)
	lowerMessage := strings.ToLower(message)
	switch {
	case strings.Contains(lowerCode, "invalid_api_key") || strings.Contains(lowerCode, "auth"):
		return llm.ErrAuth
	case strings.Contains(lowerCode, "rate") || strings.Contains(lowerMessage, "rate limit"):
		return llm.ErrRateLimited
	case strings.Contains(lowerCode, "credit") || strings.Contains(lowerCode, "quota") || strings.Contains(lowerCode, "billing"):
		return llm.ErrInsufficientCredits
	case strings.Contains(lowerCode, "context_length") ||
		strings.Contains(lowerMessage, "context window") ||
		strings.Contains(lowerMessage, "context length") ||
		strings.Contains(lowerMessage, "context limit") ||
		strings.Contains(lowerMessage, "maximum context"):
		return llm.ErrContextTooLong
	case strings.Contains(lowerCode, "moderation") ||
		strings.Contains(lowerCode, "content_filter") ||
		strings.Contains(lowerCode, "content_policy") ||
		strings.Contains(lowerMessage, "moderation") ||
		strings.Contains(lowerMessage, "moderated") ||
		strings.Contains(lowerMessage, "content filter") ||
		strings.Contains(lowerMessage, "content policy") ||
		strings.Contains(lowerMessage, "policy violation") ||
		strings.Contains(lowerMessage, "safety policy"):
		return llm.ErrContentFiltered
	case strings.Contains(lowerCode, "timeout"):
		return llm.ErrTimeout
	default:
		return nil
	}
}

// weakCodeKind resolves status-less errors whose code still names a class.
func weakCodeKind(code string) error {
	lowerCode := strings.ToLower(code)
	switch {
	case strings.Contains(lowerCode, "invalid_prompt") || strings.Contains(lowerCode, "invalid_request"):
		return llm.ErrBadRequest
	case strings.Contains(lowerCode, "server_error"):
		return llm.ErrServer
	default:
		return nil
	}
}
