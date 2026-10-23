package provideroauth

import (
	"net/http"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

// TokenError normalizes a non-2xx response from an OAuth token endpoint;
// operation ("refresh", "login token exchange") labels the message. 429
// (token endpoint rate limit) maps to llm.ErrRateLimited and 408 (request timeout) to llm.ErrTimeout — both are transient and say
// nothing about credential validity. Every other 4xx (400 invalid_grant,
// 401, 403, ...) means the stored credential can no longer be exchanged — an
// auth failure the caller must resolve by re-authenticating — so it maps to
// llm.ErrAuth. 5xx and anything else stays llm.ErrServer (retryable).
func TokenError(providerName, operation string, status int) error {
	kind := llm.ErrServer
	switch {
	case status == http.StatusTooManyRequests:
		kind = llm.ErrRateLimited
	case status == http.StatusRequestTimeout:
		kind = llm.ErrTimeout
	case status >= 400 && status < 500:
		kind = llm.ErrAuth
	}
	return &llm.ProviderError{
		Provider:   providerName,
		HTTPStatus: status,
		Message:    "OAuth " + operation + " failed",
		Kind:       kind,
	}
}

// ExpiryMargin is subtracted from a token's lifetime before it is stored,
// as pi does, so a credential pi and go-llm share means the same thing to
// both (pi's EXPIRY_MARGIN_MS).
const ExpiryMargin = 3 * time.Minute

// ExpiresAt converts a token endpoint's expires_in (seconds) into the
// Unix-millisecond expiry to store: now + expires_in - ExpiryMargin.
func ExpiresAt(expiresIn int64) int64 {
	return time.Now().Add(time.Duration(expiresIn)*time.Second - ExpiryMargin).UnixMilli()
}
