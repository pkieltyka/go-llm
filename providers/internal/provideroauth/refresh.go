package provideroauth

import (
	"net/http"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

// RefreshError normalizes a non-2xx response from an OAuth token refresh
// endpoint. 429 (token endpoint rate limit) maps to llm.ErrRateLimited and
// 408 (request timeout) to llm.ErrTimeout — both are transient and say
// nothing about credential validity. Every other 4xx (400 invalid_grant,
// 401, 403, ...) means the stored credential can no longer be exchanged — an
// auth failure the caller must resolve by re-authenticating — so it maps to
// llm.ErrAuth. 5xx and anything else stays llm.ErrServer (retryable).
func RefreshError(providerName string, status int) error {
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
		Message:    "OAuth token refresh failed",
		Kind:       kind,
	}
}

// ExpiresAt converts a token endpoint's expires_in (seconds) into an absolute
// Unix-millisecond expiry. It records the TRUE server expiry — no safety
// margin is subtracted here, because credentials handed to onRefresh are
// persisted verbatim; the Source applies its refresh-before-expiry margin at
// read time instead.
func ExpiresAt(expiresIn int64) int64 {
	if expiresIn <= 0 {
		return 0
	}
	return time.Now().Add(time.Duration(expiresIn) * time.Second).UnixMilli()
}
