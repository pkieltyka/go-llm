package provideroauth

import (
	"errors"
	"net/http"
)

// ErrUnsafeRedirect is returned when a token endpoint answers a
// credential-bearing POST with a redirect. Go's http.Client replays the
// request body on 307/308, which would re-send refresh tokens, authorization
// codes, or PKCE verifiers to an unvalidated origin, so credential POSTs
// refuse every redirect unconditionally. It surfaces wrapped in the
// *url.Error that http.Client.Do returns; match with errors.Is.
var ErrUnsafeRedirect = errors.New("OAuth token endpoint responded with a redirect")

// NoRedirectClient returns a shallow copy of client whose CheckRedirect
// refuses every redirect with ErrUnsafeRedirect. The caller's client is
// never mutated. Legitimate token endpoints respond directly, so valid
// flows are unaffected.
func NoRedirectClient(client *http.Client) *http.Client {
	c := &http.Client{}
	if client != nil {
		*c = *client
	}
	c.CheckRedirect = func(*http.Request, []*http.Request) error {
		return ErrUnsafeRedirect
	}
	return c
}
