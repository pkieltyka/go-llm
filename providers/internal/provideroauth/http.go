package provideroauth

import (
	"bytes"
	"context"
	"io"
	"net/http"

	llm "github.com/pkieltyka/go-llm"
)

// ApplyHeadersFunc applies provider-specific auth headers for a credential.
type ApplyHeadersFunc func(*http.Request, llm.AuthCredential)

// MiddlewareNext is the common SDK middleware continuation shape.
type MiddlewareNext func(*http.Request) (*http.Response, error)

// CredentialSource yields the current OAuth credential and renews it after
// a rejected access token. Resolver implements it.
type CredentialSource interface {
	Credential(ctx context.Context) (llm.AuthCredential, error)
	ForceRefreshIfCurrent(ctx context.Context, failedAccess string) (llm.AuthCredential, error)
}

// DoWithAuthRetry applies OAuth headers and retries exactly once after a 401
// with a forced refresh.
func DoWithAuthRetry(req *http.Request, next MiddlewareNext, source CredentialSource, apply ApplyHeadersFunc) (*http.Response, error) {
	if source == nil {
		return next(req)
	}
	body, err := readRequestBody(req)
	if err != nil {
		return nil, err
	}

	cred, err := source.Credential(requestContext(req))
	if err != nil {
		return nil, err
	}
	first := cloneRequest(req, body)
	apply(first, cred)
	resp, err := next(first)
	if err != nil || resp == nil || resp.StatusCode != http.StatusUnauthorized {
		return resp, err
	}
	closeUnauthorizedBody(resp.Body)

	cred, refreshErr := source.ForceRefreshIfCurrent(requestContext(req), cred.Access)
	if refreshErr != nil {
		return nil, refreshErr
	}
	second := cloneRequest(req, body)
	apply(second, cred)
	return next(second)
}

func requestContext(req *http.Request) context.Context {
	if req == nil || req.Context() == nil {
		return context.Background()
	}
	return req.Context()
}

func readRequestBody(req *http.Request) ([]byte, error) {
	if req == nil || req.Body == nil {
		return nil, nil
	}
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	_ = req.Body.Close()
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(body)), nil
	}
	return body, nil
}

func cloneRequest(req *http.Request, body []byte) *http.Request {
	if req == nil {
		return nil
	}
	cloned := req.Clone(requestContext(req))
	cloned.Header = req.Header.Clone()
	if body != nil {
		cloned.Body = io.NopCloser(bytes.NewReader(body))
		cloned.ContentLength = int64(len(body))
		cloned.GetBody = func() (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewReader(body)), nil
		}
	}
	return cloned
}

// Closing without draining avoids blocking on a stalled authentication response.
func closeUnauthorizedBody(body io.ReadCloser) {
	if body != nil {
		_ = body.Close()
	}
}
