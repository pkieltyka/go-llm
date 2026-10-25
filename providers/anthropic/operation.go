package anthropic

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	sdkoption "github.com/anthropics/anthropic-sdk-go/option"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/provideroauth"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

type operationKey struct{}

// An operation fixes the authentication mode before request conversion. It
// contains no cached authority: every transport attempt rereads the owned store.
type preparedOperation struct {
	provider     *Provider
	oauth        bool
	stored       bool
	reverseTools map[string]string
}

func (p *Provider) prepareOperation(ctx context.Context) (context.Context, *preparedOperation, error) {
	op := &preparedOperation{provider: p}
	if err := ctx.Err(); err != nil {
		return ctx, nil, err
	}
	if p.credentials != nil {
		cred, ok, err := p.credentials.Read(ctx, providerName)
		if err != nil {
			return ctx, nil, safeAuthError(err)
		}
		op.stored = ok
		op.oauth = ok && cred.Type == "oauth"
		if op.oauth {
			if err := p.validateOAuth(cred); err != nil {
				return ctx, nil, err
			}
		} else if _, err := providerutil.APIKeyFromCredential(providerName, cred, ok, p.ambient); err != nil {
			return ctx, nil, safeAuthError(err)
		}
	}
	return context.WithValue(ctx, operationKey{}, op), op, nil
}

func (p *Provider) validateOAuth(cred llm.AuthCredential) error {
	if !oauthBaseURL(p.baseURL) || !oauthBaseURL(cred.BaseURL) {
		return fmt.Errorf("%w: Anthropic OAuth requires the official API origin", llm.ErrUnsupported)
	}
	return validateOAuthCredential(cred)
}

func oauthBaseURL(raw string) bool {
	if raw == "" {
		return true
	}
	u, err := url.Parse(raw)
	return err == nil && u.Scheme == "https" && u.Host == "api.anthropic.com" && u.User == nil && (u.Path == "" || u.Path == "/") && u.RawQuery == "" && u.Fragment == ""
}

func (op *preparedOperation) read(ctx context.Context) (llm.AuthCredential, error) {
	if err := ctx.Err(); err != nil {
		return llm.AuthCredential{}, err
	}
	cred, ok, err := op.provider.credentials.Read(ctx, providerName)
	if err != nil {
		return llm.AuthCredential{}, safeAuthError(err)
	}
	if ok != op.stored || (ok && cred.Type == "oauth") != op.oauth {
		return llm.AuthCredential{}, fmt.Errorf("%w: Anthropic credential ownership or authentication mode changed", llm.ErrAuth)
	}
	if op.oauth {
		if err := op.provider.validateOAuth(cred); err != nil {
			return llm.AuthCredential{}, err
		}
	} else {
		key, err := providerutil.APIKeyFromCredential(providerName, cred, ok, op.provider.ambient)
		if err != nil {
			return llm.AuthCredential{}, safeAuthError(err)
		}
		cred.Key = key
	}
	return cred, nil
}

func (op *preparedOperation) resolver() *provideroauth.Resolver {
	p := op.provider
	tokenURL := p.oauthTokenURL
	if tokenURL == "" {
		tokenURL = anthropicOAuthTokenURL
	}
	return &provideroauth.Resolver{
		Store: operationStore{op}, ProviderID: providerName, RefreshBeforeSet: true,
		Refresh: func(ctx context.Context, cred llm.AuthCredential) (llm.AuthCredential, error) {
			if err := p.validateOAuth(cred); err != nil {
				return llm.AuthCredential{}, err
			}
			return refreshAnthropicOAuth(ctx, p.httpClient, tokenURL, cred)
		},
	}
}

func (op *preparedOperation) Credential(ctx context.Context) (llm.AuthCredential, error) {
	cred, err := op.read(ctx)
	if err == nil && op.oauth {
		cred, err = op.resolver().WithStored(cred).Credential(ctx)
	}
	if err != nil {
		return llm.AuthCredential{}, op.mapError(err)
	}
	if err := ctx.Err(); err != nil {
		return llm.AuthCredential{}, err
	}
	return cred, nil
}

func (op *preparedOperation) ForceRefreshIfCurrent(ctx context.Context, failed string) (llm.AuthCredential, error) {
	if _, err := op.read(ctx); err != nil {
		return llm.AuthCredential{}, err
	}
	cred, err := op.resolver().ForceRefreshIfCurrent(ctx, failed)
	if err != nil {
		return llm.AuthCredential{}, op.mapError(err)
	}
	if err := ctx.Err(); err != nil {
		return llm.AuthCredential{}, err
	}
	return cred, nil
}

// Validate the authoritative value again inside Modify, including when another
// request renewed the credential and the resolver decides no refresh is needed.
type operationStore struct{ op *preparedOperation }

func (s operationStore) Delete(context.Context, string) error { return llm.ErrUnsupported }

func (s operationStore) Read(ctx context.Context, id string) (llm.AuthCredential, bool, error) {
	cred, err := s.op.read(ctx)
	return cred, err == nil, err
}
func (s operationStore) Modify(ctx context.Context, id string, fn func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error)) (llm.AuthCredential, bool, error) {
	var callbackErr error
	cred, ok, err := s.op.provider.credentials.Modify(ctx, id, func(current llm.AuthCredential, exists bool) (llm.AuthCredential, bool, error) {
		if !exists {
			callbackErr = llm.ErrAuth
			return llm.AuthCredential{}, false, callbackErr
		}
		if callbackErr = s.op.provider.validateOAuth(current); callbackErr != nil {
			return llm.AuthCredential{}, false, callbackErr
		}
		next, write, fnErr := fn(current, exists)
		callbackErr = fnErr
		return next, write, fnErr
	})
	if err != nil {
		if callbackErr != nil {
			return llm.AuthCredential{}, false, s.op.mapError(callbackErr)
		}
		return llm.AuthCredential{}, false, safeAuthError(err)
	}
	if ok {
		err = s.op.provider.validateOAuth(cred)
	}
	return cred, ok, err
}

func (p *Provider) authMiddleware(req *http.Request, next sdkoption.MiddlewareNext) (*http.Response, error) {
	op, _ := req.Context().Value(operationKey{}).(*preparedOperation)
	if op == nil || op.provider != p {
		ctx, raw, err := p.prepareOperation(req.Context())
		if err != nil {
			return nil, err
		}
		if raw.oauth {
			return nil, fmt.Errorf("%w: Anthropic OAuth requires prepared provider operations", llm.ErrUnsupported)
		}
		op = raw
		req = req.Clone(ctx)
	}
	if !op.oauth {
		return next(req)
	}
	resp, err := provideroauth.DoWithAuthRetry(req, provideroauth.MiddlewareNext(next), op, applyOAuthHeaders)
	return resp, op.mapError(err)
}

type ownedAuthTransport struct {
	provider *Provider
	next     http.RoundTripper
}

func (t *ownedAuthTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	op, _ := req.Context().Value(operationKey{}).(*preparedOperation)
	if op == nil || op.provider != t.provider {
		return nil, llm.ErrUnsupported
	}
	// SDK per-request options cannot redirect OAuth credentials to another origin.
	if op.oauth && (req.URL.Scheme != "https" || req.URL.Host != "api.anthropic.com" || req.URL.User != nil) {
		return nil, fmt.Errorf("%w: Anthropic OAuth requires the official API origin", llm.ErrUnsupported)
	}
	cred, err := op.read(req.Context())
	if err != nil {
		return nil, err
	}
	if op.oauth && cred.Access == "" {
		return nil, llm.ErrAuth
	}
	cloned := req.Clone(req.Context())
	if op.oauth {
		applyOAuthHeaders(cloned, cred)
	} else {
		cloned.Header.Del("Authorization")
		cloned.Header.Del("anthropic-dangerous-direct-browser-access")
		cloned.Header.Del("x-app")
		cloned.Header.Set("X-Api-Key", cred.Key)
	}
	if err := req.Context().Err(); err != nil {
		return nil, err
	}
	if op.oauth {
		provideroauth.RecordDispatchedAccess(cloned.Context(), cred.Access)
	}
	return t.next.RoundTrip(cloned)
}

func applyOAuthHeaders(req *http.Request, cred llm.AuthCredential) {
	req.Header.Del("X-Api-Key")
	req.Header.Set("Authorization", "Bearer "+cred.Access)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("anthropic-dangerous-direct-browser-access", "true")
	req.Header.Set("User-Agent", "claude-cli/2.1.280")
	req.Header.Set("x-app", "cli")
	// Request options are additive in this SDK. Keep positional/tool betas and
	// deduplicate features without changing their order.
	var betas []string
	seen := map[string]bool{}
	for _, value := range append([]string{"claude-code-20250219", "oauth-2025-04-20"}, req.Header.Values("anthropic-beta")...) {
		for _, beta := range strings.Split(value, ",") {
			beta = strings.TrimSpace(beta)
			if beta != "" && !seen[beta] {
				betas = append(betas, beta)
				seen[beta] = true
			}
		}
	}
	req.Header.Set("anthropic-beta", strings.Join(betas, ","))
}

func safeAuthError(err error) error {
	if errors.Is(err, context.Canceled) {
		return context.Canceled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return context.DeadlineExceeded
	}
	if errors.Is(err, llm.ErrUnsupported) {
		return fmt.Errorf("%w: unsupported Anthropic OAuth configuration", llm.ErrUnsupported)
	}
	return fmt.Errorf("%w: Anthropic credential unavailable", llm.ErrAuth)
}

func (op *preparedOperation) mapError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.Canceled) {
		return context.Canceled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return context.DeadlineExceeded
	}
	mapped := mapError(err)
	if !op.oauth {
		return mapped
	}
	var pe *llm.ProviderError
	if errors.As(mapped, &pe) {
		return &llm.ProviderError{Provider: providerName, HTTPStatus: pe.HTTPStatus, Kind: oauthErrorKind(mapped), RetryAfter: pe.RetryAfter, Message: "Anthropic OAuth request failed"}
	}
	return fmt.Errorf("%w: Anthropic OAuth request failed", oauthErrorKind(mapped))
}

func (op *preparedOperation) originalTool(name string) string {
	if original, ok := op.reverseTools[strings.ToLower(name)]; ok {
		return original
	}
	return name
}

func (op *preparedOperation) restoreResponse(resp *llm.Response) {
	for i, part := range resp.Parts {
		if call, ok := part.(llm.ToolCallPart); ok {
			call.Name = op.originalTool(call.Name)
			resp.Parts[i] = call
		}
	}
}

func oauthErrorKind(err error) error {
	for _, kind := range []error{llm.ErrAuth, llm.ErrUnsupported, llm.ErrBadRequest, llm.ErrTimeout, llm.ErrPermission, llm.ErrNotFound, llm.ErrRateLimited, llm.ErrInsufficientCredits, llm.ErrOverloaded, llm.ErrContentFiltered, llm.ErrContextTooLong} {
		if errors.Is(err, kind) {
			return kind
		}
	}
	return llm.ErrServer
}

// A per-operation wrapper keeps OAuth diagnostic policy independent of mutable
// credential state and of other concurrent requests on the same provider.
type operationCaptureTransport struct {
	next    http.RoundTripper
	capture func(llm.WireCapture)
}

func (t *operationCaptureTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	capture := t.capture
	if op, ok := req.Context().Value(operationKey{}).(*preparedOperation); ok && op.oauth && capture != nil {
		original := capture
		capture = func(value llm.WireCapture) {
			value.RequestHeaders = nil
			value.ResponseHeaders = nil
			value.RequestBody = nil
			value.ResponseBody = nil
			value.URL = req.URL.Scheme + "://" + req.URL.Host + req.URL.EscapedPath()
			value.Err = op.mapError(value.Err)
			original(value)
		}
	}
	return llm.NewWireTap(t.next, providerName, capture).RoundTrip(req)
}
