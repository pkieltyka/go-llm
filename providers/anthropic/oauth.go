package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"strings"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/provideroauth"
)

// These fixed-client protocol values follow Pi's Anthropic OAuth adapter.
const (
	anthropicOAuthClientID     = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
	anthropicOAuthTokenURL     = "https://platform.claude.com/v1/oauth/token"
	anthropicOAuthAuthorizeURL = "https://claude.ai/oauth/authorize"
	anthropicOAuthScope        = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
	maxAnthropicTokenBytes     = 1 << 20
	anthropicExpiryMargin      = 5 * time.Minute
)

// ValidateOAuthCredential checks the credential's type and fixed-client grant
// identity without I/O. Pi's fixed-client credentials omit ClientID; an explicit
// conflicting identity is rejected. Token presence/expiry is checked when the
// provider resolves the credential, allowing refresh-only stored credentials.
func ValidateOAuthCredential(cred llm.AuthCredential) error {
	if cred.Type != "oauth" {
		return fmt.Errorf("%w: Anthropic requires an OAuth credential", llm.ErrAuth)
	}
	if cred.ClientID != "" && cred.ClientID != anthropicOAuthClientID {
		return fmt.Errorf("%w: unsupported Anthropic OAuth client identity", llm.ErrAuth)
	}
	return nil
}

func validateOAuthCredential(cred llm.AuthCredential) error { return ValidateOAuthCredential(cred) }

type anthropicTokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int64  `json:"expires_in"`
}

func refreshAnthropicOAuth(ctx context.Context, client *http.Client, tokenURL string, cred llm.AuthCredential) (llm.AuthCredential, error) {
	if err := validateOAuthCredential(cred); err != nil {
		return llm.AuthCredential{}, err
	}
	if strings.TrimSpace(cred.Refresh) == "" {
		return llm.AuthCredential{}, fmt.Errorf("%w: Anthropic OAuth refresh token is missing", llm.ErrAuth)
	}
	token, err := requestAnthropicToken(ctx, client, tokenURL, map[string]string{
		"grant_type": "refresh_token", "client_id": anthropicOAuthClientID, "refresh_token": cred.Refresh,
	}, "refresh")
	if err != nil {
		return llm.AuthCredential{}, err
	}
	return anthropicCredential(token, time.Now())
}

func requestAnthropicToken(ctx context.Context, client *http.Client, tokenURL string, payload map[string]string, operation string) (anthropicTokenResponse, error) {
	if client == nil {
		client = llm.DefaultHTTPClient()
	}
	if tokenURL == "" {
		tokenURL = anthropicOAuthTokenURL
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return anthropicTokenResponse{}, fmt.Errorf("%w: Anthropic OAuth request could not be encoded", llm.ErrBadRequest)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, bytes.NewReader(body))
	if err != nil {
		return anthropicTokenResponse{}, fmt.Errorf("%w: Anthropic OAuth request could not be built", llm.ErrBadRequest)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/json")
	resp, err := provideroauth.NoRedirectClient(client).Do(req)
	if err != nil {
		return anthropicTokenResponse{}, sanitizeAnthropicOAuthError(operation, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return anthropicTokenResponse{}, provideroauth.TokenError(providerName, operation, resp.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxAnthropicTokenBytes+1))
	if err != nil {
		return anthropicTokenResponse{}, sanitizeAnthropicOAuthError(operation, err)
	}
	if len(data) > maxAnthropicTokenBytes {
		return anthropicTokenResponse{}, fmt.Errorf("%w: Anthropic OAuth response exceeds limit", llm.ErrServer)
	}
	var token anthropicTokenResponse
	if json.Unmarshal(data, &token) != nil {
		return anthropicTokenResponse{}, fmt.Errorf("%w: Anthropic OAuth response is not valid JSON", llm.ErrServer)
	}
	return token, nil
}

func anthropicCredential(token anthropicTokenResponse, now time.Time) (llm.AuthCredential, error) {
	if strings.TrimSpace(token.AccessToken) == "" || strings.TrimSpace(token.RefreshToken) == "" {
		return llm.AuthCredential{}, fmt.Errorf("%w: Anthropic OAuth response is missing tokens", llm.ErrAuth)
	}
	if token.ExpiresIn <= 0 || token.ExpiresIn > math.MaxInt64/int64(time.Second) {
		return llm.AuthCredential{}, fmt.Errorf("%w: Anthropic OAuth response has invalid expires_in", llm.ErrAuth)
	}
	return llm.AuthCredential{Type: "oauth", Access: token.AccessToken, Refresh: token.RefreshToken,
		Expires: now.Add(time.Duration(token.ExpiresIn)*time.Second - anthropicExpiryMargin).UnixMilli()}, nil
}

func sanitizeAnthropicOAuthError(operation string, err error) error {
	if errors.Is(err, context.Canceled) {
		return context.Canceled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return context.DeadlineExceeded
	}
	if errors.Is(err, provideroauth.ErrUnsafeRedirect) {
		return fmt.Errorf("%w: Anthropic OAuth endpoint redirected", llm.ErrServer)
	}
	var timed net.Error
	if errors.As(err, &timed) && timed.Timeout() {
		return fmt.Errorf("%w: Anthropic OAuth %s timed out", llm.ErrTimeout, operation)
	}
	return fmt.Errorf("%w: Anthropic OAuth %s transport failed", llm.ErrServer, operation)
}
