package openaicodex

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/provideroauth"
)

const (
	openAICodexOAuthClientID = "app_EMoamEEZ73f0CkXaXp7hrann"
	openAICodexOAuthTokenURL = "https://auth.openai.com/oauth/token"
	codexAccountClaimPath    = "https://api.openai.com/auth"
	maxCodexTokenBodyBytes   = 1 << 20
	maxCodexJWTBytes         = 256 << 10
)

type codexTokenResponse struct {
	IDToken      string `json:"id_token"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int64  `json:"expires_in"`
}

func newCodexOAuthSource(cfg config) (*provideroauth.Source, error) {
	return provideroauth.New(cfg.oauthCred, func(ctx context.Context, cred llm.AuthCredential) (llm.AuthCredential, error) {
		return refreshCodexOAuth(ctx, cfg.httpClient, cfg.tokenURL, cred)
	}, cfg.persistence)
}

func refreshCodexOAuth(ctx context.Context, client *http.Client, tokenURL string, cred llm.AuthCredential) (llm.AuthCredential, error) {
	if client == nil {
		client = llm.DefaultHTTPClient()
	}
	client = provideroauth.NoRedirectClient(client)
	if tokenURL == "" {
		tokenURL = openAICodexOAuthTokenURL
	}
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {cred.Refresh},
		"client_id":     {openAICodexOAuthClientID},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return llm.AuthCredential{}, sanitizeCodexOAuthRequestError("refresh", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return llm.AuthCredential{}, sanitizeCodexOAuthTransportError("refresh", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return llm.AuthCredential{}, provideroauth.RefreshError(providerName, resp.StatusCode)
	}
	token, err := decodeCodexTokenResponse(resp.Body, "refresh")
	if err != nil {
		return llm.AuthCredential{}, err
	}
	if token.AccessToken == "" {
		return llm.AuthCredential{}, fmt.Errorf("%w: OpenAI Codex OAuth refresh response missing access token", llm.ErrAuth)
	}
	accountID := extractCodexAccountID(token.IDToken)
	if accountID == "" {
		accountID = extractCodexAccountID(token.AccessToken)
	}
	if accountID == "" {
		accountID = cred.AccountID
	}
	return llm.AuthCredential{
		Type:      "oauth",
		Access:    token.AccessToken,
		Refresh:   token.RefreshToken,
		Expires:   provideroauth.ExpiresAt(token.ExpiresIn),
		AccountID: accountID,
	}, nil
}

func decodeCodexTokenResponse(body io.Reader, operation string) (codexTokenResponse, error) {
	data, err := io.ReadAll(io.LimitReader(body, maxCodexTokenBodyBytes+1))
	if err != nil {
		return codexTokenResponse{}, fmt.Errorf("%w: OpenAI Codex OAuth %s response could not be read", llm.ErrServer, operation)
	}
	if len(data) > maxCodexTokenBodyBytes {
		return codexTokenResponse{}, fmt.Errorf("%w: OpenAI Codex OAuth %s response was too large", llm.ErrAuth, operation)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	var token codexTokenResponse
	if err := decoder.Decode(&token); err != nil {
		return codexTokenResponse{}, fmt.Errorf("%w: OpenAI Codex OAuth %s response was invalid", llm.ErrAuth, operation)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return codexTokenResponse{}, fmt.Errorf("%w: OpenAI Codex OAuth %s response had trailing data", llm.ErrAuth, operation)
	}
	if token.ExpiresIn < 0 {
		return codexTokenResponse{}, fmt.Errorf("%w: OpenAI Codex OAuth %s response had invalid expiry", llm.ErrAuth, operation)
	}
	return token, nil
}

func sanitizeCodexOAuthRequestError(operation string, err error) error {
	if errors.Is(err, context.Canceled) {
		return fmt.Errorf("openai-codex: OAuth %s request cancelled: %w", operation, context.Canceled)
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("openai-codex: OAuth %s request timed out: %w", operation, llm.ErrTimeout)
	}
	return fmt.Errorf("%w: OpenAI Codex OAuth %s request could not be constructed", llm.ErrBadRequest, operation)
}

func sanitizeCodexOAuthTransportError(operation string, err error) error {
	switch {
	case errors.Is(err, provideroauth.ErrUnsafeRedirect):
		return fmt.Errorf("openai-codex: OAuth %s redirect refused: %w", operation, provideroauth.ErrUnsafeRedirect)
	case errors.Is(err, context.Canceled):
		return fmt.Errorf("openai-codex: OAuth %s cancelled: %w", operation, context.Canceled)
	case errors.Is(err, context.DeadlineExceeded), isTimeoutError(err), errors.Is(err, llm.ErrTimeout):
		return fmt.Errorf("openai-codex: OAuth %s timed out: %w", operation, llm.ErrTimeout)
	case errors.Is(err, llm.ErrAuth):
		return fmt.Errorf("openai-codex: OAuth %s transport rejected authentication: %w", operation, llm.ErrAuth)
	case errors.Is(err, llm.ErrRateLimited):
		return fmt.Errorf("openai-codex: OAuth %s transport was rate limited: %w", operation, llm.ErrRateLimited)
	default:
		return fmt.Errorf("openai-codex: OAuth %s transport failed: %w", operation, llm.ErrServer)
	}
}

func isTimeoutError(err error) bool {
	var netErr net.Error
	return errors.As(err, &netErr) && netErr.Timeout()
}

func codexAccountID(cred llm.AuthCredential) string {
	if cred.AccountID != "" {
		return cred.AccountID
	}
	return extractCodexAccountID(cred.Access)
}

func extractCodexAccountID(accessToken string) string {
	if len(accessToken) == 0 || len(accessToken) > maxCodexJWTBytes {
		return ""
	}
	parts := strings.Split(accessToken, ".")
	if len(parts) != 3 {
		return ""
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		payload, err = base64.URLEncoding.DecodeString(parts[1])
		if err != nil {
			return ""
		}
	}
	var claims map[string]json.RawMessage
	if err := json.Unmarshal(payload, &claims); err != nil {
		return ""
	}
	var auth struct {
		AccountID string `json:"chatgpt_account_id"`
	}
	if err := json.Unmarshal(claims[codexAccountClaimPath], &auth); err != nil {
		return ""
	}
	return auth.AccountID
}
