package openaicodex

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
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
)

func newCodexOAuthSource(cfg config) (*provideroauth.Source, error) {
	return provideroauth.New(cfg.oauthCred, func(ctx context.Context, cred llm.AuthCredential) (llm.AuthCredential, error) {
		return refreshCodexOAuth(ctx, cfg.httpClient, cfg.tokenURL, cred)
	}, cfg.persistence)
}

func refreshCodexOAuth(ctx context.Context, client *http.Client, tokenURL string, cred llm.AuthCredential) (llm.AuthCredential, error) {
	if client == nil {
		client = llm.DefaultHTTPClient()
	}
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
		return llm.AuthCredential{}, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return llm.AuthCredential{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return llm.AuthCredential{}, provideroauth.RefreshError(providerName, resp.StatusCode)
	}
	var token struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
		ExpiresIn    int64  `json:"expires_in"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&token); err != nil {
		return llm.AuthCredential{}, fmt.Errorf("%w: OpenAI Codex OAuth refresh response was invalid", llm.ErrAuth)
	}
	if token.AccessToken == "" {
		return llm.AuthCredential{}, fmt.Errorf("%w: OpenAI Codex OAuth refresh response missing access token", llm.ErrAuth)
	}
	accountID := extractCodexAccountID(token.AccessToken)
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

func codexAccountID(cred llm.AuthCredential) string {
	if cred.AccountID != "" {
		return cred.AccountID
	}
	return extractCodexAccountID(cred.Access)
}

func extractCodexAccountID(accessToken string) string {
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
