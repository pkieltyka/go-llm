package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/provideroauth"
)

const (
	anthropicOAuthBeta = "oauth-2025-04-20"
	// Public OAuth client ID for Anthropic's first-party Claude Code flow.
	// This is not a client secret.
	anthropicOAuthClientID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e" // gitleaks:allow public OAuth client ID
	anthropicOAuthTokenURL = "https://platform.claude.com/v1/oauth/token"

	// Claude Code identity (FS §17C): subscription OAuth tokens are only
	// served to Claude-Code-identified traffic. Values mirror pi's
	// api/anthropic-messages.ts (claudeCodeVersion 2.1.75).
	anthropicClaudeCodeBeta = "claude-code-20250219"
	anthropicOAuthUserAgent = "claude-cli/2.1.75"

	// claudeCodeSystemPrompt must be the FIRST system block on every OAuth
	// request; the caller's System text becomes the second block.
	claudeCodeSystemPrompt = "You are Claude Code, Anthropic's official CLI for Claude."
)

func newAnthropicOAuthSource(cfg config) (*provideroauth.Source, error) {
	return provideroauth.New(cfg.oauthCred, func(ctx context.Context, cred llm.AuthCredential) (llm.AuthCredential, error) {
		return refreshAnthropicOAuth(ctx, cfg.httpClient, cfg.oauthTokenURL, cred)
	}, cfg.oauthPersistence)
}

func refreshAnthropicOAuth(ctx context.Context, client *http.Client, tokenURL string, cred llm.AuthCredential) (llm.AuthCredential, error) {
	if client == nil {
		client = llm.DefaultHTTPClient()
	}
	client = provideroauth.NoRedirectClient(client)
	if tokenURL == "" {
		tokenURL = anthropicOAuthTokenURL
	}
	body, err := json.Marshal(map[string]string{
		"grant_type":    "refresh_token",
		"client_id":     anthropicOAuthClientID,
		"refresh_token": cred.Refresh,
	})
	if err != nil {
		return llm.AuthCredential{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, bytes.NewReader(body))
	if err != nil {
		return llm.AuthCredential{}, err
	}
	req.Header.Set("Content-Type", "application/json")
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
		return llm.AuthCredential{}, fmt.Errorf("%w: Anthropic OAuth refresh response was invalid", llm.ErrAuth)
	}
	if token.AccessToken == "" {
		return llm.AuthCredential{}, fmt.Errorf("%w: Anthropic OAuth refresh response missing access token", llm.ErrAuth)
	}
	return llm.AuthCredential{
		Type:    "oauth",
		Access:  token.AccessToken,
		Refresh: token.RefreshToken,
		Expires: provideroauth.ExpiresAt(token.ExpiresIn),
	}, nil
}
