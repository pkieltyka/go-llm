package openai

import (
	"context"
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

// Sign in with ChatGPT: a public-client OAuth flow whose access token is
// sent directly to api.openai.com as a bearer token, so the ordinary
// Responses API is billed to the user's ChatGPT subscription.
const (
	chatGPTAuthorizeURL = "https://auth.openai.com/api/accounts/authorize"
	chatGPTTokenURL     = "https://auth.openai.com/api/accounts/oauth/token"
	// chatGPTResource is the audience every token request names.
	chatGPTResource = "https://api.openai.com/v1"
	// chatGPTDynamicClientID registers a new client per login; OpenAI returns
	// the issued client ID in the callback, and refresh requires it.
	chatGPTDynamicClientID = "dynamic_agent_client"
	// chatGPTDirectScope is the grant that lets the token reach the API.
	chatGPTDirectScope   = "chatgpt.tokens.use.direct"
	chatGPTScope         = "openid profile email offline_access resource.invoke " + chatGPTDirectScope
	maxChatGPTTokenBytes = 1 << 20
)

type chatGPTTokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int64  `json:"expires_in"`
	IDToken      string `json:"id_token"`
	Scope        string `json:"scope"`
}

func refreshChatGPTOAuth(ctx context.Context, client *http.Client, tokenURL string, cred llm.AuthCredential) (llm.AuthCredential, error) {
	if strings.TrimSpace(cred.ClientID) == "" {
		return llm.AuthCredential{}, fmt.Errorf("%w: OpenAI OAuth credential has no issued client ID; sign in with ChatGPT again", llm.ErrAuth)
	}
	token, err := requestChatGPTToken(ctx, client, tokenURL, url.Values{
		"grant_type":    {"refresh_token"},
		"client_id":     {cred.ClientID},
		"refresh_token": {cred.Refresh},
		"resource":      {chatGPTResource},
	}, "refresh")
	if err != nil {
		return llm.AuthCredential{}, err
	}
	return chatGPTCredential(token, cred.ClientID, "refresh")
}

func requestChatGPTToken(ctx context.Context, client *http.Client, tokenURL string, form url.Values, operation string) (chatGPTTokenResponse, error) {
	if client == nil {
		client = llm.DefaultHTTPClient()
	}
	if tokenURL == "" {
		tokenURL = chatGPTTokenURL
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return chatGPTTokenResponse{}, fmt.Errorf("%w: OpenAI OAuth %s request could not be built", llm.ErrBadRequest, operation)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := provideroauth.NoRedirectClient(client).Do(req)
	if err != nil {
		return chatGPTTokenResponse{}, sanitizeChatGPTTransportError(operation, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return chatGPTTokenResponse{}, provideroauth.TokenError(providerName, operation, resp.StatusCode)
	}
	var token chatGPTTokenResponse
	decoder := json.NewDecoder(io.LimitReader(resp.Body, maxChatGPTTokenBytes))
	if err := decoder.Decode(&token); err != nil {
		return chatGPTTokenResponse{}, fmt.Errorf("%w: OpenAI OAuth %s response was not valid JSON", llm.ErrServer, operation)
	}
	return token, nil
}

// chatGPTCredential validates a token response the way pi does: access and
// refresh tokens, a positive lifetime, and the direct-use scope are all
// required, so a grant that cannot reach the API fails at login time.
func chatGPTCredential(token chatGPTTokenResponse, clientID, operation string) (llm.AuthCredential, error) {
	switch {
	case strings.TrimSpace(token.AccessToken) == "":
		return llm.AuthCredential{}, fmt.Errorf("%w: OpenAI OAuth %s response missing access token", llm.ErrAuth, operation)
	case strings.TrimSpace(token.RefreshToken) == "":
		return llm.AuthCredential{}, fmt.Errorf("%w: OpenAI OAuth %s response missing refresh token", llm.ErrAuth, operation)
	case token.ExpiresIn <= 0:
		return llm.AuthCredential{}, fmt.Errorf("%w: OpenAI OAuth %s response has an invalid expires_in", llm.ErrAuth, operation)
	case !hasScope(token.Scope, chatGPTDirectScope):
		return llm.AuthCredential{}, fmt.Errorf("%w: OpenAI OAuth grant did not include %s", llm.ErrAuth, chatGPTDirectScope)
	}
	return llm.AuthCredential{
		Type:     "oauth",
		Access:   token.AccessToken,
		Refresh:  token.RefreshToken,
		Expires:  provideroauth.ExpiresAt(token.ExpiresIn),
		ClientID: clientID,
	}, nil
}

func hasScope(scopes, want string) bool {
	for _, scope := range strings.Fields(scopes) {
		if scope == want {
			return true
		}
	}
	return false
}

// sanitizeChatGPTTransportError keeps the failure class without echoing a
// URL whose query or form could carry tokens.
func sanitizeChatGPTTransportError(operation string, err error) error {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return err
	}
	if errors.Is(err, provideroauth.ErrUnsafeRedirect) {
		return fmt.Errorf("%w: OpenAI OAuth %s endpoint redirected", llm.ErrServer, operation)
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return fmt.Errorf("%w: OpenAI OAuth %s timed out", llm.ErrTimeout, operation)
	}
	return fmt.Errorf("%w: OpenAI OAuth %s transport failed", llm.ErrServer, operation)
}

// withOAuthTokenURL points OAuth refresh at a test token endpoint.
func withOAuthTokenURL(tokenURL string) Option {
	return func(c *config) { c.oauthTokenURL = tokenURL }
}
