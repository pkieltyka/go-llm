package anthropic

import (
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/provideroauth"
)

// LoginMethod selects the provider's browser callback or manual copy-code flow.
type LoginMethod string

const (
	LoginBrowser  LoginMethod = "browser"
	LoginCopyCode LoginMethod = "copy_code"
)

// LoginOption configures a single-use Anthropic login.
type LoginOption func(*loginConfig)
type loginConfig struct {
	method          LoginMethod
	httpClient      *http.Client
	flowTimeout     time.Duration
	exchangeTimeout time.Duration
	authorizeURL    string
	tokenURL        string
	random          io.Reader
	listen          func(string, string) (net.Listener, error)
	ports           []int
}

// WithLoginMethod selects browser login (default) or the cloud copy-code page.
func WithLoginMethod(method LoginMethod) LoginOption {
	return func(c *loginConfig) { c.method = method }
}

// WithLoginHTTPClient sets the token exchange client. Redirects are refused.
func WithLoginHTTPClient(client *http.Client) LoginOption {
	return func(c *loginConfig) { c.httpClient = client }
}

// WithLoginTimeout bounds the browser interaction (default five minutes).
func WithLoginTimeout(timeout time.Duration) LoginOption {
	return func(c *loginConfig) { c.flowTimeout = timeout }
}

func defaultLoginConfig() loginConfig {
	return loginConfig{method: LoginBrowser, httpClient: llm.DefaultHTTPClient(), flowTimeout: 5 * time.Minute,
		exchangeTimeout: 30 * time.Second, authorizeURL: anthropicOAuthAuthorizeURL, tokenURL: anthropicOAuthTokenURL,
		random: rand.Reader, listen: net.Listen, ports: []int{53692, 0}}
}

// NewLoginFlow prepares a Pi-compatible Anthropic OAuth login. Begin exposes
// only a private loopback launch target, never the provider URL or PKCE state.
// A browser on another machine requires port forwarding to that launch port,
// including in copy-code mode. If no listener can bind, login fails locally.
// Complete returns the credential without storing it; the host owns persistence.
func NewLoginFlow(options ...LoginOption) (llm.LoginFlow, error) {
	cfg := defaultLoginConfig()
	for _, option := range options {
		if option != nil {
			option(&cfg)
		}
	}
	return newLoginFlow(cfg)
}

func newLoginFlow(cfg loginConfig) (*provideroauth.LoopbackFlow, error) {
	if cfg.method != LoginBrowser && cfg.method != LoginCopyCode {
		return nil, fmt.Errorf("%w: unsupported Anthropic login method", llm.ErrBadRequest)
	}
	if cfg.httpClient == nil {
		return nil, fmt.Errorf("%w: nil Anthropic login HTTP client", llm.ErrBadRequest)
	}
	if !provideroauth.ValidEndpoint(cfg.authorizeURL) || !provideroauth.ValidEndpoint(cfg.tokenURL) {
		return nil, fmt.Errorf("%w: invalid Anthropic login endpoint", llm.ErrBadRequest)
	}
	path := "/callback"
	redirect := func(port int) string { return "http://localhost:" + strconv.Itoa(port) + path }
	instructions := "Open the local launch URL to sign in with Claude. Remote browsers require port forwarding to this launch port. Complete in the browser or paste its final redirect URL."
	if cfg.method == LoginCopyCode {
		path = "/oauth/code/callback"
		redirect = func(int) string { return "https://platform.claude.com/oauth/code/callback" }
		instructions = "Open the local launch URL, then paste the code Anthropic shows. Remote browsers still require port forwarding to the launch port."
	}
	return provideroauth.NewLoopbackFlow(provideroauth.LoopbackConfig{
		Label: "Anthropic", Ports: cfg.ports, AllowEphemeralPort: true, StateIsVerifier: true, AllowManualCode: true,
		RedirectURI: redirect, CallbackPath: path, LaunchPath: "/authorize", Instructions: instructions,
		SuccessMessage: "Anthropic sign-in received. You may close this window.",
		AuthorizationURL: func(auth provideroauth.LoopbackAuthorization) (string, error) {
			parsed, err := url.Parse(cfg.authorizeURL)
			if err != nil {
				return "", fmt.Errorf("%w: invalid Anthropic authorization endpoint", llm.ErrBadRequest)
			}
			q := parsed.Query()
			for k, v := range map[string]string{"code": "true", "client_id": anthropicOAuthClientID, "response_type": "code", "redirect_uri": auth.RedirectURI, "scope": anthropicOAuthScope, "code_challenge": auth.Challenge, "code_challenge_method": "S256", "state": auth.State} {
				q.Set(k, v)
			}
			parsed.RawQuery = q.Encode()
			return parsed.String(), nil
		},
		Exchange: func(ctx context.Context, code, verifier, redirectURI string, callback url.Values) (llm.AuthCredential, error) {
			token, err := requestAnthropicToken(ctx, cfg.httpClient, cfg.tokenURL, map[string]string{
				"grant_type": "authorization_code", "client_id": anthropicOAuthClientID, "code": code, "state": callback.Get("state"), "redirect_uri": redirectURI, "code_verifier": verifier,
			}, "login")
			if err != nil {
				return llm.AuthCredential{}, err
			}
			return anthropicCredential(token, time.Now())
		}, FlowTimeout: cfg.flowTimeout, ExchangeTimeout: cfg.exchangeTimeout, Random: cfg.random, Listen: cfg.listen,
	})
}
