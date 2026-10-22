package openai

import (
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/provideroauth"
)

const (
	chatGPTLoginCallbackPath    = "/auth/callback"
	chatGPTLoginLaunchPath      = "/authorize"
	chatGPTLoginPort            = 1455
	defaultChatGPTAgentName     = "go-llm"
	defaultLoginTimeout         = 5 * time.Minute
	defaultLoginExchangeTimeout = 30 * time.Second
)

var deviceIDPattern = regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`)

// LoginOption configures a Sign in with ChatGPT login flow.
type LoginOption func(*loginConfig)

type loginConfig struct {
	deviceID        string
	agentName       string
	httpClient      *http.Client
	flowTimeout     time.Duration
	exchangeTimeout time.Duration
	authorizeURL    string
	tokenURL        string
	random          io.Reader
	listen          func(network, address string) (net.Listener, error)
	port            int
}

// WithLoginDeviceID sets the stable ID of this application installation, a
// UUID sent to OpenAI as the agent host ID (urn:uuid:<id>). It is required:
// create it once per installation, store it with the app's settings, and
// pass the same value on every later login.
func WithLoginDeviceID(id string) LoginOption {
	return func(config *loginConfig) { config.deviceID = id }
}

// WithLoginAgentName sets the name the app introduces itself with on the
// ChatGPT consent screen. The default is "go-llm".
func WithLoginAgentName(name string) LoginOption {
	return func(config *loginConfig) { config.agentName = name }
}

// WithLoginHTTPClient replaces the client used for the authorization-code
// exchange. The client is shallow-copied and redirects are always refused.
func WithLoginHTTPClient(client *http.Client) LoginOption {
	return func(config *loginConfig) { config.httpClient = client }
}

// WithLoginTimeout bounds the complete browser interaction. The default is
// five minutes; the token exchange has its own shorter bound.
func WithLoginTimeout(timeout time.Duration) LoginOption {
	return func(config *loginConfig) { config.flowTimeout = timeout }
}

func defaultLoginConfig() loginConfig {
	return loginConfig{
		agentName:       defaultChatGPTAgentName,
		httpClient:      llm.DefaultHTTPClient(),
		flowTimeout:     defaultLoginTimeout,
		exchangeTimeout: defaultLoginExchangeTimeout,
		authorizeURL:    chatGPTAuthorizeURL,
		tokenURL:        chatGPTTokenURL,
		random:          rand.Reader,
		listen:          net.Listen,
		port:            chatGPTLoginPort,
	}
}

// NewLoginFlow returns a single-use Sign in with ChatGPT login: a browser
// authorization-code PKCE flow that lets a ChatGPT subscription use the
// OpenAI API. Persist the resulting credential under "openai" in a
// llm.CredentialStore and pass the store to WithCredentials.
//
// The flow binds only 127.0.0.1:1455 (the registered redirect), so it fails
// with ErrServer while another login (or the Codex CLI) holds that port. It
// registers a new OAuth client per login and stores the issued client ID on
// the returned credential, which refresh requires. Begin returns a loopback
// launch URL; Complete waits for the automatic callback, and Submit accepts
// a copied callback URL when the browser cannot reach loopback. The
// credential is returned, never persisted: the host stores it.
func NewLoginFlow(options ...LoginOption) (llm.LoginFlow, error) {
	config := defaultLoginConfig()
	for _, option := range options {
		if option != nil {
			option(&config)
		}
	}
	return newLoginFlow(config)
}

func newLoginFlow(config loginConfig) (*provideroauth.LoopbackFlow, error) {
	if !deviceIDPattern.MatchString(config.deviceID) {
		return nil, fmt.Errorf("%w: Sign in with ChatGPT requires WithLoginDeviceID with a UUID for this installation", llm.ErrBadRequest)
	}
	if strings.TrimSpace(config.agentName) == "" {
		return nil, fmt.Errorf("%w: OpenAI login agent name must not be empty", llm.ErrBadRequest)
	}
	if config.httpClient == nil {
		return nil, fmt.Errorf("%w: nil OpenAI login HTTP client", llm.ErrBadRequest)
	}
	if !provideroauth.ValidEndpoint(config.authorizeURL) || !provideroauth.ValidEndpoint(config.tokenURL) {
		return nil, fmt.Errorf("%w: OpenAI login endpoints must be absolute HTTPS URLs", llm.ErrBadRequest)
	}
	hostID := "urn:uuid:" + strings.ToLower(config.deviceID)
	return provideroauth.NewLoopbackFlow(provideroauth.LoopbackConfig{
		Label:          "OpenAI",
		Ports:          []int{config.port},
		RedirectURI:    chatGPTRedirectURI,
		CallbackPath:   chatGPTLoginCallbackPath,
		LaunchPath:     chatGPTLoginLaunchPath,
		Instructions:   "Open the launch URL and sign in with ChatGPT. If the callback cannot connect, copy the final redirect URL and pass it to Submit.",
		SuccessMessage: "ChatGPT authentication completed. You may close this window.",
		AuthorizationURL: func(auth provideroauth.LoopbackAuthorization) (string, error) {
			parsed, err := url.Parse(config.authorizeURL)
			if err != nil {
				return "", fmt.Errorf("%w: invalid OpenAI authorization endpoint", llm.ErrBadRequest)
			}
			query := parsed.Query()
			query.Set("client_id", chatGPTDynamicClientID)
			query.Set("agent_name_hint", config.agentName)
			query.Set("ext_agent_host_id", hostID)
			query.Set("response_type", "code")
			query.Set("redirect_uri", auth.RedirectURI)
			query.Set("resource", chatGPTResource)
			query.Set("scope", chatGPTScope)
			query.Set("state", auth.State)
			query.Set("code_challenge", auth.Challenge)
			query.Set("code_challenge_method", "S256")
			query.Set("nonce", auth.Nonce)
			parsed.RawQuery = query.Encode()
			return parsed.String(), nil
		},
		Exchange: func(ctx context.Context, code, verifier, redirectURI string, callback url.Values) (llm.AuthCredential, error) {
			clientID := strings.TrimSpace(callback.Get("client_id"))
			if clientID == "" || len(callback["client_id"]) != 1 {
				return llm.AuthCredential{}, fmt.Errorf("%w: OpenAI login callback did not contain an issued client ID", llm.ErrAuth)
			}
			token, err := requestChatGPTToken(ctx, config.httpClient, config.tokenURL, url.Values{
				"grant_type":    {"authorization_code"},
				"client_id":     {clientID},
				"code":          {code},
				"code_verifier": {verifier},
				"redirect_uri":  {redirectURI},
				"resource":      {chatGPTResource},
			}, "login token exchange")
			if err != nil {
				return llm.AuthCredential{}, err
			}
			if strings.TrimSpace(token.IDToken) == "" {
				return llm.AuthCredential{}, fmt.Errorf("%w: OpenAI login response did not contain an ID token", llm.ErrAuth)
			}
			return chatGPTCredential(token, clientID, "login")
		},
		FlowTimeout:     config.flowTimeout,
		ExchangeTimeout: config.exchangeTimeout,
		Random:          config.random,
		Listen:          config.listen,
	})
}

func chatGPTRedirectURI(port int) string {
	return "http://127.0.0.1:" + strconv.Itoa(port) + chatGPTLoginCallbackPath
}
