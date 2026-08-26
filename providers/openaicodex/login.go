package openaicodex

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/provideroauth"
)

const (
	openAICodexOAuthAuthorizeURL = "https://auth.openai.com/oauth/authorize"
	openAICodexOAuthScopes       = "openid profile email offline_access api.connectors.read api.connectors.invoke"
	codexLoginCallbackPath       = "/auth/callback"
	codexLoginLaunchPath         = "/authorize"
	codexLoginPreferredPort      = 1455
	codexLoginFallbackPort       = 1457
	defaultLoginTimeout          = 5 * time.Minute
	defaultLoginExchangeTimeout  = 30 * time.Second
	loginReadHeaderTimeout       = 5 * time.Second
	maxLoginRequestBytes         = 16 << 10
	maxLoginCodeBytes            = 12 << 10
	maxLoginStateBytes           = 1 << 10
	maxLoginErrorBytes           = 1 << 10
	maxLoginResponseBytes        = 2 << 10
)

// LoginOption configures a Codex browser login flow.
type LoginOption func(*loginConfig)

// WithLoginHTTPClient replaces the client used for the authorization-code
// exchange. The client is shallow-copied and redirects are always refused.
func WithLoginHTTPClient(client *http.Client) LoginOption {
	return func(config *loginConfig) { config.httpClient = client }
}

// WithLoginTimeout bounds the complete browser interaction. The default is
// five minutes. The token exchange has its own shorter provider-owned bound.
func WithLoginTimeout(timeout time.Duration) LoginOption {
	return func(config *loginConfig) { config.flowTimeout = timeout }
}

type loginConfig struct {
	httpClient      *http.Client
	flowTimeout     time.Duration
	exchangeTimeout time.Duration
	authorizeURL    string
	tokenURL        string
	random          io.Reader
	listen          func(network, address string) (net.Listener, error)
	callbackPorts   [2]int
	beforeFinalize  func()
}

func defaultLoginConfig() loginConfig {
	return loginConfig{
		httpClient:      llm.DefaultHTTPClient(),
		flowTimeout:     defaultLoginTimeout,
		exchangeTimeout: defaultLoginExchangeTimeout,
		authorizeURL:    openAICodexOAuthAuthorizeURL,
		tokenURL:        openAICodexOAuthTokenURL,
		random:          rand.Reader,
		listen:          net.Listen,
		callbackPorts:   [2]int{codexLoginPreferredPort, codexLoginFallbackPort},
	}
}

// NewLoginFlow returns a single-use browser authorization-code PKCE login for
// ChatGPT-backed Codex credentials. The flow listens only on IPv4 loopback,
// handles the callback automatically, and returns the credential without
// persisting it. Submit is available as a copied-callback fallback.
func NewLoginFlow(options ...LoginOption) (llm.LoginFlow, error) {
	config := defaultLoginConfig()
	for _, option := range options {
		if option != nil {
			option(&config)
		}
	}
	return newLoginFlow(config)
}

func newLoginFlow(config loginConfig) (*loginFlow, error) {
	if config.httpClient == nil {
		return nil, fmt.Errorf("%w: nil OpenAI Codex login HTTP client", llm.ErrBadRequest)
	}
	if config.flowTimeout <= 0 || config.exchangeTimeout <= 0 {
		return nil, fmt.Errorf("%w: OpenAI Codex login timeouts must be positive", llm.ErrBadRequest)
	}
	if config.random == nil || config.listen == nil || config.callbackPorts[0] <= 0 || config.callbackPorts[1] <= 0 {
		return nil, fmt.Errorf("%w: incomplete OpenAI Codex login configuration", llm.ErrBadRequest)
	}
	if !validLoginEndpoint(config.authorizeURL) || !validLoginEndpoint(config.tokenURL) {
		return nil, fmt.Errorf("%w: OpenAI Codex login endpoints must be absolute HTTPS URLs", llm.ErrBadRequest)
	}
	return &loginFlow{
		config:    config,
		status:    loginFresh,
		responses: make(chan loginResponse, 1),
		done:      make(chan struct{}),
	}, nil
}

func validLoginEndpoint(value string) bool {
	parsed, err := url.Parse(value)
	if err != nil || parsed.Host == "" || parsed.User != nil || parsed.Fragment != "" {
		return false
	}
	if parsed.Scheme == "https" {
		return true
	}
	return parsed.Scheme == "http" && (parsed.Hostname() == "127.0.0.1" || parsed.Hostname() == "localhost" || parsed.Hostname() == "::1")
}

type loginStatus uint8

const (
	loginFresh loginStatus = iota
	loginStarting
	loginBegun
	loginExchanging
	loginCompleted
	loginCancelled
	loginExpired
)

type loginResponse struct {
	code   string
	denied bool
}

type loginFlow struct {
	mu sync.Mutex

	config         loginConfig
	status         loginStatus
	state          string
	verifier       string
	redirectURI    string
	responses      chan loginResponse
	responseSet    bool
	completeCalled bool
	done           chan struct{}
	doneClosed     bool
	timer          *time.Timer
	server         *http.Server
	listener       net.Listener
	activeCancel   context.CancelFunc
}

func (*loginFlow) String() string   { return "openai-codex: login flow (redacted)" }
func (*loginFlow) GoString() string { return "openaicodex.loginFlow{redacted}" }
func (*loginFlow) LogValue() slog.Value {
	return slog.StringValue("openai-codex: login flow (redacted)")
}

func (flow *loginFlow) Begin(ctx context.Context) (llm.LoginAuthorization, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return llm.LoginAuthorization{}, err
	}
	flow.mu.Lock()
	if flow.status != loginFresh {
		flow.mu.Unlock()
		return llm.LoginAuthorization{}, llm.ErrLoginClosed
	}
	flow.status = loginStarting
	flow.mu.Unlock()

	verifier, err := randomBase64URL(flow.config.random, 32)
	if err != nil {
		flow.failStart()
		return llm.LoginAuthorization{}, errors.New("openai-codex: generate login PKCE verifier")
	}
	state, err := randomBase64URL(flow.config.random, 32)
	if err != nil {
		flow.failStart()
		return llm.LoginAuthorization{}, errors.New("openai-codex: generate login state")
	}
	if err := ctx.Err(); err != nil {
		flow.failStart()
		return llm.LoginAuthorization{}, err
	}

	listener, port, err := flow.listenLoopback()
	if err != nil {
		flow.failStart()
		return llm.LoginAuthorization{}, fmt.Errorf("%w: OpenAI Codex login callback ports are unavailable", llm.ErrServer)
	}
	redirectURI := codexRedirectURI(port)
	authorizationURL, err := flow.authorizationURL(redirectURI, state, verifier)
	if err != nil {
		_ = listener.Close()
		flow.failStart()
		return llm.LoginAuthorization{}, err
	}
	launchURL := "http://127.0.0.1:" + strconv.Itoa(port) + codexLoginLaunchPath
	authorization, err := llm.NewLoginAuthorization(launchURL,
		"Open the launch URL and complete OpenAI sign-in. If the callback cannot connect, copy its complete URL and pass it to Submit.")
	if err != nil {
		_ = listener.Close()
		flow.failStart()
		return llm.LoginAuthorization{}, err
	}

	server := &http.Server{
		ReadHeaderTimeout: loginReadHeaderTimeout,
		WriteTimeout:      loginReadHeaderTimeout,
		IdleTimeout:       loginReadHeaderTimeout,
		MaxHeaderBytes:    maxLoginRequestBytes,
	}
	server.Handler = flow.callbackHandler(authorizationURL)

	flow.mu.Lock()
	if flow.status != loginStarting {
		flow.mu.Unlock()
		_ = listener.Close()
		return llm.LoginAuthorization{}, llm.ErrLoginClosed
	}
	flow.state = state
	flow.verifier = verifier
	flow.redirectURI = redirectURI
	flow.listener = listener
	flow.server = server
	flow.status = loginBegun
	flow.timer = time.AfterFunc(flow.config.flowTimeout, flow.expire)
	flow.mu.Unlock()

	go func() { _ = server.Serve(listener) }()
	return authorization, nil
}

func (flow *loginFlow) Complete(ctx context.Context) (llm.AuthCredential, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	flow.mu.Lock()
	if flow.status != loginBegun || flow.completeCalled {
		err := flow.statusErrorLocked()
		flow.mu.Unlock()
		return llm.AuthCredential{}, err
	}
	flow.completeCalled = true
	responses := flow.responses
	done := flow.done
	flow.mu.Unlock()

	var response loginResponse
	select {
	case response = <-responses:
	case <-done:
		flow.mu.Lock()
		err := flow.statusErrorLocked()
		flow.mu.Unlock()
		return llm.AuthCredential{}, err
	case <-ctx.Done():
		flow.Cancel()
		return llm.AuthCredential{}, ctx.Err()
	}

	flow.mu.Lock()
	if flow.status != loginBegun {
		err := flow.statusErrorLocked()
		flow.mu.Unlock()
		return llm.AuthCredential{}, err
	}
	flow.status = loginExchanging
	verifier := flow.verifier
	redirectURI := flow.redirectURI
	exchangeCtx, cancel := context.WithTimeout(ctx, flow.config.exchangeTimeout)
	flow.activeCancel = cancel
	flow.mu.Unlock()

	var credential llm.AuthCredential
	var exchangeErr error
	if response.denied {
		exchangeErr = fmt.Errorf("%w: OpenAI Codex login was denied", llm.ErrAuth)
	} else {
		credential, exchangeErr = flow.exchange(exchangeCtx, response.code, verifier, redirectURI)
	}
	if flow.config.beforeFinalize != nil {
		flow.config.beforeFinalize()
	}
	cancel()

	flow.mu.Lock()
	flow.activeCancel = nil
	status := flow.status
	if status == loginExchanging {
		flow.finishLocked(loginCompleted)
	}
	flow.mu.Unlock()
	flow.closeResources()

	switch {
	case status == loginExpired:
		return llm.AuthCredential{}, llm.ErrLoginExpired
	case ctx.Err() != nil:
		return llm.AuthCredential{}, ctx.Err()
	case status == loginCancelled:
		return llm.AuthCredential{}, llm.ErrLoginClosed
	default:
		return credential, exchangeErr
	}
}

func (flow *loginFlow) Submit(ctx context.Context, response string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	values, err := parseSubmittedCallback(response)
	if err != nil {
		return err
	}
	if err := flow.deliver(values); err != nil {
		return err
	}
	go flow.closeResources()
	return nil
}

func (flow *loginFlow) Cancel() {
	flow.mu.Lock()
	switch flow.status {
	case loginCompleted, loginCancelled, loginExpired:
		flow.mu.Unlock()
		return
	default:
		flow.finishLocked(loginCancelled)
	}
	flow.mu.Unlock()
	flow.closeResources()
}

func (flow *loginFlow) listenLoopback() (net.Listener, int, error) {
	var lastErr error
	for _, port := range flow.config.callbackPorts {
		listener, err := flow.config.listen("tcp4", "127.0.0.1:"+strconv.Itoa(port))
		if err == nil {
			return listener, port, nil
		}
		lastErr = err
	}
	return nil, 0, lastErr
}

func codexRedirectURI(port int) string {
	return "http://localhost:" + strconv.Itoa(port) + codexLoginCallbackPath
}

func (flow *loginFlow) authorizationURL(redirectURI, state, verifier string) (string, error) {
	parsed, err := url.Parse(flow.config.authorizeURL)
	if err != nil {
		return "", fmt.Errorf("%w: invalid OpenAI Codex authorization endpoint", llm.ErrBadRequest)
	}
	challengeBytes := sha256.Sum256([]byte(verifier))
	query := parsed.Query()
	query.Set("response_type", "code")
	query.Set("client_id", openAICodexOAuthClientID)
	query.Set("redirect_uri", redirectURI)
	query.Set("scope", openAICodexOAuthScopes)
	query.Set("code_challenge", base64.RawURLEncoding.EncodeToString(challengeBytes[:]))
	query.Set("code_challenge_method", "S256")
	query.Set("id_token_add_organizations", "true")
	query.Set("codex_cli_simplified_flow", "true")
	query.Set("originator", defaultOriginator)
	query.Set("state", state)
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}

func (flow *loginFlow) callbackHandler(authorizationURL string) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Cache-Control", "no-store")
		writer.Header().Set("Referrer-Policy", "no-referrer")
		writer.Header().Set("Content-Type", "text/plain; charset=utf-8")
		if len(request.RequestURI) > maxLoginRequestBytes {
			writeLoginResponse(writer, http.StatusRequestURITooLong, "Login callback was too large.")
			return
		}
		if request.Method != http.MethodGet {
			writer.Header().Set("Allow", http.MethodGet)
			writeLoginResponse(writer, http.StatusMethodNotAllowed, "Only GET is supported.")
			return
		}
		if request.URL.Path == codexLoginLaunchPath {
			http.Redirect(writer, request, authorizationURL, http.StatusFound)
			return
		}
		if request.URL.Path != codexLoginCallbackPath {
			writeLoginResponse(writer, http.StatusNotFound, "Not found.")
			return
		}
		values, err := url.ParseQuery(request.URL.RawQuery)
		if err != nil {
			writeLoginResponse(writer, http.StatusBadRequest, "Login callback was invalid.")
			return
		}
		if err := flow.deliver(values); err != nil {
			status := http.StatusBadRequest
			if errors.Is(err, llm.ErrLoginClosed) {
				status = http.StatusConflict
			}
			writeLoginResponse(writer, status, "Login callback was not accepted.")
			return
		}
		if values.Get("error") != "" {
			writeLoginResponse(writer, http.StatusUnauthorized, "OpenAI sign-in was denied. You may close this window.")
		} else {
			writeLoginResponse(writer, http.StatusOK, "OpenAI Codex sign-in completed. You may close this window.")
		}
		go flow.closeResources()
	})
}

func writeLoginResponse(writer http.ResponseWriter, status int, message string) {
	if len(message) > maxLoginResponseBytes {
		message = "Login response unavailable."
	}
	writer.WriteHeader(status)
	_, _ = io.WriteString(writer, message)
}

func parseSubmittedCallback(response string) (url.Values, error) {
	if len(response) == 0 || len(response) > maxLoginRequestBytes {
		return nil, fmt.Errorf("%w: OpenAI Codex login callback has invalid size", llm.ErrBadRequest)
	}
	response = strings.TrimSpace(response)
	if response == "" {
		return nil, fmt.Errorf("%w: OpenAI Codex login callback is empty", llm.ErrBadRequest)
	}
	rawQuery := response
	if strings.HasPrefix(response, "http://") || strings.HasPrefix(response, "https://") {
		parsed, err := url.Parse(response)
		if err != nil || parsed.Path != codexLoginCallbackPath || parsed.RawQuery == "" || parsed.Fragment != "" {
			return nil, fmt.Errorf("%w: OpenAI Codex login callback URL is invalid", llm.ErrBadRequest)
		}
		rawQuery = parsed.RawQuery
	} else {
		rawQuery = strings.TrimPrefix(rawQuery, "?")
	}
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return nil, fmt.Errorf("%w: OpenAI Codex login callback query is invalid", llm.ErrBadRequest)
	}
	return values, nil
}

func (flow *loginFlow) deliver(values url.Values) error {
	flow.mu.Lock()
	if flow.status != loginBegun || flow.responseSet {
		err := flow.statusErrorLocked()
		flow.mu.Unlock()
		return err
	}
	expectedState := flow.state
	flow.mu.Unlock()

	stateValues := values["state"]
	codeValues := values["code"]
	errorValues := values["error"]
	descriptionValues := values["error_description"]
	if len(stateValues) != 1 || len(stateValues[0]) == 0 || len(stateValues[0]) > maxLoginStateBytes ||
		len(codeValues) > 1 || len(errorValues) > 1 || len(descriptionValues) > 1 {
		return fmt.Errorf("%w: OpenAI Codex login callback fields are invalid", llm.ErrBadRequest)
	}
	state := stateValues[0]
	if !constantTimeEqual(state, expectedState) {
		return llm.ErrLoginStateMismatch
	}
	code := values.Get("code")
	providerError := values.Get("error")
	errorDescription := values.Get("error_description")
	if len(code) > maxLoginCodeBytes || len(providerError) > maxLoginErrorBytes || len(errorDescription) > maxLoginErrorBytes {
		return fmt.Errorf("%w: OpenAI Codex login callback fields are too large", llm.ErrBadRequest)
	}
	if (code == "") == (providerError == "") {
		return fmt.Errorf("%w: OpenAI Codex login callback must contain one result", llm.ErrBadRequest)
	}

	flow.mu.Lock()
	defer flow.mu.Unlock()
	if flow.status != loginBegun || flow.responseSet {
		return flow.statusErrorLocked()
	}
	if !constantTimeEqual(state, flow.state) {
		return llm.ErrLoginStateMismatch
	}
	flow.responseSet = true
	flow.responses <- loginResponse{code: code, denied: providerError != ""}
	return nil
}

func constantTimeEqual(left, right string) bool {
	return len(left) == len(right) && subtle.ConstantTimeCompare([]byte(left), []byte(right)) == 1
}

func (flow *loginFlow) exchange(ctx context.Context, code, verifier, redirectURI string) (llm.AuthCredential, error) {
	form := url.Values{
		"grant_type":    {"authorization_code"},
		"code":          {code},
		"redirect_uri":  {redirectURI},
		"client_id":     {openAICodexOAuthClientID},
		"code_verifier": {verifier},
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, flow.config.tokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return llm.AuthCredential{}, sanitizeCodexOAuthRequestError("login token exchange", err)
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.Header.Set(originatorHeader, defaultOriginator)
	client := provideroauth.NoRedirectClient(flow.config.httpClient)
	response, err := client.Do(request)
	if err != nil {
		return llm.AuthCredential{}, sanitizeCodexOAuthTransportError("login token exchange", err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return llm.AuthCredential{}, provideroauth.RefreshError(providerName, response.StatusCode)
	}
	token, err := decodeCodexTokenResponse(response.Body, "login token exchange")
	if err != nil {
		return llm.AuthCredential{}, err
	}
	if token.AccessToken == "" {
		return llm.AuthCredential{}, fmt.Errorf("%w: OpenAI Codex OAuth login response missing access token", llm.ErrAuth)
	}
	accountID := extractCodexAccountID(token.IDToken)
	if accountID == "" {
		accountID = extractCodexAccountID(token.AccessToken)
	}
	if accountID == "" {
		return llm.AuthCredential{}, fmt.Errorf("%w: OpenAI Codex OAuth login response missing account ID", llm.ErrAuth)
	}
	return llm.AuthCredential{
		Type:      "oauth",
		Access:    token.AccessToken,
		Refresh:   token.RefreshToken,
		Expires:   provideroauth.ExpiresAt(token.ExpiresIn),
		AccountID: accountID,
	}, nil
}

func randomBase64URL(reader io.Reader, size int) (string, error) {
	data := make([]byte, size)
	if _, err := io.ReadFull(reader, data); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(data), nil
}

func (flow *loginFlow) expire() {
	flow.mu.Lock()
	if flow.status != loginBegun && flow.status != loginExchanging {
		flow.mu.Unlock()
		return
	}
	flow.finishLocked(loginExpired)
	flow.mu.Unlock()
	flow.closeResources()
}

func (flow *loginFlow) failStart() {
	flow.mu.Lock()
	if flow.status == loginStarting {
		flow.finishLocked(loginCompleted)
	}
	flow.mu.Unlock()
}

func (flow *loginFlow) finishLocked(status loginStatus) {
	flow.status = status
	flow.state = ""
	flow.verifier = ""
	flow.redirectURI = ""
	select {
	case <-flow.responses:
	default:
	}
	if flow.timer != nil {
		flow.timer.Stop()
		flow.timer = nil
	}
	if flow.activeCancel != nil {
		flow.activeCancel()
	}
	if !flow.doneClosed {
		close(flow.done)
		flow.doneClosed = true
	}
}

func (flow *loginFlow) closeResources() {
	flow.mu.Lock()
	server := flow.server
	listener := flow.listener
	flow.server = nil
	flow.listener = nil
	flow.mu.Unlock()
	if listener != nil {
		_ = listener.Close()
	}
	if server != nil {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
		if err := server.Shutdown(shutdownCtx); err != nil {
			_ = server.Close()
		}
		cancel()
	}
}

func (flow *loginFlow) statusErrorLocked() error {
	if flow.status == loginExpired {
		return llm.ErrLoginExpired
	}
	return llm.ErrLoginClosed
}
