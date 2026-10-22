package provideroauth

import (
	"context"
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
)

const (
	loopbackReadHeaderTimeout = 5 * time.Second
	maxLoopbackRequestBytes   = 16 << 10
	maxLoopbackCodeBytes      = 12 << 10
	maxLoopbackStateBytes     = 1 << 10
	maxLoopbackErrorBytes     = 1 << 10
	maxLoopbackResponseBytes  = 2 << 10
)

// LoopbackAuthorization carries the per-login values an authorization URL
// is built from.
type LoopbackAuthorization struct {
	RedirectURI string
	State       string
	// Challenge is the S256 PKCE code challenge for the flow's verifier.
	Challenge string
	// Nonce is a fresh random value for flows that send an OpenID nonce.
	Nonce string
}

// LoopbackExchangeFunc trades an authorization code for a credential. The
// complete callback query is passed so flows can read provider-issued values
// such as a dynamically registered client ID.
type LoopbackExchangeFunc func(ctx context.Context, code, verifier, redirectURI string, callback url.Values) (llm.AuthCredential, error)

// LoopbackConfig configures a single-use browser authorization-code PKCE
// login served on IPv4 loopback.
type LoopbackConfig struct {
	// Label names the provider in errors and redacted formatting, such as
	// "OpenAI".
	Label string
	// Ports are tried in order on 127.0.0.1.
	Ports []int
	// RedirectURI builds the registered redirect URI for the bound port.
	RedirectURI func(port int) string
	// CallbackPath is the redirect path; LaunchPath, when set, serves a
	// redirect to the authorization URL so the launch URL stays short and
	// loopback-only.
	CallbackPath string
	LaunchPath   string
	// Instructions accompany the launch URL.
	Instructions string
	// SuccessMessage is shown in the browser after an accepted callback.
	SuccessMessage string
	// AuthorizationURL builds the provider authorization URL.
	AuthorizationURL func(LoopbackAuthorization) (string, error)
	// Exchange trades the code for a credential.
	Exchange LoopbackExchangeFunc

	FlowTimeout     time.Duration
	ExchangeTimeout time.Duration
	Random          io.Reader
	Listen          func(network, address string) (net.Listener, error)
	// BeforeFinalize is a test hook run after the exchange returns.
	BeforeFinalize func()
}

type loopbackStatus uint8

const (
	loopbackFresh loopbackStatus = iota
	loopbackStarting
	loopbackBegun
	loopbackExchanging
	loopbackCompleted
	loopbackCancelled
	loopbackExpired
)

type loopbackResponse struct {
	code   string
	denied bool
	values url.Values
}

// LoopbackFlow implements llm.LoginFlow for a LoopbackConfig. It listens
// only on IPv4 loopback, compares state in constant time, bounds every
// callback field, refuses reuse, expires after FlowTimeout, and never
// persists the credential it returns.
type LoopbackFlow struct {
	mu sync.Mutex

	config         LoopbackConfig
	status         loopbackStatus
	state          string
	verifier       string
	redirectURI    string
	responses      chan loopbackResponse
	responseSet    bool
	completeCalled bool
	done           chan struct{}
	doneClosed     bool
	timer          *time.Timer
	server         *http.Server
	listener       net.Listener
	activeCancel   context.CancelFunc
}

var _ llm.LoginFlow = (*LoopbackFlow)(nil)

// NewLoopbackFlow validates config and returns an unstarted flow.
func NewLoopbackFlow(config LoopbackConfig) (*LoopbackFlow, error) {
	if config.Label == "" || config.AuthorizationURL == nil || config.Exchange == nil || config.RedirectURI == nil ||
		config.CallbackPath == "" || len(config.Ports) == 0 || config.Random == nil || config.Listen == nil {
		return nil, fmt.Errorf("%w: incomplete %s login configuration", llm.ErrBadRequest, config.Label)
	}
	if config.FlowTimeout <= 0 || config.ExchangeTimeout <= 0 {
		return nil, fmt.Errorf("%w: %s login timeouts must be positive", llm.ErrBadRequest, config.Label)
	}
	for _, port := range config.Ports {
		if port <= 0 || port > 65535 {
			return nil, fmt.Errorf("%w: invalid %s login callback port %d", llm.ErrBadRequest, config.Label, port)
		}
	}
	return &LoopbackFlow{
		config:    config,
		status:    loopbackFresh,
		responses: make(chan loopbackResponse, 1),
		done:      make(chan struct{}),
	}, nil
}

// ValidEndpoint reports whether value is an absolute HTTPS URL, or HTTP on
// loopback (for tests), without credentials or fragment.
func ValidEndpoint(value string) bool {
	parsed, err := url.Parse(value)
	if err != nil || parsed.Host == "" || parsed.User != nil || parsed.Fragment != "" {
		return false
	}
	if parsed.Scheme == "https" {
		return true
	}
	return parsed.Scheme == "http" && (parsed.Hostname() == "127.0.0.1" || parsed.Hostname() == "localhost" || parsed.Hostname() == "::1")
}

func (flow *LoopbackFlow) String() string { return flow.redacted() }

// GoString redacts the flow.
func (flow *LoopbackFlow) GoString() string { return flow.redacted() }

// LogValue redacts the flow.
func (flow *LoopbackFlow) LogValue() slog.Value { return slog.StringValue(flow.redacted()) }

func (flow *LoopbackFlow) redacted() string {
	return strings.ToLower(flow.config.Label) + ": login flow (redacted)"
}

// Begin binds the loopback callback and returns the launch authorization.
func (flow *LoopbackFlow) Begin(ctx context.Context) (llm.LoginAuthorization, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return llm.LoginAuthorization{}, err
	}
	flow.mu.Lock()
	if flow.status != loopbackFresh {
		flow.mu.Unlock()
		return llm.LoginAuthorization{}, llm.ErrLoginClosed
	}
	flow.status = loopbackStarting
	flow.mu.Unlock()

	verifier, err := randomBase64URL(flow.config.Random, 32)
	if err != nil {
		flow.failStart()
		return llm.LoginAuthorization{}, fmt.Errorf("%s: generate login PKCE verifier", flow.config.Label)
	}
	state, err := randomBase64URL(flow.config.Random, 32)
	if err != nil {
		flow.failStart()
		return llm.LoginAuthorization{}, fmt.Errorf("%s: generate login state", flow.config.Label)
	}
	nonce, err := randomBase64URL(flow.config.Random, 32)
	if err != nil {
		flow.failStart()
		return llm.LoginAuthorization{}, fmt.Errorf("%s: generate login nonce", flow.config.Label)
	}
	launchToken, err := randomBase64URL(flow.config.Random, 32)
	if err != nil {
		flow.failStart()
		return llm.LoginAuthorization{}, fmt.Errorf("%s: generate login launch token", flow.config.Label)
	}
	if err := ctx.Err(); err != nil {
		flow.failStart()
		return llm.LoginAuthorization{}, err
	}

	listener, port, err := flow.listenLoopback()
	if err != nil {
		flow.failStart()
		return llm.LoginAuthorization{}, fmt.Errorf("%w: %s login callback port %v is unavailable (another login may be in progress)", llm.ErrServer, flow.config.Label, flow.config.Ports)
	}
	redirectURI := flow.config.RedirectURI(port)
	challenge := sha256.Sum256([]byte(verifier))
	authorizationURL, err := flow.config.AuthorizationURL(LoopbackAuthorization{
		RedirectURI: redirectURI,
		State:       state,
		Challenge:   base64.RawURLEncoding.EncodeToString(challenge[:]),
		Nonce:       nonce,
	})
	if err != nil {
		_ = listener.Close()
		flow.failStart()
		return llm.LoginAuthorization{}, err
	}
	launchURL := authorizationURL
	if flow.config.LaunchPath != "" {
		launchURL = "http://127.0.0.1:" + strconv.Itoa(port) + flow.config.LaunchPath + "?t=" + launchToken
	}
	authorization, err := llm.NewLoginAuthorization(launchURL, flow.config.Instructions)
	if err != nil {
		_ = listener.Close()
		flow.failStart()
		return llm.LoginAuthorization{}, err
	}

	server := &http.Server{
		ReadHeaderTimeout: loopbackReadHeaderTimeout,
		WriteTimeout:      loopbackReadHeaderTimeout,
		IdleTimeout:       loopbackReadHeaderTimeout,
		MaxHeaderBytes:    maxLoopbackRequestBytes,
	}
	server.Handler = flow.callbackHandler(authorizationURL, launchToken)

	flow.mu.Lock()
	if flow.status != loopbackStarting {
		flow.mu.Unlock()
		_ = listener.Close()
		return llm.LoginAuthorization{}, llm.ErrLoginClosed
	}
	flow.state = state
	flow.verifier = verifier
	flow.redirectURI = redirectURI
	flow.listener = listener
	flow.server = server
	flow.status = loopbackBegun
	flow.timer = time.AfterFunc(flow.config.FlowTimeout, flow.expire)
	flow.mu.Unlock()

	go func() { _ = server.Serve(listener) }()
	return authorization, nil
}

// Complete waits for the callback (or Submit), exchanges the code, and
// returns the credential without persisting it.
func (flow *LoopbackFlow) Complete(ctx context.Context) (llm.AuthCredential, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	flow.mu.Lock()
	if flow.status != loopbackBegun || flow.completeCalled {
		err := flow.statusErrorLocked()
		flow.mu.Unlock()
		return llm.AuthCredential{}, err
	}
	flow.completeCalled = true
	responses := flow.responses
	done := flow.done
	flow.mu.Unlock()

	var response loopbackResponse
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
	if flow.status != loopbackBegun {
		err := flow.statusErrorLocked()
		flow.mu.Unlock()
		return llm.AuthCredential{}, err
	}
	flow.status = loopbackExchanging
	verifier := flow.verifier
	redirectURI := flow.redirectURI
	exchangeCtx, cancel := context.WithTimeout(ctx, flow.config.ExchangeTimeout)
	flow.activeCancel = cancel
	flow.mu.Unlock()

	var credential llm.AuthCredential
	var exchangeErr error
	if response.denied {
		exchangeErr = fmt.Errorf("%w: %s login was denied", llm.ErrAuth, flow.config.Label)
	} else {
		credential, exchangeErr = flow.config.Exchange(exchangeCtx, response.code, verifier, redirectURI, response.values)
	}
	if flow.config.BeforeFinalize != nil {
		flow.config.BeforeFinalize()
	}
	cancel()

	flow.mu.Lock()
	flow.activeCancel = nil
	status := flow.status
	if status == loopbackExchanging {
		flow.finishLocked(loopbackCompleted)
	}
	flow.mu.Unlock()
	flow.closeResources()

	switch {
	case status == loopbackExpired:
		return llm.AuthCredential{}, llm.ErrLoginExpired
	case exchangeErr == nil && status == loopbackExchanging:
		// A minted credential (and its server-side refresh token) is
		// returned even if ctx ended after the exchange succeeded.
		return credential, nil
	case ctx.Err() != nil:
		return llm.AuthCredential{}, ctx.Err()
	case status == loopbackCancelled:
		return llm.AuthCredential{}, llm.ErrLoginClosed
	default:
		return credential, exchangeErr
	}
}

// Submit accepts a copied complete callback URL or query string when the
// browser cannot reach the loopback callback.
func (flow *LoopbackFlow) Submit(ctx context.Context, response string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	values, err := flow.parseSubmittedCallback(response)
	if err != nil {
		return err
	}
	if err := flow.deliver(values); err != nil {
		return err
	}
	go flow.closeResources()
	return nil
}

// Cancel ends the flow and releases the loopback listener.
func (flow *LoopbackFlow) Cancel() {
	flow.mu.Lock()
	switch flow.status {
	case loopbackCompleted, loopbackCancelled, loopbackExpired:
		flow.mu.Unlock()
		return
	default:
		flow.finishLocked(loopbackCancelled)
	}
	flow.mu.Unlock()
	flow.closeResources()
}

func (flow *LoopbackFlow) listenLoopback() (net.Listener, int, error) {
	var lastErr error
	for _, port := range flow.config.Ports {
		listener, err := flow.config.Listen("tcp4", "127.0.0.1:"+strconv.Itoa(port))
		if err == nil {
			return listener, port, nil
		}
		lastErr = err
	}
	return nil, 0, lastErr
}

// callbackHandler serves the callback and, once, the launch redirect. The
// redirect carries state, so it requires the launch URL's random token:
// another local process cannot fetch it to forge a callback.
func (flow *LoopbackFlow) callbackHandler(authorizationURL, launchToken string) http.Handler {
	var launched sync.Once
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Cache-Control", "no-store")
		writer.Header().Set("Referrer-Policy", "no-referrer")
		writer.Header().Set("Content-Type", "text/plain; charset=utf-8")
		if len(request.RequestURI) > maxLoopbackRequestBytes {
			writeLoopbackResponse(writer, http.StatusRequestURITooLong, "Login callback was too large.")
			return
		}
		if request.Method != http.MethodGet {
			writer.Header().Set("Allow", http.MethodGet)
			writeLoopbackResponse(writer, http.StatusMethodNotAllowed, "Only GET is supported.")
			return
		}
		if flow.config.LaunchPath != "" && request.URL.Path == flow.config.LaunchPath {
			served := false
			if constantTimeEqual(request.URL.Query().Get("t"), launchToken) {
				launched.Do(func() { served = true })
			}
			if !served {
				writeLoopbackResponse(writer, http.StatusGone, "Login link is invalid or already used.")
				return
			}
			http.Redirect(writer, request, authorizationURL, http.StatusFound)
			return
		}
		if request.URL.Path != flow.config.CallbackPath {
			writeLoopbackResponse(writer, http.StatusNotFound, "Not found.")
			return
		}
		values, err := url.ParseQuery(request.URL.RawQuery)
		if err != nil {
			writeLoopbackResponse(writer, http.StatusBadRequest, "Login callback was invalid.")
			return
		}
		if err := flow.deliver(values); err != nil {
			status := http.StatusBadRequest
			if errors.Is(err, llm.ErrLoginClosed) {
				status = http.StatusConflict
			}
			writeLoopbackResponse(writer, status, "Login callback was not accepted.")
			return
		}
		if values.Get("error") != "" {
			writeLoopbackResponse(writer, http.StatusUnauthorized, flow.config.Label+" sign-in was denied. You may close this window.")
		} else {
			message := flow.config.SuccessMessage
			if message == "" {
				message = flow.config.Label + " sign-in completed. You may close this window."
			}
			writeLoopbackResponse(writer, http.StatusOK, message)
		}
		go flow.closeResources()
	})
}

func writeLoopbackResponse(writer http.ResponseWriter, status int, message string) {
	if len(message) > maxLoopbackResponseBytes {
		message = "Login response unavailable."
	}
	writer.WriteHeader(status)
	_, _ = io.WriteString(writer, message)
}

func (flow *LoopbackFlow) parseSubmittedCallback(response string) (url.Values, error) {
	label := flow.config.Label
	if len(response) == 0 || len(response) > maxLoopbackRequestBytes {
		return nil, fmt.Errorf("%w: %s login callback has invalid size", llm.ErrBadRequest, label)
	}
	response = strings.TrimSpace(response)
	if response == "" {
		return nil, fmt.Errorf("%w: %s login callback is empty", llm.ErrBadRequest, label)
	}
	rawQuery := response
	if strings.HasPrefix(response, "http://") || strings.HasPrefix(response, "https://") {
		parsed, err := url.Parse(response)
		if err != nil || parsed.Path != flow.config.CallbackPath || parsed.RawQuery == "" || parsed.Fragment != "" || !flow.sameRedirectOrigin(parsed) {
			return nil, fmt.Errorf("%w: %s login callback URL is invalid", llm.ErrBadRequest, label)
		}
		rawQuery = parsed.RawQuery
	} else {
		rawQuery = strings.TrimPrefix(rawQuery, "?")
	}
	values, err := url.ParseQuery(rawQuery)
	if err != nil {
		return nil, fmt.Errorf("%w: %s login callback query is invalid", llm.ErrBadRequest, label)
	}
	return values, nil
}

// sameRedirectOrigin reports whether a pasted callback URL has the redirect
// URI's scheme and host, as pi checks.
func (flow *LoopbackFlow) sameRedirectOrigin(callback *url.URL) bool {
	flow.mu.Lock()
	redirectURI := flow.redirectURI
	flow.mu.Unlock()
	redirect, err := url.Parse(redirectURI)
	return err == nil && callback.Scheme == redirect.Scheme && strings.EqualFold(callback.Host, redirect.Host)
}

func (flow *LoopbackFlow) deliver(values url.Values) error {
	label := flow.config.Label
	flow.mu.Lock()
	if flow.status != loopbackBegun || flow.responseSet {
		err := flow.statusErrorLocked()
		flow.mu.Unlock()
		return err
	}
	expectedState := flow.state
	flow.mu.Unlock()

	stateValues := values["state"]
	if len(stateValues) != 1 || len(stateValues[0]) == 0 || len(stateValues[0]) > maxLoopbackStateBytes ||
		len(values["code"]) > 1 || len(values["error"]) > 1 || len(values["error_description"]) > 1 {
		return fmt.Errorf("%w: %s login callback fields are invalid", llm.ErrBadRequest, label)
	}
	state := stateValues[0]
	if !constantTimeEqual(state, expectedState) {
		return llm.ErrLoginStateMismatch
	}
	code := values.Get("code")
	providerError := values.Get("error")
	if len(code) > maxLoopbackCodeBytes || len(providerError) > maxLoopbackErrorBytes || len(values.Get("error_description")) > maxLoopbackErrorBytes {
		return fmt.Errorf("%w: %s login callback fields are too large", llm.ErrBadRequest, label)
	}
	if (code == "") == (providerError == "") {
		return fmt.Errorf("%w: %s login callback must contain one result", llm.ErrBadRequest, label)
	}

	flow.mu.Lock()
	defer flow.mu.Unlock()
	if flow.status != loopbackBegun || flow.responseSet {
		return flow.statusErrorLocked()
	}
	if !constantTimeEqual(state, flow.state) {
		return llm.ErrLoginStateMismatch
	}
	flow.responseSet = true
	flow.responses <- loopbackResponse{code: code, denied: providerError != "", values: values}
	return nil
}

func constantTimeEqual(left, right string) bool {
	return len(left) == len(right) && subtle.ConstantTimeCompare([]byte(left), []byte(right)) == 1
}

func randomBase64URL(reader io.Reader, size int) (string, error) {
	data := make([]byte, size)
	if _, err := io.ReadFull(reader, data); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(data), nil
}

func (flow *LoopbackFlow) expire() {
	flow.mu.Lock()
	if flow.status != loopbackBegun && flow.status != loopbackExchanging {
		flow.mu.Unlock()
		return
	}
	flow.finishLocked(loopbackExpired)
	flow.mu.Unlock()
	flow.closeResources()
}

func (flow *LoopbackFlow) failStart() {
	flow.mu.Lock()
	if flow.status == loopbackStarting {
		flow.finishLocked(loopbackCompleted)
	}
	flow.mu.Unlock()
}

func (flow *LoopbackFlow) finishLocked(status loopbackStatus) {
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

func (flow *LoopbackFlow) closeResources() {
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
		// Shutdown also closes idle keep-alive connections a browser opened
		// ahead of time, so a later login's callback cannot land on this
		// server and be rejected as a state mismatch.
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
		if err := server.Shutdown(shutdownCtx); err != nil {
			_ = server.Close()
		}
		cancel()
	}
}

func (flow *LoopbackFlow) statusErrorLocked() error {
	if flow.status == loopbackExpired {
		return llm.ErrLoginExpired
	}
	return llm.ErrLoginClosed
}
