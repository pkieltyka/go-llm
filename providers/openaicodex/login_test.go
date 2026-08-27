package openaicodex

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/provideroauth"
)

func TestCodexLoginAuthorizationPKCEAndExchangeShape(t *testing.T) {
	type tokenRequest struct {
		form       url.Values
		originator string
	}
	requests := make(chan tokenRequest, 1)
	tokenServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if err := request.ParseForm(); err != nil {
			t.Errorf("ParseForm: %v", err)
		}
		requests <- tokenRequest{form: request.PostForm, originator: request.Header.Get(originatorHeader)}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id_token":"`+fakeCodexJWT(t, "account-from-id")+`","access_token":"access-secret","refresh_token":"refresh-secret","expires_in":3600}`)
	}))
	defer tokenServer.Close()

	flow := newLoginTestFlow(t, tokenServer.Client(), tokenServer.URL, time.Minute, time.Second)
	authorization, providerURL := beginLoginTestFlow(t, flow)
	if strings.Contains(authorization.URL(), "state=") || strings.Contains(authorization.URL(), "code_challenge=") {
		t.Fatal("loopback launch URL exposed OAuth state")
	}
	query := providerURL.Query()
	if got := providerURL.Scheme + "://" + providerURL.Host + providerURL.Path; got != openAICodexOAuthAuthorizeURL {
		t.Fatalf("authorization endpoint = %q", got)
	}
	for key, want := range map[string]string{
		"response_type":              "code",
		"client_id":                  openAICodexOAuthClientID,
		"scope":                      openAICodexOAuthScopes,
		"code_challenge_method":      "S256",
		"id_token_add_organizations": "true",
		"codex_cli_simplified_flow":  "true",
		"originator":                 defaultOriginator,
	} {
		if got := query.Get(key); got != want {
			t.Fatalf("authorization %s = %q, want %q", key, got, want)
		}
	}
	if query.Get("state") == "" || query.Get("code_challenge") == "" {
		t.Fatal("authorization state or PKCE challenge is empty")
	}
	if query.Get("state") == query.Get("code_challenge") {
		t.Fatal("CSRF state was reused as the PKCE challenge")
	}

	result := completeLoginAsync(flow)
	callback := query.Get("redirect_uri") + "?code=authorization-secret&state=" + url.QueryEscape(query.Get("state"))
	response, err := http.Get(callback)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK || response.Header.Get("Cache-Control") != "no-store" || response.Header.Get("Referrer-Policy") != "no-referrer" {
		t.Fatalf("callback response = %d, cache=%q, referrer=%q", response.StatusCode, response.Header.Get("Cache-Control"), response.Header.Get("Referrer-Policy"))
	}
	completed := <-result
	if completed.err != nil {
		t.Fatal(completed.err)
	}
	if completed.credential.Type != "oauth" || completed.credential.Access != "access-secret" || completed.credential.Refresh != "refresh-secret" || completed.credential.AccountID != "account-from-id" || completed.credential.Expires <= time.Now().UnixMilli() {
		t.Fatalf("credential = %#v", completed.credential)
	}
	assertLoopbackClosed(t, authorization.URL())

	token := <-requests
	for key, want := range map[string]string{
		"grant_type":   "authorization_code",
		"code":         "authorization-secret",
		"redirect_uri": query.Get("redirect_uri"),
		"client_id":    openAICodexOAuthClientID,
	} {
		if got := token.form.Get(key); got != want {
			t.Fatalf("token form %s = %q, want %q", key, got, want)
		}
	}
	if token.originator != defaultOriginator {
		t.Fatalf("token originator = %q", token.originator)
	}
	verifier := token.form.Get("code_verifier")
	challenge := sha256.Sum256([]byte(verifier))
	if verifier == "" || verifier == query.Get("state") || base64.RawURLEncoding.EncodeToString(challenge[:]) != query.Get("code_challenge") {
		t.Fatal("token PKCE verifier did not match the independent authorization challenge")
	}
	if _, err := flow.Complete(context.Background()); !errors.Is(err, llm.ErrLoginClosed) {
		t.Fatalf("second Complete = %v, want ErrLoginClosed", err)
	}
}

func TestCodexLoginDefaultCallbackPortsAndPath(t *testing.T) {
	config := defaultLoginConfig()
	if config.callbackPorts != [2]int{1455, 1457} {
		t.Fatalf("callback ports = %v", config.callbackPorts)
	}
	if got := codexRedirectURI(1455); got != "http://localhost:1455/auth/callback" {
		t.Fatalf("preferred redirect URI = %q", got)
	}
	if got := codexRedirectURI(1457); got != "http://localhost:1457/auth/callback" {
		t.Fatalf("fallback redirect URI = %q", got)
	}
}

func TestCodexLoginPortCollisionFallbackAndFailure(t *testing.T) {
	ports := loginTestPorts(t)
	preferred := listenOnTestPort(t, ports[0])
	defer preferred.Close()

	flow := newLoginTestFlowWithPorts(t, http.DefaultClient, "https://example.test/token", time.Minute, time.Second, ports)
	authorization, providerURL := beginLoginTestFlow(t, flow)
	if !strings.Contains(authorization.URL(), ":"+strconv.Itoa(ports[1])+codexLoginLaunchPath) {
		t.Fatalf("fallback launch URL = %q", authorization.URL())
	}
	if got := providerURL.Query().Get("redirect_uri"); got != codexRedirectURI(ports[1]) {
		t.Fatalf("fallback redirect URI = %q", got)
	}
	flow.Cancel()

	fallback := listenOnTestPort(t, ports[1])
	defer fallback.Close()
	flow = newLoginTestFlowWithPorts(t, http.DefaultClient, "https://example.test/token", time.Minute, time.Second, ports)
	if _, err := flow.Begin(context.Background()); !errors.Is(err, llm.ErrServer) {
		t.Fatalf("both ports unavailable error = %v, want ErrServer", err)
	}
}

func TestCodexLoginWrongStateAndStrayRequestsAreNonTerminal(t *testing.T) {
	tokenServer := successfulLoginTokenServer(t, fakeCodexJWT(t, "account"), fakeCodexJWT(t, "access-account"))
	defer tokenServer.Close()
	flow := newLoginTestFlow(t, tokenServer.Client(), tokenServer.URL, time.Minute, time.Second)
	authorization, providerURL := beginLoginTestFlow(t, flow)
	state := providerURL.Query().Get("state")

	client := &http.Client{}
	for _, request := range []struct {
		method string
		path   string
		want   int
	}{
		{method: http.MethodPost, path: codexLoginCallbackPath, want: http.StatusMethodNotAllowed},
		{method: http.MethodGet, path: "/stray", want: http.StatusNotFound},
		{method: http.MethodGet, path: codexLoginCallbackPath + "?code=secret&state=wrong-state", want: http.StatusBadRequest},
		{method: http.MethodGet, path: codexLoginCallbackPath + "?state=" + url.QueryEscape(state), want: http.StatusBadRequest},
	} {
		req, err := http.NewRequest(request.method, loopbackOrigin(authorization.URL())+request.path, nil)
		if err != nil {
			t.Fatal(err)
		}
		response, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		response.Body.Close()
		if response.StatusCode != request.want {
			t.Fatalf("%s %s status = %d, want %d", request.method, request.path, response.StatusCode, request.want)
		}
	}

	result := completeLoginAsync(flow)
	response, err := http.Get(loopbackOrigin(authorization.URL()) + codexLoginCallbackPath + "?code=valid-secret&state=" + url.QueryEscape(state))
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if completed := <-result; completed.err != nil {
		t.Fatalf("valid callback after stray requests = %v", completed.err)
	}
}

func TestCodexLoginDuplicateCallbackIsRejected(t *testing.T) {
	tokenServer := successfulLoginTokenServer(t, fakeCodexJWT(t, "account"), "access")
	defer tokenServer.Close()
	flow := newLoginTestFlow(t, tokenServer.Client(), tokenServer.URL, time.Minute, time.Second)
	_, providerURL := beginLoginTestFlow(t, flow)
	state := url.QueryEscape(providerURL.Query().Get("state"))
	handler := flow.callbackHandler(providerURL.String())
	result := completeLoginAsync(flow)

	first := httptest.NewRecorder()
	handler.ServeHTTP(first, httptest.NewRequest(http.MethodGet, codexLoginCallbackPath+"?code=first-secret&state="+state, nil))
	if first.Code != http.StatusOK {
		t.Fatalf("first callback status = %d, want %d", first.Code, http.StatusOK)
	}
	duplicate := httptest.NewRecorder()
	handler.ServeHTTP(duplicate, httptest.NewRequest(http.MethodGet, codexLoginCallbackPath+"?code=second-secret&state="+state, nil))
	if duplicate.Code != http.StatusConflict {
		t.Fatalf("duplicate callback status = %d, want %d", duplicate.Code, http.StatusConflict)
	}
	if completed := <-result; completed.err != nil {
		t.Fatal(completed.err)
	}
}

func TestCodexLoginTerminalCleanupDrainsQueuedCode(t *testing.T) {
	tests := []struct {
		name      string
		terminate func(*loginFlow)
		wantErr   error
	}{
		{name: "cancelled", terminate: func(flow *loginFlow) { flow.Cancel() }, wantErr: llm.ErrLoginClosed},
		{name: "expired", terminate: func(flow *loginFlow) { flow.expire() }, wantErr: llm.ErrLoginExpired},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			flow := newLoginTestFlow(t, http.DefaultClient, "https://example.test/token", time.Minute, time.Second)
			_, providerURL := beginLoginTestFlow(t, flow)
			if err := flow.Submit(context.Background(), validSubmit(providerURL)); err != nil {
				t.Fatal(err)
			}
			if len(flow.responses) != 1 {
				t.Fatal("authorization code was not queued before terminal cleanup")
			}

			test.terminate(flow)
			select {
			case <-flow.responses:
				t.Fatal("terminal cleanup retained a queued authorization code")
			default:
			}
			if _, err := flow.Complete(context.Background()); !errors.Is(err, test.wantErr) {
				t.Fatalf("Complete error = %v, want %v", err, test.wantErr)
			}
		})
	}
}

func TestCodexLoginSubmitFallbackLimitsDenialAndDuplicates(t *testing.T) {
	t.Run("fallback query", func(t *testing.T) {
		tokenServer := successfulLoginTokenServer(t, fakeCodexJWT(t, "account"), "access")
		defer tokenServer.Close()
		flow := newLoginTestFlow(t, tokenServer.Client(), tokenServer.URL, time.Minute, time.Second)
		_, providerURL := beginLoginTestFlow(t, flow)
		result := completeLoginAsync(flow)
		query := "?code=manual-secret&state=" + url.QueryEscape(providerURL.Query().Get("state"))
		if err := flow.Submit(context.Background(), query); err != nil {
			t.Fatal(err)
		}
		if err := flow.Submit(context.Background(), query); !errors.Is(err, llm.ErrLoginClosed) {
			t.Fatalf("duplicate Submit = %v, want ErrLoginClosed", err)
		}
		if completed := <-result; completed.err != nil {
			t.Fatal(completed.err)
		}
	})

	t.Run("complete callback URL", func(t *testing.T) {
		tokenServer := successfulLoginTokenServer(t, fakeCodexJWT(t, "account"), "access")
		defer tokenServer.Close()
		flow := newLoginTestFlow(t, tokenServer.Client(), tokenServer.URL, time.Minute, time.Second)
		_, providerURL := beginLoginTestFlow(t, flow)
		result := completeLoginAsync(flow)
		callback := providerURL.Query().Get("redirect_uri") + "?code=manual-secret&state=" + url.QueryEscape(providerURL.Query().Get("state"))
		if err := flow.Submit(context.Background(), callback); err != nil {
			t.Fatal(err)
		}
		if completed := <-result; completed.err != nil {
			t.Fatal(completed.err)
		}
	})

	t.Run("wrong state and input limits remain non-terminal", func(t *testing.T) {
		flow := newLoginTestFlow(t, http.DefaultClient, "https://example.test/token", time.Minute, time.Second)
		_, providerURL := beginLoginTestFlow(t, flow)
		state := providerURL.Query().Get("state")
		if err := flow.Submit(context.Background(), "?code=secret&state=wrong-secret"); !errors.Is(err, llm.ErrLoginStateMismatch) || strings.Contains(err.Error(), "wrong-secret") {
			t.Fatalf("wrong state Submit = %v", err)
		}
		for _, input := range []string{
			strings.Repeat("x", maxLoginRequestBytes+1),
			"?code=" + strings.Repeat("c", maxLoginCodeBytes+1) + "&state=" + state,
			"?code=secret&state=" + strings.Repeat("s", maxLoginStateBytes+1),
			"http://localhost:1455/wrong?code=secret&state=" + state,
		} {
			if err := flow.Submit(context.Background(), input); !errors.Is(err, llm.ErrBadRequest) {
				t.Fatalf("invalid Submit = %v, want ErrBadRequest", err)
			}
		}
		flow.Cancel()
	})

	t.Run("provider denial", func(t *testing.T) {
		flow := newLoginTestFlow(t, http.DefaultClient, "https://example.test/token", time.Minute, time.Second)
		_, providerURL := beginLoginTestFlow(t, flow)
		result := completeLoginAsync(flow)
		if err := flow.Submit(context.Background(), "?error=access_denied&error_description=provider-secret&state="+url.QueryEscape(providerURL.Query().Get("state"))); err != nil {
			t.Fatal(err)
		}
		completed := <-result
		if !errors.Is(completed.err, llm.ErrAuth) || strings.Contains(completed.err.Error(), "provider-secret") || strings.Contains(completed.err.Error(), "access_denied") {
			t.Fatalf("provider denial = %v", completed.err)
		}
	})
}

func TestCodexLoginDenialPreservesConcurrentTerminalState(t *testing.T) {
	tests := []struct {
		name    string
		action  func(*loginFlow, context.CancelFunc)
		wantErr error
	}{
		{name: "flow cancellation", action: func(flow *loginFlow, _ context.CancelFunc) { flow.Cancel() }, wantErr: llm.ErrLoginClosed},
		{name: "total expiry", action: func(flow *loginFlow, _ context.CancelFunc) { flow.expire() }, wantErr: llm.ErrLoginExpired},
		{name: "caller cancellation", action: func(_ *loginFlow, cancel context.CancelFunc) { cancel() }, wantErr: context.Canceled},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			entered := make(chan struct{})
			release := make(chan struct{})
			var releaseOnce sync.Once
			releaseFlow := func() { releaseOnce.Do(func() { close(release) }) }
			defer releaseFlow()
			flow := newLoginTestFlow(t, http.DefaultClient, "https://example.test/token", time.Minute, time.Second)
			flow.config.beforeFinalize = func() {
				close(entered)
				<-release
			}
			_, providerURL := beginLoginTestFlow(t, flow)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			result := completeLoginAsyncContext(flow, ctx)
			denial := "?error=access_denied&state=" + url.QueryEscape(providerURL.Query().Get("state"))
			if err := flow.Submit(context.Background(), denial); err != nil {
				t.Fatal(err)
			}
			select {
			case <-entered:
			case <-time.After(time.Second):
				t.Fatal("Complete did not reach denial finalization")
			}
			test.action(flow, cancel)
			releaseFlow()
			if completed := <-result; !errors.Is(completed.err, test.wantErr) {
				t.Fatalf("Complete error = %v, want %v", completed.err, test.wantErr)
			}
		})
	}
}

func TestCodexLoginCancellationExpiryAndListenerShutdown(t *testing.T) {
	t.Run("callback wait cancellation", func(t *testing.T) {
		flow := newLoginTestFlow(t, http.DefaultClient, "https://example.test/token", time.Minute, time.Second)
		authorization, _ := beginLoginTestFlow(t, flow)
		ctx, cancel := context.WithCancel(context.Background())
		result := make(chan error, 1)
		go func() {
			_, err := flow.Complete(ctx)
			result <- err
		}()
		cancel()
		if err := <-result; !errors.Is(err, context.Canceled) {
			t.Fatalf("Complete cancellation = %v", err)
		}
		assertLoopbackClosed(t, authorization.URL())
	})

	t.Run("total expiry", func(t *testing.T) {
		flow := newLoginTestFlow(t, http.DefaultClient, "https://example.test/token", 10*time.Millisecond, time.Second)
		authorization, _ := beginLoginTestFlow(t, flow)
		_, err := flow.Complete(context.Background())
		if !errors.Is(err, llm.ErrLoginExpired) {
			t.Fatalf("Complete expiry = %v", err)
		}
		assertLoopbackClosed(t, authorization.URL())
	})

	t.Run("active exchange cancellation", func(t *testing.T) {
		started := make(chan struct{})
		release := make(chan struct{})
		tokenServer := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
			close(started)
			<-release
		}))
		defer tokenServer.Close()
		defer tokenServer.CloseClientConnections()
		defer close(release)
		flow := newLoginTestFlow(t, tokenServer.Client(), tokenServer.URL, time.Minute, time.Second)
		_, providerURL := beginLoginTestFlow(t, flow)
		ctx, cancel := context.WithCancel(context.Background())
		result := completeLoginAsyncContext(flow, ctx)
		if err := flow.Submit(context.Background(), validSubmit(providerURL)); err != nil {
			t.Fatal(err)
		}
		<-started
		cancel()
		if completed := <-result; !errors.Is(completed.err, context.Canceled) {
			t.Fatalf("active exchange cancellation = %v", completed.err)
		}
	})
}

func TestCodexLoginExchangeTimeoutRedirectAndRedaction(t *testing.T) {
	t.Run("exchange timeout", func(t *testing.T) {
		started := make(chan struct{})
		release := make(chan struct{})
		tokenServer := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
			close(started)
			<-release
		}))
		defer tokenServer.Close()
		defer tokenServer.CloseClientConnections()
		defer close(release)
		flow := newLoginTestFlow(t, tokenServer.Client(), tokenServer.URL, time.Minute, 20*time.Millisecond)
		_, providerURL := beginLoginTestFlow(t, flow)
		result := completeLoginAsync(flow)
		if err := flow.Submit(context.Background(), validSubmit(providerURL)); err != nil {
			t.Fatal(err)
		}
		<-started
		if completed := <-result; !errors.Is(completed.err, llm.ErrTimeout) {
			t.Fatalf("exchange timeout = %v", completed.err)
		}
	})

	t.Run("redirect refused", func(t *testing.T) {
		var trapHits atomic.Int32
		trap := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { trapHits.Add(1) }))
		defer trap.Close()
		redirector := httptest.NewServer(http.RedirectHandler(trap.URL, http.StatusTemporaryRedirect))
		defer redirector.Close()
		flow := newLoginTestFlow(t, redirector.Client(), redirector.URL, time.Minute, time.Second)
		_, providerURL := beginLoginTestFlow(t, flow)
		result := completeLoginAsync(flow)
		if err := flow.Submit(context.Background(), validSubmit(providerURL)); err != nil {
			t.Fatal(err)
		}
		completed := <-result
		if !errors.Is(completed.err, provideroauth.ErrUnsafeRedirect) || trapHits.Load() != 0 {
			t.Fatalf("redirect result = %v, trap hits=%d", completed.err, trapHits.Load())
		}
	})

	t.Run("provider body redacted", func(t *testing.T) {
		tokenServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(writer, `{"error":"authorization-secret provider-secret"}`)
		}))
		defer tokenServer.Close()
		flow := newLoginTestFlow(t, tokenServer.Client(), tokenServer.URL, time.Minute, time.Second)
		_, providerURL := beginLoginTestFlow(t, flow)
		result := completeLoginAsync(flow)
		if err := flow.Submit(context.Background(), validSubmit(providerURL)); err != nil {
			t.Fatal(err)
		}
		completed := <-result
		if !errors.Is(completed.err, llm.ErrAuth) || strings.Contains(completed.err.Error(), "authorization-secret") || strings.Contains(completed.err.Error(), "provider-secret") {
			t.Fatalf("provider error leaked body: %v", completed.err)
		}
	})

	t.Run("custom transport redacted", func(t *testing.T) {
		const leaked = "authorization-secret"
		client := &http.Client{Transport: loginRoundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, errors.New("transport echoed " + leaked)
		})}
		flow := newLoginTestFlow(t, client, "https://example.test/token", time.Minute, time.Second)
		_, providerURL := beginLoginTestFlow(t, flow)
		result := completeLoginAsync(flow)
		if err := flow.Submit(context.Background(), validSubmit(providerURL)); err != nil {
			t.Fatal(err)
		}
		completed := <-result
		if !errors.Is(completed.err, llm.ErrServer) || strings.Contains(completed.err.Error(), leaked) {
			t.Fatalf("transport error = %v", completed.err)
		}
	})
}

func TestCodexLoginStrictTokenResponseAndAccountFallback(t *testing.T) {
	tests := []struct {
		name        string
		body        func() string
		wantAccount string
		wantErr     error
	}{
		{name: "id token account", body: func() string { return `{"id_token":"` + fakeCodexJWT(t, "id-account") + `","access_token":"opaque"}` }, wantAccount: "id-account"},
		{name: "access token fallback", body: func() string {
			return `{"id_token":"malformed","access_token":"` + fakeCodexJWT(t, "access-account") + `"}`
		}, wantAccount: "access-account"},
		{name: "missing account", body: func() string { return `{"access_token":"opaque"}` }, wantErr: llm.ErrAuth},
		{name: "malformed", body: func() string { return `{"access_token":` }, wantErr: llm.ErrAuth},
		{name: "trailing", body: func() string { return `{"access_token":"opaque"}{}` }, wantErr: llm.ErrAuth},
		{name: "oversized", body: func() string { return strings.Repeat("x", maxCodexTokenBodyBytes+1) }, wantErr: llm.ErrAuth},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			tokenServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(writer, test.body())
			}))
			defer tokenServer.Close()
			flow := newLoginTestFlow(t, tokenServer.Client(), tokenServer.URL, time.Minute, time.Second)
			_, providerURL := beginLoginTestFlow(t, flow)
			result := completeLoginAsync(flow)
			if err := flow.Submit(context.Background(), validSubmit(providerURL)); err != nil {
				t.Fatal(err)
			}
			completed := <-result
			if test.wantErr != nil {
				if !errors.Is(completed.err, test.wantErr) {
					t.Fatalf("error = %v, want %v", completed.err, test.wantErr)
				}
				return
			}
			if completed.err != nil || completed.credential.AccountID != test.wantAccount {
				t.Fatalf("credential/error = %#v/%v", completed.credential, completed.err)
			}
		})
	}
}

func TestCodexLoginFormattingRedactsEphemeralState(t *testing.T) {
	flow := newLoginTestFlow(t, http.DefaultClient, "https://example.test/token", time.Minute, time.Second)
	_, providerURL := beginLoginTestFlow(t, flow)
	state := providerURL.Query().Get("state")
	for _, formatted := range []string{fmt.Sprint(flow), fmt.Sprintf("%+v", flow), fmt.Sprintf("%#v", flow)} {
		if strings.Contains(formatted, state) || strings.Contains(formatted, "verifier") {
			t.Fatalf("formatted flow leaked ephemeral data: %s", formatted)
		}
	}
	var out bytes.Buffer
	slog.New(slog.NewJSONHandler(&out, nil)).Info("flow", "login", flow)
	if strings.Contains(out.String(), state) {
		t.Fatalf("structured log leaked state: %s", out.String())
	}
	flow.Cancel()
}

type loginCompletion struct {
	credential llm.AuthCredential
	err        error
}

type loginRoundTripFunc func(*http.Request) (*http.Response, error)

func (roundTrip loginRoundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return roundTrip(request)
}

func newLoginTestFlow(t *testing.T, client *http.Client, tokenURL string, timeout, exchangeTimeout time.Duration) *loginFlow {
	t.Helper()
	return newLoginTestFlowWithPorts(t, client, tokenURL, timeout, exchangeTimeout, loginTestPorts(t))
}

func newLoginTestFlowWithPorts(t *testing.T, client *http.Client, tokenURL string, timeout, exchangeTimeout time.Duration, ports [2]int) *loginFlow {
	t.Helper()
	random := make([]byte, 64)
	for index := range random {
		random[index] = byte(index + 1)
	}
	flow, err := newLoginFlow(loginConfig{
		httpClient:      client,
		flowTimeout:     timeout,
		exchangeTimeout: exchangeTimeout,
		authorizeURL:    openAICodexOAuthAuthorizeURL,
		tokenURL:        tokenURL,
		random:          bytes.NewReader(random),
		listen:          net.Listen,
		callbackPorts:   ports,
	})
	if err != nil {
		t.Fatal(err)
	}
	return flow
}

func beginLoginTestFlow(t *testing.T, flow *loginFlow) (llm.LoginAuthorization, *url.URL) {
	t.Helper()
	authorization, err := flow.Begin(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	response, err := client.Get(authorization.URL())
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusFound {
		t.Fatalf("launch status = %d", response.StatusCode)
	}
	providerURL, err := url.Parse(response.Header.Get("Location"))
	if err != nil {
		t.Fatal(err)
	}
	return authorization, providerURL
}

func completeLoginAsync(flow *loginFlow) <-chan loginCompletion {
	return completeLoginAsyncContext(flow, context.Background())
}

func completeLoginAsyncContext(flow *loginFlow, ctx context.Context) <-chan loginCompletion {
	result := make(chan loginCompletion, 1)
	go func() {
		credential, err := flow.Complete(ctx)
		result <- loginCompletion{credential: credential, err: err}
	}()
	return result
}

func validSubmit(providerURL *url.URL) string {
	return "?code=authorization-secret&state=" + url.QueryEscape(providerURL.Query().Get("state"))
}

func successfulLoginTokenServer(t *testing.T, idToken, accessToken string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(writer, `{"id_token":"`+idToken+`","access_token":"`+accessToken+`","refresh_token":"refresh"}`)
	}))
}

func loginTestPorts(t *testing.T) [2]int {
	t.Helper()
	first := freeLoginTestPort(t)
	second := freeLoginTestPort(t)
	for second == first {
		second = freeLoginTestPort(t)
	}
	return [2]int{first, second}
}

func freeLoginTestPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	return port
}

func listenOnTestPort(t *testing.T, port int) net.Listener {
	t.Helper()
	listener, err := net.Listen("tcp4", "127.0.0.1:"+strconv.Itoa(port))
	if err != nil {
		t.Fatal(err)
	}
	return listener
}

func loopbackOrigin(launchURL string) string {
	parsed, _ := url.Parse(launchURL)
	return parsed.Scheme + "://" + parsed.Host
}

func assertLoopbackClosed(t *testing.T, launchURL string) {
	t.Helper()
	parsed, err := url.Parse(launchURL)
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for {
		connection, err := net.DialTimeout("tcp4", parsed.Host, 10*time.Millisecond)
		if err != nil {
			return
		}
		connection.Close()
		if time.Now().After(deadline) {
			t.Fatalf("loopback listener %s remained open", parsed.Host)
		}
		time.Sleep(time.Millisecond)
	}
}
