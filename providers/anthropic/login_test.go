package anthropic

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

func TestAnthropicLoginMethods(t *testing.T) {
	for _, method := range []LoginMethod{LoginBrowser, LoginCopyCode} {
		t.Run(string(method), func(t *testing.T) {
			var exchange map[string]string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&exchange); err != nil {
					t.Error(err)
				}
				_, _ = io.WriteString(w, `{"access_token":"login-access","refresh_token":"login-refresh","expires_in":3600}`)
			}))
			defer server.Close()
			cfg := defaultLoginConfig()
			cfg.method = method
			cfg.ports = []int{0}
			cfg.tokenURL = server.URL
			cfg.httpClient = server.Client()
			flow, err := newLoginFlow(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer flow.Cancel()
			authorization, err := flow.Begin(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			launch, err := url.Parse(authorization.URL())
			if err != nil {
				t.Fatal(err)
			}
			if launch.Hostname() != "127.0.0.1" || launch.Port() == "0" || strings.Contains(authorization.URL(), "client_id") || !strings.Contains(authorization.Instructions(), "forwarding") {
				t.Fatal("unsafe launch target or missing remote limitation")
			}
			client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
			response, err := client.Get(authorization.URL())
			if err != nil {
				t.Fatal(err)
			}
			response.Body.Close()
			target, err := url.Parse(response.Header.Get("Location"))
			if err != nil {
				t.Fatal(err)
			}
			q := target.Query()
			if target.Scheme != "https" || target.Host != "claude.ai" || q.Get("client_id") != anthropicOAuthClientID || q.Get("scope") != anthropicOAuthScope || q.Get("code") != "true" || q.Get("response_type") != "code" || q.Get("code_challenge_method") != "S256" {
				t.Fatal("authorization parameters mismatch")
			}
			hashed := sha256.Sum256([]byte(q.Get("state")))
			if q.Get("code_challenge") != base64.RawURLEncoding.EncodeToString(hashed[:]) {
				t.Fatal("Pi verifier/state contract mismatch")
			}
			redirect, err := url.Parse(q.Get("redirect_uri"))
			if err != nil {
				t.Fatal(err)
			}
			if method == LoginCopyCode {
				if redirect.String() != "https://platform.claude.com/oauth/code/callback" {
					t.Fatal("wrong headless redirect")
				}
			} else if redirect.Hostname() != "localhost" || redirect.Port() != launch.Port() || redirect.Path != "/callback" {
				t.Fatal("wrong browser redirect")
			}
			if err := flow.Submit(context.Background(), "code#wrong"); !errors.Is(err, llm.ErrLoginStateMismatch) {
				t.Fatal("wrong state accepted")
			}
			input := "test-code#" + q.Get("state")
			if method == LoginBrowser {
				redirect.RawQuery = url.Values{"code": {"test-code"}, "state": {q.Get("state")}}.Encode()
				input = redirect.String()
			}
			if err := flow.Submit(context.Background(), input); err != nil {
				t.Fatal(err)
			}
			got, err := flow.Complete(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if got.Access != "login-access" || got.Refresh != "login-refresh" || got.ClientID != "" {
				t.Fatal("credential mismatch")
			}
			for k, v := range map[string]string{"grant_type": "authorization_code", "client_id": anthropicOAuthClientID, "code": "test-code", "state": q.Get("state"), "code_verifier": q.Get("state"), "redirect_uri": q.Get("redirect_uri")} {
				if exchange[k] != v {
					t.Errorf("exchange field %s mismatch", k)
				}
			}
			if _, err := flow.Complete(context.Background()); !errors.Is(err, llm.ErrLoginClosed) {
				t.Fatal("flow reused")
			}
		})
	}
}

func TestAnthropicLoginPortFallbackAndFailure(t *testing.T) {
	cfg := defaultLoginConfig()
	calls := 0
	cfg.listen = func(network, address string) (net.Listener, error) {
		calls++
		if calls == 1 {
			return nil, errors.New("synthetic occupied")
		}
		return net.Listen(network, address)
	}
	flow, err := newLoginFlow(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer flow.Cancel()
	auth, err := flow.Begin(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	parsed, _ := url.Parse(auth.URL())
	if calls != 2 || parsed.Port() == "0" {
		t.Fatal("ephemeral fallback not bound")
	}
	cfg.listen = func(string, string) (net.Listener, error) { return nil, errors.New("marker-secret") }
	failed, err := newLoginFlow(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := failed.Begin(context.Background()); !errors.Is(err, llm.ErrServer) || strings.Contains(err.Error(), "marker-secret") {
		t.Fatal("unsafe listener failure")
	}
}

func TestAnthropicLoginOptionsAndCancellation(t *testing.T) {
	for _, options := range [][]LoginOption{{WithLoginMethod("bad")}, {WithLoginHTTPClient(nil)}, {WithLoginTimeout(0)}} {
		if _, err := NewLoginFlow(options...); !errors.Is(err, llm.ErrBadRequest) {
			t.Fatalf("invalid config=%v", err)
		}
	}
	cfg := defaultLoginConfig()
	cfg.ports = []int{0}
	cfg.flowTimeout = time.Second
	flow, err := newLoginFlow(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := flow.Begin(context.Background()); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := flow.Complete(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled complete=%v", err)
	}
	flow.Cancel()
	flow.Cancel()
}
