package openai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
)

type fakeOpenAI struct {
	mu        sync.Mutex
	auths     []string
	reject    map[string]bool // access tokens answered with 401
	refreshes []url.Values
	scope     string
	server    *httptest.Server
}

func newFakeOpenAI(t *testing.T) *fakeOpenAI {
	f := &fakeOpenAI{reject: map[string]bool{}, scope: chatGPTScope}
	f.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		defer f.mu.Unlock()
		if r.URL.Path == "/oauth/token" {
			_ = r.ParseForm()
			f.refreshes = append(f.refreshes, r.PostForm)
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"access_token":"access-%d","refresh_token":"refresh-%d","expires_in":3600,"scope":%q,"id_token":"id"}`, len(f.refreshes), len(f.refreshes), f.scope)
			return
		}
		auth := r.Header.Get("Authorization")
		f.auths = append(f.auths, auth)
		if f.reject[strings.TrimPrefix(auth, "Bearer ")] {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = io.WriteString(w, `{"error":{"message":"expired","type":"invalid_request_error"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[]}`)
	}))
	t.Cleanup(f.server.Close)
	return f
}

func (f *fakeOpenAI) provider(t *testing.T, store llm.CredentialStore) *Provider {
	t.Helper()
	p, err := New(WithCredentials(store), WithBaseURL(f.server.URL+"/v1"), WithMaxRetries(0), withOAuthTokenURL(f.server.URL+"/oauth/token"))
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func oauthStore(access string, expires time.Time) *authstore.Memory {
	return authstore.NewMemory(llm.AuthFile{"openai": {Type: "oauth", Access: access, Refresh: "refresh-0", ClientID: "issued-client", Expires: expires.UnixMilli()}})
}

func TestWithCredentialsUsesStoredOAuthToken(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	f := newFakeOpenAI(t)
	if _, err := f.provider(t, oauthStore("valid", time.Now().Add(time.Hour))).Models(context.Background()); err != nil {
		t.Fatal(err)
	}
	if f.auths[0] != "Bearer valid" || len(f.refreshes) != 0 {
		t.Fatalf("auths=%v refreshes=%d", f.auths, len(f.refreshes))
	}
}

func TestWithCredentialsRefreshesExpiringTokenAndStoresIt(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	f := newFakeOpenAI(t)
	store := oauthStore("stale", time.Now().Add(time.Minute))
	if _, err := f.provider(t, store).Models(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(f.refreshes) != 1 || f.auths[0] != "Bearer access-1" {
		t.Fatalf("refreshes=%d auths=%v", len(f.refreshes), f.auths)
	}
	form := f.refreshes[0]
	for key, want := range map[string]string{"grant_type": "refresh_token", "client_id": "issued-client", "refresh_token": "refresh-0", "resource": chatGPTResource} {
		if got := form.Get(key); got != want {
			t.Errorf("refresh form %s = %q, want %q", key, got, want)
		}
	}
	stored, _, _ := store.Read(context.Background(), "openai")
	if stored.Access != "access-1" || stored.Refresh != "refresh-1" || stored.ClientID != "issued-client" {
		t.Fatalf("stored = %+v, want the renewed credential", stored)
	}
}

func TestWithCredentialsRefreshesOnceAfter401(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	f := newFakeOpenAI(t)
	f.reject["revoked"] = true
	if _, err := f.provider(t, oauthStore("revoked", time.Now().Add(time.Hour))).Models(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(f.refreshes) != 1 || len(f.auths) != 2 || f.auths[1] != "Bearer access-1" {
		t.Fatalf("refreshes=%d auths=%v, want one refresh and a retry with the new token", len(f.refreshes), f.auths)
	}
}

func TestWithCredentialsAPIKeyAndAmbientFallback(t *testing.T) {
	f := newFakeOpenAI(t)
	t.Setenv("OPENAI_API_KEY", "env-key")
	stored := authstore.NewMemory(llm.AuthFile{"openai": {Type: "api_key", Key: "stored-key"}})
	if _, err := f.provider(t, stored).Models(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := f.provider(t, authstore.NewMemory(nil)).Models(context.Background()); err != nil {
		t.Fatal(err)
	}
	if f.auths[0] != "Bearer stored-key" || f.auths[1] != "Bearer env-key" {
		t.Fatalf("auths = %v, want the stored key, then the environment fallback", f.auths)
	}
	t.Setenv("OPENAI_API_KEY", "")
	if _, err := f.provider(t, authstore.NewMemory(nil)).Models(context.Background()); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("nothing stored, no env = %v, want ErrAuth", err)
	}
}

func TestWithCredentialsIsExclusiveWithAPIKey(t *testing.T) {
	if _, err := New(WithCredentials(authstore.NewMemory(nil)), WithAPIKey("k")); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("err = %v, want ErrBadRequest", err)
	}
}

func TestRefreshRequiresDirectScopeAndClientID(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	f := newFakeOpenAI(t)
	f.scope = "openid profile"
	if _, err := f.provider(t, oauthStore("stale", time.Now())).Models(context.Background()); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("grant without %s = %v, want ErrAuth", chatGPTDirectScope, err)
	}
	noClient := authstore.NewMemory(llm.AuthFile{"openai": {Type: "oauth", Access: "a", Refresh: "r", Expires: 1}})
	if _, err := f.provider(t, noClient).Models(context.Background()); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("refresh without client ID = %v, want ErrAuth", err)
	}
}

func freePort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	_ = listener.Close()
	return port
}

func TestLoginFlowSignInWithChatGPT(t *testing.T) {
	f := newFakeOpenAI(t)
	cfg := defaultLoginConfig()
	cfg.deviceID = "0F8FAD5B-D9CB-469F-A165-70867728950E"
	cfg.agentName = "test-agent"
	cfg.authorizeURL = f.server.URL + "/authorize"
	cfg.tokenURL = f.server.URL + "/oauth/token"
	cfg.port = freePort(t)
	flow, err := newLoginFlow(cfg)
	if err != nil {
		t.Fatal(err)
	}
	auth, err := flow.Begin(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	noRedirect := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	launch, err := noRedirect.Get(auth.URL())
	if err != nil {
		t.Fatal(err)
	}
	launch.Body.Close()
	authorize, err := url.Parse(launch.Header.Get("Location"))
	if err != nil {
		t.Fatal(err)
	}
	q := authorize.Query()
	redirect := fmt.Sprintf("http://127.0.0.1:%d/auth/callback", cfg.port)
	for key, want := range map[string]string{
		"client_id": chatGPTDynamicClientID, "agent_name_hint": "test-agent",
		"ext_agent_host_id": "urn:uuid:0f8fad5b-d9cb-469f-a165-70867728950e", "response_type": "code",
		"redirect_uri": redirect, "resource": chatGPTResource, "scope": chatGPTScope, "code_challenge_method": "S256",
	} {
		if got := q.Get(key); got != want {
			t.Errorf("authorize %s = %q, want %q", key, got, want)
		}
	}
	if q.Get("nonce") == "" || q.Get("code_challenge") == "" || q.Get("state") == "" {
		t.Fatalf("authorize URL missing nonce/challenge/state: %s", authorize)
	}

	go func() {
		resp, err := http.Get(redirect + "?code=auth-code&state=" + url.QueryEscape(q.Get("state")) + "&client_id=issued-at-login")
		if err == nil {
			resp.Body.Close()
		}
	}()
	cred, err := flow.Complete(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if cred.Type != "oauth" || cred.Access != "access-1" || cred.ClientID != "issued-at-login" || cred.Expires == 0 {
		t.Fatalf("credential = %+v", cred)
	}
	exchange := f.refreshes[0]
	for key, want := range map[string]string{"grant_type": "authorization_code", "client_id": "issued-at-login", "code": "auth-code", "redirect_uri": redirect, "resource": chatGPTResource} {
		if got := exchange.Get(key); got != want {
			t.Errorf("exchange %s = %q, want %q", key, got, want)
		}
	}
	if exchange.Get("code_verifier") == "" {
		t.Fatal("exchange omitted the PKCE verifier")
	}
}

func TestLoginFlowRequiresDeviceIDAndIssuedClientID(t *testing.T) {
	if _, err := NewLoginFlow(); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("missing device ID = %v, want ErrBadRequest", err)
	}
	if _, err := NewLoginFlow(WithLoginDeviceID("not-a-uuid")); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("invalid device ID = %v, want ErrBadRequest", err)
	}
	f := newFakeOpenAI(t)
	cfg := defaultLoginConfig()
	cfg.deviceID = "0f8fad5b-d9cb-469f-a165-70867728950e"
	cfg.authorizeURL = f.server.URL + "/authorize"
	cfg.tokenURL = f.server.URL + "/oauth/token"
	cfg.port = freePort(t)
	flow, err := newLoginFlow(cfg)
	if err != nil {
		t.Fatal(err)
	}
	auth, err := flow.Begin(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	noRedirect := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	launch, err := noRedirect.Get(auth.URL())
	if err != nil {
		t.Fatal(err)
	}
	launch.Body.Close()
	location, _ := url.Parse(launch.Header.Get("Location"))
	state := location.Query().Get("state")
	redirect := fmt.Sprintf("http://127.0.0.1:%d/auth/callback", cfg.port)
	if err := flow.Submit(context.Background(), redirect+"?code=c&state="+url.QueryEscape(state)); err != nil {
		t.Fatalf("Submit = %v", err)
	}
	if _, err := flow.Complete(context.Background()); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("callback without client_id = %v, want ErrAuth", err)
	}
}
