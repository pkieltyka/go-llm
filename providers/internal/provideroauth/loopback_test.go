package provideroauth

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

type loopbackHarness struct {
	flow      *LoopbackFlow
	port      int
	launchURL string
	exchanges []url.Values
}

func freeLoopbackPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	_ = listener.Close()
	return port
}

func newLoopbackHarness(t *testing.T, mutate func(*LoopbackConfig)) *loopbackHarness {
	t.Helper()
	h := &loopbackHarness{port: freeLoopbackPort(t)}
	cfg := LoopbackConfig{
		Label:        "Test",
		Ports:        []int{h.port},
		RedirectURI:  func(port int) string { return "http://127.0.0.1:" + strconv.Itoa(port) + "/cb" },
		CallbackPath: "/cb",
		LaunchPath:   "/go",
		AuthorizationURL: func(a LoopbackAuthorization) (string, error) {
			return "https://auth.example/authorize?state=" + url.QueryEscape(a.State) + "&challenge=" + a.Challenge + "&nonce=" + a.Nonce + "&redirect_uri=" + url.QueryEscape(a.RedirectURI), nil
		},
		Exchange: func(_ context.Context, code, verifier, redirectURI string, callback url.Values) (llm.AuthCredential, error) {
			h.exchanges = append(h.exchanges, url.Values{"code": {code}, "verifier": {verifier}, "redirect": {redirectURI}, "extra": {callback.Get("extra")}})
			return llm.AuthCredential{Type: "oauth", Access: "minted"}, nil
		},
		FlowTimeout:     5 * time.Second,
		ExchangeTimeout: time.Second,
		Random:          rand.Reader,
		Listen:          net.Listen,
	}
	if mutate != nil {
		mutate(&cfg)
	}
	flow, err := NewLoopbackFlow(cfg)
	if err != nil {
		t.Fatal(err)
	}
	h.flow = flow
	t.Cleanup(flow.Cancel)
	return h
}

func (h *loopbackHarness) begin(t *testing.T) url.Values {
	t.Helper()
	auth, err := h.flow.Begin(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	h.launchURL = auth.URL()
	client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	resp, err := client.Get(auth.URL())
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusFound {
		t.Fatalf("launch status = %d, want 302", resp.StatusCode)
	}
	location, err := url.Parse(resp.Header.Get("Location"))
	if err != nil {
		t.Fatal(err)
	}
	return location.Query()
}

func (h *loopbackHarness) callback(t *testing.T, query string) (int, string) {
	t.Helper()
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/cb?%s", h.port, query))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(body)
}

func TestLoopbackFlowCompletesWithCallback(t *testing.T) {
	h := newLoopbackHarness(t, nil)
	q := h.begin(t)
	if q.Get("challenge") == "" || q.Get("nonce") == "" || q.Get("redirect_uri") != fmt.Sprintf("http://127.0.0.1:%d/cb", h.port) {
		t.Fatalf("authorization query = %v", q)
	}
	status, body := h.callback(t, "code=the-code&extra=issued&state="+url.QueryEscape(q.Get("state")))
	if status != http.StatusOK || !strings.Contains(body, "completed") {
		t.Fatalf("callback = %d %q", status, body)
	}
	cred, err := h.flow.Complete(context.Background())
	if err != nil || cred.Access != "minted" {
		t.Fatalf("Complete = %+v, %v", cred, err)
	}
	got := h.exchanges[0]
	if got.Get("code") != "the-code" || got.Get("verifier") == "" || got.Get("extra") != "issued" {
		t.Fatalf("exchange = %v", got)
	}
	if _, err := h.flow.Complete(context.Background()); !errors.Is(err, llm.ErrLoginClosed) {
		t.Fatalf("second Complete = %v, want ErrLoginClosed", err)
	}
	if _, err := h.flow.Begin(context.Background()); !errors.Is(err, llm.ErrLoginClosed) {
		t.Fatalf("reused Begin = %v, want ErrLoginClosed", err)
	}
}

func TestLoopbackLaunchRequiresTokenAndIsSingleUse(t *testing.T) {
	h := newLoopbackHarness(t, nil)
	h.begin(t) // consumes the launch with the right token
	launch := func(query string) int {
		client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
		resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/go%s", h.port, query))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		return resp.StatusCode
	}
	if status := launch(""); status != http.StatusGone {
		t.Fatalf("launch without token = %d, want 410", status)
	}
	if status := launch("?t=guess"); status != http.StatusGone {
		t.Fatalf("launch with wrong token = %d, want 410", status)
	}
	auth, _ := url.Parse(h.launchURL)
	if status := launch("?" + auth.RawQuery); status != http.StatusGone {
		t.Fatalf("second launch = %d, want 410 (single use)", status)
	}
}

func TestLoopbackFlowRejectsWrongStateWithoutEndingTheFlow(t *testing.T) {
	h := newLoopbackHarness(t, nil)
	q := h.begin(t)
	if status, _ := h.callback(t, "code=x&state=wrong"); status != http.StatusBadRequest {
		t.Fatalf("wrong state status = %d, want 400", status)
	}
	if status, _ := h.callback(t, "state="+url.QueryEscape(q.Get("state"))); status != http.StatusBadRequest {
		t.Fatalf("missing code status = %d, want 400", status)
	}
	if err := h.flow.Submit(context.Background(), "https://elsewhere.example/other?code=x"); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("Submit wrong path = %v, want ErrBadRequest", err)
	}
	if err := h.flow.Submit(context.Background(), "https://elsewhere.example/cb?code=x&state="+url.QueryEscape(q.Get("state"))); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("Submit wrong origin = %v, want ErrBadRequest", err)
	}
	if err := h.flow.Submit(context.Background(), "?code=x&state="+url.QueryEscape(q.Get("state"))); err != nil {
		t.Fatalf("Submit valid query = %v", err)
	}
	if _, err := h.flow.Complete(context.Background()); err != nil {
		t.Fatalf("Complete after a rejected stray callback = %v", err)
	}
}

func TestLoopbackFlowDenial(t *testing.T) {
	h := newLoopbackHarness(t, nil)
	q := h.begin(t)
	if status, _ := h.callback(t, "error=access_denied&state="+url.QueryEscape(q.Get("state"))); status != http.StatusUnauthorized {
		t.Fatalf("denial status = %d", status)
	}
	if _, err := h.flow.Complete(context.Background()); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("denied Complete = %v, want ErrAuth", err)
	}
	if len(h.exchanges) != 0 {
		t.Fatal("a denied login must not exchange a code")
	}
}

func TestLoopbackFlowCancelAndExpire(t *testing.T) {
	h := newLoopbackHarness(t, nil)
	h.begin(t)
	done := make(chan error, 1)
	go func() {
		_, err := h.flow.Complete(context.Background())
		done <- err
	}()
	time.Sleep(20 * time.Millisecond)
	h.flow.Cancel()
	if err := <-done; !errors.Is(err, llm.ErrLoginClosed) {
		t.Fatalf("cancelled Complete = %v, want ErrLoginClosed", err)
	}

	expiring := newLoopbackHarness(t, func(c *LoopbackConfig) { c.FlowTimeout = 30 * time.Millisecond })
	expiring.begin(t)
	if _, err := expiring.flow.Complete(context.Background()); !errors.Is(err, llm.ErrLoginExpired) {
		t.Fatalf("expired Complete = %v, want ErrLoginExpired", err)
	}

	ctxFlow := newLoopbackHarness(t, nil)
	ctxFlow.begin(t)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := ctxFlow.flow.Complete(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Complete ctx deadline = %v", err)
	}
}

func TestLoopbackFlowPortInUseAndConfigValidation(t *testing.T) {
	busy, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer busy.Close()
	h := newLoopbackHarness(t, func(c *LoopbackConfig) { c.Ports = []int{busy.Addr().(*net.TCPAddr).Port} })
	if _, err := h.flow.Begin(context.Background()); !errors.Is(err, llm.ErrServer) {
		t.Fatalf("busy port Begin = %v, want ErrServer", err)
	}
	if _, err := NewLoopbackFlow(LoopbackConfig{Label: "Test"}); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("incomplete config = %v, want ErrBadRequest", err)
	}
	if got := fmt.Sprint(h.flow); strings.Contains(got, "state") && !strings.Contains(got, "redacted") {
		t.Fatalf("flow formatting leaks state: %q", got)
	}
	for _, tt := range []struct {
		url  string
		want bool
	}{{"https://auth.example/x", true}, {"http://127.0.0.1:1/x", true}, {"http://evil.example/x", false}, {"https://u:p@auth.example", false}} {
		if got := ValidEndpoint(tt.url); got != tt.want {
			t.Errorf("ValidEndpoint(%q) = %v", tt.url, got)
		}
	}
}

func TestLoopbackFlowKeepsMintedCredentialWhenContextEndsAfterExchange(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h := newLoopbackHarness(t, func(c *LoopbackConfig) {
		c.Exchange = func(context.Context, string, string, string, url.Values) (llm.AuthCredential, error) {
			cancel() // the caller gives up just as the token arrives
			return llm.AuthCredential{Type: "oauth", Access: "minted"}, nil
		}
	})
	q := h.begin(t)
	h.callback(t, "code=c&state="+url.QueryEscape(q.Get("state")))
	if cred, err := h.flow.Complete(ctx); err != nil || cred.Access != "minted" {
		t.Fatalf("Complete = %+v, %v; a minted credential must not be dropped", cred, err)
	}
}
