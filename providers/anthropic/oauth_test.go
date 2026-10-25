package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

type oauthRoundTripFunc func(*http.Request) (*http.Response, error)

func (f oauthRoundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestAnthropicTokenExchangeAndExpiry(t *testing.T) {
	var request map[string]string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.Header.Get("Content-Type") != "application/json" || r.Header.Get("Accept") != "application/json" {
			t.Error("wrong token request protocol")
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		_, _ = io.WriteString(w, `{"access_token":"access-new","refresh_token":"refresh-new","expires_in":3600}`)
	}))
	defer server.Close()
	before := time.Now().Add(55 * time.Minute).UnixMilli()
	got, err := refreshAnthropicOAuth(context.Background(), server.Client(), server.URL, llm.AuthCredential{Type: "oauth", Refresh: "refresh-old"})
	if err != nil {
		t.Fatal(err)
	}
	if request["client_id"] != anthropicOAuthClientID || request["grant_type"] != "refresh_token" || request["refresh_token"] != "refresh-old" {
		t.Fatal("refresh JSON mismatch")
	}
	if got.Access != "access-new" || got.Refresh != "refresh-new" || got.ClientID != "" || got.Expires < before || got.Expires > time.Now().Add(55*time.Minute).UnixMilli() {
		t.Fatal("minted credential or Pi expiry mismatch")
	}
}

func TestAnthropicOAuthIdentityFailsBeforeTransport(t *testing.T) {
	calls := 0
	client := &http.Client{Transport: oauthRoundTripFunc(func(*http.Request) (*http.Response, error) { calls++; return nil, errors.New("must not run") })}
	for _, cred := range []llm.AuthCredential{{Type: "api_key"}, {Type: "oauth", ClientID: "foreign", Access: "valid", Refresh: "refresh"}, {Type: "oauth"}} {
		_, err := refreshAnthropicOAuth(context.Background(), client, "", cred)
		if !errors.Is(err, llm.ErrAuth) {
			t.Fatalf("invalid credential error=%v", err)
		}
	}
	if calls != 0 {
		t.Fatal("invalid grant reached transport")
	}
	for _, id := range []string{"", anthropicOAuthClientID} {
		if err := ValidateOAuthCredential(llm.AuthCredential{Type: "oauth", ClientID: id}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestAnthropicOAuthTokenErrorsAreBoundedAndRedacted(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		status     int
		want       error
	}{
		{"invalid", `{"access_token":"marker-secret"`, 200, llm.ErrServer},
		{"extra", `{} {"marker-secret":true}`, 200, llm.ErrServer},
		{"large", strings.Repeat("marker-secret", 100000), 200, llm.ErrServer},
		{"auth", "marker-secret", 400, llm.ErrAuth}, {"forbidden", "marker-secret", 403, llm.ErrAuth},
		{"limit", "marker-secret", 429, llm.ErrRateLimited}, {"timeout", "marker-secret", 408, llm.ErrTimeout}, {"server", "marker-secret", 503, llm.ErrServer},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			}))
			defer server.Close()
			_, err := requestAnthropicToken(context.Background(), server.Client(), server.URL, nil, "test")
			if !errors.Is(err, tc.want) || strings.Contains(fmt.Sprint(err), "marker-secret") {
				t.Fatalf("unsafe/wrong token error %v", err)
			}
		})
	}
}

func TestAnthropicOAuthTransportCancellationIsCanonical(t *testing.T) {
	for _, kind := range []error{context.Canceled, context.DeadlineExceeded} {
		client := &http.Client{Transport: oauthRoundTripFunc(func(*http.Request) (*http.Response, error) { return nil, fmt.Errorf("marker-secret: %w", kind) })}
		_, err := requestAnthropicToken(context.Background(), client, "", nil, "test")
		if err != kind || strings.Contains(fmt.Sprint(err), "marker-secret") {
			t.Fatalf("not canonical cancellation: %v", err)
		}
	}
}

func TestAnthropicTokenRedirectRejected(t *testing.T) {
	calls := 0
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls++ }))
	defer target.Close()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer server.Close()
	_, err := requestAnthropicToken(context.Background(), server.Client(), server.URL, nil, "test")
	if !errors.Is(err, llm.ErrServer) || calls != 0 {
		t.Fatal("token redirect followed")
	}
}

func TestAnthropicCredentialValidation(t *testing.T) {
	now := time.Unix(1700000000, 0)
	for _, token := range []anthropicTokenResponse{{}, {AccessToken: "a", RefreshToken: "r"}, {AccessToken: "a", RefreshToken: "r", ExpiresIn: -1}, {AccessToken: "a", RefreshToken: "r", ExpiresIn: math.MaxInt64}} {
		if _, err := anthropicCredential(token, now); !errors.Is(err, llm.ErrAuth) {
			t.Fatalf("invalid token error=%v", err)
		}
	}
	got, err := anthropicCredential(anthropicTokenResponse{AccessToken: "a", RefreshToken: "r", ExpiresIn: 60}, now)
	if err != nil || got.Expires != now.Add(-4*time.Minute).UnixMilli() {
		t.Fatal("short-lived adjusted expiry changed")
	}
}
