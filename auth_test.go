package llm

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAuthCredentialFormattingAndLoggingAreRedacted(t *testing.T) {
	credential := AuthCredential{
		Type:      "oauth",
		Key:       "key-secret",
		Access:    "access-secret",
		Refresh:   "refresh-secret",
		AccountID: "account-secret",
	}
	for name, formatted := range map[string]string{
		"String":   fmt.Sprint(credential),
		"value":    fmt.Sprintf("%v", credential),
		"detailed": fmt.Sprintf("%+v", credential),
		"GoString": fmt.Sprintf("%#v", credential),
	} {
		assertNoCredentialSecret(t, name, formatted)
	}

	for _, format := range []struct {
		name string
		new  func(*bytes.Buffer) slog.Handler
	}{
		{name: "text", new: func(out *bytes.Buffer) slog.Handler { return slog.NewTextHandler(out, nil) }},
		{name: "json", new: func(out *bytes.Buffer) slog.Handler { return slog.NewJSONHandler(out, nil) }},
	} {
		t.Run(format.name, func(t *testing.T) {
			var out bytes.Buffer
			slog.New(format.new(&out)).Info("credential", "auth", credential)
			assertNoCredentialSecret(t, format.name, out.String())
		})
	}
}

func TestAuthCredentialExplicitJSONPersistenceStillWorks(t *testing.T) {
	credential := AuthCredential{Type: "oauth", Access: "access-secret", Refresh: "refresh-secret", AccountID: "account-secret"}
	encoded, err := json.Marshal(credential)
	if err != nil {
		t.Fatal(err)
	}
	for _, secret := range []string{credential.Access, credential.Refresh, credential.AccountID} {
		if !bytes.Contains(encoded, []byte(secret)) {
			t.Fatalf("explicit JSON = %s, want persisted %q", encoded, secret)
		}
	}
}

func assertNoCredentialSecret(t *testing.T, name, value string) {
	t.Helper()
	for _, secret := range []string{"key-secret", "access-secret", "refresh-secret", "account-secret"} {
		if strings.Contains(value, secret) {
			t.Fatalf("%s output leaked %q: %s", name, secret, value)
		}
	}
}

func TestParseAuthFileNestedAndBare(t *testing.T) {
	nested, err := ParseAuthFile([]byte(`{
		"providers": {
			"anthropic": {"type": "api_key", "api_key": "key-1", "model": "claude-test", "base_url": "https://example.test"},
			"openai-codex": {"type": "oauth", "access": "access-1", "refresh": "refresh-1", "expires": 123, "accountId": "acct-1"}
		}
	}`))
	if err != nil {
		t.Fatalf("ParseAuthFile nested returned error: %v", err)
	}
	if nested["anthropic"].Key != "key-1" || nested["anthropic"].BaseURL != "https://example.test" {
		t.Fatalf("nested anthropic credential = %+v", nested["anthropic"])
	}
	if got := nested["openai-codex"]; got.Type != "oauth" || got.Access != "access-1" || got.Refresh != "refresh-1" || got.Expires != 123 || got.AccountID != "acct-1" {
		t.Fatalf("nested codex credential = %+v", got)
	}

	bare, err := ParseAuthFile([]byte(`{
		"anthropic": {"type": "api_key", "key": "key-2"},
		"future": {"type": "unknown", "new_field": true}
	}`))
	if err != nil {
		t.Fatalf("ParseAuthFile bare returned error: %v", err)
	}
	if bare["anthropic"].Key != "key-2" || bare["future"].Type != "unknown" {
		t.Fatalf("bare credentials = %+v", bare)
	}
}

func TestAuthCredentialUnmarshalClearsStaleExpires(t *testing.T) {
	credential := AuthCredential{Expires: 123, Access: "old"}
	if err := json.Unmarshal([]byte(`{"type":"api_key","key":"new"}`), &credential); err != nil {
		t.Fatalf("Unmarshal returned error: %v", err)
	}
	if credential.Expires != 0 || credential.Access != "" || credential.Key != "new" {
		t.Fatalf("credential = %+v, want stale fields cleared", credential)
	}
}

func TestLoadAuthFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "auth.json")
	if err := os.WriteFile(path, []byte(`{"providers":{"openai":{"type":"api_key","key":"key"}}}`), 0o600); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	auth, err := LoadAuthFile(path)
	if err != nil {
		t.Fatalf("LoadAuthFile returned error: %v", err)
	}
	if auth["openai"].Key != "key" {
		t.Fatalf("auth = %+v", auth)
	}
}

func TestLoginAuthorizationValidatesAndRedactsLaunchURL(t *testing.T) {
	const launchURL = "http://127.0.0.1:43210/authorize?ephemeral=secret"
	authorization, err := NewLoginAuthorization(launchURL, "Open it")
	if err != nil {
		t.Fatal(err)
	}
	if authorization.URL() != launchURL || authorization.Instructions() != "Open it" {
		t.Fatal("authorization accessors did not preserve their explicit values")
	}
	for _, formatted := range []string{
		fmt.Sprint(authorization),
		fmt.Sprintf("%+v", authorization),
		fmt.Sprintf("%#v", authorization),
	} {
		if strings.Contains(formatted, "ephemeral") || strings.Contains(formatted, "secret") {
			t.Fatal("formatted authorization leaked its launch URL")
		}
	}
	encoded, err := json.Marshal(authorization)
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != "{}" {
		t.Fatalf("JSON authorization = %s, want no exported data", encoded)
	}
	if _, err := NewLoginAuthorization("javascript:alert(1)", ""); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("unsafe login URL error = %v, want ErrBadRequest", err)
	}
}
