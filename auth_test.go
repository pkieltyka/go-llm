package llm

import (
	"os"
	"path/filepath"
	"testing"
)

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
