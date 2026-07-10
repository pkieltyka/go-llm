package e2e

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

// All onRefresh persistence tests run against temp files with FAKE
// credentials only — never against the real gollm-test.json.

func TestPersistRefreshedCredentialPreservesOtherEntriesAndUnknownFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fake-creds.json")
	original := `{
  "anthropic": {"type": "api_key", "api_key": "fake-anthropic-key"},
  "openai-codex": {
    "type": "oauth",
    "access": "fake-old-access",
    "refresh": "fake-old-refresh",
    "expires": 100,
    "accountId": "fake-acct",
    "custom_unknown_field": {"keep": true}
  },
  "top_level_unknown": {"note": "keep-me"}
}`
	if err := os.WriteFile(path, []byte(original), 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	persist := PersistOnRefresh(path, "openai-codex", t.Logf, NewSecretSet())
	persist(llm.AuthCredential{
		Type:      "oauth",
		Access:    "fake-new-access",
		Refresh:   "fake-new-refresh",
		Expires:   999,
		AccountID: "fake-acct-2",
	})

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read rewritten file: %v", err)
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(data, &root); err != nil {
		t.Fatalf("rewritten file is invalid JSON: %v\n%s", err, data)
	}
	var unknown map[string]any
	if err := json.Unmarshal(root["top_level_unknown"], &unknown); err != nil || unknown["note"] != "keep-me" {
		t.Fatalf("top-level unknown field lost (err=%v): %s", err, data)
	}

	var entry map[string]any
	if err := json.Unmarshal(root["openai-codex"], &entry); err != nil {
		t.Fatalf("parse rewritten entry: %v", err)
	}
	if entry["access"] != "fake-new-access" || entry["refresh"] != "fake-new-refresh" || entry["accountId"] != "fake-acct-2" {
		t.Fatalf("rewritten entry = %+v", entry)
	}
	if expires, _ := entry["expires"].(float64); expires != 999 {
		t.Fatalf("expires = %v, want 999", entry["expires"])
	}
	if _, ok := entry["custom_unknown_field"]; !ok {
		t.Fatalf("entry unknown field lost: %+v", entry)
	}

	var anthropicEntry map[string]any
	if err := json.Unmarshal(root["anthropic"], &anthropicEntry); err != nil {
		t.Fatalf("parse anthropic entry: %v", err)
	}
	if anthropicEntry["api_key"] != "fake-anthropic-key" {
		t.Fatalf("other provider entry changed: %+v", anthropicEntry)
	}

	if runtime.GOOS != "windows" {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatalf("stat rewritten file: %v", err)
		}
		if info.Mode().Perm() != 0o600 {
			t.Fatalf("file mode = %v, want 0600", info.Mode().Perm())
		}
	}

	// The rewritten file must still load through the normal auth path.
	auth, err := llm.LoadAuthFile(path)
	if err != nil {
		t.Fatalf("LoadAuthFile on rewritten file: %v", err)
	}
	if auth["openai-codex"].Access != "fake-new-access" || auth["openai-codex"].Refresh != "fake-new-refresh" {
		t.Fatalf("reloaded credential = %+v", auth["openai-codex"])
	}
}

func TestPersistRefreshedCredentialProvidersWrapper(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fake-creds.json")
	original := `{"providers": {"openai-codex": {"type": "oauth", "access": "fake-old", "refresh": "fake-old-refresh"}}, "version": 2}`
	if err := os.WriteFile(path, []byte(original), 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	if err := persistRefreshedCredential(path, "openai-codex", llm.AuthCredential{
		Type:    "oauth",
		Access:  "fake-new",
		Refresh: "fake-new-refresh",
	}); err != nil {
		t.Fatalf("persistRefreshedCredential returned error: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read rewritten file: %v", err)
	}
	var root struct {
		Providers map[string]map[string]any `json:"providers"`
		Version   int                       `json:"version"`
	}
	if err := json.Unmarshal(data, &root); err != nil {
		t.Fatalf("rewritten file is invalid JSON: %v\n%s", err, data)
	}
	if root.Version != 2 {
		t.Fatalf("wrapper sibling field lost: %s", data)
	}
	entry := root.Providers["openai-codex"]
	if entry["access"] != "fake-new" || entry["refresh"] != "fake-new-refresh" {
		t.Fatalf("rewritten wrapped entry = %+v", entry)
	}
}

func TestPersistRefreshedCredentialKeepsRefreshWhenRotatedEmpty(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fake-creds.json")
	original := `{"openai-codex": {"type": "oauth", "access": "fake-old", "refresh": "fake-keep-refresh"}}`
	if err := os.WriteFile(path, []byte(original), 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	if err := persistRefreshedCredential(path, "openai-codex", llm.AuthCredential{
		Type:   "oauth",
		Access: "fake-new",
	}); err != nil {
		t.Fatalf("persistRefreshedCredential returned error: %v", err)
	}

	auth, err := llm.LoadAuthFile(path)
	if err != nil {
		t.Fatalf("LoadAuthFile: %v", err)
	}
	if auth["openai-codex"].Access != "fake-new" || auth["openai-codex"].Refresh != "fake-keep-refresh" {
		t.Fatalf("credential = %+v", auth["openai-codex"])
	}
}

func TestPersistOnRefreshAddsEveryRotatedCredentialToSecretSet(t *testing.T) {
	secrets := NewSecretSet("initial-secret")
	var logged bool
	persist := PersistOnRefresh(filepath.Join(t.TempDir(), "missing", "credentials.json"), "openai-codex", func(string, ...any) {
		logged = true
	}, secrets)
	rotated := llm.AuthCredential{
		Key:       "rotated-key-secret",
		Access:    "rotated-access-secret",
		Refresh:   "rotated-refresh-secret",
		AccountID: "rotated-account-secret",
	}
	persist(rotated)
	if !logged {
		t.Fatal("persistence failure was not logged")
	}
	values := strings.Join(secrets.Values(), "\n")
	for _, want := range []string{"initial-secret", rotated.Key, rotated.Access, rotated.Refresh, rotated.AccountID} {
		if !strings.Contains(values, want) {
			t.Fatalf("secret set %q missing %q", values, want)
		}
	}
}

func TestPersistOnRefreshRequiresSecretSet(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("PersistOnRefresh accepted a nil recording secret set")
		}
	}()
	PersistOnRefresh("unused", "openai-codex", nil, nil)
}
