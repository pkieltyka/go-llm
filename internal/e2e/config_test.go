package e2e

import (
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestProviderTreatsPlaceholdersAsMissing(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "env-openai-key")
	cfg := Config{Providers: llm.AuthFile{
		"openai": {
			Type:    "api_key",
			Key:     "replace-with-openai-api-key",
			Model:   "optional-model-override",
			BaseURL: "optional-base-url",
		},
	}}

	provider := cfg.Provider("openai", "OPENAI_API_KEY")
	if provider.Auth.Key != "env-openai-key" {
		t.Fatalf("API key = %q, want env fallback", provider.Auth.Key)
	}
	if provider.Model != "" || provider.BaseURL != "" {
		t.Fatalf("model/base_url placeholders were not cleared: %+v", provider)
	}
}

// TestConfiguredKeylessBaseURL pins the keyless self-hosted loader rule: an
// entry carrying only a base_url (vLLM) counts as configured, while an
// empty entry does not.
func TestConfiguredKeylessBaseURL(t *testing.T) {
	cfg := Config{Providers: llm.AuthFile{
		"vllm": {
			BaseURL: "http://pax.local:8000/v1",
			Model:   "Qwen/Qwen3.6-27B-FP8",
		},
	}}

	provider := cfg.Provider("vllm", "")
	if !provider.Configured() {
		t.Fatalf("keyless base_url entry should be configured: %+v", provider)
	}
	if provider.Auth.Key != "" || provider.BaseURL == "" || provider.Model == "" {
		t.Fatalf("keyless entry fields = %+v", provider)
	}

	missing := cfg.Provider("zai", "")
	if missing.Configured() {
		t.Fatalf("empty entry should not be configured: %+v", missing)
	}
}

func TestProviderReturnsOAuthCredential(t *testing.T) {
	cfg := Config{Providers: llm.AuthFile{
		"openai-codex": {
			Type:      "oauth",
			Access:    "access",
			Refresh:   "refresh",
			Expires:   123,
			AccountID: "acct",
			Model:     "gpt-5.4-mini",
		},
	}}

	provider := cfg.Provider("openai-codex", "")
	auth := provider.Auth
	if auth.Type != "oauth" || auth.Access != "access" || auth.Refresh != "refresh" || auth.AccountID != "acct" || auth.Expires != 123 {
		t.Fatalf("oauth provider = %+v", provider)
	}
	if provider.Model != "gpt-5.4-mini" {
		t.Fatalf("model = %q", provider.Model)
	}
}
