package e2e

import (
	"errors"
	"os"
	"path/filepath"
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
		"openai": {
			Type:     "oauth",
			Access:   "access",
			Refresh:  "refresh",
			Expires:  123,
			ClientID: "client",
			Model:    "gpt-5.4-mini",
		},
	}}

	provider := cfg.Provider("openai", "")
	auth := provider.Auth
	if auth.Type != "oauth" || auth.Access != "access" || auth.Refresh != "refresh" || auth.ClientID != "client" || auth.Expires != 123 {
		t.Fatalf("oauth provider = %+v", provider)
	}
	if provider.Model != "gpt-5.4-mini" {
		t.Fatalf("model = %q", provider.Model)
	}
}

func TestValidateLiveProviderConfigDistinguishesMissingAndInvalid(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		config   ProviderConfig
		missing  bool
		invalid  bool
	}{
		{name: "missing API key", provider: "anthropic", missing: true},
		{name: "valid API key", provider: "anthropic", config: ProviderConfig{Auth: llm.AuthCredential{Type: "api_key", Key: "key"}}},
		{name: "configured rejected API key is not missing", provider: "openai", config: ProviderConfig{Auth: llm.AuthCredential{Type: "api_key", Key: "invalid-key"}}},
		{name: "unsupported Anthropic OAuth", provider: "anthropic", config: ProviderConfig{Auth: llm.AuthCredential{Type: "oauth", Access: "access"}}, invalid: true},
		{name: "missing OpenAI credential", provider: "openai", missing: true},
		{name: "OpenAI OAuth without client ID", provider: "openai", config: ProviderConfig{Auth: llm.AuthCredential{Type: "oauth", Access: "a", Refresh: "refresh"}}, invalid: true},
		{name: "OpenAI OAuth without tokens", provider: "openai", config: ProviderConfig{Auth: llm.AuthCredential{Type: "oauth"}}, invalid: true},
		{name: "valid OpenAI OAuth", provider: "openai", config: ProviderConfig{Auth: llm.AuthCredential{Type: "oauth", Access: "access", Refresh: "r", ClientID: "c"}}},
		{name: "missing vLLM URL", provider: "vllm", missing: true},
		{name: "vLLM key without URL", provider: "vllm", config: ProviderConfig{Auth: llm.AuthCredential{Key: "key"}}, invalid: true},
		{name: "valid keyless vLLM", provider: "vllm", config: ProviderConfig{BaseURL: "http://localhost:8000/v1"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := ValidateLiveProviderConfig(test.provider, test.config)
			switch {
			case test.missing && !errors.Is(err, errLiveCredentialsMissing):
				t.Fatalf("error = %v, want missing credentials", err)
			case test.invalid && (err == nil || errors.Is(err, errLiveCredentialsMissing)):
				t.Fatalf("error = %v, want configured-invalid failure", err)
			case !test.missing && !test.invalid && err != nil:
				t.Fatalf("error = %v, want nil", err)
			}
		})
	}
}

func TestLoadConfigUsesAuthFileFormatsForAnthropicAndOpenAIOAuth(t *testing.T) {
	root := t.TempDir()
	data := []byte(`{
		"providers": {
			"anthropic": {"type":"api_key","key":"anthropic-test","model":"claude-test"},
			"openai": {"type":"oauth","access":"chatgpt-access","refresh":"chatgpt-refresh","expires":123,"clientId":"client-test","scopes":["chatgpt.tokens.use.direct"],"model":"gpt-test"}
		}
	}`)
	if err := os.WriteFile(filepath.Join(root, "gollm-test.json"), data, 0o600); err != nil {
		t.Fatalf("write auth file: %v", err)
	}
	config, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("LoadConfig returned error: %v", err)
	}
	anthropic := config.Provider("anthropic", "")
	if anthropic.Auth.Type != "api_key" || anthropic.Auth.Key != "anthropic-test" || anthropic.Model != "claude-test" {
		t.Fatalf("anthropic config = %+v", anthropic)
	}
	chatgpt := config.Provider("openai", "")
	if chatgpt.Auth.Type != "oauth" || chatgpt.Auth.Access != "chatgpt-access" || chatgpt.Auth.Refresh != "chatgpt-refresh" || chatgpt.Auth.Expires != 123 || chatgpt.Auth.ClientID != "client-test" || chatgpt.Model != "gpt-test" {
		t.Fatalf("openai oauth config = %+v", chatgpt)
	}
}

// The main and handoff OpenAI suites share one default: the override, else
// the subscription model for Sign in with ChatGPT, else the cheap API model.
func TestOpenAIDefaultModel(t *testing.T) {
	for name, tc := range map[string]struct {
		cfg  ProviderConfig
		want string
	}{
		"api key":  {ProviderConfig{Auth: llm.AuthCredential{Type: "api_key", Key: "k"}}, openAICheapModel},
		"oauth":    {ProviderConfig{Auth: llm.AuthCredential{Type: "oauth", Access: "a"}}, openAISignInModel},
		"override": {ProviderConfig{Auth: llm.AuthCredential{Type: "oauth", Access: "a"}, Model: "gpt-6-luna"}, "gpt-6-luna"},
	} {
		if got := openAIDefaultModel(tc.cfg); got != tc.want {
			t.Errorf("%s: default = %q, want %q", name, got, tc.want)
		}
	}
}
