package e2e

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	llm "github.com/pkieltyka/go-llm"
)

// Config mirrors the gitignored gollm-test.json file used only by live tests.
type Config struct {
	Providers llm.AuthFile
}

// ProviderConfig holds per-provider live-test credentials and overrides.
// Auth carries the full credential; Model/BaseURL are the two per-suite
// overrides lifted out for convenience.
type ProviderConfig struct {
	Auth    llm.AuthCredential
	Model   string
	BaseURL string
}

var errLiveCredentialsMissing = errors.New("live credentials missing")

// Configured reports whether the entry carries enough to construct a live
// provider: an API key, an OAuth access token, or — for keyless self-hosted
// servers (vLLM) — a base URL alone.
func (pc ProviderConfig) Configured() bool {
	return pc.Auth.Key != "" || pc.Auth.Access != "" || pc.BaseURL != ""
}

// ValidateLiveProviderConfig distinguishes an absent live-test setup from a
// present but malformed one. Callers may skip only errLiveCredentialsMissing;
// malformed or remotely rejected configured credentials are test failures.
func ValidateLiveProviderConfig(provider string, pc ProviderConfig) error {
	switch provider {
	case "anthropic":
		switch pc.Auth.Type {
		case "", "api_key":
			if pc.Auth.Key == "" {
				return fmt.Errorf("%w: anthropic API key", errLiveCredentialsMissing)
			}
		case "oauth":
			if pc.Auth.Access == "" {
				return errors.New("anthropic OAuth credential is configured without an access token")
			}
		default:
			return fmt.Errorf("anthropic credential has unsupported type %q", pc.Auth.Type)
		}
	case "openai", "openrouter":
		if pc.Auth.Type != "" && pc.Auth.Type != "api_key" {
			return fmt.Errorf("%s credential has unsupported type %q", provider, pc.Auth.Type)
		}
		if pc.Auth.Key == "" {
			return fmt.Errorf("%w: %s API key", errLiveCredentialsMissing, provider)
		}
	case "openai-codex":
		if pc.Auth.Type == "" && pc.Auth.Access == "" && pc.Auth.Refresh == "" && pc.Auth.AccountID == "" {
			return fmt.Errorf("%w: openai-codex OAuth credential", errLiveCredentialsMissing)
		}
		if pc.Auth.Type != "oauth" {
			return fmt.Errorf("openai-codex credential must use oauth, got %q", pc.Auth.Type)
		}
		if pc.Auth.Access == "" {
			return errors.New("openai-codex OAuth credential is configured without an access token")
		}
	case "vllm":
		if pc.BaseURL == "" {
			if pc.Auth.Key != "" {
				return errors.New("vllm API key is configured without a base URL")
			}
			return fmt.Errorf("%w: vllm base URL", errLiveCredentialsMissing)
		}
	default:
		return fmt.Errorf("unknown live provider %q", provider)
	}
	return nil
}

// LoadConfig reads gollm-test.json from repoRoot. Missing files return an empty
// config; live tests decide whether to skip.
func LoadConfig(repoRoot string) (Config, error) {
	path := filepath.Join(repoRoot, "gollm-test.json")
	auth, err := llm.LoadAuthFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Config{}, nil
		}
		return Config{}, err
	}
	return Config{Providers: auth}, nil
}

// Provider returns config for name with env fallback for API-key providers.
func (c Config) Provider(name, envVar string) ProviderConfig {
	cred := llm.AuthCredential{}
	if c.Providers != nil {
		cred = c.Providers[name]
	}
	cred.Key = liveConfigValue(cred.Key)
	cred.Access = liveConfigValue(cred.Access)
	cred.Refresh = liveConfigValue(cred.Refresh)
	cred.AccountID = liveConfigValue(cred.AccountID)
	cred.Model = liveConfigValue(cred.Model)
	cred.BaseURL = liveConfigValue(cred.BaseURL)
	if cred.Type == "" && cred.Key != "" {
		cred.Type = "api_key"
	}
	if (cred.Type == "" || cred.Type == "api_key") && cred.Key == "" {
		cred.Key = liveConfigValue(os.Getenv(envVar))
		if cred.Key != "" && cred.Type == "" {
			cred.Type = "api_key"
		}
	}
	return ProviderConfig{
		Auth:    cred,
		Model:   cred.Model,
		BaseURL: cred.BaseURL,
	}
}

func liveConfigValue(value string) string {
	value = strings.TrimSpace(value)
	placeholder := strings.ToLower(strings.ReplaceAll(value, "_", "-"))
	if value == "" || value == "..." || strings.HasPrefix(placeholder, "replace-with-") || strings.HasPrefix(placeholder, "optional-") {
		return ""
	}
	return value
}

// RepoRoot walks upward from start until it finds go.mod.
func RepoRoot(start string) (string, error) {
	dir, err := filepath.Abs(start)
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", os.ErrNotExist
		}
		dir = parent
	}
}
