package e2e

import (
	"errors"
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

// Configured reports whether the entry carries enough to construct a live
// provider: an API key, an OAuth access token, or — for keyless self-hosted
// servers (vLLM) — a base URL alone.
func (pc ProviderConfig) Configured() bool {
	return pc.Auth.Key != "" || pc.Auth.Access != "" || pc.BaseURL != ""
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
