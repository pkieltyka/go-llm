package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/anthropic"
	"github.com/pkieltyka/go-llm/providers/openai"
	"github.com/pkieltyka/go-llm/providers/openaicodex"
	"github.com/pkieltyka/go-llm/providers/openrouter"
)

func newProvider(_ context.Context, cfg providerConfig) (llm.Provider, error) {
	if cfg.authFile != "" && cfg.name != "openai-codex" {
		return nil, fmt.Errorf("%w: --auth-file is only supported for openai-codex", llm.ErrBadRequest)
	}
	switch cfg.name {
	case "anthropic":
		opts := []anthropic.Option{}
		if cfg.apiKey != "" {
			opts = append(opts, anthropic.WithAPIKey(cfg.apiKey))
		}
		if cfg.baseURL != "" {
			opts = append(opts, anthropic.WithBaseURL(cfg.baseURL))
		}
		if cfg.timeout > 0 {
			opts = append(opts, anthropic.WithTimeout(cfg.timeout))
		}
		if logger := debugLogger(cfg); logger != nil {
			opts = append(opts, anthropic.WithLogger(logger), anthropic.WithWireCapture(llm.WireCaptureToLogger(logger)))
		}
		return anthropic.New(opts...)
	case "openai":
		opts := []openai.Option{}
		if cfg.apiKey != "" {
			opts = append(opts, openai.WithAPIKey(cfg.apiKey))
		}
		if cfg.baseURL != "" {
			opts = append(opts, openai.WithBaseURL(cfg.baseURL))
		}
		if cfg.timeout > 0 {
			opts = append(opts, openai.WithTimeout(cfg.timeout))
		}
		if logger := debugLogger(cfg); logger != nil {
			opts = append(opts, openai.WithLogger(logger), openai.WithWireCapture(llm.WireCaptureToLogger(logger)))
		}
		return openai.New(opts...)
	case "openai-codex":
		cred, persist, err := resolveCodexAuth(cfg)
		if err != nil {
			return nil, err
		}
		opts := []openaicodex.Option{openaicodex.WithOAuth(cred, persist)}
		if cfg.baseURL != "" {
			opts = append(opts, openaicodex.WithBaseURL(cfg.baseURL))
		}
		if cfg.timeout > 0 {
			opts = append(opts, openaicodex.WithTimeout(cfg.timeout))
		}
		if logger := debugLogger(cfg); logger != nil {
			opts = append(opts, openaicodex.WithLogger(logger), openaicodex.WithWireCapture(llm.WireCaptureToLogger(logger)))
		}
		return openaicodex.New(opts...)
	case "openrouter":
		opts := []openrouter.Option{}
		if cfg.apiKey != "" {
			opts = append(opts, openrouter.WithAPIKey(cfg.apiKey))
		}
		if cfg.baseURL != "" {
			opts = append(opts, openrouter.WithBaseURL(cfg.baseURL))
		}
		if cfg.timeout > 0 {
			opts = append(opts, openrouter.WithTimeout(cfg.timeout))
		}
		if logger := debugLogger(cfg); logger != nil {
			opts = append(opts, openrouter.WithLogger(logger), openrouter.WithWireCapture(llm.WireCaptureToLogger(logger)))
		}
		return openrouter.New(opts...)
	case "zai":
		return nil, fmt.Errorf("%w: provider zai is deferred in this module", llm.ErrUnsupported)
	case "":
		return nil, fmt.Errorf("%w: missing provider; set -p/--provider", llm.ErrBadRequest)
	default:
		return nil, fmt.Errorf("%w: unknown provider %q", llm.ErrBadRequest, cfg.name)
	}
}

func debugLogger(cfg providerConfig) *slog.Logger {
	if !cfg.debug {
		return nil
	}
	w := cfg.stderr
	if w == nil {
		w = os.Stderr
	}
	return slog.New(slog.NewTextHandler(w, &slog.HandlerOptions{Level: slog.LevelDebug}))
}

func providerConfigFromChat(cfg chatConfig, stderr io.Writer) providerConfig {
	return providerConfig{
		name:     cfg.provider,
		apiKey:   cfg.apiKey,
		authFile: cfg.authFile,
		baseURL:  cfg.baseURL,
		timeout:  cfg.timeout,
		debug:    cfg.debug,
		stderr:   stderr,
	}
}

func providerConfigFromModels(cfg modelsConfig, stderr io.Writer) providerConfig {
	return providerConfig{
		name:     cfg.provider,
		apiKey:   cfg.apiKey,
		authFile: cfg.authFile,
		baseURL:  cfg.baseURL,
		timeout:  cfg.timeout,
		debug:    cfg.debug,
		stderr:   stderr,
	}
}

func resolveCodexAuth(cfg providerConfig) (llm.AuthCredential, llm.OAuthPersistenceFunc, error) {
	if cfg.authFile != "" {
		auth, err := llm.LoadAuthFile(cfg.authFile)
		if err != nil {
			return llm.AuthCredential{}, nil, fmt.Errorf("load auth file: %w", err)
		}
		cred, ok := auth["openai-codex"]
		if !ok {
			return llm.AuthCredential{}, nil, fmt.Errorf("%w: auth file has no openai-codex credential", llm.ErrAuth)
		}
		return cred, codexAuthFilePersistence(cfg.authFile), nil
	}
	if access := strings.TrimSpace(os.Getenv("OPENAI_CODEX_ACCESS_TOKEN")); access != "" {
		return llm.AuthCredential{Type: "oauth", Access: access}, nil, nil
	}
	return llm.AuthCredential{Type: "oauth", Access: cfg.apiKey}, nil, nil
}

func codexAuthFilePersistence(path string) llm.OAuthPersistenceFunc {
	return func(ctx context.Context, cred llm.AuthCredential) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		var root map[string]json.RawMessage
		if err := json.Unmarshal(data, &root); err != nil {
			return err
		}
		providers := root
		wrapped := false
		if raw, ok := root["providers"]; ok {
			wrapped = true
			if err := json.Unmarshal(raw, &providers); err != nil {
				return fmt.Errorf("parse auth providers: %w", err)
			}
		}
		encoded, err := json.Marshal(authCredentialJSON(cred))
		if err != nil {
			return err
		}
		providers["openai-codex"] = encoded
		if wrapped {
			encoded, err = json.Marshal(providers)
			if err != nil {
				return err
			}
			root["providers"] = encoded
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		data, err = json.MarshalIndent(root, "", "  ")
		if err != nil {
			return err
		}
		data = append(data, '\n')
		return writeAuthFileAtomic(ctx, path, data)
	}
}

type authCredentialJSON llm.AuthCredential

func (c authCredentialJSON) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type      string `json:"type,omitempty"`
		Key       string `json:"key,omitempty"`
		Access    string `json:"access,omitempty"`
		Refresh   string `json:"refresh,omitempty"`
		Expires   int64  `json:"expires,omitempty"`
		AccountID string `json:"accountId,omitempty"`
		Model     string `json:"model,omitempty"`
		BaseURL   string `json:"base_url,omitempty"`
	}{c.Type, c.Key, c.Access, c.Refresh, c.Expires, c.AccountID, c.Model, c.BaseURL})
}

func writeAuthFileAtomic(ctx context.Context, path string, data []byte) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".llm-auth-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	mode := info.Mode().Perm() & 0o600
	if mode == 0 {
		mode = 0o600
	}
	if err := tmp.Chmod(mode); err != nil {
		tmp.Close()
		return err
	}
	if err := ctx.Err(); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	dir, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}
