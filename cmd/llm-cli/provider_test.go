package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

// clearProviderEnv blanks every ambient credential env var the providers
// read, so table rows exercise exactly the configured resolution path.
func clearProviderEnv(t *testing.T) {
	t.Helper()
	for _, key := range []string{"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"} {
		t.Setenv(key, "")
	}
}

// TestNewProviderSelectionAndCredentials tables provider selection ×
// credential resolution: explicit --api-key, env-var fallback, missing
// credentials, the codex --api-key-as-access-token overload, and the
// rejection rows. No live calls — only constructors run.
func TestNewProviderSelectionAndCredentials(t *testing.T) {
	tests := []struct {
		name     string
		cfg      providerConfig
		env      map[string]string
		wantName string
		wantErr  error
	}{
		{name: "anthropic api key", cfg: providerConfig{name: "anthropic", apiKey: "test-key"}, wantName: "anthropic"},
		{name: "anthropic env fallback", cfg: providerConfig{name: "anthropic"}, env: map[string]string{"ANTHROPIC_API_KEY": "env-key"}, wantName: "anthropic"},
		{name: "anthropic missing key", cfg: providerConfig{name: "anthropic"}, wantErr: llm.ErrAuth},
		{name: "openai api key", cfg: providerConfig{name: "openai", apiKey: "test-key"}, wantName: "openai"},
		{name: "openai env fallback", cfg: providerConfig{name: "openai"}, env: map[string]string{"OPENAI_API_KEY": "env-key"}, wantName: "openai"},
		{name: "openai missing key", cfg: providerConfig{name: "openai"}, wantErr: llm.ErrAuth},
		{name: "openrouter api key", cfg: providerConfig{name: "openrouter", apiKey: "test-key"}, wantName: "openrouter"},
		{name: "openrouter env fallback", cfg: providerConfig{name: "openrouter"}, env: map[string]string{"OPENROUTER_API_KEY": "env-key"}, wantName: "openrouter"},
		{name: "openrouter missing key", cfg: providerConfig{name: "openrouter"}, wantErr: llm.ErrAuth},
		// The --api-key flag doubles as the OAuth access token for the codex
		// subscription backend (flags.go documents the overload).
		{name: "codex api key as access token", cfg: providerConfig{name: "openai-codex", apiKey: "oauth-access-token"}, wantName: "openai-codex"},
		{name: "codex missing credential", cfg: providerConfig{name: "openai-codex"}, wantErr: llm.ErrAuth},
		{name: "zai deferred", cfg: providerConfig{name: "zai", apiKey: "key"}, wantErr: llm.ErrUnsupported},
		{name: "missing provider", cfg: providerConfig{}, wantErr: llm.ErrBadRequest},
		{name: "unknown provider", cfg: providerConfig{name: "eliza"}, wantErr: llm.ErrBadRequest},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clearProviderEnv(t)
			for key, value := range tt.env {
				t.Setenv(key, value)
			}
			p, err := newProvider(context.Background(), tt.cfg)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("newProvider error = %v, want %v", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("newProvider returned error: %v", err)
			}
			if p.Name() != tt.wantName {
				t.Fatalf("provider name = %q, want %q", p.Name(), tt.wantName)
			}
		})
	}
}

// TestNewProviderOptionWiring covers the non-credential knobs newProvider
// forwards to every constructor: base URL, timeout, and the debug logger.
func TestNewProviderOptionWiring(t *testing.T) {
	clearProviderEnv(t)
	var stderr bytes.Buffer
	for _, name := range []string{"anthropic", "openai", "openai-codex", "openrouter"} {
		t.Run(name, func(t *testing.T) {
			p, err := newProvider(context.Background(), providerConfig{
				name:    name,
				apiKey:  "test-key",
				baseURL: "https://config.invalid/v1",
				timeout: 5 * time.Second,
				debug:   true,
				stderr:  &stderr,
			})
			if err != nil {
				t.Fatalf("newProvider(%s) returned error: %v", name, err)
			}
			if p == nil || p.Name() != name {
				t.Fatalf("provider = %v", p)
			}
		})
	}
}

// TestNewProviderFromAuthFile drives the documented auth-file workflow with
// a fake temp credential file: parse a pi-compatible auth file with
// llm.LoadAuthFile, feed the openai-codex OAuth access token through the
// --api-key overload, and feed an API-key credential to a key provider.
func TestNewProviderFromAuthFile(t *testing.T) {
	clearProviderEnv(t)
	path := filepath.Join(t.TempDir(), "auth.json")
	payload := `{
		"providers": {
			"openai-codex": {"type": "oauth", "access": "fake-access-token", "refresh": "fake-refresh-token", "accountId": "acct_1"},
			"openrouter": {"type": "api", "key": "fake-openrouter-key"}
		}
	}`
	if err := os.WriteFile(path, []byte(payload), 0o600); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}

	auth, err := llm.LoadAuthFile(path)
	if err != nil {
		t.Fatalf("LoadAuthFile returned error: %v", err)
	}

	codexCred, ok := auth["openai-codex"]
	if !ok || codexCred.Access != "fake-access-token" || codexCred.Refresh != "fake-refresh-token" || codexCred.AccountID != "acct_1" {
		t.Fatalf("codex credential = %+v", codexCred)
	}
	p, err := newProvider(context.Background(), providerConfig{name: "openai-codex", apiKey: codexCred.Access})
	if err != nil {
		t.Fatalf("newProvider(openai-codex) returned error: %v", err)
	}
	if p.Name() != "openai-codex" {
		t.Fatalf("provider name = %q", p.Name())
	}

	routerCred, ok := auth["openrouter"]
	if !ok || routerCred.Key != "fake-openrouter-key" {
		t.Fatalf("openrouter credential = %+v", routerCred)
	}
	p, err = newProvider(context.Background(), providerConfig{name: "openrouter", apiKey: routerCred.Key})
	if err != nil {
		t.Fatalf("newProvider(openrouter) returned error: %v", err)
	}
	if p.Name() != "openrouter" {
		t.Fatalf("provider name = %q", p.Name())
	}
}

func TestDebugLogger(t *testing.T) {
	if logger := debugLogger(providerConfig{debug: false}); logger != nil {
		t.Fatalf("debugLogger without debug = %v, want nil", logger)
	}
	var buf bytes.Buffer
	logger := debugLogger(providerConfig{debug: true, stderr: &buf})
	if logger == nil {
		t.Fatalf("debugLogger with debug returned nil")
	}
	logger.Debug("wired")
	if !strings.Contains(buf.String(), "wired") {
		t.Fatalf("debug logger did not write to configured stderr: %q", buf.String())
	}
	// Nil stderr falls back to os.Stderr without panicking.
	if logger := debugLogger(providerConfig{debug: true}); logger == nil {
		t.Fatalf("debugLogger with nil stderr returned nil")
	}
}

func TestProviderConfigMapping(t *testing.T) {
	var stderr bytes.Buffer
	chat := chatConfig{provider: "anthropic", apiKey: "k", baseURL: "https://b", timeout: 3 * time.Second, debug: true}
	got := providerConfigFromChat(chat, &stderr)
	want := providerConfig{name: "anthropic", apiKey: "k", baseURL: "https://b", timeout: 3 * time.Second, debug: true, stderr: &stderr}
	if got != want {
		t.Fatalf("providerConfigFromChat = %+v, want %+v", got, want)
	}

	models := modelsConfig{provider: "openrouter", apiKey: "k2", baseURL: "https://m", timeout: time.Second, debug: false}
	gotModels := providerConfigFromModels(models, &stderr)
	wantModels := providerConfig{name: "openrouter", apiKey: "k2", baseURL: "https://m", timeout: time.Second, stderr: &stderr}
	if gotModels != wantModels {
		t.Fatalf("providerConfigFromModels = %+v, want %+v", gotModels, wantModels)
	}
}

func TestShouldReadStdin(t *testing.T) {
	// Non-file readers (piped test harnesses) always read.
	if !shouldReadStdin(strings.NewReader("piped")) {
		t.Fatalf("shouldReadStdin(non-file reader) = false, want true")
	}

	// A regular file is not a character device: read it.
	path := filepath.Join(t.TempDir(), "stdin.txt")
	if err := os.WriteFile(path, []byte("data"), 0o600); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("Open returned error: %v", err)
	}
	defer file.Close()
	if !shouldReadStdin(file) {
		t.Fatalf("shouldReadStdin(regular file) = false, want true")
	}

	// A character device (an interactive terminal) is not read.
	if tty, err := os.Open(os.DevNull); err == nil {
		defer tty.Close()
		if shouldReadStdin(tty) {
			t.Fatalf("shouldReadStdin(%s) = true, want false", os.DevNull)
		}
	}

	// Stat failure (closed file) means no stdin read.
	closed, err := os.Open(path)
	if err != nil {
		t.Fatalf("Open returned error: %v", err)
	}
	_ = closed.Close()
	if shouldReadStdin(closed) {
		t.Fatalf("shouldReadStdin(closed file) = true, want false")
	}
}
