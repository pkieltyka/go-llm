package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

// clearProviderEnv blanks every ambient credential env var the providers
// read, so table rows exercise exactly the configured resolution path.
func clearProviderEnv(t *testing.T) {
	t.Helper()
	for _, key := range []string{"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
		"TYPESAFE_API_KEY", "AI_GATEWAY_API_KEY", "OPENCODE_API_KEY", "CLOUDFLARE_API_KEY", "CLOUDFLARE_ACCOUNT_ID"} {
		t.Setenv(key, "")
	}
}

// TestNewProviderSelectionAndCredentials tables provider selection ×
// credential resolution: explicit --api-key, env-var fallback, missing
// credentials, and the rejection rows. No live calls — only constructors run.
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
		{name: "auth file and api key are exclusive", cfg: providerConfig{name: "openai", apiKey: "k", authFile: "auth.json"}, wantErr: llm.ErrBadRequest},
		{name: "removed openai-codex provider", cfg: providerConfig{name: "openai-codex", apiKey: "k"}, wantErr: llm.ErrBadRequest},
		{name: "vllm keyless", cfg: providerConfig{name: "vllm", baseURL: "http://localhost:8000/v1"}, wantName: "vllm"},
		{name: "vllm api key", cfg: providerConfig{name: "vllm", baseURL: "http://localhost:8000/v1", apiKey: "k"}, wantName: "vllm"},
		{name: "vllm ignores the environment", cfg: providerConfig{name: "vllm", baseURL: "http://localhost:8000/v1"}, env: map[string]string{"OPENAI_API_KEY": "env-key"}, wantName: "vllm"},
		{name: "vllm needs a base URL", cfg: providerConfig{name: "vllm"}, wantErr: llm.ErrBadRequest},
		{name: "vllm base URL needs a scheme", cfg: providerConfig{name: "vllm", baseURL: "gpu-box:8000/v1"}, wantErr: llm.ErrBadRequest},
		{name: "chatcompletions base URL needs a host", cfg: providerConfig{name: "chatcompletions", baseURL: "http:///v1"}, wantErr: llm.ErrBadRequest},
		{name: "chatcompletions keyless", cfg: providerConfig{name: "chatcompletions", baseURL: "http://localhost:11434/v1"}, wantName: "chatcompletions"},
		{name: "chatcompletions needs a base URL", cfg: providerConfig{name: "chatcompletions"}, wantErr: llm.ErrBadRequest},
		{name: "zai is not a provider", cfg: providerConfig{name: "zai", apiKey: "key"}, wantErr: llm.ErrBadRequest},
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
	for _, name := range []string{"anthropic", "openai", "openrouter"} {
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

// TestNewProviderFromAuthFile drives the documented auth-file workflow: a
// pi-compatible auth.json serves every provider, and openai sends a stored
// Sign in with ChatGPT access token as its bearer token.
func TestNewProviderFromAuthFile(t *testing.T) {
	clearProviderEnv(t)
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[]}`)
	}))
	defer server.Close()

	path := filepath.Join(t.TempDir(), "auth.json")
	expires := time.Now().Add(time.Hour).UnixMilli()
	payload := `{
		"anthropic": {"type": "api_key", "key": "fake-anthropic-key"},
		"openai": {"type": "oauth", "access": "fake-chatgpt-access", "refresh": "fake-refresh", "clientId": "client_1", "expires": ` + strconv.FormatInt(expires, 10) + `},
		"openrouter": {"type": "api_key", "key": "fake-openrouter-key"}
	}`
	if err := os.WriteFile(path, []byte(payload), 0o600); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	for _, name := range []string{"anthropic", "openai", "openrouter"} {
		p, err := newProvider(context.Background(), providerConfig{name: name, authFile: path})
		if err != nil || p.Name() != name {
			t.Fatalf("newProvider(%s) = %v, %v", name, p, err)
		}
	}

	p, err := newProvider(context.Background(), providerConfig{name: "openai", authFile: path, baseURL: server.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Models(context.Background()); err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	if gotAuth != "Bearer fake-chatgpt-access" {
		t.Fatalf("Authorization = %q, want the stored ChatGPT access token", gotAuth)
	}
}

func TestAuthFileIsOnlyReadWhenExplicit(t *testing.T) {
	clearProviderEnv(t)
	t.Setenv("OPENAI_API_KEY", "env-key")
	if _, err := newProvider(context.Background(), providerConfig{name: "openai"}); err != nil {
		t.Fatalf("env fallback without --auth-file returned error: %v", err)
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
	chat := chatConfig{provider: "anthropic", apiKey: "k", authFile: "/auth", baseURL: "https://b", timeout: 3 * time.Second, debug: true}
	got := providerConfigFromChat(chat, &stderr)
	want := providerConfig{name: "anthropic", apiKey: "k", authFile: "/auth", baseURL: "https://b", timeout: 3 * time.Second, debug: true, stderr: &stderr}
	if got != want {
		t.Fatalf("providerConfigFromChat = %+v, want %+v", got, want)
	}

	models := modelsConfig{provider: "openrouter", apiKey: "k2", authFile: "/models-auth", baseURL: "https://m", timeout: time.Second, debug: false}
	gotModels := providerConfigFromModels(models, &stderr)
	wantModels := providerConfig{name: "openrouter", apiKey: "k2", authFile: "/models-auth", baseURL: "https://m", timeout: time.Second, stderr: &stderr}
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

// TestSelfHostedAuthFileKey: with --auth-file a self-hosted server gets the
// "api_key" entry named after the provider, no key when the entry is absent,
// and a typed error for an entry of another kind.
func TestSelfHostedAuthFileKey(t *testing.T) {
	clearProviderEnv(t)
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[]}`)
	}))
	defer server.Close()
	path := filepath.Join(t.TempDir(), "auth.json")
	payload := `{
		"vllm": {"type": "api_key", "key": "fake-vllm-key"},
		"chatcompletions": {"type": "oauth", "access": "x", "refresh": "y"}
	}`
	if err := os.WriteFile(path, []byte(payload), 0o600); err != nil {
		t.Fatal(err)
	}
	models := func(name, authFile string) error {
		gotAuth = ""
		p, err := newProvider(context.Background(), providerConfig{name: name, authFile: authFile, baseURL: server.URL + "/v1"})
		if err != nil {
			return err
		}
		_, err = p.Models(context.Background())
		return err
	}
	if err := models("vllm", path); err != nil || gotAuth != "Bearer fake-vllm-key" {
		t.Fatalf("vllm with stored key: auth %q, err %v", gotAuth, err)
	}
	if err := models("chatcompletions", path); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("chatcompletions with an oauth entry = %v, want ErrAuth", err)
	}
	empty := filepath.Join(t.TempDir(), "empty.json")
	if err := os.WriteFile(empty, []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := models("vllm", empty); err != nil || gotAuth != "" {
		t.Fatalf("vllm without an entry: auth %q, err %v", gotAuth, err)
	}
}

// TestAuthFileKeyIndirection: pi's "$VAR" keys resolve; "!command" keys run
// only with --auth-file-commands.
func TestAuthFileKeyIndirection(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("sh-specific command")
	}
	clearProviderEnv(t)
	t.Setenv("CLI_TEST_VLLM_KEY", "env-indirect-key")
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[]}`)
	}))
	defer server.Close()
	write := func(key string) string {
		path := filepath.Join(t.TempDir(), "auth.json")
		if err := os.WriteFile(path, []byte(`{"vllm":{"type":"api_key","key":`+strconv.Quote(key)+`}}`), 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	models := func(path string, commands bool) error {
		gotAuth = ""
		p, err := newProvider(context.Background(), providerConfig{name: "vllm", authFile: path, authFileCommands: commands, baseURL: server.URL + "/v1"})
		if err != nil {
			return err
		}
		_, err = p.Models(context.Background())
		return err
	}
	if err := models(write("$CLI_TEST_VLLM_KEY"), false); err != nil || gotAuth != "Bearer env-indirect-key" {
		t.Fatalf("env key: auth %q, err %v", gotAuth, err)
	}
	command := write("!printf command-key")
	if err := models(command, false); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("command key without the flag = %v, want ErrAuth", err)
	}
	if err := models(command, true); err != nil || gotAuth != "Bearer command-key" {
		t.Fatalf("command key with the flag: auth %q, err %v", gotAuth, err)
	}
	if _, err := newProvider(context.Background(), providerConfig{name: "vllm", baseURL: server.URL, authFileCommands: true}); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("--auth-file-commands without --auth-file = %v", err)
	}
}
