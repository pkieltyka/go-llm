package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
)

// fakeLoginFlow completes with a fixed credential, either at once or once
// a pasted callback was submitted.
type fakeLoginFlow struct {
	waitForSubmit bool
	submitted     chan string
	once          sync.Once
	cancelled     bool
}

func (f *fakeLoginFlow) Begin(context.Context) (llm.LoginAuthorization, error) {
	return llm.NewLoginAuthorization("http://127.0.0.1:1455/authorize", "sign in")
}

func (f *fakeLoginFlow) Complete(ctx context.Context) (llm.AuthCredential, error) {
	if f.waitForSubmit {
		select {
		case <-f.submitted:
		case <-ctx.Done():
			return llm.AuthCredential{}, ctx.Err()
		}
	}
	return llm.AuthCredential{Type: "oauth", Access: "fake-access", Refresh: "fake-refresh", ClientID: "client_1",
		Expires: time.Now().Add(time.Hour).UnixMilli()}, nil
}

func (f *fakeLoginFlow) Submit(_ context.Context, response string) error {
	if !strings.HasPrefix(response, "http://") {
		return errors.New("not a callback URL")
	}
	f.once.Do(func() { f.submitted <- response })
	return nil
}

func (f *fakeLoginFlow) Cancel() { f.cancelled = true }

func TestLoginStoresCredential(t *testing.T) {
	configDir := t.TempDir()
	authPath := filepath.Join(t.TempDir(), "auth.json")
	if err := os.WriteFile(authPath, []byte(`{"anthropic":{"type":"api_key","key":"keep-me"},"openai":{"type":"api_key","key":"old","model":"gpt-5.5"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var deviceIDs []string
	flow := &fakeLoginFlow{waitForSubmit: true, submitted: make(chan string, 1)}
	var stderr bytes.Buffer
	a := app{
		stdin:  strings.NewReader("\nnot-a-url\nhttp://localhost:1455/auth/callback?code=x\n"),
		stdout: &bytes.Buffer{}, stderr: &stderr,
		configDir: func() (string, error) { return configDir, nil },
		loginFlowFactory: func(deviceID string, _ time.Duration) (llm.LoginFlow, error) {
			deviceIDs = append(deviceIDs, deviceID)
			return flow, nil
		},
	}
	if err := a.run(context.Background(), []string{"login", "openai", "--auth-file", authPath}); err != nil {
		t.Fatalf("login = %v\n%s", err, stderr.String())
	}
	if !flow.cancelled {
		t.Fatal("flow not released")
	}
	if strings.Contains(stderr.String(), "fake-access") || !strings.Contains(stderr.String(), "http://127.0.0.1:1455/authorize") {
		t.Fatalf("stderr = %q", stderr.String())
	}
	store := authstore.NewFile(authPath)
	cred, ok, err := store.Read(context.Background(), "openai")
	if err != nil || !ok || cred.Type != "oauth" || cred.Access != "fake-access" || cred.Model != "gpt-5.5" {
		t.Fatalf("stored openai = %v %v (model %q)", ok, err, cred.Model)
	}
	if other, ok, err := store.Read(context.Background(), "anthropic"); err != nil || !ok || other.Key != "keep-me" {
		t.Fatalf("other entry = %v %v", ok, err)
	}

	// The device ID is created once and reused.
	flow2 := &fakeLoginFlow{}
	a.stdin = nil
	a.loginFlowFactory = func(deviceID string, _ time.Duration) (llm.LoginFlow, error) {
		deviceIDs = append(deviceIDs, deviceID)
		return flow2, nil
	}
	if err := a.run(context.Background(), []string{"login", "openai", "--auth-file", authPath}); err != nil {
		t.Fatal(err)
	}
	if len(deviceIDs) != 2 || deviceIDs[0] != deviceIDs[1] || !uuidPattern.MatchString(deviceIDs[0]) {
		t.Fatalf("device IDs = %v", deviceIDs)
	}
	info, err := os.Stat(filepath.Join(configDir, "llm-cli", "device-id"))
	if err != nil || (runtime.GOOS != "windows" && info.Mode().Perm() != 0o600) {
		t.Fatalf("device-id file = %v, %v", info, err)
	}

	// An explicit --device-id is used as given and never stored.
	var explicit string
	a.loginFlowFactory = func(deviceID string, _ time.Duration) (llm.LoginFlow, error) {
		explicit = deviceID
		return &fakeLoginFlow{}, nil
	}
	const given = "6f1e5a0e-2b7c-4c1d-9a3e-0b8f7d6c5e4a"
	if err := a.run(context.Background(), []string{"login", "openai", "--auth-file", authPath, "--device-id", given}); err != nil || explicit != given {
		t.Fatalf("--device-id = %q, %v", explicit, err)
	}
}

func TestLoginRejectsBadInput(t *testing.T) {
	a := app{stdout: &bytes.Buffer{}, stderr: &bytes.Buffer{},
		configDir: func() (string, error) { return t.TempDir(), nil },
		loginFlowFactory: func(string, time.Duration) (llm.LoginFlow, error) {
			t.Fatal("flow built for invalid input")
			return nil, nil
		},
	}
	notJSON := filepath.Join(t.TempDir(), "auth.json")
	if err := os.WriteFile(notJSON, []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	// A bad auth file fails before any sign-in starts.
	if err := a.run(context.Background(), []string{"login", "openai", "--auth-file", notJSON}); err == nil || !strings.Contains(err.Error(), "auth file") {
		t.Fatalf("unparsable auth file = %v", err)
	}
	for _, args := range [][]string{
		{"login", "openai", "--auth-file", "a.json", "extra"},
		{"logout", "openai", "extra"},
		{"auth", "extra"},
	} {
		if err := a.run(context.Background(), args); !errors.Is(err, errUsage) {
			t.Fatalf("%v = %v, want a usage error", args, err)
		}
	}
	for _, args := range [][]string{
		{"login"},
		{"login", "nope", "--auth-file", "a.json"},
		{"login", "anthropic", "--auth", "oauth", "--auth-file", "a.json"},
		{"login", "openai", "--auth", "password", "--auth-file", "a.json"},
		{"login", "openai", "--auth-file", "a.json", "--timeout", "-1s"},
		{"logout"},
	} {
		if err := a.run(context.Background(), args); !errors.Is(err, llm.ErrBadRequest) {
			t.Fatalf("%v = %v, want ErrBadRequest", args, err)
		}
	}
	bad := t.TempDir()
	if err := os.MkdirAll(filepath.Join(bad, "llm-cli"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bad, "llm-cli", "device-id"), []byte("nope"), 0o600); err != nil {
		t.Fatal(err)
	}
	a.configDir = func() (string, error) { return bad, nil }
	if err := a.run(context.Background(), []string{"login", "openai", "--auth-file", filepath.Join(t.TempDir(), "a.json")}); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("corrupt device id = %v", err)
	}
}

func TestNewUUID(t *testing.T) {
	id, err := newUUID()
	if err != nil || !uuidPattern.MatchString(id) || id[14] != '4' {
		t.Fatalf("newUUID = %q, %v", id, err)
	}
}

// TestLoginFlowErrors: a flow that cannot be built, begun or completed
// stores nothing.
func TestLoginFlowErrors(t *testing.T) {
	authPath := filepath.Join(t.TempDir(), "auth.json")
	base := app{stdout: &bytes.Buffer{}, stderr: &bytes.Buffer{},
		configDir: func() (string, error) { return t.TempDir(), nil }}
	for name, factory := range map[string]loginFlowFactory{
		"build":    func(string, time.Duration) (llm.LoginFlow, error) { return nil, llm.ErrServer },
		"begin":    func(string, time.Duration) (llm.LoginFlow, error) { return failingFlow{begin: true}, nil },
		"complete": func(string, time.Duration) (llm.LoginFlow, error) { return failingFlow{}, nil },
	} {
		a := base
		a.loginFlowFactory = factory
		if err := a.run(context.Background(), []string{"login", "openai", "--auth-file", authPath}); !errors.Is(err, llm.ErrServer) {
			t.Fatalf("%s failure = %v", name, err)
		}
	}
	if _, ok, _ := authstore.NewFile(authPath).Read(context.Background(), "openai"); ok {
		t.Fatal("a failed login stored a credential")
	}
	failingConfig := base
	failingConfig.configDir = func() (string, error) { return "", errors.New("no home") }
	if err := failingConfig.run(context.Background(), []string{"login", "openai", "--auth-file", authPath}); err == nil || !strings.Contains(err.Error(), "--device-id") {
		t.Fatalf("missing config dir = %v", err)
	}
}

type failingFlow struct{ begin bool }

func (f failingFlow) Begin(context.Context) (llm.LoginAuthorization, error) {
	if f.begin {
		return llm.LoginAuthorization{}, llm.ErrServer
	}
	return llm.NewLoginAuthorization("http://127.0.0.1:1455/authorize", "")
}
func (failingFlow) Complete(context.Context) (llm.AuthCredential, error) {
	return llm.AuthCredential{}, llm.ErrServer
}
func (failingFlow) Submit(context.Context, string) error { return nil }
func (failingFlow) Cancel()                              {}

// TestLoginAPIKey: providers without OAuth default to an API key read from
// stdin; openai takes one with --auth api-key or api_key; the value is stored
// as pi stores it, and a reference stays a reference.
func TestLoginAPIKey(t *testing.T) {
	authPath := filepath.Join(t.TempDir(), "nested", "auth.json") // directory created on demand
	run := func(stdin string, args ...string) (string, error) {
		var stderr bytes.Buffer
		a := app{stdin: strings.NewReader(stdin), stdout: &bytes.Buffer{}, stderr: &stderr,
			configDir: func() (string, error) { return t.TempDir(), nil },
			loginFlowFactory: func(string, time.Duration) (llm.LoginFlow, error) {
				t.Fatal("API-key login started a browser flow")
				return nil, nil
			}}
		err := a.run(context.Background(), append(args, "--auth-file", authPath))
		return stderr.String(), err
	}
	out, err := run("sk-or-fake\n", "login", "-p", "openrouter") // -p works as an alias

	if err != nil || !strings.Contains(out, "Saved API key for OpenRouter. Credentials saved to "+authPath) || strings.Contains(out, "sk-or-fake") {
		t.Fatalf("openrouter login = %q, %v", out, err)
	}
	for _, spelling := range []string{"api-key", "api_key", "API-KEY"} {
		if _, err := run("$OPENAI_API_KEY\n", "login", "openai", "--auth", spelling); err != nil {
			t.Fatalf("--auth %s = %v", spelling, err)
		}
	}
	if _, err := run("", "login", "anthropic"); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("empty key = %v", err)
	}
	auth, err := llm.LoadAuthFile(authPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := auth["openrouter"]; got.Type != "api_key" || got.Key != "sk-or-fake" {
		t.Fatalf("openrouter entry = %q", got.Type)
	}
	if got := auth["openai"]; got.Type != "api_key" || got.Key != "$OPENAI_API_KEY" {
		t.Fatalf("openai entry kept a reference? type %q", got.Type)
	}
	if _, ok := auth["anthropic"]; ok {
		t.Fatal("a failed login stored a credential")
	}
}

// TestLogoutAndAuthStatus covers pi's /logout wording and the status table.
func TestLogoutAndAuthStatus(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "env-key")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENROUTER_API_KEY", "")
	authPath := filepath.Join(t.TempDir(), "auth.json")
	payload := `{"openai":{"type":"oauth","access":"a","refresh":"r","expires":1},"openrouter":{"type":"api_key","key":"$OR_KEY"},"vllm":{"type":"api_key","key":"!op read x"},"chatcompletions":{"type":"oauth","access":"a"}}`
	if err := os.WriteFile(authPath, []byte(payload), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	a := app{stdout: &stdout, stderr: &stderr}
	if err := a.run(context.Background(), []string{"auth", "--auth-file", authPath}); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"Auth file: " + authPath,
		"openai", "subscription (default), API key", "✓ subscription configured",
		"✓ env: ANTHROPIC_API_KEY",
		"✓ API key configured (reference)",
		"✓ API key configured (command)",
		"✗ subscription stored, which Chat Completions does not accept here",
	} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("auth status missing %q:\n%s", want, stdout.String())
		}
	}
	for _, tc := range []struct{ provider, want string }{
		{"openai", "Logged out of OpenAI"},
		{"openrouter", "Removed stored API key for OpenRouter. Environment variables are unchanged."},
		{"openrouter", "No stored credentials for OpenRouter"},
	} {
		stderr.Reset()
		if err := a.run(context.Background(), []string{"logout", tc.provider, "--auth-file", authPath}); err != nil || !strings.Contains(stderr.String(), tc.want) {
			t.Fatalf("logout %s = %q, %v", tc.provider, stderr.String(), err)
		}
	}
	auth, err := llm.LoadAuthFile(authPath)
	if err != nil || len(auth) != 2 || auth["vllm"].Type != "api_key" {
		t.Fatalf("after logout = %d entries, %v", len(auth), err)
	}
}

// TestDefaultAuthFile: without --auth-file, commands use $LLM_AUTH_FILE,
// else llm-cli/auth.json in the user config directory; --api-key bypasses it.
func TestDefaultAuthFile(t *testing.T) {
	configDir := t.TempDir()
	a := app{configDir: func() (string, error) { return configDir, nil }}
	t.Setenv("LLM_AUTH_FILE", "")
	if cfg, err := a.withAuthFile(providerConfig{name: "openai"}); err != nil || cfg.authFile != filepath.Join(configDir, "llm-cli", "auth.json") {
		t.Fatalf("default = %q, %v", cfg.authFile, err)
	}
	t.Setenv("LLM_AUTH_FILE", "/elsewhere/auth.json")
	if cfg, _ := a.withAuthFile(providerConfig{name: "openai"}); cfg.authFile != "/elsewhere/auth.json" {
		t.Fatalf("env = %q", cfg.authFile)
	}
	if cfg, _ := a.withAuthFile(providerConfig{name: "openai", authFile: "flag.json"}); cfg.authFile != "flag.json" {
		t.Fatalf("flag = %q", cfg.authFile)
	}
	if cfg, _ := a.withAuthFile(providerConfig{name: "openai", apiKey: "k"}); cfg.authFile != "" {
		t.Fatalf("--api-key kept an auth file %q", cfg.authFile)
	}
}

// TestLoginHint: an authentication failure without --api-key says how to
// fix it; other errors and --api-key runs are unchanged.
func TestLoginHint(t *testing.T) {
	authErr := fmt.Errorf("%w: no openai API key is stored or configured", llm.ErrAuth)
	err := withLoginHint("openai", "", authErr)
	if !errors.Is(err, llm.ErrAuth) || !strings.Contains(err.Error(), "run 'llm-cli login openai' or set OPENAI_API_KEY") {
		t.Fatalf("hint = %v", err)
	}
	if err := withLoginHint("vllm", "", authErr); !strings.Contains(err.Error(), "run 'llm-cli login vllm'") || strings.Contains(err.Error(), " or set ") {
		t.Fatalf("self-hosted hint = %v", err)
	}
	if err := withLoginHint("openai", "k", authErr); err != authErr {
		t.Fatalf("--api-key run rewrapped: %v", err)
	}
	other := errors.New("boom")
	if err := withLoginHint("openai", "", other); err != other {
		t.Fatalf("non-auth error rewrapped: %v", err)
	}
}
