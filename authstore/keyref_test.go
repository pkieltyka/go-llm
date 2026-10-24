package authstore

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

func writeAuth(t *testing.T, payload string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "auth.json")
	if err := os.WriteFile(path, []byte(payload), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestResolveKeyTemplates pins pi's template rules (resolve-config-value.ts).
func TestResolveKeyTemplates(t *testing.T) {
	env := map[string]string{"A": "alpha", "B_2": "beta", "EMPTY": ""}
	f := &File{}
	f.lookupEnv = func(name string) (string, bool) { value, ok := env[name]; return value, ok }
	for _, tc := range []struct{ in, want string }{
		{"plain-key", "plain-key"},
		{"$A", "alpha"},
		{"${A}", "alpha"},
		{"pre-${A}-$B_2-post", "pre-alpha-beta-post"},
		{"$A$B_2", "alphabeta"},
		{"cost$$5", "cost$5"},
		{"$!bang", "!bang"},
		{"${not-a-name}", "${not-a-name}"},
		{"${A", "${A"},
		{"end$", "end$"},
		{"$1", "$1"},
		{"a $ b", "a $ b"},
	} {
		got, err := f.resolveKey(context.Background(), "p", tc.in, nil)
		if err != nil || got != tc.want {
			t.Errorf("resolveKey(%q) = %q, %v; want %q", tc.in, got, err, tc.want)
		}
	}
	// The entry's env map wins over the process environment.
	if got, err := f.resolveKey(context.Background(), "p", "$A", map[string]string{"A": "override"}); err != nil || got != "override" {
		t.Fatalf("env map = %q, %v", got, err)
	}
	// Unset and empty variables are missing; each is named once; nothing
	// resolved is echoed.
	_, err := f.resolveKey(context.Background(), "p", "$A-$MISSING-${MISSING}-$EMPTY", nil)
	if !errors.Is(err, llm.ErrAuth) || !strings.Contains(err.Error(), "MISSING, EMPTY") || strings.Contains(err.Error(), "alpha") {
		t.Fatalf("missing = %v", err)
	}
}

func TestResolveKeyCommands(t *testing.T) {
	ctx := context.Background()
	f := &File{}
	if _, err := f.resolveKey(ctx, "openai", "!op read op://v/k", nil); !errors.Is(err, llm.ErrAuth) || !strings.Contains(err.Error(), "WithKeyCommands") {
		t.Fatalf("command without opt-in = %v", err)
	}
	runs := 0
	WithKeyCommands()(f)
	f.runCommand = func(_ context.Context, command string) (string, error) {
		runs++
		switch command {
		case "good":
			return "from-command", nil
		case "empty":
			return "", nil
		}
		return "", errors.New("exit status 1")
	}
	for i := 0; i < 2; i++ {
		if got, err := f.resolveKey(ctx, "openai", "!good", nil); err != nil || got != "from-command" {
			t.Fatalf("command = %q, %v", got, err)
		}
	}
	if runs != 1 {
		t.Fatalf("successful command ran %d times; want cached", runs)
	}
	if _, err := f.resolveKey(ctx, "openai", "!empty", nil); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("empty output = %v", err)
	}
	if _, err := f.resolveKey(ctx, "openai", "!bad", nil); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("failing command = %v", err)
	}
	if _, err := f.resolveKey(ctx, "openai", "!bad", nil); runs != 4 || err == nil {
		t.Fatalf("a failure was cached (runs %d)", runs)
	}
}

func TestRunKeyCommandShell(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("sh-specific command")
	}
	got, err := runKeyCommand(context.Background(), `printf '  shell-secret\n' && echo noise >&2`)
	if err != nil || got != "shell-secret" {
		t.Fatalf("runKeyCommand = %q, %v", got, err)
	}
	if _, err := runKeyCommand(context.Background(), "echo leaked; exit 3"); err == nil || strings.Contains(err.Error(), "leaked") {
		t.Fatalf("failing command error = %v; must not include output", err)
	}
}

// TestFileReadResolvesOnlyAPIKeys: Read resolves api_key indirection; OAuth
// fields stay literal; Modify sees and keeps the raw value, so a resolved
// secret is never written back.
func TestFileReadResolvesOnlyAPIKeys(t *testing.T) {
	ctx := context.Background()
	t.Setenv("KEYREF_TEST_OPENROUTER", "resolved-secret")
	path := writeAuth(t, `{
		"openrouter": {"type": "api_key", "key": "$KEYREF_TEST_OPENROUTER", "scopes": ["x"]},
		"vllm": {"type": "api_key", "key": "${TOKEN}", "env": {"TOKEN": "from-entry-env"}},
		"openai": {"type": "oauth", "access": "$NOT_RESOLVED", "refresh": "r", "expires": 1}
	}`)
	store := NewFile(path)
	cred, ok, err := store.Read(ctx, "openrouter")
	if err != nil || !ok || cred.Key != "resolved-secret" {
		t.Fatalf("openrouter = %q %v %v", cred.Key, ok, err)
	}
	if cred, _, err := store.Read(ctx, "vllm"); err != nil || cred.Key != "from-entry-env" {
		t.Fatalf("vllm = %q %v", cred.Key, err)
	}
	if cred, _, err := store.Read(ctx, "openai"); err != nil || cred.Access != "$NOT_RESOLVED" {
		t.Fatalf("oauth = %q %v", cred.Access, err)
	}
	var seen string
	if _, _, err := store.Modify(ctx, "openrouter", func(current llm.AuthCredential, _ bool) (llm.AuthCredential, bool, error) {
		seen = current.Key
		current.Model = "openai/gpt-5.5"
		return current, true, nil
	}); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if seen != "$KEYREF_TEST_OPENROUTER" || strings.Contains(string(raw), "resolved-secret") || !strings.Contains(string(raw), `"scopes"`) {
		t.Fatalf("Modify saw %q; file = %s", seen, raw)
	}
	if _, _, err := NewFile(writeAuth(t, `{"x":{"type":"api_key","key":"!true"}}`)).Read(ctx, "x"); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("command without opt-in via Read = %v", err)
	}
}

// TestKeyCommandSingleFlight: concurrent first reads run a key command once.
func TestKeyCommandSingleFlight(t *testing.T) {
	f := &File{}
	WithKeyCommands()(f)
	release := make(chan struct{})
	var runs atomic.Int32
	f.runCommand = func(context.Context, string) (string, error) {
		runs.Add(1)
		<-release
		return "shared", nil
	}
	var wg sync.WaitGroup
	results := make(chan string, 8)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			value, err := f.resolveKey(context.Background(), "p", "!cmd", nil)
			if err != nil {
				t.Error(err)
			}
			results <- value
		}()
	}
	time.Sleep(20 * time.Millisecond)
	close(release)
	wg.Wait()
	close(results)
	for value := range results {
		if value != "shared" {
			t.Fatalf("value = %q", value)
		}
	}
	if runs.Load() != 1 {
		t.Fatalf("command ran %d times; want 1", runs.Load())
	}
}

func TestMalformedEntryEnv(t *testing.T) {
	path := writeAuth(t, `{"x":{"type":"api_key","key":"$A","env":{"A":1}}}`)
	// pi's loader rejects such an entry; so does the credential parser.
	// encoding/json names the field as "env" or "env.A" depending on the Go
	// release.
	if _, _, err := NewFile(path).Read(context.Background(), "x"); err == nil || !strings.Contains(err.Error(), `"env`) {
		t.Fatalf("malformed env = %v", err)
	}
}
