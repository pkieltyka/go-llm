package authstore_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
)

func stores(t *testing.T) map[string]func() llm.CredentialStore {
	path := filepath.Join(t.TempDir(), "auth.json")
	return map[string]func() llm.CredentialStore{
		"memory": func() llm.CredentialStore { return authstore.NewMemory(nil) },
		"file":   func() llm.CredentialStore { return authstore.NewFile(path) },
	}
}

func TestStoreReadModifyDelete(t *testing.T) {
	ctx := context.Background()
	for name, newStore := range stores(t) {
		t.Run(name, func(t *testing.T) {
			store := newStore()
			if _, ok, err := store.Read(ctx, "openai"); ok || err != nil {
				t.Fatalf("empty Read = %v, %v", ok, err)
			}
			want := llm.AuthCredential{Type: "oauth", Access: "a", Refresh: "r", Expires: 42, ClientID: "c"}
			got, ok, err := store.Modify(ctx, "openai", func(current llm.AuthCredential, ok bool) (llm.AuthCredential, bool, error) {
				if ok {
					t.Error("fn saw a credential in an empty store")
				}
				return want, true, nil
			})
			if err != nil || !ok || !reflect.DeepEqual(got, want) {
				t.Fatalf("Modify = %+v, %v, %v", got, ok, err)
			}
			if read, ok, err := store.Read(ctx, "openai"); err != nil || !ok || !reflect.DeepEqual(read, want) {
				t.Fatalf("Read = %+v, %v, %v", read, ok, err)
			}
			unchanged, ok, err := store.Modify(ctx, "openai", func(current llm.AuthCredential, _ bool) (llm.AuthCredential, bool, error) {
				return llm.AuthCredential{Type: "oauth", Access: "ignored"}, false, nil
			})
			if err != nil || !ok || !reflect.DeepEqual(unchanged, want) {
				t.Fatalf("no-write Modify = %+v, %v, %v; want the current credential", unchanged, ok, err)
			}
			boom := errors.New("boom")
			if _, _, err := store.Modify(ctx, "openai", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
				return llm.AuthCredential{}, true, boom
			}); !errors.Is(err, boom) {
				t.Fatalf("fn error = %v", err)
			}
			if err := store.Delete(ctx, "openai"); err != nil {
				t.Fatal(err)
			}
			if _, ok, _ := store.Read(ctx, "openai"); ok {
				t.Fatal("credential survived Delete")
			}
		})
	}
}

func TestStoreModifySerializesConcurrentWriters(t *testing.T) {
	for name, newStore := range stores(t) {
		t.Run(name, func(t *testing.T) {
			// Memory writers share one store; File writers each get their
			// own value, modelling separate processes on one file.
			shared := newStore()
			writer := func() llm.CredentialStore {
				if name == "memory" {
					return shared
				}
				return newStore()
			}
			var wg sync.WaitGroup
			for range 20 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					_, _, err := writer().Modify(context.Background(), "counter", func(current llm.AuthCredential, _ bool) (llm.AuthCredential, bool, error) {
						n, _ := strconv.Atoi(current.Key)
						return llm.AuthCredential{Type: "api_key", Key: strconv.Itoa(n + 1)}, true, nil
					})
					if err != nil {
						t.Error(err)
					}
				}()
			}
			wg.Wait()
			cred, _, err := writer().Read(context.Background(), "counter")
			if err != nil || cred.Key != "20" {
				t.Fatalf("counter = %q, %v; want 20 (no lost updates)", cred.Key, err)
			}
		})
	}
}

func TestFilePreservesWrapperOtherEntriesAndUnknownFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gollm-test.json")
	initial := `{
  "meta": {"keep": true},
  "providers": {
    "openai": {"type": "oauth", "access": "old", "refresh": "r1", "clientId": "c1", "scopes": ["chatgpt.tokens.use.direct"], "model": "gpt-x"},
    "vllm": {"base_url": "http://box:8000/v1", "model": "qwen"}
  }
}`
	if err := os.WriteFile(path, []byte(initial), 0o600); err != nil {
		t.Fatal(err)
	}
	store := authstore.NewFile(path)
	if _, _, err := store.Modify(context.Background(), "openai", func(current llm.AuthCredential, _ bool) (llm.AuthCredential, bool, error) {
		if current.ClientID != "c1" || current.Model != "gpt-x" {
			t.Errorf("current = %+v", current)
		}
		current.Access, current.Refresh, current.Expires = "new", "r2", 99
		return current, true, nil
	}); err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Meta      map[string]bool                       `json:"meta"`
		Providers map[string]map[string]json.RawMessage `json:"providers"`
	}
	data, _ := os.ReadFile(path)
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatalf("rewritten file is not valid JSON: %v", err)
	}
	if !doc.Meta["keep"] {
		t.Fatal("top-level fields outside providers were dropped")
	}
	openai := doc.Providers["openai"]
	for key, want := range map[string]string{"access": `"new"`, "refresh": `"r2"`, "expires": `99`, "clientId": `"c1"`, "model": `"gpt-x"`, "scopes": `["chatgpt.tokens.use.direct"]`} {
		var compact bytes.Buffer
		_ = json.Compact(&compact, openai[key])
		if got := compact.String(); got != want {
			t.Errorf("openai.%s = %s, want %s", key, got, want)
		}
	}
	if string(doc.Providers["vllm"]["base_url"]) != `"http://box:8000/v1"` {
		t.Fatalf("unrelated entry changed: %s", doc.Providers["vllm"]["base_url"])
	}
	if runtime.GOOS != "windows" {
		if info, err := os.Stat(path); err != nil || info.Mode().Perm() != 0o600 {
			t.Fatalf("mode = %v, %v; want 0600", info.Mode().Perm(), err)
		}
	}
}

func TestFileTypeChangeDropsStaleCredentialFieldsButKeepsConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "auth.json")
	if err := os.WriteFile(path, []byte(`{"openai":{"type":"oauth","access":"a","refresh":"r","clientId":"c","scopes":["x"],"model":"m"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := authstore.NewFile(path).Modify(context.Background(), "openai", func(current llm.AuthCredential, _ bool) (llm.AuthCredential, bool, error) {
		return llm.AuthCredential{Type: "api_key", Key: "k"}, true, nil
	}); err != nil {
		t.Fatal(err)
	}
	var entry map[string]json.RawMessage
	data, _ := os.ReadFile(path)
	var root map[string]map[string]json.RawMessage
	_ = json.Unmarshal(data, &root)
	entry = root["openai"]
	for _, stale := range []string{"access", "refresh", "clientId", "scopes"} {
		if _, ok := entry[stale]; ok {
			t.Errorf("stale OAuth field %q survived a switch to api_key", stale)
		}
	}
	if string(entry["key"]) != `"k"` || string(entry["model"]) != `"m"` {
		t.Fatalf("entry = %s", data)
	}
}

func TestFileCreatesPrivateFileAndTreatsMissingAsEmpty(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "auth.json")
	store := authstore.NewFile(path)
	if _, ok, err := store.Read(context.Background(), "x"); ok || err != nil {
		t.Fatalf("missing file Read = %v, %v", ok, err)
	}
	if _, _, err := store.Modify(context.Background(), "anthropic", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
		return llm.AuthCredential{Type: "api_key", Key: "k"}, true, nil
	}); err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" {
		if info, err := os.Stat(path); err != nil || info.Mode().Perm() != 0o600 {
			t.Fatalf("new file mode = %v, %v", info.Mode().Perm(), err)
		}
	}
}

func TestFileParseErrorsDoNotEchoContent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "auth.json")
	if err := os.WriteFile(path, []byte(`{"openai": {"key": "sk-SECRET-VALUE",}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	_, _, err := authstore.NewFile(path).Read(context.Background(), "openai")
	if err == nil || strings.Contains(err.Error(), "SECRET") {
		t.Fatalf("parse error = %v, want an error without file content", err)
	}
}

func TestModifyHonorsContextWhileWaiting(t *testing.T) {
	store := authstore.NewMemory(nil)
	started, release := make(chan struct{}), make(chan struct{})
	go func() {
		_, _, _ = store.Modify(context.Background(), "p", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
			close(started)
			<-release
			return llm.AuthCredential{}, false, nil
		})
	}()
	<-started
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := store.Modify(ctx, "p", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
		t.Error("fn must not run after the wait was cancelled")
		return llm.AuthCredential{}, false, nil
	}); !errors.Is(err, context.Canceled) {
		t.Fatalf("waiting Modify = %v, want context.Canceled", err)
	}
	close(release)
}

func TestFileEdgeCases(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	bom := filepath.Join(dir, "bom.json")
	if err := os.WriteFile(bom, []byte("\xef\xbb\xbf{\"openai\":{\"type\":\"api_key\",\"key\":\"k\"}}"), 0o600); err != nil {
		t.Fatal(err)
	}
	if cred, ok, err := authstore.NewFile(bom).Read(ctx, "openai"); err != nil || !ok || cred.Key != "k" {
		t.Fatalf("BOM file Read = %+v, %v, %v", cred, ok, err)
	}
	empty := filepath.Join(dir, "empty.json")
	if err := os.WriteFile(empty, []byte("  \n"), 0o600); err != nil {
		t.Fatal(err)
	}
	store := authstore.NewFile(empty)
	if _, ok, err := store.Read(ctx, "x"); ok || err != nil {
		t.Fatalf("whitespace file Read = %v, %v", ok, err)
	}
	if err := store.Delete(ctx, "absent"); err != nil {
		t.Fatalf("Delete of an absent entry = %v", err)
	}
	if store.Path() != empty || !strings.Contains(store.String(), "redacted") || !strings.Contains(authstore.NewMemory(nil).String(), "redacted") {
		t.Fatal("store formatting must be redacted")
	}
	if _, _, err := store.Modify(ctx, "x", nil); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("nil fn = %v", err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if _, _, err := store.Read(cancelled, "x"); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled Read = %v", err)
	}
	badEntry := filepath.Join(dir, "bad-entry.json")
	if err := os.WriteFile(badEntry, []byte(`{"openai":{"expires":"soon"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := authstore.NewFile(badEntry).Modify(ctx, "openai", func(c llm.AuthCredential, ok bool) (llm.AuthCredential, bool, error) {
		return c, false, nil
	}); err == nil {
		t.Fatal("an unparsable entry must fail Modify")
	}
}

func TestFileModifyHonorsContextWhileAnotherHolderHasTheLock(t *testing.T) {
	path := filepath.Join(t.TempDir(), "auth.json")
	holder := authstore.NewFile(path)
	started, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		_, _, _ = holder.Modify(context.Background(), "p", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
			close(started)
			<-release
			return llm.AuthCredential{}, false, nil
		})
	}()
	<-started
	defer func() { close(release); <-done }()
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	// A second File value contends on the OS lock, as another process would.
	if _, _, err := authstore.NewFile(path).Modify(ctx, "p", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
		return llm.AuthCredential{}, false, nil
	}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("contended Modify = %v, want DeadlineExceeded", err)
	}
}

// Formatting the public store handle never reveals credential material,
// whatever the verb (%#v would otherwise walk the private map).
func TestMemoryFormattingRedactsCredentials(t *testing.T) {
	canaries := llm.AuthCredential{
		Type: "oauth", Key: "canary-key", Access: "canary-access", Refresh: "canary-refresh",
		AccountID: "canary-account", ClientID: "canary-client",
	}
	store := authstore.NewMemory(llm.AuthFile{"openai": canaries})
	for _, verb := range []string{"%v", "%+v", "%#v", "%s"} {
		out := fmt.Sprintf(verb, store)
		for _, secret := range []string{"canary-key", "canary-access", "canary-refresh", "canary-account", "canary-client"} {
			if strings.Contains(out, secret) {
				t.Fatalf("%s formatting leaked %s: %s", verb, secret, out)
			}
		}
	}
}

// An already-cancelled context never starts fn or deletes: select between a
// free lock and ctx.Done would otherwise pick either at random.
func TestCancelledContextNeverModifies(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for name, newStore := range stores(t) {
		t.Run(name, func(t *testing.T) {
			store := newStore()
			if _, _, err := store.Modify(context.Background(), "p", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
				return llm.AuthCredential{Type: "api_key", Key: "k"}, true, nil
			}); err != nil {
				t.Fatal(err)
			}
			for range 200 {
				_, _, err := store.Modify(ctx, "p", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
					t.Fatal("fn ran under a cancelled context")
					return llm.AuthCredential{}, false, nil
				})
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("Modify = %v, want context.Canceled", err)
				}
				if err := store.Delete(ctx, "p"); !errors.Is(err, context.Canceled) {
					t.Fatalf("Delete = %v, want context.Canceled", err)
				}
			}
			if _, ok, err := store.Read(context.Background(), "p"); !ok || err != nil {
				t.Fatalf("credential gone after cancelled calls: %v %v", ok, err)
			}
		})
	}
}

// A nil context means no deadline, as for Memory: File never panics on it.
func TestNilContextIsBackground(t *testing.T) {
	for name, newStore := range stores(t) {
		t.Run(name, func(t *testing.T) {
			store := newStore()
			if _, _, err := store.Modify(nil, "p", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) { //nolint:staticcheck
				return llm.AuthCredential{Type: "api_key", Key: "k"}, true, nil
			}); err != nil {
				t.Fatal(err)
			}
			if err := store.Delete(nil, "p"); err != nil { //nolint:staticcheck
				t.Fatal(err)
			}
		})
	}
}

// TestCredentialEnvRoundTrip: pi's per-entry env (Cloudflare's account ID)
// survives a write and a read, and Memory never aliases the map.
func TestCredentialEnvRoundTrip(t *testing.T) {
	ctx := context.Background()
	cred := llm.AuthCredential{Type: "api_key", Key: "cf-key", Env: map[string]string{"CLOUDFLARE_ACCOUNT_ID": "acct-1"}}
	file := authstore.NewFile(filepath.Join(t.TempDir(), "auth.json"))
	if _, _, err := file.Modify(ctx, "cloudflare", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) { return cred, true, nil }); err != nil {
		t.Fatal(err)
	}
	read, ok, err := file.Read(ctx, "cloudflare")
	if err != nil || !ok || read.Env["CLOUDFLARE_ACCOUNT_ID"] != "acct-1" {
		t.Fatalf("file read = %v %v env %v", ok, err, read.Env)
	}
	memory := authstore.NewMemory(llm.AuthFile{"cloudflare": cred})
	cred.Env["CLOUDFLARE_ACCOUNT_ID"] = "mutated"
	got, _, _ := memory.Read(ctx, "cloudflare")
	got.Env["CLOUDFLARE_ACCOUNT_ID"] = "mutated again"
	again, _, _ := memory.Read(ctx, "cloudflare")
	if again.Env["CLOUDFLARE_ACCOUNT_ID"] != "acct-1" {
		t.Fatalf("memory env aliased: %v", again.Env)
	}
}
