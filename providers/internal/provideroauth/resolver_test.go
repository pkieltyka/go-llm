package provideroauth

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
)

const testProvider = "test-provider"

func newTestResolver(t *testing.T, cred llm.AuthCredential, refresh RefreshFunc) *Resolver {
	t.Helper()
	return &Resolver{Store: authstore.NewMemory(llm.AuthFile{testProvider: cred}), ProviderID: testProvider, Refresh: refresh}
}

func TestResolverRefreshesExpiringCredentialOnceUnderConcurrency(t *testing.T) {
	now := time.Now()
	store := authstore.NewMemory(llm.AuthFile{testProvider: {Type: "oauth", Access: "old", Refresh: "r1", ClientID: "client", Expires: now.Add(time.Minute).UnixMilli()}})
	var calls atomic.Int64
	resolver := &Resolver{Store: store, ProviderID: testProvider, Refresh: func(_ context.Context, cred llm.AuthCredential) (llm.AuthCredential, error) {
		calls.Add(1)
		time.Sleep(10 * time.Millisecond)
		return llm.AuthCredential{Access: "new", Refresh: "r2", Expires: now.Add(time.Hour).UnixMilli()}, nil
	}}
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			cred, err := resolver.Credential(context.Background())
			if err != nil || cred.Access != "new" {
				t.Errorf("Credential = %+v, %v", cred, err)
			}
		}()
	}
	wg.Wait()
	if calls.Load() != 1 {
		t.Fatalf("refresh calls = %d, want exactly 1", calls.Load())
	}
	stored, _, _ := store.Read(context.Background(), testProvider)
	if stored.Access != "new" || stored.Refresh != "r2" || stored.ClientID != "client" || stored.Type != "oauth" {
		t.Fatalf("stored = %+v, want renewed tokens with identity fields kept", stored)
	}
}

func TestResolverUsesValidCredentialWithoutRefresh(t *testing.T) {
	resolver := newTestResolver(t, llm.AuthCredential{Type: "oauth", Access: "ok", Expires: time.Now().Add(time.Hour).UnixMilli()},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			t.Fatal("refresh must not run for a valid credential")
			return llm.AuthCredential{}, nil
		})
	if cred, err := resolver.Credential(context.Background()); err != nil || cred.Access != "ok" {
		t.Fatalf("Credential = %+v, %v", cred, err)
	}
}

func TestResolverErrors(t *testing.T) {
	ctx := context.Background()
	expired := llm.AuthCredential{Type: "oauth", Access: "old", Expires: time.Now().Add(-time.Minute).UnixMilli()}
	if _, err := newTestResolver(t, expired, nil).Credential(ctx); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("expired without refresh token = %v, want ErrAuth", err)
	}
	missing := &Resolver{Store: authstore.NewMemory(nil), ProviderID: testProvider}
	if _, err := missing.Credential(ctx); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("missing credential = %v, want ErrAuth", err)
	}
	apiKey := &Resolver{Store: authstore.NewMemory(llm.AuthFile{testProvider: {Type: "api_key", Key: "k"}}), ProviderID: testProvider}
	if _, err := apiKey.Credential(ctx); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("api_key credential = %v, want ErrAuth (not OAuth)", err)
	}
	failing := newTestResolver(t, llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "r", Expires: 1},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			return llm.AuthCredential{}, llm.ErrServer
		})
	if _, err := failing.Credential(ctx); !errors.Is(err, llm.ErrServer) {
		t.Fatalf("refresh failure = %v, want the refresh error", err)
	}
}

func TestResolverStartedRefreshSurvivesCallerCancellation(t *testing.T) {
	store := authstore.NewMemory(llm.AuthFile{testProvider: {Type: "oauth", Access: "old", Refresh: "r1", Expires: 1}})
	ctx, cancel := context.WithCancel(context.Background())
	resolver := &Resolver{Store: store, ProviderID: testProvider, Refresh: func(refreshCtx context.Context, _ llm.AuthCredential) (llm.AuthCredential, error) {
		cancel()
		if refreshCtx.Err() != nil {
			t.Error("refresh context must not inherit the caller's cancellation")
		}
		return llm.AuthCredential{Access: "new", Refresh: "r2"}, nil
	}}
	_, _ = resolver.Credential(ctx)
	stored, _, _ := store.Read(context.Background(), testProvider)
	if stored.Refresh != "r2" {
		t.Fatalf("rotated refresh token lost after caller cancellation: %+v", stored)
	}
}

func TestResolverAcrossFileStoreInstancesRefreshesOnce(t *testing.T) {
	path := filepath.Join(t.TempDir(), "auth.json")
	seed := authstore.NewFile(path)
	if _, _, err := seed.Modify(context.Background(), testProvider, func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
		return llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "r1", Expires: 1}, true, nil
	}); err != nil {
		t.Fatal(err)
	}
	var calls atomic.Int64
	refresh := func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
		calls.Add(1)
		return llm.AuthCredential{Access: "new", Refresh: "r2", Expires: time.Now().Add(time.Hour).UnixMilli()}, nil
	}
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Separate File values model separate processes sharing the file.
			resolver := &Resolver{Store: authstore.NewFile(path), ProviderID: testProvider, Refresh: refresh}
			if cred, err := resolver.Credential(context.Background()); err != nil || cred.Access != "new" {
				t.Errorf("Credential = %+v, %v", cred, err)
			}
		}()
	}
	wg.Wait()
	if calls.Load() != 1 {
		t.Fatalf("refresh calls across stores = %d, want 1", calls.Load())
	}
}

func TestForceRefreshIfCurrentSkipsWhenAlreadyRotated(t *testing.T) {
	var calls int
	resolver := newTestResolver(t, llm.AuthCredential{Type: "oauth", Access: "newer", Refresh: "r"},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			calls++
			return llm.AuthCredential{Access: "x"}, nil
		})
	cred, err := resolver.ForceRefreshIfCurrent(context.Background(), "stale")
	if err != nil || cred.Access != "newer" || calls != 0 {
		t.Fatalf("ForceRefreshIfCurrent = %+v, %v (calls %d), want the stored newer credential", cred, err, calls)
	}
}

func inAnHour() int64 { return time.Now().Add(time.Hour).UnixMilli() }
