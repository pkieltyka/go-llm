package provideroauth

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

func TestSourceRefreshesExpiredCredential(t *testing.T) {
	now := time.Unix(100, 0)
	var refreshed llm.AuthCredential
	source := New(
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh", Expires: now.Add(-time.Second).UnixMilli()},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			return llm.AuthCredential{Type: "oauth", Access: "new", Refresh: "next-refresh", Expires: now.Add(time.Hour).UnixMilli()}, nil
		},
		func(cred llm.AuthCredential) { refreshed = cred },
		WithNow(func() time.Time { return now }),
	)

	token, err := source.Token(context.Background())
	if err != nil {
		t.Fatalf("Token returned error: %v", err)
	}
	if token != "new" || refreshed.Access != "new" || refreshed.Refresh != "next-refresh" {
		t.Fatalf("token/refreshed = %q/%+v", token, refreshed)
	}
}

func TestSourceForceRefresh(t *testing.T) {
	var calls int
	source := New(
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh"},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			calls++
			return llm.AuthCredential{Type: "oauth", Access: "new"}, nil
		},
		nil,
	)

	cred, err := source.ForceRefresh(context.Background())
	if err != nil {
		t.Fatalf("ForceRefresh returned error: %v", err)
	}
	if cred.Access != "new" || cred.Refresh != "refresh" || calls != 1 {
		t.Fatalf("credential/calls = %+v/%d", cred, calls)
	}
}

func TestSourceForceRefreshIfCurrentSkipsAlreadyRefreshedToken(t *testing.T) {
	var calls int
	source := New(
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh"},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			calls++
			return llm.AuthCredential{Type: "oauth", Access: "new"}, nil
		},
		nil,
	)

	cred, err := source.ForceRefreshIfCurrent(context.Background(), "old")
	if err != nil {
		t.Fatalf("ForceRefreshIfCurrent returned error: %v", err)
	}
	if cred.Access != "new" || calls != 1 {
		t.Fatalf("credential/calls = %+v/%d", cred, calls)
	}
	cred, err = source.ForceRefreshIfCurrent(context.Background(), "old")
	if err != nil {
		t.Fatalf("second ForceRefreshIfCurrent returned error: %v", err)
	}
	if cred.Access != "new" || calls != 1 {
		t.Fatalf("second credential/calls = %+v/%d, want no extra refresh", cred, calls)
	}
}

func TestSourceSingleFlight(t *testing.T) {
	var calls atomic.Int64
	block := make(chan struct{})
	source := New(
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh", Expires: time.Unix(1, 0).UnixMilli()},
		func(ctx context.Context, cred llm.AuthCredential) (llm.AuthCredential, error) {
			calls.Add(1)
			select {
			case <-block:
			case <-ctx.Done():
				return llm.AuthCredential{}, ctx.Err()
			}
			return llm.AuthCredential{Type: "oauth", Access: "new", Refresh: cred.Refresh, Expires: time.Now().Add(time.Hour).UnixMilli()}, nil
		},
		nil,
	)

	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			token, err := source.Token(context.Background())
			if err != nil {
				errs <- err
				return
			}
			if token != "new" {
				errs <- errors.New("unexpected token")
			}
		}()
	}
	close(block)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("refresh calls = %d, want 1", calls.Load())
	}
}
