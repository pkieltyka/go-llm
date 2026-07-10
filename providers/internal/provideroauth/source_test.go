package provideroauth

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

func noOpPersistence(ctx context.Context, _ llm.AuthCredential) error {
	return ctx.Err()
}

func mustNewSource(t *testing.T, cred llm.AuthCredential, refresh RefreshFunc, persist llm.OAuthPersistenceFunc, opts ...Option) *Source {
	t.Helper()
	source, err := New(cred, refresh, persist, opts...)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	return source
}

func TestNewPersistenceContract(t *testing.T) {
	t.Run("refreshable requires persistence", func(t *testing.T) {
		source, err := New(
			llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh"},
			func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
				t.Fatal("refresh ran during source construction")
				return llm.AuthCredential{}, nil
			},
			nil,
		)
		if !errors.Is(err, llm.ErrBadRequest) {
			t.Fatalf("New error = %v, want ErrBadRequest", err)
		}
		if source != nil {
			t.Fatalf("New source = %+v, want nil", source)
		}
	})

	t.Run("access only permits nil persistence", func(t *testing.T) {
		source, err := New(
			llm.AuthCredential{Type: "oauth", Access: "access-only"},
			nil,
			nil,
		)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		token, err := source.Token(context.Background())
		if err != nil || token != "access-only" {
			t.Fatalf("Token = %q, %v", token, err)
		}
	})

	t.Run("explicit no-op permits in-memory renewal", func(t *testing.T) {
		persistenceCalls := 0
		persist := func(ctx context.Context, _ llm.AuthCredential) error {
			persistenceCalls++
			return ctx.Err()
		}
		source, err := New(
			llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh"},
			func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
				return llm.AuthCredential{Access: "new", Refresh: "rotated"}, nil
			},
			persist,
		)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		cred, err := source.ForceRefresh(context.Background())
		if err != nil {
			t.Fatalf("ForceRefresh returned error: %v", err)
		}
		if cred.Access != "new" || cred.Refresh != "rotated" || persistenceCalls != 1 {
			t.Fatalf("credential/persistence calls = %+v/%d", cred, persistenceCalls)
		}
	})
}

func TestSourceRefreshesExpiredCredential(t *testing.T) {
	now := time.Unix(100, 0)
	var refreshed llm.AuthCredential
	source := mustNewSource(t,
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh", Expires: now.Add(-time.Second).UnixMilli()},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			return llm.AuthCredential{Type: "oauth", Access: "new", Refresh: "next-refresh", Expires: now.Add(time.Hour).UnixMilli()}, nil
		},
		func(_ context.Context, cred llm.AuthCredential) error {
			refreshed = cred
			return nil
		},
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
	source := mustNewSource(t,
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh"},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			calls++
			return llm.AuthCredential{Type: "oauth", Access: "new"}, nil
		},
		noOpPersistence,
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
	source := mustNewSource(t,
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh"},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			calls++
			return llm.AuthCredential{Type: "oauth", Access: "new"}, nil
		},
		noOpPersistence,
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
	source := mustNewSource(t,
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
		noOpPersistence,
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

func TestSourcePublishesAfterPersistenceAndWakesWaitersTogether(t *testing.T) {
	now := time.Unix(100, 0)
	refreshStarted := make(chan struct{})
	persistStarted := make(chan llm.AuthCredential, 1)
	releasePersist := make(chan struct{})
	var refreshCalls atomic.Int64

	var source *Source
	source = mustNewSource(t,
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh-1", Expires: now.Add(-time.Second).UnixMilli()},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			refreshCalls.Add(1)
			close(refreshStarted)
			return llm.AuthCredential{Access: "new", Refresh: "refresh-2", Expires: now.Add(time.Hour).UnixMilli()}, nil
		},
		nil,
		WithNow(func() time.Time { return now }),
		WithPersistence(func(_ context.Context, cred llm.AuthCredential) error {
			persistStarted <- cred
			if got := source.Snapshot().Access; got != "old" {
				t.Errorf("Snapshot during persistence = %q, want old", got)
			}
			<-releasePersist
			return nil
		}),
	)

	type result struct {
		cred llm.AuthCredential
		err  error
	}
	results := make(chan result, 2)
	go func() {
		cred, err := source.Credential(context.Background())
		results <- result{cred: cred, err: err}
	}()
	<-refreshStarted
	persisted := <-persistStarted
	if persisted.Access != "new" || persisted.Refresh != "refresh-2" {
		t.Fatalf("persisted credential = %+v", persisted)
	}
	waiterJoined := make(chan struct{})
	go func() {
		cred, err := source.credential(context.Background(), func(llm.AuthCredential) bool {
			close(waiterJoined)
			return true
		})
		results <- result{cred: cred, err: err}
	}()
	<-waiterJoined

	select {
	case got := <-results:
		t.Fatalf("caller returned before persistence completed: %+v", got)
	case <-time.After(20 * time.Millisecond):
	}
	if got := refreshCalls.Load(); got != 1 {
		t.Fatalf("refresh calls during persistence = %d, want 1", got)
	}
	if got := source.Snapshot().Access; got != "old" {
		t.Fatalf("Snapshot before persistence completed = %q, want old", got)
	}

	close(releasePersist)
	for range 2 {
		got := <-results
		if got.err != nil {
			t.Fatalf("credential returned error: %v", got.err)
		}
		if got.cred.Access != "new" || got.cred.Refresh != "refresh-2" {
			t.Fatalf("credential = %+v", got.cred)
		}
	}
	if got := refreshCalls.Load(); got != 1 {
		t.Fatalf("refresh calls = %d, want 1", got)
	}
}

func TestSourceWaiterCancellationDoesNotCancelRefresh(t *testing.T) {
	now := time.Unix(100, 0)
	refreshStarted := make(chan struct{})
	releaseRefresh := make(chan struct{})
	var refreshCalls atomic.Int64
	source := mustNewSource(t,
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh", Expires: now.Add(-time.Second).UnixMilli()},
		func(ctx context.Context, cred llm.AuthCredential) (llm.AuthCredential, error) {
			refreshCalls.Add(1)
			close(refreshStarted)
			select {
			case <-releaseRefresh:
				return llm.AuthCredential{Access: "new", Refresh: cred.Refresh, Expires: now.Add(time.Hour).UnixMilli()}, nil
			case <-ctx.Done():
				return llm.AuthCredential{}, ctx.Err()
			}
		},
		noOpPersistence,
		WithNow(func() time.Time { return now }),
	)

	canceledCtx, cancel := context.WithCancel(context.Background())
	canceled := make(chan error, 1)
	go func() {
		_, err := source.Token(canceledCtx)
		canceled <- err
	}()
	<-refreshStarted

	waiter := make(chan error, 1)
	go func() {
		token, err := source.Token(context.Background())
		if err == nil && token != "new" {
			err = errors.New("unexpected token")
		}
		waiter <- err
	}()
	cancel()
	if err := <-canceled; !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled waiter error = %v, want context.Canceled", err)
	}

	select {
	case err := <-waiter:
		t.Fatalf("shared refresh stopped with canceled waiter: %v", err)
	case <-time.After(20 * time.Millisecond):
	}
	close(releaseRefresh)
	if err := <-waiter; err != nil {
		t.Fatalf("healthy waiter returned error: %v", err)
	}
	if got := refreshCalls.Load(); got != 1 {
		t.Fatalf("refresh calls = %d, want 1", got)
	}
}

func TestSourceWaiterDeadlineDoesNotCancelRefresh(t *testing.T) {
	now := time.Unix(100, 0)
	refreshStarted := make(chan struct{})
	releaseRefresh := make(chan struct{})
	type contextKey struct{}
	key := contextKey{}
	source := mustNewSource(t,
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh", Expires: now.Add(-time.Second).UnixMilli()},
		func(ctx context.Context, cred llm.AuthCredential) (llm.AuthCredential, error) {
			if got := ctx.Value(key); got != "refresh-value" {
				t.Errorf("refresh context value = %v", got)
			}
			deadline, ok := ctx.Deadline()
			if !ok || time.Until(deadline) < 100*time.Millisecond {
				t.Errorf("refresh deadline = %v, want independent generation deadline", deadline)
			}
			close(refreshStarted)
			select {
			case <-releaseRefresh:
				return llm.AuthCredential{Access: "new", Refresh: cred.Refresh, Expires: now.Add(time.Hour).UnixMilli()}, nil
			case <-ctx.Done():
				return llm.AuthCredential{}, ctx.Err()
			}
		},
		noOpPersistence,
		WithNow(func() time.Time { return now }),
		WithRefreshTimeout(time.Second),
	)

	base := context.WithValue(context.Background(), key, "refresh-value")
	ctx, cancel := context.WithTimeout(base, 20*time.Millisecond)
	defer cancel()
	errCh := make(chan error, 1)
	go func() {
		_, err := source.Token(ctx)
		errCh <- err
	}()
	<-refreshStarted
	if err := <-errCh; !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("deadline waiter error = %v, want context.DeadlineExceeded", err)
	}

	close(releaseRefresh)
	token, err := source.Token(context.Background())
	if err != nil || token != "new" {
		t.Fatalf("Token after waiter deadline = %q, %v", token, err)
	}
}

func TestSourcePersistenceFailureDoesNotPublishAndCanRetry(t *testing.T) {
	now := time.Unix(100, 0)
	persistErr := errors.New("write credential file")
	firstPersistStarted := make(chan struct{})
	releaseFirstPersist := make(chan struct{})
	var refreshCalls atomic.Int64
	var persistCalls atomic.Int64
	source := mustNewSource(t,
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh-0", Expires: now.Add(-time.Second).UnixMilli()},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			generation := refreshCalls.Add(1)
			return llm.AuthCredential{
				Access:  fmt.Sprintf("new-%d", generation),
				Refresh: fmt.Sprintf("refresh-%d", generation),
				Expires: now.Add(time.Hour).UnixMilli(),
			}, nil
		},
		nil,
		WithNow(func() time.Time { return now }),
		WithPersistence(func(context.Context, llm.AuthCredential) error {
			if persistCalls.Add(1) == 1 {
				close(firstPersistStarted)
				<-releaseFirstPersist
				return persistErr
			}
			return nil
		}),
	)

	errs := make(chan error, 2)
	go func() {
		_, err := source.Token(context.Background())
		errs <- err
	}()
	<-firstPersistStarted
	secondWaiterJoined := make(chan struct{})
	go func() {
		_, err := source.credential(context.Background(), func(llm.AuthCredential) bool {
			close(secondWaiterJoined)
			return true
		})
		errs <- err
	}()
	<-secondWaiterJoined
	close(releaseFirstPersist)
	for range 2 {
		if err := <-errs; !errors.Is(err, persistErr) {
			t.Fatalf("persistence error = %v, want %v", err, persistErr)
		}
	}
	if got := source.Snapshot(); got.Access != "old" || got.Refresh != "refresh-0" {
		t.Fatalf("credential published after persistence failure: %+v", got)
	}
	if got := refreshCalls.Load(); got != 1 {
		t.Fatalf("refresh calls after failed generation = %d, want 1", got)
	}

	cred, err := source.Credential(context.Background())
	if err != nil {
		t.Fatalf("retry Credential returned error: %v", err)
	}
	if cred.Access != "new-2" || cred.Refresh != "refresh-2" {
		t.Fatalf("retry credential = %+v", cred)
	}
	if got := source.Snapshot(); got != cred {
		t.Fatalf("Snapshot = %+v, want %+v", got, cred)
	}
	if got := refreshCalls.Load(); got != 2 {
		t.Fatalf("refresh calls after retry = %d, want 2", got)
	}
}

func TestSourceAccessOnlyCredentialUsesActualExpiry(t *testing.T) {
	now := time.Unix(100, 0)
	expires := now.Add(time.Minute)
	var refreshCalls atomic.Int64
	source := mustNewSource(t,
		llm.AuthCredential{Type: "oauth", Access: "access-only", Expires: expires.UnixMilli()},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			refreshCalls.Add(1)
			return llm.AuthCredential{}, errors.New("refresh must not run")
		},
		nil,
		WithNow(func() time.Time { return now }),
	)

	token, err := source.Token(context.Background())
	if err != nil || token != "access-only" {
		t.Fatalf("Token inside refresh margin = %q, %v", token, err)
	}
	now = expires
	token, err = source.Token(context.Background())
	if !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("Token at expiry error = %v, want ErrAuth", err)
	}
	if token != "" {
		t.Fatalf("Token at expiry = %q, want empty", token)
	}
	if got := refreshCalls.Load(); got != 0 {
		t.Fatalf("refresh calls = %d, want 0 without refresh token", got)
	}
}

func TestSourceRefreshGenerationTimeoutAllowsRetryAndRejectsLateResult(t *testing.T) {
	now := time.Unix(100, 0)
	firstStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	firstReturned := make(chan struct{})
	persisted := make(chan string, 2)
	var refreshCalls atomic.Int64
	source := mustNewSource(t,
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh-0", Expires: now.Add(-time.Second).UnixMilli()},
		func(ctx context.Context, _ llm.AuthCredential) (llm.AuthCredential, error) {
			if _, ok := ctx.Deadline(); !ok {
				t.Error("refresh context has no generation deadline")
			}
			switch refreshCalls.Add(1) {
			case 1:
				close(firstStarted)
				<-releaseFirst
				close(firstReturned)
				return llm.AuthCredential{Access: "stale", Refresh: "refresh-stale", Expires: now.Add(time.Hour).UnixMilli()}, nil
			default:
				return llm.AuthCredential{Access: "fresh", Refresh: "refresh-fresh", Expires: now.Add(time.Hour).UnixMilli()}, nil
			}
		},
		nil,
		WithNow(func() time.Time { return now }),
		WithRefreshTimeout(20*time.Millisecond),
		WithPersistence(func(_ context.Context, cred llm.AuthCredential) error {
			persisted <- cred.Access
			return nil
		}),
	)

	errCh := make(chan error, 1)
	go func() {
		_, err := source.Token(context.Background())
		errCh <- err
	}()
	<-firstStarted
	if err := <-errCh; !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("generation timeout error = %v, want context.DeadlineExceeded", err)
	}
	if got := source.Snapshot().Access; got != "old" {
		t.Fatalf("credential after timed-out refresh = %q, want old", got)
	}

	cred, err := source.Credential(context.Background())
	if err != nil {
		t.Fatalf("retry Credential returned error: %v", err)
	}
	if cred.Access != "fresh" || cred.Refresh != "refresh-fresh" {
		t.Fatalf("retry credential = %+v", cred)
	}
	if got := <-persisted; got != "fresh" {
		t.Fatalf("persisted credential = %q, want fresh", got)
	}

	close(releaseFirst)
	<-firstReturned
	if got := source.Snapshot(); got != cred {
		t.Fatalf("late refresh overwrote credential: got %+v, want %+v", got, cred)
	}
	select {
	case got := <-persisted:
		t.Fatalf("late refresh ran stale persistence for %q", got)
	default:
	}
	if got := refreshCalls.Load(); got != 2 {
		t.Fatalf("refresh calls = %d, want 2", got)
	}
}

func TestSourcePersistenceGenerationTimeoutAllowsRetryAndRejectsLateResult(t *testing.T) {
	now := time.Unix(100, 0)
	firstPersistStarted := make(chan context.Context, 1)
	firstPersistExited := make(chan struct{})
	var refreshCalls atomic.Int64
	var persistedMu sync.Mutex
	var persisted []string
	source := mustNewSource(t,
		llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "refresh-0", Expires: now.Add(-time.Second).UnixMilli()},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			generation := refreshCalls.Add(1)
			return llm.AuthCredential{
				Access:  fmt.Sprintf("new-%d", generation),
				Refresh: fmt.Sprintf("refresh-%d", generation),
				Expires: now.Add(time.Hour).UnixMilli(),
			}, nil
		},
		nil,
		WithNow(func() time.Time { return now }),
		WithRefreshTimeout(20*time.Millisecond),
		WithPersistence(func(ctx context.Context, cred llm.AuthCredential) error {
			if cred.Access == "new-1" {
				firstPersistStarted <- ctx
				<-ctx.Done()
				close(firstPersistExited)
				return ctx.Err()
			}
			persistedMu.Lock()
			persisted = append(persisted, cred.Access)
			persistedMu.Unlock()
			return nil
		}),
	)

	errCh := make(chan error, 1)
	go func() {
		_, err := source.Token(context.Background())
		errCh <- err
	}()
	persistCtx := <-firstPersistStarted
	if _, ok := persistCtx.Deadline(); !ok {
		t.Fatal("persistence context has no generation deadline")
	}
	if err := <-errCh; !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("persistence timeout error = %v, want context.DeadlineExceeded", err)
	}
	if !errors.Is(persistCtx.Err(), context.DeadlineExceeded) {
		t.Fatalf("persistence context error = %v, want context.DeadlineExceeded", persistCtx.Err())
	}
	if got := source.Snapshot().Access; got != "old" {
		t.Fatalf("credential after timed-out persistence = %q, want old", got)
	}
	select {
	case <-firstPersistExited:
	case <-time.After(time.Second):
		t.Fatal("timed-out persistence callback did not exit")
	}

	cred, err := source.Credential(context.Background())
	if err != nil {
		t.Fatalf("retry Credential returned error: %v", err)
	}
	if cred.Access != "new-2" || cred.Refresh != "refresh-2" {
		t.Fatalf("retry credential = %+v", cred)
	}

	if got := source.Snapshot(); got != cred {
		t.Fatalf("timed-out persistence overwrote credential: got %+v, want %+v", got, cred)
	}
	persistedMu.Lock()
	defer persistedMu.Unlock()
	if len(persisted) != 1 || persisted[0] != "new-2" {
		t.Fatalf("successful persisted credentials = %v, want [new-2]", persisted)
	}
}
