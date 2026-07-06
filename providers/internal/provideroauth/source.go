package provideroauth

import (
	"context"
	"fmt"
	"sync"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

// defaultRefreshBefore is the single expiry safety margin, applied only when
// the Source decides whether a credential still has useful life. Refresh
// endpoints must persist the TRUE server expiry (see ExpiresAt) so the margin
// is never baked into stored credentials.
const defaultRefreshBefore = 5 * time.Minute

// RefreshFunc exchanges a refresh token for a renewed credential.
type RefreshFunc func(context.Context, llm.AuthCredential) (llm.AuthCredential, error)

// Source is a goroutine-safe OAuth token source with single-flight refresh.
type Source struct {
	mu            sync.Mutex
	cred          llm.AuthCredential
	refresh       RefreshFunc
	onRefresh     func(llm.AuthCredential)
	refreshBefore time.Duration
	now           func() time.Time
	inflight      chan struct{}
	refreshErr    error
}

// Option configures a Source.
type Option func(*Source)

// WithRefreshBefore changes how early a token is refreshed before expiry.
func WithRefreshBefore(d time.Duration) Option {
	return func(s *Source) {
		if d >= 0 {
			s.refreshBefore = d
		}
	}
}

// WithNow overrides the source clock for tests.
func WithNow(fn func() time.Time) Option {
	return func(s *Source) {
		if fn != nil {
			s.now = fn
		}
	}
}

// New constructs a token source around an initial credential.
func New(cred llm.AuthCredential, refresh RefreshFunc, onRefresh func(llm.AuthCredential), opts ...Option) *Source {
	s := &Source{
		cred:          cred,
		refresh:       refresh,
		onRefresh:     onRefresh,
		refreshBefore: defaultRefreshBefore,
		now:           time.Now,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Token returns a valid bearer token, refreshing first when needed.
func (s *Source) Token(ctx context.Context) (string, error) {
	cred, err := s.Credential(ctx)
	if err != nil {
		return "", err
	}
	return cred.Access, nil
}

// Credential returns a valid credential snapshot.
func (s *Source) Credential(ctx context.Context) (llm.AuthCredential, error) {
	return s.credential(ctx, func(llm.AuthCredential) bool { return s.needsRefreshLocked() })
}

// ForceRefresh renews the credential regardless of its current expiry.
func (s *Source) ForceRefresh(ctx context.Context) (llm.AuthCredential, error) {
	return s.credential(ctx, func(llm.AuthCredential) bool { return true })
}

// ForceRefreshIfCurrent renews the credential only when failedAccess is still
// the current access token. If another request already refreshed it, the fresh
// credential is returned without another token exchange.
func (s *Source) ForceRefreshIfCurrent(ctx context.Context, failedAccess string) (llm.AuthCredential, error) {
	return s.credential(ctx, func(cred llm.AuthCredential) bool {
		return failedAccess == "" || cred.Access == failedAccess
	})
}

// Snapshot returns the current credential without refreshing.
func (s *Source) Snapshot() llm.AuthCredential {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cred
}

// credential returns the current credential, refreshing it first when
// needsRefresh (evaluated under s.mu) reports the snapshot is unusable.
// Concurrent callers share a single in-flight refresh.
func (s *Source) credential(ctx context.Context, needsRefresh func(llm.AuthCredential) bool) (llm.AuthCredential, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	if !needsRefresh(s.cred) {
		cred := s.cred
		s.mu.Unlock()
		return credentialWithAccess(cred)
	}
	if done := s.inflight; done != nil {
		s.mu.Unlock()
		select {
		case <-done:
			s.mu.Lock()
			err := s.refreshErr
			cred := s.cred
			s.mu.Unlock()
			if err != nil {
				return llm.AuthCredential{}, err
			}
			return credentialWithAccess(cred)
		case <-ctx.Done():
			return llm.AuthCredential{}, ctx.Err()
		}
	}
	done := make(chan struct{})
	s.inflight = done
	s.mu.Unlock()

	cred, err := s.refreshCredential(ctx)
	s.mu.Lock()
	if err == nil {
		s.cred = mergeCredential(s.cred, cred)
	}
	s.refreshErr = err
	s.inflight = nil
	close(done)
	cred = s.cred
	onRefresh := s.onRefresh
	s.mu.Unlock()
	if err != nil {
		return llm.AuthCredential{}, err
	}
	if onRefresh != nil {
		onRefresh(cred)
	}
	return cred, nil
}

func credentialWithAccess(cred llm.AuthCredential) (llm.AuthCredential, error) {
	if cred.Access == "" {
		return llm.AuthCredential{}, fmt.Errorf("%w: missing OAuth access token", llm.ErrAuth)
	}
	return cred, nil
}

func (s *Source) needsRefreshLocked() bool {
	if s.cred.Access == "" {
		return true
	}
	if s.cred.Expires == 0 {
		return false
	}
	expires := time.UnixMilli(s.cred.Expires)
	return !s.now().Add(s.refreshBefore).Before(expires)
}

func (s *Source) refreshCredential(ctx context.Context) (llm.AuthCredential, error) {
	s.mu.Lock()
	cred := s.cred
	refresh := s.refresh
	s.mu.Unlock()
	if cred.Refresh == "" {
		return llm.AuthCredential{}, fmt.Errorf("%w: missing OAuth refresh token", llm.ErrAuth)
	}
	if refresh == nil {
		return llm.AuthCredential{}, fmt.Errorf("%w: OAuth refresh is not configured", llm.ErrAuth)
	}
	next, err := refresh(ctx, cred)
	if err != nil {
		return llm.AuthCredential{}, err
	}
	if next.Access == "" {
		return llm.AuthCredential{}, fmt.Errorf("%w: OAuth refresh response missing access token", llm.ErrAuth)
	}
	return next, nil
}

func mergeCredential(prev, next llm.AuthCredential) llm.AuthCredential {
	if next.Type == "" {
		next.Type = prev.Type
	}
	if next.Refresh == "" {
		next.Refresh = prev.Refresh
	}
	if next.AccountID == "" {
		next.AccountID = prev.AccountID
	}
	if next.Model == "" {
		next.Model = prev.Model
	}
	if next.BaseURL == "" {
		next.BaseURL = prev.BaseURL
	}
	if next.Key == "" {
		next.Key = prev.Key
	}
	return next
}
