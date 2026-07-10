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
const (
	defaultRefreshBefore  = 5 * time.Minute
	defaultRefreshTimeout = 30 * time.Second
)

// RefreshFunc exchanges a refresh token for a renewed credential.
type RefreshFunc func(context.Context, llm.AuthCredential) (llm.AuthCredential, error)

// Source is a goroutine-safe OAuth token source with single-flight refresh.
type Source struct {
	mu             sync.Mutex
	cred           llm.AuthCredential
	refresh        RefreshFunc
	persist        llm.OAuthPersistenceFunc
	refreshBefore  time.Duration
	refreshTimeout time.Duration
	now            func() time.Time
	inflight       *refreshGeneration
}

type refreshGeneration struct {
	ctx    context.Context
	cancel context.CancelFunc
	done   chan struct{}
	err    error
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

// WithRefreshTimeout bounds a complete refresh generation, including
// credential persistence. It is internal to provider implementations.
func WithRefreshTimeout(d time.Duration) Option {
	return func(s *Source) {
		if d > 0 {
			s.refreshTimeout = d
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

// WithPersistence installs the callback that durably persists renewed
// credentials before they become visible to callers.
func WithPersistence(fn llm.OAuthPersistenceFunc) Option {
	return func(s *Source) {
		s.persist = fn
	}
}

// New constructs a token source around an initial credential.
func New(cred llm.AuthCredential, refresh RefreshFunc, persist llm.OAuthPersistenceFunc, opts ...Option) (*Source, error) {
	s := &Source{
		cred:           cred,
		refresh:        refresh,
		persist:        persist,
		refreshBefore:  defaultRefreshBefore,
		refreshTimeout: defaultRefreshTimeout,
		now:            time.Now,
	}
	for _, opt := range opts {
		opt(s)
	}
	if err := ValidatePersistence(s.cred, s.persist); err != nil {
		return nil, err
	}
	return s, nil
}

// ValidatePersistence rejects refreshable credentials whose rotated tokens
// cannot be durably stored before publication.
func ValidatePersistence(cred llm.AuthCredential, persist llm.OAuthPersistenceFunc) error {
	if cred.Refresh != "" && persist == nil {
		return fmt.Errorf("%w: OAuthPersistenceFunc is required for a credential with a refresh token", llm.ErrBadRequest)
	}
	return nil
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
	generation := s.inflight
	if generation == nil {
		base := context.WithoutCancel(ctx)
		generationCtx, cancel := context.WithTimeout(base, s.refreshTimeout)
		generation = &refreshGeneration{
			ctx:    generationCtx,
			cancel: cancel,
			done:   make(chan struct{}),
		}
		s.inflight = generation
		go s.runRefresh(generation)
	}
	s.mu.Unlock()

	select {
	case <-generation.done:
		if generation.err != nil {
			return llm.AuthCredential{}, generation.err
		}
		s.mu.Lock()
		cred := s.cred
		s.mu.Unlock()
		return credentialWithAccess(cred)
	case <-ctx.Done():
		return llm.AuthCredential{}, ctx.Err()
	}
}

type refreshResult struct {
	cred llm.AuthCredential
	err  error
}

func (s *Source) runRefresh(generation *refreshGeneration) {
	result := make(chan refreshResult, 1)
	go func() {
		cred, err := s.refreshCredential(generation.ctx)
		result <- refreshResult{cred: cred, err: err}
	}()

	select {
	case <-generation.ctx.Done():
		s.finishRefresh(generation, llm.AuthCredential{}, generation.ctx.Err())
	case refreshed := <-result:
		if err := generation.ctx.Err(); err != nil {
			s.finishRefresh(generation, llm.AuthCredential{}, err)
			return
		}
		if refreshed.err != nil {
			s.finishRefresh(generation, llm.AuthCredential{}, refreshed.err)
			return
		}

		s.mu.Lock()
		cred := mergeCredential(s.cred, refreshed.cred)
		persist := s.persist
		s.mu.Unlock()
		if persist == nil {
			s.finishRefresh(generation, cred, nil)
			return
		}
		if err := generation.ctx.Err(); err != nil {
			s.finishRefresh(generation, llm.AuthCredential{}, err)
			return
		}

		persisted := make(chan error, 1)
		go func() {
			if err := generation.ctx.Err(); err != nil {
				persisted <- err
				return
			}
			persisted <- persist(generation.ctx, cred)
		}()
		select {
		case <-generation.ctx.Done():
			s.finishRefresh(generation, llm.AuthCredential{}, generation.ctx.Err())
		case err := <-persisted:
			if ctxErr := generation.ctx.Err(); ctxErr != nil {
				s.finishRefresh(generation, llm.AuthCredential{}, ctxErr)
				return
			}
			if err != nil {
				err = fmt.Errorf("persist refreshed OAuth credential: %w", err)
			}
			s.finishRefresh(generation, cred, err)
		}
	}
}

func (s *Source) finishRefresh(generation *refreshGeneration, cred llm.AuthCredential, err error) {
	s.mu.Lock()
	if s.inflight != generation {
		s.mu.Unlock()
		return
	}
	if err == nil {
		if ctxErr := generation.ctx.Err(); ctxErr != nil {
			err = ctxErr
		} else {
			s.cred = cred
		}
	}
	generation.err = err
	s.inflight = nil
	close(generation.done)
	s.mu.Unlock()
	generation.cancel()
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
	if s.cred.Refresh == "" {
		return !s.now().Before(expires)
	}
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
