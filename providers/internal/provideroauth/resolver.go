package provideroauth

import (
	"context"
	"fmt"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

const (
	// DefaultRefreshBefore is how much validity a token must have left;
	// anything less is refreshed first (pi: five minutes).
	DefaultRefreshBefore = 5 * time.Minute
	// DefaultRefreshTimeout bounds a started refresh, which ignores caller
	// cancellation once it begins (pi: 15 seconds).
	DefaultRefreshTimeout = 15 * time.Second
)

// RefreshFunc exchanges a stored credential's refresh token for a renewed
// credential.
type RefreshFunc func(context.Context, llm.AuthCredential) (llm.AuthCredential, error)

// Resolver resolves a provider's stored OAuth credential, refreshing it
// inside CredentialStore.Modify so concurrent requests and processes refresh
// exactly once and the renewed credential is stored before it is used.
type Resolver struct {
	Store          llm.CredentialStore
	ProviderID     string
	Refresh        RefreshFunc
	RefreshBefore  time.Duration
	RefreshTimeout time.Duration
	Now            func() time.Time
}

// Credential returns a usable OAuth credential: the stored one, or a renewed
// one when it expires within RefreshBefore. The optimistic check reads the
// store; the authoritative check repeats under the store lock, so a
// credential another request or process already renewed is used as is.
func (r *Resolver) Credential(ctx context.Context) (llm.AuthCredential, error) {
	cred, ok, err := r.Store.Read(ctx, r.ProviderID)
	if err != nil {
		return llm.AuthCredential{}, fmt.Errorf("%w: read %s credential: %w", llm.ErrAuth, r.ProviderID, err)
	}
	if !ok {
		return llm.AuthCredential{}, fmt.Errorf("%w: no %s OAuth credential is stored", llm.ErrAuth, r.ProviderID)
	}
	return r.resolve(ctx, cred)
}

// WithStored returns a CredentialSource that starts from cred, a credential
// the caller already read from Store, instead of reading it again.
func (r *Resolver) WithStored(cred llm.AuthCredential) CredentialSource {
	return storedSource{resolver: r, cred: cred}
}

type storedSource struct {
	resolver *Resolver
	cred     llm.AuthCredential
}

func (s storedSource) Credential(ctx context.Context) (llm.AuthCredential, error) {
	return s.resolver.resolve(ctx, s.cred)
}

func (s storedSource) ForceRefreshIfCurrent(ctx context.Context, failedAccess string) (llm.AuthCredential, error) {
	return s.resolver.ForceRefreshIfCurrent(ctx, failedAccess)
}

func (r *Resolver) resolve(ctx context.Context, cred llm.AuthCredential) (llm.AuthCredential, error) {
	if cred.Type != "oauth" {
		return llm.AuthCredential{}, fmt.Errorf("%w: no %s OAuth credential is stored", llm.ErrAuth, r.ProviderID)
	}
	if !r.expiresSoon(cred) {
		return withAccess(cred)
	}
	return r.refreshLocked(ctx, r.expiresSoon)
}

// ForceRefreshIfCurrent renews the credential after the provider rejected
// failedAccess (HTTP 401), unless the stored access token has already
// changed, in which case the newer stored credential is returned.
func (r *Resolver) ForceRefreshIfCurrent(ctx context.Context, failedAccess string) (llm.AuthCredential, error) {
	return r.refreshLocked(ctx, func(cred llm.AuthCredential) bool { return cred.Access == failedAccess })
}

func (r *Resolver) refreshLocked(ctx context.Context, needsRefresh func(llm.AuthCredential) bool) (llm.AuthCredential, error) {
	stored, ok, err := r.Store.Modify(ctx, r.ProviderID, func(current llm.AuthCredential, ok bool) (llm.AuthCredential, bool, error) {
		if !ok || current.Type != "oauth" {
			return llm.AuthCredential{}, false, nil // logged out meanwhile
		}
		if !needsRefresh(current) {
			return llm.AuthCredential{}, false, nil // already renewed elsewhere
		}
		if current.Refresh == "" {
			return llm.AuthCredential{}, false, fmt.Errorf("%w: %s OAuth credential expired and has no refresh token", llm.ErrAuth, r.ProviderID)
		}
		if r.Refresh == nil {
			return llm.AuthCredential{}, false, fmt.Errorf("%w: %s OAuth refresh is not configured", llm.ErrAuth, r.ProviderID)
		}
		// The provider may rotate the refresh token the moment the request is
		// sent, so a started refresh must finish and be stored even if the
		// caller gives up; only the timeout bounds it.
		refreshCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), r.refreshTimeout())
		defer cancel()
		next, err := r.Refresh(refreshCtx, current)
		if err != nil {
			return llm.AuthCredential{}, false, err
		}
		if next.Access == "" {
			return llm.AuthCredential{}, false, fmt.Errorf("%w: %s OAuth refresh returned no access token", llm.ErrAuth, r.ProviderID)
		}
		return merge(current, next), true, nil
	})
	if err != nil {
		return llm.AuthCredential{}, err
	}
	if !ok || stored.Type != "oauth" {
		return llm.AuthCredential{}, fmt.Errorf("%w: no %s OAuth credential is stored", llm.ErrAuth, r.ProviderID)
	}
	return withAccess(stored)
}

func (r *Resolver) expiresSoon(cred llm.AuthCredential) bool {
	if cred.Access == "" {
		return true
	}
	before := r.RefreshBefore
	if before <= 0 {
		before = DefaultRefreshBefore
	}
	now := time.Now
	if r.Now != nil {
		now = r.Now
	}
	return !now().Add(before).Before(time.UnixMilli(cred.Expires))
}

func (r *Resolver) refreshTimeout() time.Duration {
	if r.RefreshTimeout > 0 {
		return r.RefreshTimeout
	}
	return DefaultRefreshTimeout
}

func withAccess(cred llm.AuthCredential) (llm.AuthCredential, error) {
	if cred.Access == "" {
		return llm.AuthCredential{}, fmt.Errorf("%w: missing OAuth access token", llm.ErrAuth)
	}
	return cred, nil
}

// merge keeps identity and configuration fields a refresh response omits.
func merge(prev, next llm.AuthCredential) llm.AuthCredential {
	if next.Type == "" {
		next.Type = prev.Type
	}
	if next.Refresh == "" {
		next.Refresh = prev.Refresh
	}
	if next.AccountID == "" {
		next.AccountID = prev.AccountID
	}
	if next.ClientID == "" {
		next.ClientID = prev.ClientID
	}
	if next.Model == "" {
		next.Model = prev.Model
	}
	if next.BaseURL == "" {
		next.BaseURL = prev.BaseURL
	}
	return next
}
