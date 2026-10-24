// Package authstore provides llm.CredentialStore implementations: Memory for
// tests and in-process use, and File, a cross-process store for pi-compatible
// auth.json files.
//
// Memory serializes Modify per provider ID and File per file (pi's lock is
// per file); both pass fn the current credential, so an OAuth refresh
// performed inside Modify happens exactly once even when many requests (or
// processes, for File) notice an expiring token at the same time. Credential values never appear in errors or formatted output.
package authstore

import (
	"context"
	"fmt"
	"sync"

	llm "github.com/pkieltyka/go-llm"
)

// Memory is an in-process llm.CredentialStore.
type Memory struct {
	mu          sync.Mutex
	credentials map[string]llm.AuthCredential
	locks       map[string]chan struct{}
}

var _ llm.CredentialStore = (*Memory)(nil)

// NewMemory returns a store seeded with a copy of initial (which may be nil),
// for example the result of llm.LoadAuthFile.
func NewMemory(initial llm.AuthFile) *Memory {
	m := &Memory{credentials: map[string]llm.AuthCredential{}, locks: map[string]chan struct{}{}}
	for id, cred := range initial {
		cred.Env = cred.CloneEnv()
		m.credentials[id] = cred
	}
	return m
}

func (m *Memory) String() string { return "authstore: memory (redacted)" }

// GoString redacts the stored credentials from Go-syntax formatted output,
// which would otherwise print the private map field by field.
func (m *Memory) GoString() string { return "authstore.Memory{redacted}" }

// Read returns the stored credential for providerID.
func (m *Memory) Read(ctx context.Context, providerID string) (llm.AuthCredential, bool, error) {
	if err := contextErr(ctx); err != nil {
		return llm.AuthCredential{}, false, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	cred, ok := m.credentials[providerID]
	cred.Env = cred.CloneEnv()
	return cred, ok, nil
}

// Modify serializes fn per provider ID. ctx bounds only the wait for the
// provider's turn.
func (m *Memory) Modify(ctx context.Context, providerID string, fn func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error)) (llm.AuthCredential, bool, error) {
	if fn == nil {
		return llm.AuthCredential{}, false, fmt.Errorf("%w: nil modify function", llm.ErrBadRequest)
	}
	release, err := m.acquire(ctx, providerID)
	if err != nil {
		return llm.AuthCredential{}, false, err
	}
	defer release()
	m.mu.Lock()
	current, ok := m.credentials[providerID]
	current.Env = current.CloneEnv()
	m.mu.Unlock()
	next, write, err := fn(current, ok)
	if err != nil {
		return llm.AuthCredential{}, false, err
	}
	if !write {
		return current, ok, nil
	}
	stored := next
	stored.Env = next.CloneEnv()
	m.mu.Lock()
	m.credentials[providerID] = stored
	m.mu.Unlock()
	return next, true, nil
}

// Delete removes the credential for providerID (logout).
func (m *Memory) Delete(ctx context.Context, providerID string) error {
	release, err := m.acquire(ctx, providerID)
	if err != nil {
		return err
	}
	defer release()
	m.mu.Lock()
	delete(m.credentials, providerID)
	m.mu.Unlock()
	return nil
}

func (m *Memory) acquire(ctx context.Context, providerID string) (func(), error) {
	m.mu.Lock()
	lock, ok := m.locks[providerID]
	if !ok {
		lock = make(chan struct{}, 1)
		m.locks[providerID] = lock
	}
	m.mu.Unlock()
	return acquireChannel(ctx, lock)
}

// acquireChannel takes a one-slot semaphore, honoring ctx while waiting. An
// already-cancelled ctx never acquires: select picks randomly between ready
// cases, so cancellation is checked before and after.
func acquireChannel(ctx context.Context, lock chan struct{}) (func(), error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	select {
	case lock <- struct{}{}:
		if err := ctx.Err(); err != nil {
			<-lock
			return nil, err
		}
		return func() { <-lock }, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func contextErr(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}
