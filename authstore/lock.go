package authstore

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sync"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

// The file lock speaks pi's protocol (proper-lockfile): "<auth file>.lock" is
// created as a directory, the holder refreshes its mtime while it works, and
// a lock left untouched for staleLockAge belonged to a dead process and may
// be removed. pi and go-llm processes sharing one auth.json therefore
// exclude each other. Like proper-lockfile, removing a stale lock races with
// another waiter doing the same; the window needs a holder to have died.
// A holder whose lock was removed or replaced anyway (it stalled past a
// waiter's stale age) is compromised: it writes nothing and leaves the
// other process's lock alone.
// The timings are variables only so tests can shorten them.
var (
	staleLockAge = 30 * time.Second // pi's auth-storage staleMs
	// lockUpdateEvery keeps a held lock fresh for pi's synchronous reload,
	// which uses proper-lockfile's default 10 s stale age (and its 5 s
	// update interval for that age).
	lockUpdateEvery = 5 * time.Second
	// maxFileLockWait outlasts staleLockAge, so a waiter recovers a lock
	// whose holder died, and the longest live holder (an OAuth refresh,
	// bounded at 15 s).
	maxFileLockWait = 45 * time.Second
	fileLockPoll    = 10 * time.Millisecond
)

type fileLock struct {
	path        string
	stop        chan struct{}
	done        chan struct{}
	mu          sync.Mutex
	mtime       time.Time // last mtime this holder observed on its lock
	compromised bool
}

var errLockCompromised = errors.New("authstore: lock was taken over by another process; nothing was written")

func acquireFileLock(ctx context.Context, lockPath string) (*fileLock, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := os.MkdirAll(filepath.Dir(lockPath), 0o700); err != nil {
		return nil, err
	}
	deadline := time.NewTimer(maxFileLockWait)
	defer deadline.Stop()
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		lock, err := tryFileLock(lockPath)
		if err != nil {
			return nil, fmt.Errorf("authstore: lock %s: %w", lockPath, err)
		}
		if lock != nil {
			return lock, nil
		}
		poll := time.NewTimer(fileLockPoll)
		select {
		case <-poll.C:
		case <-ctx.Done():
			poll.Stop()
			return nil, ctx.Err()
		case <-deadline.C:
			poll.Stop()
			return nil, fmt.Errorf("%w: authstore: %s is held by another process", llm.ErrTimeout, lockPath)
		}
	}
}

// tryFileLock returns a held lock, or nil when another process holds it.
func tryFileLock(path string) (*fileLock, error) {
	err := os.Mkdir(path, 0o700)
	if err == nil {
		info, err := os.Stat(path)
		if err != nil {
			_ = os.Remove(path)
			return nil, err
		}
		lock := &fileLock{path: path, stop: make(chan struct{}), done: make(chan struct{}), mtime: info.ModTime()}
		go lock.keepFresh()
		return lock, nil
	}
	if !errors.Is(err, fs.ErrExist) {
		return nil, err
	}
	info, err := os.Stat(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil // released in between; retry
	}
	if err != nil {
		return nil, err
	}
	if time.Since(info.ModTime()) > staleLockAge {
		// Abandoned.
		if err := os.Remove(path); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return nil, err
		}
	}
	return nil, nil
}

func (l *fileLock) keepFresh() {
	defer close(l.done)
	ticker := time.NewTicker(lockUpdateEvery)
	defer ticker.Stop()
	for {
		select {
		case <-l.stop:
			return
		case <-ticker.C:
			if !l.touch() {
				return
			}
		}
	}
}

// touch refreshes the lock's mtime, reporting false once the lock is
// compromised.
func (l *fileLock) touch() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.ownedLocked() {
		return false
	}
	now := time.Now()
	err := os.Chtimes(l.path, now, now)
	var info os.FileInfo
	if err == nil {
		info, err = os.Stat(l.path)
	}
	if err != nil {
		l.compromised = true
		return false
	}
	l.mtime = info.ModTime()
	return true
}

// ownedLocked reports whether the lock still carries the mtime this holder
// last set; a removed or replaced lock marks the holder compromised.
func (l *fileLock) ownedLocked() bool {
	if l.compromised {
		return false
	}
	info, err := os.Stat(l.path)
	if err != nil || !info.ModTime().Equal(l.mtime) {
		l.compromised = true
		return false
	}
	return true
}

// Check returns errLockCompromised when the lock is no longer this
// holder's; a write must not proceed then.
func (l *fileLock) Check() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.ownedLocked() {
		return errLockCompromised
	}
	return nil
}

// Release stops the heartbeat and removes the lock directory, unless the
// lock was compromised and may now belong to another process.
func (l *fileLock) Release() error {
	close(l.stop)
	<-l.done
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.ownedLocked() {
		return nil
	}
	return os.Remove(l.path)
}
