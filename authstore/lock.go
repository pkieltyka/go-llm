package authstore

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

// The file lock speaks pi's protocol (proper-lockfile): "<auth file>.lock" is
// created as a directory, the holder refreshes its mtime while it works, and
// a lock left untouched for staleLockAge belonged to a dead process and may
// be removed. pi and go-llm processes sharing one auth.json therefore
// exclude each other. Like proper-lockfile, removing a stale lock races with
// another waiter doing the same; the window needs a holder to have died.
// The timings are variables only so tests can shorten them.
var (
	staleLockAge    = 30 * time.Second // pi's auth-storage staleMs
	lockUpdateEvery = staleLockAge / 2 // proper-lockfile's default update interval
	// maxFileLockWait outlasts staleLockAge, so a waiter recovers a lock
	// whose holder died, and the longest live holder (an OAuth refresh,
	// bounded at 30 s).
	maxFileLockWait = 45 * time.Second
	fileLockPoll    = 10 * time.Millisecond
)

type fileLock struct {
	path string
	stop chan struct{}
	done chan struct{}
}

func acquireFileLock(ctx context.Context, lockPath string) (*fileLock, error) {
	if err := os.MkdirAll(filepath.Dir(lockPath), 0o700); err != nil {
		return nil, err
	}
	deadline := time.NewTimer(maxFileLockWait)
	defer deadline.Stop()
	for {
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
		lock := &fileLock{path: path, stop: make(chan struct{}), done: make(chan struct{})}
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
		// Abandoned. This also clears a regular file left by older builds.
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
			now := time.Now()
			_ = os.Chtimes(l.path, now, now)
		}
	}
}

// Release stops the heartbeat and removes the lock directory.
func (l *fileLock) Release() error {
	close(l.stop)
	<-l.done
	return os.Remove(l.path)
}
