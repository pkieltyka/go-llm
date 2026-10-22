package authstore

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

func shortenLockTimings(t *testing.T) {
	t.Helper()
	stale, update, wait := staleLockAge, lockUpdateEvery, maxFileLockWait
	staleLockAge, lockUpdateEvery, maxFileLockWait = 200*time.Millisecond, 20*time.Millisecond, 400*time.Millisecond
	t.Cleanup(func() { staleLockAge, lockUpdateEvery, maxFileLockWait = stale, update, wait })
}

func writeKey(ctx context.Context, f *File) error {
	_, _, err := f.Modify(ctx, "p", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
		return llm.AuthCredential{Type: "api_key", Key: "k"}, true, nil
	})
	return err
}

// A pi process holds the lock as a "<path>.lock" directory (proper-lockfile).
func TestFileWaitsForPiHeldLockDirectory(t *testing.T) {
	shortenLockTimings(t)
	path := filepath.Join(t.TempDir(), "auth.json")
	if err := os.Mkdir(path+".lock", 0o700); err != nil {
		t.Fatal(err)
	}
	released := make(chan struct{})
	go func() {
		time.Sleep(50 * time.Millisecond)
		_ = os.Remove(path + ".lock") // pi's rmdir on release
		close(released)
	}()
	if err := writeKey(context.Background(), NewFile(path)); err != nil {
		t.Fatalf("Modify after pi released = %v", err)
	}
	select {
	case <-released:
	default:
		t.Fatal("Modify acquired the lock while pi held it")
	}
	if _, err := os.Stat(path + ".lock"); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("lock directory not removed on release: %v", err)
	}
}

func TestFileRecoversStaleLockAndLegacyLockFile(t *testing.T) {
	shortenLockTimings(t)
	old := time.Now().Add(-time.Minute)
	for name, create := range map[string]func(string) error{
		"directory":    func(p string) error { return os.Mkdir(p, 0o700) },
		"regular file": func(p string) error { return os.WriteFile(p, nil, 0o600) },
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "auth.json")
			if err := create(path + ".lock"); err != nil {
				t.Fatal(err)
			}
			if err := os.Chtimes(path+".lock", old, old); err != nil {
				t.Fatal(err)
			}
			if err := writeKey(context.Background(), NewFile(path)); err != nil {
				t.Fatalf("Modify over a stale lock = %v", err)
			}
		})
	}
}

func TestFileLockWaitTimesOutAsErrTimeout(t *testing.T) {
	shortenLockTimings(t)
	path := filepath.Join(t.TempDir(), "auth.json")
	if err := os.Mkdir(path+".lock", 0o700); err != nil {
		t.Fatal(err)
	}
	stop := make(chan struct{})
	defer close(stop)
	go func() { // a live holder keeps the lock fresh
		for {
			select {
			case <-stop:
				return
			case <-time.After(20 * time.Millisecond):
				now := time.Now()
				_ = os.Chtimes(path+".lock", now, now)
			}
		}
	}()
	if err := writeKey(context.Background(), NewFile(path)); !errors.Is(err, llm.ErrTimeout) {
		t.Fatalf("Modify against a live holder = %v, want ErrTimeout", err)
	}
}

func TestHeldLockStaysFresh(t *testing.T) {
	shortenLockTimings(t)
	lockPath := filepath.Join(t.TempDir(), "auth.json.lock")
	lock, err := acquireFileLock(context.Background(), lockPath)
	if err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-time.Minute)
	if err := os.Chtimes(lockPath, old, old); err != nil {
		t.Fatal(err)
	}
	time.Sleep(100 * time.Millisecond)
	info, err := os.Stat(lockPath)
	if err != nil || time.Since(info.ModTime()) > staleLockAge {
		t.Fatalf("held lock went stale: %v %v", info, err)
	}
	if err := lock.Release(); err != nil {
		t.Fatal(err)
	}
}

func TestFileWritesThroughSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "real.json")
	link := filepath.Join(dir, "auth.json")
	if err := os.WriteFile(target, []byte(`{}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, link); err != nil {
		t.Skip("symlinks unavailable:", err)
	}
	if err := writeKey(context.Background(), NewFile(link)); err != nil {
		t.Fatal(err)
	}
	if info, err := os.Lstat(link); err != nil || info.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("symlink replaced by a regular file: %v %v", info, err)
	}
	if cred, ok, err := NewFile(target).Read(context.Background(), "p"); err != nil || !ok || cred.Key != "k" {
		t.Fatalf("target not updated: %+v %v %v", cred, ok, err)
	}
}
