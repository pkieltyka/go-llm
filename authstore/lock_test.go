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

func TestFileRecoversStaleLock(t *testing.T) {
	shortenLockTimings(t)
	path := filepath.Join(t.TempDir(), "auth.json")
	if err := os.Mkdir(path+".lock", 0o700); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-time.Minute)
	if err := os.Chtimes(path+".lock", old, old); err != nil {
		t.Fatal(err)
	}
	if err := writeKey(context.Background(), NewFile(path)); err != nil {
		t.Fatalf("Modify over a stale lock = %v", err)
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
	time.Sleep(2 * staleLockAge)
	info, err := os.Stat(lockPath)
	if err != nil || time.Since(info.ModTime()) > staleLockAge/2 {
		t.Fatalf("held lock went stale: %v %v", info, err)
	}
	if err := lock.Check(); err != nil {
		t.Fatalf("Check on a held lock = %v", err)
	}
	if err := lock.Release(); err != nil {
		t.Fatal(err)
	}
}

// pi's heartbeat cadence must suit its synchronous reload, which treats a
// lock older than proper-lockfile's default 10 s as stale.
func TestLockUpdateOutpacesPiSyncStaleAge(t *testing.T) {
	if lockUpdateEvery*2 > 10*time.Second {
		t.Fatalf("lockUpdateEvery = %v, want at most 5s", lockUpdateEvery)
	}
}

// A holder that stalled while another process judged its lock stale and
// took it over must neither write nor remove the new holder's lock.
func TestCompromisedLockWritesNothingAndKeepsForeignLock(t *testing.T) {
	shortenLockTimings(t)
	path := filepath.Join(t.TempDir(), "auth.json")
	f := NewFile(path)
	if err := writeKey(context.Background(), f); err != nil {
		t.Fatal(err)
	}
	_, _, err := f.Modify(context.Background(), "p", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
		// Another process removes the "stale" lock and takes it.
		if err := os.Remove(path + ".lock"); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(path+".lock", 0o700); err != nil {
			t.Fatal(err)
		}
		past := time.Now().Add(-time.Second)
		if err := os.Chtimes(path+".lock", past, past); err != nil {
			t.Fatal(err)
		}
		return llm.AuthCredential{Type: "api_key", Key: "overwritten"}, true, nil
	})
	if !errors.Is(err, errLockCompromised) {
		t.Fatalf("Modify under a compromised lock = %v, want errLockCompromised", err)
	}
	if _, err := os.Stat(path + ".lock"); err != nil {
		t.Fatalf("the other process's lock was removed: %v", err)
	}
	if cred, _, err := f.Read(context.Background(), "p"); err != nil || cred.Key != "k" {
		t.Fatalf("stored credential = %+v, %v; want the original", cred, err)
	}
}

// pi rewrites auth.json in place under its lock; a read that catches the
// truncated file waits for the lock and sees the finished write.
func TestFileReadWaitsOutInPlaceWriter(t *testing.T) {
	shortenLockTimings(t)
	path := filepath.Join(t.TempDir(), "auth.json")
	for name, partial := range map[string]string{"empty": "", "partial": `{"p": {"type": "api_`} {
		t.Run(name, func(t *testing.T) {
			if err := os.Mkdir(path+".lock", 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, []byte(partial), 0o600); err != nil {
				t.Fatal(err)
			}
			go func() {
				time.Sleep(50 * time.Millisecond)
				_ = os.WriteFile(path, []byte(`{"p": {"type": "api_key", "key": "k"}}`), 0o600)
				_ = os.Remove(path + ".lock")
			}()
			cred, ok, err := NewFile(path).Read(context.Background(), "p")
			if err != nil || !ok || cred.Key != "k" {
				t.Fatalf("Read during an in-place write = %+v, %v, %v", cred, ok, err)
			}
		})
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

// The first write through a dangling symlink creates its target and keeps
// the link (a dotfile pointing at a not-yet-created store).
func TestFileWritesThroughDanglingSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "store", "real.json")
	middle := filepath.Join(dir, "middle.json")
	link := filepath.Join(dir, "auth.json")
	if err := os.Symlink(filepath.Join("store", "real.json"), middle); err != nil {
		t.Skip("symlinks unavailable:", err)
	}
	if err := os.Symlink(middle, link); err != nil {
		t.Fatal(err)
	}
	if err := writeKey(context.Background(), NewFile(link)); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{link, middle} {
		if info, err := os.Lstat(path); err != nil || info.Mode()&os.ModeSymlink == 0 {
			t.Fatalf("%s replaced by a regular file: %v %v", path, info, err)
		}
	}
	if cred, ok, err := NewFile(target).Read(context.Background(), "p"); err != nil || !ok || cred.Key != "k" {
		t.Fatalf("target not created: %+v %v %v", cred, ok, err)
	}
}

// A relative link under a symlinked directory resolves ".." against the
// real directory, as the kernel does: the first write lands on the link's
// real target, keeps the link, and reads back.
func TestFileDanglingSymlinkResolvesDotDotPhysically(t *testing.T) {
	for name, tc := range map[string]struct {
		setup        func(t *testing.T, dir string)
		link, target string
	}{
		"symlinked parent": {
			// dotfiles/pi linked in as pi, pi/auth.json -> ../auth.json
			setup: func(t *testing.T, dir string) {
				mustMkdir(t, filepath.Join(dir, "dotfiles", "pi"))
				mustSymlink(t, filepath.Join(dir, "dotfiles", "pi"), filepath.Join(dir, "pi"))
				mustSymlink(t, "../auth.json", filepath.Join(dir, "pi", "auth.json"))
			},
			link: "pi/auth.json", target: "dotfiles/auth.json",
		},
		"dot-dot in the link value": {
			setup: func(t *testing.T, dir string) {
				mustMkdir(t, filepath.Join(dir, "real", "child"))
				mustSymlink(t, filepath.Join(dir, "real", "child"), filepath.Join(dir, "alias"))
				mustSymlink(t, "alias/../target.json", filepath.Join(dir, "auth.json"))
			},
			link: "auth.json", target: "real/target.json",
		},
	} {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			tc.setup(t, dir)
			link := filepath.Join(dir, tc.link)
			store := NewFile(link)
			if err := writeKey(context.Background(), store); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(filepath.Join(dir, tc.target)); err != nil {
				t.Fatalf("real target not written: %v", err)
			}
			if info, err := os.Lstat(link); err != nil || info.Mode()&os.ModeSymlink == 0 {
				t.Fatalf("link replaced: %v %v", info, err)
			}
			if cred, ok, err := store.Read(context.Background(), "p"); err != nil || !ok || cred.Key != "k" {
				t.Fatalf("read back through the link: %+v %v %v", cred, ok, err)
			}
		})
	}
}

// ".." after a missing element has no real directory to climb from: the
// write fails, as the kernel's lookup would, instead of landing where a read
// through the link cannot find it.
func TestFileDotDotAfterMissingElementFails(t *testing.T) {
	dir := t.TempDir()
	link := filepath.Join(dir, "auth.json")
	if err := os.Symlink("missing/../real.json", link); err != nil {
		t.Skip("symlinks unavailable:", err)
	}
	if err := writeKey(context.Background(), NewFile(link)); err == nil {
		t.Fatal("write through missing/.. succeeded")
	}
	if _, err := os.Stat(filepath.Join(dir, "real.json")); err == nil {
		t.Fatal("a file was written at the lexically cleaned path")
	}
}

// A relative store path with no directory (llm-cli --auth-file auth.json)
// is written as given on first login.
func TestFileRelativePathFirstWrite(t *testing.T) {
	t.Chdir(t.TempDir())
	store := NewFile("auth.json")
	if err := writeKey(context.Background(), store); err != nil {
		t.Fatal(err)
	}
	if cred, ok, err := store.Read(context.Background(), "p"); err != nil || !ok || cred.Key != "k" {
		t.Fatalf("read back: %+v %v %v", cred, ok, err)
	}
}

func mustMkdir(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o700); err != nil {
		t.Fatal(err)
	}
}

func mustSymlink(t *testing.T, value, link string) {
	t.Helper()
	if err := os.Symlink(value, link); err != nil {
		t.Skip("symlinks unavailable:", err)
	}
}

// A symlink loop fails the write and leaves the links alone.
func TestFileSymlinkLoopFailsWithoutWriting(t *testing.T) {
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a.json"), filepath.Join(dir, "b.json")
	if err := os.Symlink("b.json", a); err != nil {
		t.Skip("symlinks unavailable:", err)
	}
	if err := os.Symlink("a.json", b); err != nil {
		t.Fatal(err)
	}
	if err := writeKey(context.Background(), NewFile(a)); err == nil {
		t.Fatal("write through a symlink loop succeeded")
	}
	for _, path := range []string{a, b} {
		if info, err := os.Lstat(path); err != nil || info.Mode()&os.ModeSymlink == 0 {
			t.Fatalf("%s replaced: %v %v", path, info, err)
		}
	}
}
