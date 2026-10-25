package e2e

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
)

// namedAnthropicStore is confined to the opt-in runner and synthetic tests.
// File.Modify preserves raw unknown fields and holds its cross-process lock
// across token rotation and persistence. This lease rejects foreign changes.
type namedAnthropicStore struct {
	file    *authstore.File
	current llm.AuthCredential
	closed  bool
	lock    chan struct{}
}

var errNamedProfileChanged = errors.New("named OAuth test profile changed or unavailable")

func newNamedAnthropicStore(ctx context.Context, path string) (*namedAnthropicStore, error) {
	file := authstore.NewFile(path)
	cred, ok, err := file.Read(ctx, "anthropic2")
	if err != nil || !ok || cred.Type != "oauth" {
		return nil, errNamedProfileChanged
	}
	return &namedAnthropicStore{file: file, current: cred, lock: make(chan struct{}, 1)}, nil
}

func (s *namedAnthropicStore) acquire(ctx context.Context, id string) error {
	if id != "anthropic" {
		return errNamedProfileChanged
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case s.lock <- struct{}{}:
	case <-ctx.Done():
		return ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		<-s.lock
		return err
	}
	if s.closed {
		<-s.lock
		return errNamedProfileChanged
	}
	return nil
}
func (s *namedAnthropicStore) Read(ctx context.Context, id string) (llm.AuthCredential, bool, error) {
	if err := s.acquire(ctx, id); err != nil {
		return llm.AuthCredential{}, false, err
	}
	defer func() { <-s.lock }()
	cred, ok, err := s.file.Read(ctx, "anthropic2")
	if err != nil || !ok || !reflect.DeepEqual(cred, s.current) {
		s.closed = true
		return llm.AuthCredential{}, false, errNamedProfileChanged
	}
	return cred, true, nil
}
func (s *namedAnthropicStore) Modify(ctx context.Context, id string, fn func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error)) (llm.AuthCredential, bool, error) {
	if fn == nil {
		return llm.AuthCredential{}, false, errNamedProfileChanged
	}
	if err := s.acquire(ctx, id); err != nil {
		return llm.AuthCredential{}, false, err
	}
	defer func() { <-s.lock }()
	next, ok, err := s.file.Modify(ctx, "anthropic2", func(current llm.AuthCredential, exists bool) (llm.AuthCredential, bool, error) {
		if !exists || !reflect.DeepEqual(current, s.current) {
			return llm.AuthCredential{}, false, errNamedProfileChanged
		}
		return fn(current, true)
	})
	if err != nil {
		s.closed = true
		return llm.AuthCredential{}, false, errNamedProfileChanged
	}
	s.current = next
	s.current.Env = next.CloneEnv()
	return next, ok, nil
}
func (*namedAnthropicStore) Delete(context.Context, string) error {
	return errors.New("named OAuth runner does not delete credentials")
}

func TestNamedAnthropicStorePreservesOtherProfilesAndUnknownFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "auth.json")
	original := `{"metadata":{"keep":true},"providers":{"other":{"type":"api_key","key":"synthetic-other","extra":42},"anthropic2":{"type":"oauth","access":"old","refresh":"old-refresh","expires":1,"unknown":{"keep":true}}}}`
	if err := os.WriteFile(path, []byte(original), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	store, err := newNamedAnthropicStore(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.Read(ctx, "other"); err == nil {
		t.Fatal("wrong provider accepted")
	}
	_, _, err = store.Modify(ctx, "anthropic", func(c llm.AuthCredential, _ bool) (llm.AuthCredential, bool, error) {
		cancel()
		c.Access = "renewed"
		c.Refresh = "rotated"
		return c, true, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	cred, ok, err := store.Read(context.Background(), "anthropic")
	if err != nil || !ok || cred.Refresh != "rotated" {
		t.Fatal("rotation not persisted after cancel")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var before, after map[string]any
	if json.Unmarshal([]byte(original), &before) != nil || json.Unmarshal(data, &after) != nil {
		t.Fatal("invalid JSON")
	}
	bp := before["providers"].(map[string]any)
	ap := after["providers"].(map[string]any)
	if !reflect.DeepEqual(before["metadata"], after["metadata"]) || !reflect.DeepEqual(bp["other"], ap["other"]) || !reflect.DeepEqual(bp["anthropic2"].(map[string]any)["unknown"], ap["anthropic2"].(map[string]any)["unknown"]) {
		t.Fatal("unowned data changed")
	}
}

func TestNamedAnthropicStoreRejectsReplacement(t *testing.T) {
	path := filepath.Join(t.TempDir(), "auth.json")
	file := authstore.NewFile(path)
	ctx := context.Background()
	_, _, err := file.Modify(ctx, "anthropic2", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
		return llm.AuthCredential{Type: "oauth", Access: "original"}, true, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := newNamedAnthropicStore(ctx, path)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = file.Modify(ctx, "anthropic2", func(c llm.AuthCredential, _ bool) (llm.AuthCredential, bool, error) {
		c.Access = "replacement"
		return c, true, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	called := false
	if _, _, err := store.Modify(ctx, "anthropic", func(c llm.AuthCredential, _ bool) (llm.AuthCredential, bool, error) {
		called = true
		return c, true, nil
	}); err == nil || called {
		t.Fatal("replacement was overwritten")
	}
	if _, _, err := store.Read(ctx, "anthropic"); err == nil {
		t.Fatal("lost lease recovered silently")
	}
}
