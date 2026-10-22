package authstore

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"

	llm "github.com/pkieltyka/go-llm"
)

// credentialKeys are the entry fields a credential write owns. Every other
// key in an entry (pi's "scopes", go-llm's "model"/"base_url", future
// fields) is preserved, unless the credential type changes.
var credentialKeys = []string{"type", "key", "api_key", "access", "refresh", "expires", "accountId", "clientId"}

// configKeys survive even a credential type change: they configure the
// provider, not the credential.
var configKeys = map[string]bool{"model": true, "base_url": true, "baseUrl": true}

// File is an llm.CredentialStore backed by a pi-compatible auth.json file:
// either a bare {"<provider>": {...}} map or a {"providers": {...}} wrapper,
// whichever the file already uses (new files use the bare form, like pi).
//
// Read parses the file on every call, so it sees credentials refreshed by
// other processes. Modify and Delete hold an in-process lock and pi's
// cross-process lock on "<path>.lock" (a directory whose mtime the holder
// refreshes; see lock.go), so pi and go-llm exclude each other on a shared
// file. Under the lock they re-read the file and write atomically (temporary
// file, fsync, rename, directory fsync) with mode 0600, through a symlinked
// path to its target. Other providers' entries and unknown fields are
// preserved byte-for-byte in meaning.
type File struct {
	path string
	mu   chan struct{}
}

var _ llm.CredentialStore = (*File)(nil)

// NewFile returns a store for the auth file at path. The file need not
// exist until the first write.
func NewFile(path string) *File {
	return &File{path: path, mu: make(chan struct{}, 1)}
}

// Path returns the backing file path.
func (f *File) Path() string { return f.path }

func (f *File) String() string { return "authstore: file " + f.path + " (redacted)" }

// Read returns the stored credential for providerID. A missing file is an
// empty store.
func (f *File) Read(ctx context.Context, providerID string) (llm.AuthCredential, bool, error) {
	if err := contextErr(ctx); err != nil {
		return llm.AuthCredential{}, false, err
	}
	doc, err := f.load()
	if err != nil {
		return llm.AuthCredential{}, false, err
	}
	raw, ok := doc.providers[providerID]
	if !ok {
		return llm.AuthCredential{}, false, nil
	}
	var cred llm.AuthCredential
	if err := json.Unmarshal(raw, &cred); err != nil {
		return llm.AuthCredential{}, false, fmt.Errorf("authstore: parse %s credential: %w", providerID, sanitizeJSONError(err))
	}
	return cred, true, nil
}

// Modify serializes fn per file across goroutines and processes. ctx bounds
// only the wait for the locks; once fn runs, its result is written even if
// ctx is cancelled.
func (f *File) Modify(ctx context.Context, providerID string, fn func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error)) (llm.AuthCredential, bool, error) {
	if fn == nil {
		return llm.AuthCredential{}, false, fmt.Errorf("%w: nil modify function", llm.ErrBadRequest)
	}
	var result llm.AuthCredential
	var exists bool
	err := f.locked(ctx, func(doc *authDocument) (bool, error) {
		current, ok, err := doc.credential(providerID)
		if err != nil {
			return false, err
		}
		next, write, err := fn(current, ok)
		if err != nil {
			return false, err
		}
		if !write {
			result, exists = current, ok
			return false, nil
		}
		if err := doc.setCredential(providerID, next); err != nil {
			return false, err
		}
		result, exists = next, true
		return true, nil
	})
	if err != nil {
		return llm.AuthCredential{}, false, err
	}
	return result, exists, nil
}

// Delete removes providerID's entry (logout), preserving the rest of the
// file.
func (f *File) Delete(ctx context.Context, providerID string) error {
	return f.locked(ctx, func(doc *authDocument) (bool, error) {
		if _, ok := doc.providers[providerID]; !ok {
			return false, nil
		}
		delete(doc.providers, providerID)
		return true, nil
	})
}

func (f *File) locked(ctx context.Context, fn func(*authDocument) (bool, error)) error {
	release, err := acquireChannel(ctx, f.mu)
	if err != nil {
		return err
	}
	defer release()
	lock, err := acquireFileLock(ctx, f.path+".lock")
	if err != nil {
		return err
	}
	defer func() { _ = lock.Release() }()

	doc, err := f.load()
	if err != nil {
		return err
	}
	changed, err := fn(&doc)
	if err != nil || !changed {
		return err
	}
	data, err := doc.encode()
	if err != nil {
		return err
	}
	target := f.path
	if resolved, err := filepath.EvalSymlinks(f.path); err == nil {
		target = resolved // keep a dotfile symlink pointing at the store
	}
	return writeAtomic(target, data)
}

type authDocument struct {
	root      map[string]json.RawMessage
	providers map[string]json.RawMessage
	wrapped   bool
}

func (f *File) load() (authDocument, error) {
	doc := authDocument{root: map[string]json.RawMessage{}, providers: map[string]json.RawMessage{}}
	data, err := os.ReadFile(f.path)
	if errors.Is(err, fs.ErrNotExist) {
		return doc, nil
	}
	if err != nil {
		return authDocument{}, err
	}
	data = bytes.TrimPrefix(data, []byte("\xef\xbb\xbf"))
	if len(bytes.TrimSpace(data)) == 0 {
		return doc, nil
	}
	if err := json.Unmarshal(data, &doc.root); err != nil {
		return authDocument{}, fmt.Errorf("authstore: parse %s: %w", f.path, sanitizeJSONError(err))
	}
	if doc.root == nil {
		doc.root = map[string]json.RawMessage{}
	}
	if raw, ok := doc.root["providers"]; ok {
		doc.wrapped = true
		if err := json.Unmarshal(raw, &doc.providers); err != nil {
			return authDocument{}, fmt.Errorf("authstore: parse %s providers: %w", f.path, sanitizeJSONError(err))
		}
		if doc.providers == nil {
			doc.providers = map[string]json.RawMessage{}
		}
	} else {
		doc.providers = doc.root
	}
	return doc, nil
}

func (doc *authDocument) credential(providerID string) (llm.AuthCredential, bool, error) {
	raw, ok := doc.providers[providerID]
	if !ok {
		return llm.AuthCredential{}, false, nil
	}
	var cred llm.AuthCredential
	if err := json.Unmarshal(raw, &cred); err != nil {
		return llm.AuthCredential{}, false, fmt.Errorf("authstore: parse %s credential: %w", providerID, sanitizeJSONError(err))
	}
	return cred, true, nil
}

func (doc *authDocument) setCredential(providerID string, cred llm.AuthCredential) error {
	entry := map[string]json.RawMessage{}
	if raw, ok := doc.providers[providerID]; ok {
		_ = json.Unmarshal(raw, &entry)
		if entry == nil {
			entry = map[string]json.RawMessage{}
		}
	}
	var previousType string
	if raw, ok := entry["type"]; ok {
		_ = json.Unmarshal(raw, &previousType)
	}
	if previousType != "" && previousType != cred.Type {
		for key := range entry {
			if !configKeys[key] {
				delete(entry, key)
			}
		}
	}
	for _, key := range credentialKeys {
		delete(entry, key)
	}
	set := func(key string, value any) error {
		encoded, err := json.Marshal(value)
		if err != nil {
			return err
		}
		entry[key] = encoded
		return nil
	}
	fields := []struct {
		key   string
		value string
	}{
		{"type", cred.Type}, {"key", cred.Key}, {"access", cred.Access}, {"refresh", cred.Refresh},
		{"accountId", cred.AccountID}, {"clientId", cred.ClientID},
	}
	for _, field := range fields {
		if field.value == "" {
			continue
		}
		if err := set(field.key, field.value); err != nil {
			return err
		}
	}
	if cred.Expires != 0 {
		if err := set("expires", cred.Expires); err != nil {
			return err
		}
	}
	if cred.Model != "" {
		if err := set("model", cred.Model); err != nil {
			return err
		}
	}
	if cred.BaseURL != "" {
		if _, hasCamel := entry["baseUrl"]; hasCamel {
			if err := set("baseUrl", cred.BaseURL); err != nil {
				return err
			}
		} else if err := set("base_url", cred.BaseURL); err != nil {
			return err
		}
	}
	encoded, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	doc.providers[providerID] = encoded
	return nil
}

func (doc *authDocument) encode() ([]byte, error) {
	root := doc.root
	if doc.wrapped {
		providers, err := json.Marshal(doc.providers)
		if err != nil {
			return nil, err
		}
		root["providers"] = providers
	} else {
		root = doc.providers
	}
	data, err := json.MarshalIndent(root, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(data, '\n'), nil
}

// writeAtomic replaces path with data via a synced temporary file in the
// same directory, keeping the file private (0600 or stricter).
func writeAtomic(path string, data []byte) error {
	mode := os.FileMode(0o600)
	if info, err := os.Stat(path); err == nil {
		if perm := info.Mode().Perm() & 0o600; perm != 0 {
			mode = perm
		}
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".authstore-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(mode); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	return syncDirectory(dir)
}

// sanitizeJSONError keeps a JSON syntax position without quoting file
// content, which may contain secrets.
func sanitizeJSONError(err error) error {
	var syntax *json.SyntaxError
	if errors.As(err, &syntax) {
		return fmt.Errorf("invalid JSON at offset %d", syntax.Offset)
	}
	var typeErr *json.UnmarshalTypeError
	if errors.As(err, &typeErr) {
		return fmt.Errorf("invalid JSON type for field %q", typeErr.Field)
	}
	return errors.New("invalid credential JSON")
}
