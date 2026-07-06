package e2e

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	llm "github.com/pkieltyka/go-llm"
)

// persistMu serializes credential-file rewrites: OAuth refreshes can fire
// from concurrent requests and read-modify-write must not interleave.
var persistMu sync.Mutex

// PersistOnRefresh returns an onRefresh callback that writes a rotated OAuth
// credential for provider back into the credential file at path, preserving
// every other provider entry and any unknown fields. Live suites MUST wire
// this into WithOAuth: providers rotate refresh tokens on every refresh, and
// discarding the rotated token strands the stored credential.
//
// Failures are reported through logf (when non-nil) rather than aborting the
// calling request.
func PersistOnRefresh(path, provider string, logf func(format string, args ...any)) func(llm.AuthCredential) {
	return func(cred llm.AuthCredential) {
		if err := persistRefreshedCredential(path, provider, cred); err != nil && logf != nil {
			logf("persist refreshed %s credential to %s: %v", provider, path, err)
		}
	}
}

// persistRefreshedCredential rewrites only the given provider entry inside
// the credential file, keeping all other entries and unknown fields intact,
// then replaces the file atomically (temp file + rename, mode 0600).
func persistRefreshedCredential(path, provider string, cred llm.AuthCredential) error {
	persistMu.Lock()
	defer persistMu.Unlock()

	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal(data, &root); err != nil {
		return fmt.Errorf("parse credential file: %w", err)
	}

	// Support both the bare provider map and the {"providers": ...} wrapper.
	entries := root
	wrapped := false
	if rawProviders, ok := root["providers"]; ok {
		providers := map[string]json.RawMessage{}
		if err := json.Unmarshal(rawProviders, &providers); err != nil {
			return fmt.Errorf("parse providers wrapper: %w", err)
		}
		entries = providers
		wrapped = true
	}

	entry := map[string]json.RawMessage{}
	if raw, ok := entries[provider]; ok {
		if err := json.Unmarshal(raw, &entry); err != nil {
			return fmt.Errorf("parse %s entry: %w", provider, err)
		}
	}
	setStringField(entry, "type", "oauth")
	setStringField(entry, "access", cred.Access)
	if cred.Refresh != "" {
		setStringField(entry, "refresh", cred.Refresh)
	}
	if cred.Expires > 0 {
		entry["expires"], err = json.Marshal(cred.Expires)
		if err != nil {
			return err
		}
	}
	if cred.AccountID != "" {
		setStringField(entry, "accountId", cred.AccountID)
	}

	rawEntry, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	entries[provider] = rawEntry
	if wrapped {
		rawProviders, err := json.Marshal(entries)
		if err != nil {
			return err
		}
		root["providers"] = rawProviders
	}
	out, err := json.MarshalIndent(root, "", "  ")
	if err != nil {
		return err
	}
	out = append(out, '\n')

	tmp, err := os.CreateTemp(filepath.Dir(path), ".credentials-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(out); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}

func setStringField(entry map[string]json.RawMessage, key, value string) {
	raw, err := json.Marshal(value)
	if err != nil {
		return
	}
	entry[key] = raw
}
