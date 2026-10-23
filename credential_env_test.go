package llm

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

// sdkCredentialEnvNames are listed by CredentialEnvNames because a wrapped
// SDK reads them in-process (openai-go DefaultClientOptions), not because
// go-llm source reads them, so the source scan cannot find them.
var sdkCredentialEnvNames = []string{"OPENAI_ADMIN_KEY", "OPENAI_WEBHOOK_SECRET"}

// TestCredentialEnvNamesMatchSources fails when a provider or CLI source file
// reads an environment variable that CredentialEnvNames does not list. Getenv
// and LookupEnv arguments may be string literals, identifiers bound to string
// constants in the same package directory, or package-qualified identifiers
// resolved in the scanned package of that name. An identifier that cannot be
// resolved fails the test rather than passing silently.
func TestCredentialEnvNamesMatchSources(t *testing.T) {
	call := regexp.MustCompile(`os\.(?:Getenv|LookupEnv)\(\s*([^)\s]+)\s*\)`)
	literal := regexp.MustCompile(`^"([A-Z][A-Z0-9_]*)"$`)
	identifier := regexp.MustCompile(`^(?:([A-Za-z_]\w*)\.)?([A-Za-z_]\w*)$`)
	binding := regexp.MustCompile(`(?m)\b([A-Za-z_]\w*)\s*=\s*"([A-Z][A-Z0-9_]*)"`)

	// bindings maps package directory -> identifier -> constant value.
	bindings := map[string]map[string]string{}
	type use struct{ arg, path string }
	var uses []use
	for _, root := range []string{"providers", "cmd"} {
		err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			dir := filepath.Dir(path)
			if bindings[dir] == nil {
				bindings[dir] = map[string]string{}
			}
			for _, match := range binding.FindAllStringSubmatch(string(data), -1) {
				bindings[dir][match[1]] = match[2]
			}
			for _, match := range call.FindAllStringSubmatch(string(data), -1) {
				uses = append(uses, use{arg: match[1], path: path})
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if len(uses) == 0 {
		t.Fatal("source scan found no environment reads; the scan is broken")
	}

	found := map[string]string{}
	for _, u := range uses {
		if m := literal.FindStringSubmatch(u.arg); m != nil {
			found[m[1]] = u.path
			continue
		}
		m := identifier.FindStringSubmatch(u.arg)
		if m == nil {
			t.Errorf("%s: cannot classify environment read argument %s", u.path, u.arg)
			continue
		}
		var name string
		var ok bool
		if m[1] == "" {
			name, ok = bindings[filepath.Dir(u.path)][m[2]]
		} else {
			for dir, values := range bindings {
				if filepath.Base(dir) == m[1] {
					name, ok = values[m[2]]
					break
				}
			}
		}
		if !ok {
			t.Errorf("%s: environment read argument %s does not resolve to a string constant", u.path, u.arg)
			continue
		}
		found[name] = u.path
	}

	names := CredentialEnvNames()
	for name, path := range found {
		if !slices.Contains(names, name) {
			t.Errorf("%s reads %s, which CredentialEnvNames does not list", path, name)
		}
	}
	for _, name := range names {
		if _, ok := found[name]; !ok && !slices.Contains(sdkCredentialEnvNames, name) {
			t.Errorf("CredentialEnvNames lists %s, but no provider or CLI source reads it", name)
		}
	}
	names[0] = "MUTATED"
	if CredentialEnvNames()[0] == "MUTATED" {
		t.Fatal("CredentialEnvNames must return a copy")
	}
}
