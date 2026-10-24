package authstore

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

// keyCommandTimeout bounds one "!command" key resolution, as pi does.
const keyCommandTimeout = 10 * time.Second

// Option configures a File.
type Option func(*File)

// WithKeyCommands lets Read resolve an api_key value that starts with "!" by
// running the rest as a shell command (sh -c; cmd /C on Windows) and using
// its trimmed stdout, as pi does for values like "!op read op://vault/key".
// The command runs with the process's environment and a 10 second bound;
// stdin is empty and stderr is discarded. A successful result is cached for
// the File's lifetime. Without this option such a value is an error, since
// it would run code named by a file.
func WithKeyCommands() Option {
	return func(f *File) { f.keyCommands = true }
}

// keyResolver holds a File's key-indirection settings and command cache.
type keyResolver struct {
	keyCommands bool
	lookupEnv   func(string) (string, bool)
	runCommand  func(ctx context.Context, command string) (string, error)

	cacheMu  sync.Mutex
	resolved map[string]string   // "!command" → its output
	inflight map[string]*keyCall // commands running now, shared by waiters
}

// keyCall is one running key command; concurrent first reads wait for it
// instead of running the command again (a biometric prompt should appear
// once).
type keyCall struct {
	done  chan struct{}
	value string
	err   error
}

// resolveKey applies pi's api_key indirection (resolve-config-value.ts):
//   - "!command" runs the command (only with WithKeyCommands);
//   - otherwise "$NAME" and "${NAME}" interpolate environment variables,
//     looked up first in the entry's "env" map, then the process
//     environment (an empty value counts as unset); "$$" is a literal "$"
//     and "$!" a literal "!"; any other "$" is literal.
//
// A literal key without "$" is returned as is. Errors name the provider and
// the unset variables, never a resolved value.
func (f *File) resolveKey(ctx context.Context, providerID, key string, env map[string]string) (string, error) {
	if command, ok := strings.CutPrefix(key, "!"); ok {
		if !f.keyCommands {
			return "", fmt.Errorf("%w: auth file key for %q is a shell command (\"!...\"); allow it with authstore.WithKeyCommands", llm.ErrAuth, providerID)
		}
		return f.commandKey(ctx, providerID, command)
	}
	if !strings.Contains(key, "$") {
		return key, nil
	}
	lookup := f.lookupEnv
	if lookup == nil {
		lookup = os.LookupEnv
	}
	var out strings.Builder
	var missing []string
	for index := 0; index < len(key); {
		dollar := strings.IndexByte(key[index:], '$')
		if dollar < 0 {
			out.WriteString(key[index:])
			break
		}
		dollar += index
		out.WriteString(key[index:dollar])
		rest := key[dollar+1:]
		switch {
		case strings.HasPrefix(rest, "$"), strings.HasPrefix(rest, "!"):
			out.WriteByte(rest[0])
			index = dollar + 2
		case strings.HasPrefix(rest, "{"):
			end := strings.IndexByte(rest, '}')
			if end < 0 {
				out.WriteByte('$')
				index = dollar + 1
				continue
			}
			name := rest[1:end]
			if envName(name) == len(name) && name != "" {
				missing = appendValue(&out, missing, name, env, lookup)
			} else {
				out.WriteString(key[dollar : dollar+2+end])
			}
			index = dollar + 2 + end
		default:
			if n := envName(rest); n > 0 {
				missing = appendValue(&out, missing, rest[:n], env, lookup)
				index = dollar + 1 + n
				continue
			}
			out.WriteByte('$')
			index = dollar + 1
		}
	}
	if len(missing) > 0 {
		return "", fmt.Errorf("%w: auth file key for %q references unset environment variable(s) %s", llm.ErrAuth, providerID, strings.Join(missing, ", "))
	}
	return out.String(), nil
}

// appendValue writes the variable's value, or records it as missing.
func appendValue(out *strings.Builder, missing []string, name string, env map[string]string, lookup func(string) (string, bool)) []string {
	if value := env[name]; value != "" {
		out.WriteString(value)
		return missing
	}
	if value, ok := lookup(name); ok && value != "" {
		out.WriteString(value)
		return missing
	}
	for _, seen := range missing {
		if seen == name {
			return missing
		}
	}
	return append(missing, name)
}

// envName returns the length of the environment-variable name
// ([A-Za-z_][A-Za-z0-9_]*) at the start of s.
func envName(s string) int {
	for i := 0; i < len(s); i++ {
		c := s[i]
		letter := c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
		if !letter && (i == 0 || c < '0' || c > '9') {
			return i
		}
	}
	return len(s)
}

func (f *File) commandKey(ctx context.Context, providerID, command string) (string, error) {
	f.cacheMu.Lock()
	if value, ok := f.resolved[command]; ok {
		f.cacheMu.Unlock()
		return value, nil
	}
	if call, ok := f.inflight[command]; ok {
		f.cacheMu.Unlock()
		select {
		case <-call.done:
			return call.value, call.err
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
	call := &keyCall{done: make(chan struct{})}
	if f.inflight == nil {
		f.inflight = make(map[string]*keyCall)
	}
	f.inflight[command] = call
	f.cacheMu.Unlock()

	call.value, call.err = f.runKey(ctx, providerID, command)
	f.cacheMu.Lock()
	delete(f.inflight, command)
	if call.err == nil {
		if f.resolved == nil {
			f.resolved = make(map[string]string)
		}
		f.resolved[command] = call.value
	}
	f.cacheMu.Unlock()
	close(call.done)
	return call.value, call.err
}

func (f *File) runKey(ctx context.Context, providerID, command string) (string, error) {
	run := f.runCommand
	if run == nil {
		run = runKeyCommand
	}
	value, err := run(ctx, command)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return "", ctxErr
		}
		return "", fmt.Errorf("%w: auth file key command for %q failed: %v", llm.ErrAuth, providerID, err)
	}
	if value == "" {
		return "", fmt.Errorf("%w: auth file key command for %q printed nothing", llm.ErrAuth, providerID)
	}
	return value, nil
}

// runKeyCommand runs command through the platform shell and returns its
// trimmed stdout. Only the failure kind is reported: the command's output
// may be the secret.
func runKeyCommand(ctx context.Context, command string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, keyCommandTimeout)
	defer cancel()
	shell, flag := "sh", "-c"
	if runtime.GOOS == "windows" {
		shell, flag = "cmd", "/C"
	}
	cmd := exec.CommandContext(ctx, shell, flag, command)
	var stdout bytes.Buffer
	cmd.Stdin = nil
	cmd.Stdout = &stdout
	cmd.Stderr = io.Discard
	cmd.WaitDelay = time.Second
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return "", fmt.Errorf("timed out or cancelled: %w", ctx.Err())
		}
		if exit, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("exit status %d", exit.ExitCode())
		}
		return "", fmt.Errorf("could not start the shell: %w", err)
	}
	return strings.TrimSpace(stdout.String()), nil
}
