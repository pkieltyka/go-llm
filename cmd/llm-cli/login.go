package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
	"github.com/pkieltyka/go-llm/providers/openai"
)

// loginFlowFactory builds the Sign in with ChatGPT flow; tests replace it.
type loginFlowFactory func(deviceID string, timeout time.Duration) (llm.LoginFlow, error)

func newOpenAILoginFlow(deviceID string, timeout time.Duration) (llm.LoginFlow, error) {
	opts := []openai.LoginOption{openai.WithLoginDeviceID(deviceID), openai.WithLoginAgentName("llm-cli")}
	if timeout > 0 {
		opts = append(opts, openai.WithLoginTimeout(timeout))
	}
	return openai.NewLoginFlow(opts...)
}

var uuidPattern = regexp.MustCompile(`^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$`)

// runLogin signs in with ChatGPT and stores the credential under "openai"
// in the pi-compatible auth file, where --auth-file (and pi) find it. The
// credential is never printed.
func (a app) runLogin(ctx context.Context, cfg loginConfig) error {
	if cfg.provider != "" && cfg.provider != "openai" {
		return fmt.Errorf("%w: login supports -p openai (Sign in with ChatGPT) only; other providers use API keys", llm.ErrBadRequest)
	}
	// Like every other command, login uses an auth file only when named.
	path := cfg.authFile
	if path == "" {
		return fmt.Errorf("%w: login needs --auth-file naming the auth.json to store the credential in", llm.ErrBadRequest)
	}
	if cfg.timeout < 0 {
		return fmt.Errorf("%w: --timeout must not be negative", llm.ErrBadRequest)
	}
	// Prove the auth file can be locked and parsed before the browser
	// sign-in, so a bad path or file never discards a minted credential.
	store := authstore.NewFile(path)
	if _, _, err := store.Modify(ctx, "openai", func(current llm.AuthCredential, ok bool) (llm.AuthCredential, bool, error) {
		return current, false, nil
	}); err != nil {
		return fmt.Errorf("auth file %s: %w", path, err)
	}
	deviceID := cfg.deviceID
	if deviceID == "" {
		id, err := a.installationDeviceID()
		if err != nil {
			return err
		}
		deviceID = id
	}
	factory := a.loginFlowFactory
	if factory == nil {
		factory = newOpenAILoginFlow
	}
	flow, err := factory(deviceID, cfg.timeout)
	if err != nil {
		return err
	}
	defer flow.Cancel()
	authorization, err := flow.Begin(ctx)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(a.stderr, "Open this URL in a browser and sign in with ChatGPT:\n\n  %s\n\n"+
		"Waiting for the sign-in to complete. If the browser cannot reach this machine,\n"+
		"paste the final redirect URL here and press Enter.\n", authorization.URL()); err != nil {
		return fmt.Errorf("write login instructions: %w", err)
	}
	if a.stdin != nil {
		go a.submitPastedCallbacks(ctx, flow)
	}
	credential, err := flow.Complete(ctx)
	if err != nil {
		return err
	}
	if _, _, err := store.Modify(ctx, "openai", func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
		return credential, true, nil
	}); err != nil {
		return fmt.Errorf("save credential: %w", err)
	}
	if _, err := fmt.Fprintf(a.stderr, "Signed in with ChatGPT; saved the \"openai\" credential to %s\n", path); err != nil {
		return fmt.Errorf("write login result: %w", err)
	}
	return nil
}

// submitPastedCallbacks forwards each non-empty stdin line to the flow as a
// manual callback, the fallback when the browser cannot reach loopback.
func (a app) submitPastedCallbacks(ctx context.Context, flow llm.LoginFlow) {
	scanner := bufio.NewScanner(a.stdin)
	scanner.Buffer(make([]byte, 0, 4096), 64<<10)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if err := flow.Submit(ctx, line); err != nil {
			_, _ = fmt.Fprintf(a.stderr, "callback not accepted: %v\n", err)
			continue
		}
		return
	}
}

// installationDeviceID returns this installation's stable login device ID,
// creating it on first use in <user config dir>/llm-cli/device-id. OpenAI
// expects the same ID on every login from one installation.
func (a app) installationDeviceID() (string, error) {
	configDir := a.configDir
	if configDir == nil {
		configDir = os.UserConfigDir
	}
	base, err := configDir()
	if err != nil {
		return "", fmt.Errorf("locate config directory for the login device ID (or pass --device-id): %w", err)
	}
	path := filepath.Join(base, "llm-cli", "device-id")
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		id := strings.TrimSpace(string(data))
		if !uuidPattern.MatchString(id) {
			return "", fmt.Errorf("%w: %s does not hold a UUID; remove it or pass --device-id", llm.ErrBadRequest, path)
		}
		return id, nil
	case !errors.Is(err, fs.ErrNotExist):
		return "", fmt.Errorf("read login device ID: %w", err)
	}
	id, err := newUUID()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return "", fmt.Errorf("create login device ID: %w", err)
	}
	if err := os.WriteFile(path, []byte(id+"\n"), 0o600); err != nil {
		return "", fmt.Errorf("create login device ID: %w", err)
	}
	return id, nil
}

// newUUID returns a random (version 4) UUID.
func newUUID() (string, error) {
	var b [16]byte
	if _, err := io.ReadFull(rand.Reader, b[:]); err != nil {
		return "", fmt.Errorf("generate login device ID: %w", err)
	}
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16]), nil
}
