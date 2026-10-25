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
	"text/tabwriter"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
	"github.com/pkieltyka/go-llm/providers/openai"
	"golang.org/x/term"
)

// Auth types, named as pi names them and as auth.json stores them.
const (
	authOAuth  = "oauth"
	authAPIKey = "api_key"
)

// loginProvider is one provider llm-cli can sign in to, described as pi
// describes it: a display name, whether it has an account (OAuth) sign-in,
// and the name of its API key.
type loginProvider struct {
	id         string
	name       string
	oauthLabel string // "" when the provider has no OAuth sign-in
	keyName    string
	// envName and envLabel ask for one more value after the key, stored in
	// the entry's env as pi stores it (Cloudflare's account ID).
	envName  string
	envLabel string
}

// loginProviders lists the providers in display order: the chat
// providers, then the classifier-only ones (FS §13C). Anthropic has no
// OAuth sign-in command here; applications use anthropic.NewLoginFlow.
// Existing Anthropic OAuth entries are consumed by the provider factory.
var loginProviders = []loginProvider{
	{id: "openai", name: "OpenAI", oauthLabel: "Sign in with ChatGPT", keyName: "OpenAI API key"},
	{id: "anthropic", name: "Anthropic", keyName: "Anthropic API key"},
	{id: "openrouter", name: "OpenRouter", keyName: "OpenRouter API key"},
	{id: "vllm", name: "vLLM", keyName: "vLLM API key"},
	{id: "chatcompletions", name: "Chat Completions", keyName: "API key"},
	{id: "typesafe", name: "TypeSafe", keyName: "TypeSafe API key"},
	{id: "cloudflare-workers-ai", name: "Cloudflare Workers AI", keyName: "Cloudflare API key", envName: "CLOUDFLARE_ACCOUNT_ID", envLabel: "Cloudflare account ID"},
	{id: "vercel-ai-gateway", name: "Vercel AI Gateway", keyName: "Vercel AI Gateway API key"},
	{id: "opencode", name: "OpenCode Zen", keyName: "OpenCode API key"},
	{id: "systemone", name: "System One server", keyName: "API key"},
}

func findLoginProvider(id string) (loginProvider, bool) {
	for _, provider := range loginProviders {
		if provider.id == id {
			return provider, true
		}
	}
	return loginProvider{}, false
}

func loginProviderIDs() string {
	ids := make([]string, len(loginProviders))
	for i, provider := range loginProviders {
		ids[i] = provider.id
	}
	return strings.Join(ids, ", ")
}

// normalizeAuthType accepts pi's "oauth"/"api_key" and the "api-key"
// spelling, and returns pi's form.
func normalizeAuthType(value string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "":
		return "", nil
	case authOAuth:
		return authOAuth, nil
	case authAPIKey, "api-key":
		return authAPIKey, nil
	}
	return "", fmt.Errorf("%w: --auth must be oauth or api-key, got %q", llm.ErrBadRequest, value)
}

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

// authFilePath is the auth file a command uses: --auth-file, else
// LLM_AUTH_FILE, else <user config dir>/llm-cli/auth.json. Point it at
// ~/.pi/agent/auth.json to share credentials with pi.
func (a app) authFilePath(flag string) (string, error) {
	if flag != "" {
		return flag, nil
	}
	if path := os.Getenv("LLM_AUTH_FILE"); path != "" {
		return path, nil
	}
	base, err := a.userConfigDir()
	if err != nil {
		return "", fmt.Errorf("locate the default auth file (or pass --auth-file): %w", err)
	}
	return filepath.Join(base, "llm-cli", "auth.json"), nil
}

// withAuthFile fills in the default auth file unless --api-key bypasses it,
// so a credential stored by 'llm-cli login' is used without flags.
func (a app) withAuthFile(cfg providerConfig) (providerConfig, error) {
	if cfg.apiKey != "" {
		return cfg, nil
	}
	path, err := a.authFilePath(cfg.authFile)
	if err != nil {
		return cfg, err
	}
	cfg.authFile = path
	return cfg, nil
}

func (a app) userConfigDir() (string, error) {
	if a.configDir != nil {
		return a.configDir()
	}
	return os.UserConfigDir()
}

// runLogin stores a credential for one provider, as pi's /login does: an
// account (OAuth) sign-in by default when the provider has one, else its
// API key, entered at a hidden prompt or piped on stdin. The credential is
// never printed.
func (a app) runLogin(ctx context.Context, cfg loginConfig) error {
	authType, err := normalizeAuthType(cfg.auth)
	if err != nil {
		return err
	}
	if cfg.timeout < 0 {
		return fmt.Errorf("%w: --timeout must not be negative", llm.ErrBadRequest)
	}
	if cfg.provider == "" {
		return fmt.Errorf("%w: name a provider to log in to: %s", llm.ErrBadRequest, loginProviderIDs())
	}
	provider, ok := findLoginProvider(cfg.provider)
	if !ok {
		return fmt.Errorf("%w: unknown provider %q; choose one of %s", llm.ErrBadRequest, cfg.provider, loginProviderIDs())
	}
	if authType == "" {
		authType = authAPIKey
		if provider.oauthLabel != "" {
			authType = authOAuth
		}
	}
	if authType == authOAuth && provider.oauthLabel == "" {
		return fmt.Errorf("%w: %s has no account sign-in; use --auth api-key", llm.ErrBadRequest, provider.name)
	}
	path, err := a.authFilePath(cfg.authFile)
	if err != nil {
		return err
	}
	store, err := openAuthFileForWrite(ctx, path, provider.id)
	if err != nil {
		return err
	}

	var credential llm.AuthCredential
	if authType == authOAuth {
		credential, err = a.signInWithChatGPT(ctx, cfg)
	} else {
		credential, err = a.readAPIKey(ctx, provider)
	}
	if err != nil {
		return err
	}
	if _, _, err := store.Modify(ctx, provider.id, func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
		return credential, true, nil
	}); err != nil {
		return fmt.Errorf("save credential: %w", err)
	}
	action := "Logged in to " + provider.name
	if authType == authAPIKey {
		action = "Saved API key for " + provider.name
	}
	return a.status("%s. Credentials saved to %s\n", action, path)
}

// openAuthFileForWrite creates the auth file's directory and proves the
// file can be locked and parsed before any sign-in, so a bad path or file
// never discards a minted credential.
func openAuthFileForWrite(ctx context.Context, path, providerID string) (*authstore.File, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, fmt.Errorf("auth file %s: %w", path, err)
	}
	store := authstore.NewFile(path)
	if _, _, err := store.Modify(ctx, providerID, func(current llm.AuthCredential, _ bool) (llm.AuthCredential, bool, error) {
		return current, false, nil
	}); err != nil {
		return nil, fmt.Errorf("auth file %s: %w", path, err)
	}
	return store, nil
}

func (a app) signInWithChatGPT(ctx context.Context, cfg loginConfig) (llm.AuthCredential, error) {
	deviceID := cfg.deviceID
	if deviceID == "" {
		id, err := a.installationDeviceID()
		if err != nil {
			return llm.AuthCredential{}, err
		}
		deviceID = id
	}
	factory := a.loginFlowFactory
	if factory == nil {
		factory = newOpenAILoginFlow
	}
	flow, err := factory(deviceID, cfg.timeout)
	if err != nil {
		return llm.AuthCredential{}, err
	}
	defer flow.Cancel()
	authorization, err := flow.Begin(ctx)
	if err != nil {
		return llm.AuthCredential{}, err
	}
	if err := a.status("Open this URL in a browser and sign in with ChatGPT:\n\n  %s\n\n"+
		"Waiting for the sign-in to complete. If the browser cannot reach this machine,\n"+
		"paste the final redirect URL here and press Enter.\n", authorization.URL()); err != nil {
		return llm.AuthCredential{}, err
	}
	if a.stdin != nil {
		go a.submitPastedCallbacks(ctx, flow)
	}
	return flow.Complete(ctx)
}

// readAPIKey asks for the provider's API key: a hidden prompt on a
// terminal, else the first line of stdin. The value is stored as entered,
// so a pi reference such as "$OPENAI_API_KEY" stays a reference. A
// provider with an env value (Cloudflare's account ID) asks for it next, at
// a visible prompt or as the second line of stdin, as pi's login does.
func (a app) readAPIKey(ctx context.Context, provider loginProvider) (llm.AuthCredential, error) {
	var key string
	var lines *bufio.Reader
	file, ok := a.stdin.(*os.File)
	terminal := ok && term.IsTerminal(int(file.Fd()))
	if terminal {
		if err := a.status("Enter %s: ", provider.keyName); err != nil {
			return llm.AuthCredential{}, err
		}
		secret, err := readHidden(ctx, int(file.Fd()))
		_ = a.status("\n")
		if err != nil {
			return llm.AuthCredential{}, fmt.Errorf("read %s: %w", provider.keyName, err)
		}
		key = secret
	} else if a.stdin != nil {
		lines = bufio.NewReader(a.stdin)
		line, err := readLine(ctx, lines)
		if err != nil {
			return llm.AuthCredential{}, fmt.Errorf("read %s: %w", provider.keyName, err)
		}
		key = line
	}
	key = strings.TrimSpace(key)
	if key == "" {
		return llm.AuthCredential{}, fmt.Errorf("%w: no %s entered", llm.ErrBadRequest, provider.keyName)
	}
	credential := llm.AuthCredential{Type: authAPIKey, Key: key}
	if provider.envName == "" {
		return credential, nil
	}
	var value string
	switch {
	case terminal:
		if err := a.status("Enter %s: ", provider.envLabel); err != nil {
			return llm.AuthCredential{}, err
		}
		line, err := readLine(ctx, bufio.NewReader(file))
		if err != nil {
			return llm.AuthCredential{}, fmt.Errorf("read %s: %w", provider.envLabel, err)
		}
		value = line
	case lines != nil:
		line, err := readLine(ctx, lines)
		if err != nil {
			return llm.AuthCredential{}, fmt.Errorf("read %s: %w", provider.envLabel, err)
		}
		value = line
	}
	value = strings.TrimSpace(value)
	if value == "" {
		return llm.AuthCredential{}, fmt.Errorf("%w: no %s entered (pipe it as the second line after the key)", llm.ErrBadRequest, provider.envLabel)
	}
	credential.Env = map[string]string{provider.envName: value}
	return credential, nil
}

// readLine reads one line, ending at EOF too, and gives up when ctx is
// canceled.
func readLine(ctx context.Context, r *bufio.Reader) (string, error) {
	type result struct {
		line string
		err  error
	}
	done := make(chan result, 1)
	go func() {
		line, err := r.ReadString('\n')
		if errors.Is(err, io.EOF) {
			err = nil
		}
		done <- result{line, err}
	}()
	select {
	case res := <-done:
		return res.line, res.err
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

// readHidden reads a line without echo. ReadPassword blocks in read(2)
// and ignores ctx, so on cancellation (Ctrl-C) the terminal state saved
// beforehand is restored here; a second Ctrl-C then cannot leave the shell
// without echo.
func readHidden(ctx context.Context, fd int) (string, error) {
	state, err := term.GetState(fd)
	if err != nil {
		return "", err
	}
	type result struct {
		secret []byte
		err    error
	}
	done := make(chan result, 1)
	go func() {
		secret, err := term.ReadPassword(fd)
		done <- result{secret, err}
	}()
	select {
	case r := <-done:
		return string(r.secret), r.err
	case <-ctx.Done():
		_ = term.Restore(fd, state)
		return "", ctx.Err()
	}
}

// runLogout removes a provider's stored credential, as pi's /logout does;
// environment variables are left alone.
func (a app) runLogout(ctx context.Context, cfg logoutConfig) error {
	if cfg.provider == "" {
		return fmt.Errorf("%w: name a provider to log out of: %s", llm.ErrBadRequest, loginProviderIDs())
	}
	provider, ok := findLoginProvider(cfg.provider)
	if !ok {
		provider = loginProvider{id: cfg.provider, name: cfg.provider}
	}
	path, err := a.authFilePath(cfg.authFile)
	if err != nil {
		return err
	}
	stored, ok, err := storedEntry(path, provider.id)
	if err != nil {
		return err
	}
	if !ok {
		return a.status("No stored credentials for %s in %s. Environment variables are unchanged.\n", provider.name, path)
	}
	if err := authstore.NewFile(path).Delete(ctx, provider.id); err != nil {
		return fmt.Errorf("remove credential: %w", err)
	}
	if stored.Type == authOAuth {
		return a.status("Logged out of %s\n", provider.name)
	}
	return a.status("Removed stored API key for %s. Environment variables are unchanged.\n", provider.name)
}

// storedEntry reads a provider's raw entry without resolving key
// references (no environment lookup, no command).
func storedEntry(path, providerID string) (llm.AuthCredential, bool, error) {
	auth, err := llm.LoadAuthFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return llm.AuthCredential{}, false, nil
	}
	if err != nil {
		return llm.AuthCredential{}, false, fmt.Errorf("auth file %s: %w", path, err)
	}
	cred, ok := auth[providerID]
	return cred, ok, nil
}

// runAuthStatus lists each provider's sign-in methods and what is
// configured, as pi's login selector shows it. Key references are never
// resolved here.
func (a app) runAuthStatus(cfg authStatusConfig) error {
	path, err := a.authFilePath(cfg.authFile)
	if err != nil {
		return err
	}
	auth, err := llm.LoadAuthFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		auth, err = llm.AuthFile{}, nil
	}
	if err != nil {
		return fmt.Errorf("auth file %s: %w", path, err)
	}
	if _, err := fmt.Fprintf(a.stdout, "Auth file: %s\n\n", path); err != nil {
		return fmt.Errorf("write auth status: %w", err)
	}
	tw := tabwriter.NewWriter(a.stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "PROVIDER\tSIGN-IN\tSTATUS")
	for _, provider := range loginProviders {
		methods := "API key"
		if provider.oauthLabel != "" {
			methods = "subscription (default), API key"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\n", provider.id, methods, authStatus(provider, auth))
	}
	return tw.Flush()
}

func authStatus(provider loginProvider, auth llm.AuthFile) string {
	if cred, ok := auth[provider.id]; ok {
		switch {
		case cred.Type == authOAuth && provider.oauthLabel == "":
			return "✗ subscription stored, which " + provider.name + " does not accept here; run 'llm-cli login " + provider.id + "'"
		case cred.Type == authOAuth:
			return "✓ subscription configured"
		case cred.Key != "" && provider.envName != "" && cred.Env[provider.envName] == "" && !envValueSet(provider.envName):
			return "✗ API key stored without " + provider.envLabel + "; run 'llm-cli login " + provider.id + "' or set " + provider.envName
		case strings.HasPrefix(cred.Key, "!"):
			return "✓ API key configured (command)"
		case strings.Contains(cred.Key, "$"):
			return "✓ API key configured (reference)"
		case cred.Key != "" && provider.envName != "" && cred.Env[provider.envName] == "":
			return "✓ API key configured (" + provider.envName + " from the environment)"
		case cred.Key != "":
			return "✓ API key configured"
		}
	}
	if name := providerEnvKey(provider.id); name != "" {
		if provider.envName != "" && !envValueSet(provider.envName) {
			return "✗ env: " + name + " without " + provider.envName
		}
		return "✓ env: " + name
	}
	return "• not configured"
}

// loginHintError keeps the authentication error and adds how to fix it.
type loginHintError struct {
	err  error
	hint string
}

func (e *loginHintError) Error() string { return e.err.Error() + "\n" + e.hint }
func (e *loginHintError) Unwrap() error { return e.err }

// withLoginHint adds a pointer to 'llm-cli login' (and the provider's
// environment variable) to an authentication failure, unless the key came
// from --api-key.
func withLoginHint(providerID, apiKey string, err error) error {
	if err == nil || apiKey != "" || !errors.Is(err, llm.ErrAuth) {
		return err
	}
	provider, ok := findLoginProvider(providerID)
	if !ok {
		return err
	}
	hint := "run 'llm-cli login " + provider.id + "'"
	if name := providerEnvName[provider.id]; name != "" {
		hint += " or set " + name
	}
	return &loginHintError{err: err, hint: hint}
}

// providerEnvName is each provider's API key environment variable; the
// self-hosted providers never read the environment.
var providerEnvName = map[string]string{
	"openai":                "OPENAI_API_KEY",
	"anthropic":             "ANTHROPIC_API_KEY",
	"openrouter":            "OPENROUTER_API_KEY",
	"typesafe":              "TYPESAFE_API_KEY",
	"cloudflare-workers-ai": "CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID",
	"vercel-ai-gateway":     "AI_GATEWAY_API_KEY",
	"opencode":              "OPENCODE_API_KEY",
}

// providerEnvKey names the provider's API key environment variable when it
// is set. The self-hosted providers never read the environment.
func providerEnvKey(providerID string) string {
	var name, value string
	switch providerID {
	case "openai":
		name, value = "OPENAI_API_KEY", os.Getenv("OPENAI_API_KEY")
	case "anthropic":
		name, value = "ANTHROPIC_API_KEY", os.Getenv("ANTHROPIC_API_KEY")
	case "openrouter":
		name, value = "OPENROUTER_API_KEY", os.Getenv("OPENROUTER_API_KEY")
	case "typesafe":
		name, value = "TYPESAFE_API_KEY", os.Getenv("TYPESAFE_API_KEY")
	case "cloudflare-workers-ai":
		name, value = "CLOUDFLARE_API_KEY", os.Getenv("CLOUDFLARE_API_KEY")
	case "vercel-ai-gateway":
		name, value = "AI_GATEWAY_API_KEY", os.Getenv("AI_GATEWAY_API_KEY")
	case "opencode":
		name, value = "OPENCODE_API_KEY", os.Getenv("OPENCODE_API_KEY")
	}
	if value == "" {
		return ""
	}
	return name
}

// envValueSet reports whether a provider's extra environment value (named
// by loginProvider.envName) is set; each name is read explicitly so the
// credential-variable scan sees it.
func envValueSet(name string) bool {
	switch name {
	case "CLOUDFLARE_ACCOUNT_ID":
		return os.Getenv("CLOUDFLARE_ACCOUNT_ID") != ""
	}
	return false
}

func (a app) status(format string, args ...any) error {
	if _, err := fmt.Fprintf(a.stderr, format, args...); err != nil {
		return fmt.Errorf("write status: %w", err)
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
			_ = a.status("callback not accepted: %v\n", err)
			continue
		}
		return
	}
}

// installationDeviceID returns this installation's stable login device ID,
// creating it on first use in <user config dir>/llm-cli/device-id. OpenAI
// expects the same ID on every login from one installation.
func (a app) installationDeviceID() (string, error) {
	base, err := a.userConfigDir()
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
