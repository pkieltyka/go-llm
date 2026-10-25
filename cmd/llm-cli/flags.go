package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

var errHelp = errors.New("help requested")

// errUsage marks a flag-parse failure that has already been reported to
// stderr (error line + usage); main exits 1 without re-printing it.
var errUsage = errors.New("usage error")

func parseChatFlags(args []string, stderr io.Writer) (chatConfig, error) {
	var cfg chatConfig
	fs := flag.NewFlagSet("llm-cli", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { chatUsage(fs) }
	bindCommonFlags(fs, &cfg.provider, &cfg.apiKey, &cfg.authFile, &cfg.baseURL, &cfg.timeout, &cfg.version, &cfg.authFileCommands)
	fs.StringVar(&cfg.model, "m", "", "model ID (verbatim provider model name)")
	fs.StringVar(&cfg.model, "model", "", "model ID (verbatim provider model name)")
	fs.StringVar(&cfg.system, "s", "", "system prompt")
	fs.StringVar(&cfg.system, "system", "", "system prompt")
	fs.Var((*effortValue)(&cfg.effort), "effort", "reasoning effort: none|minimal|low|medium|high|xhigh|max")
	fs.IntVar(&cfg.maxTokens, "max-tokens", 0, "maximum output tokens")
	fs.Var(&cfg.temperature, "temp", "sampling temperature")
	fs.Var(&cfg.temperature, "temperature", "sampling temperature (alias of -temp)")
	fs.Var(&cfg.imagePaths, "image", "image path or URL (repeatable)")
	fs.Var(&cfg.filePaths, "file", "file path or URL (repeatable)")
	fs.StringVar(&cfg.schemaPath, "schema", "", "JSON schema file for structured output; forces the non-streaming path")
	fs.Var(&cfg.toolPaths, "tool", "tool declaration JSON file (repeatable); tool calls are printed, not executed")
	fs.BoolVar(&cfg.noStream, "no-stream", false, "disable streaming; buffer and print the complete response")
	fs.BoolVar(&cfg.jsonOutput, "json", false, "emit the full canonical Response JSON (non-streaming)")
	fs.BoolVar(&cfg.usage, "usage", false, "print usage summary to stderr")
	fs.BoolVar(&cfg.reasoning, "reasoning", false, "print reasoning deltas to stderr")
	fs.BoolVar(&cfg.debug, "debug", false, "emit wire debug logs to stderr")
	fs.BoolVar(&cfg.cacheSystem, "cache-system", false, "mark system prompt cacheable")
	fs.StringVar(&cfg.sessionID, "session-id", "", "session/routing affinity ID")
	fs.StringVar(&cfg.loadPath, "load", "", "load canonical conversation file")
	fs.StringVar(&cfg.savePath, "save", "", "save canonical conversation file")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return cfg, errHelp
		}
		// The flag package already printed the error and usage to stderr.
		return cfg, errUsage
	}
	cfg.args = fs.Args()
	return cfg, nil
}

func parseModelsFlags(args []string, stderr io.Writer) (modelsConfig, error) {
	var cfg modelsConfig
	fs := flag.NewFlagSet("llm-cli models", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { modelsUsage(fs) }
	bindCommonFlags(fs, &cfg.provider, &cfg.apiKey, &cfg.authFile, &cfg.baseURL, &cfg.timeout, &cfg.version, &cfg.authFileCommands)
	fs.BoolVar(&cfg.jsonOutput, "json", false, "emit JSON instead of a table")
	fs.BoolVar(&cfg.debug, "debug", false, "emit wire debug logs to stderr")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return cfg, errHelp
		}
		// The flag package already printed the error and usage to stderr.
		return cfg, errUsage
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "llm-cli models: unexpected argument %q\n", fs.Arg(0))
		fmt.Fprintln(stderr, "Run 'llm-cli models -h' for usage.")
		return cfg, errUsage
	}
	return cfg, nil
}

func parseClassifyFlags(args []string, stderr io.Writer) (classifyConfig, error) {
	var cfg classifyConfig
	fs := flag.NewFlagSet("llm-cli classify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { classifyUsage(fs) }
	bindCommonFlags(fs, &cfg.provider, &cfg.apiKey, &cfg.authFile, &cfg.baseURL, &cfg.timeout, &cfg.version, &cfg.authFileCommands)
	fs.StringVar(&cfg.model, "m", "", "classifier model ID (verbatim provider model name)")
	fs.StringVar(&cfg.model, "model", "", "classifier model ID (verbatim provider model name)")
	fs.StringVar(&cfg.state, "state", "", "state to classify: inline JSON object, a JSON file, or - for stdin (default: piped stdin)")
	fs.StringVar(&cfg.questionsPath, "questions", "", "questions JSON file: pi's {name: {type, instructions, criteria}} object or a list")
	fs.Var(&cfg.imagePaths, "image", "image file to classify with the state (repeatable)")
	fs.BoolVar(&cfg.jsonOutput, "json", false, "emit the full ClassifyResult JSON")
	fs.BoolVar(&cfg.usage, "usage", false, "print usage summary to stderr")
	fs.BoolVar(&cfg.debug, "debug", false, "emit wire debug logs to stderr")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return cfg, errHelp
		}
		return cfg, errUsage
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "llm-cli classify: unexpected argument %q\n", fs.Arg(0))
		fmt.Fprintln(stderr, "Run 'llm-cli classify -h' for usage.")
		return cfg, errUsage
	}
	return cfg, nil
}

func classifyUsage(fs *flag.FlagSet) {
	w := fs.Output()
	fmt.Fprint(w, `Usage: llm-cli classify -p <provider> -m <model> --questions <file> [--state <json|file|->] [flags]

Ask a classifier (decision) model questions about a JSON state, such as a
conversation or a document's fields, and print each answer with its
probabilities. Question types: choice (one of several values), score (a
position on ordered levels) and bool (a probability of true).

Questions file, pi's form (names and choices keep their order):
  {"category": {"type": "choice", "instructions": "Classify the message.",
                "criteria": {"bug": "A defect report", "question": "A question"}},
   "urgency":  {"type": "score", "instructions": "How urgent is it?",
                "criteria": ["low", "medium", "high"]},
   "approved": {"type": "bool", "instructions": "Does the user approve?",
                "criteria": {"true": "Approval", "false": "No approval"}}}

Providers: openrouter (Jev, Clef, Luna and others it lists), typesafe,
cloudflare-workers-ai (CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID),
vercel-ai-gateway, opencode, openai (GPT-6 Luna over the Decisions API; API
keys only), and systemone (any System One server, with --base-url).
Credentials work as for chat; 'llm-cli login <provider>' stores one.

Flags:
`)
	fs.PrintDefaults()
	fmt.Fprint(w, `
Examples:
  llm-cli classify -p openrouter -m typesafe/jev-1.14 --questions q.json --state '{"message":"It crashes on save"}'
  cat ticket.json | llm-cli classify -p typesafe -m jev-latest --questions q.json --json
  llm-cli classify -p cloudflare-workers-ai -m @cf/cloudflare/clef --questions q.json --state ticket.json --image shot.png
`)
}

// splitProvider takes a leading provider argument, as in pi's
// "/login openai", so flags may follow it.
func splitProvider(args []string) (string, []string) {
	if len(args) > 0 && !strings.HasPrefix(args[0], "-") {
		return args[0], args[1:]
	}
	return "", args
}

// trailingProvider accepts the provider after the flags instead, and
// reports any further argument.
func trailingProvider(fs *flag.FlagSet, provider *string, stderr io.Writer) error {
	rest := fs.Args()
	if *provider == "" && len(rest) > 0 {
		*provider, rest = rest[0], rest[1:]
	}
	if len(rest) > 0 {
		fmt.Fprintf(stderr, "%s: unexpected argument %q\n", fs.Name(), rest[0])
		fmt.Fprintf(stderr, "Run '%s -h' for usage.\n", fs.Name())
		return errUsage
	}
	return nil
}

func parseLoginFlags(args []string, stderr io.Writer) (loginConfig, error) {
	var cfg loginConfig
	cfg.provider, args = splitProvider(args)
	fs := flag.NewFlagSet("llm-cli login", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { loginUsage(fs) }
	fs.StringVar(&cfg.provider, "p", cfg.provider, "provider (alternative to the positional argument)")
	fs.StringVar(&cfg.provider, "provider", cfg.provider, "provider (alternative to the positional argument)")
	fs.StringVar(&cfg.auth, "auth", "", "sign-in method: oauth (default where the provider has it) or api-key")
	fs.StringVar(&cfg.authFile, "auth-file", "", "pi-compatible auth.json (default $LLM_AUTH_FILE, else the user config dir's llm-cli/auth.json)")
	fs.StringVar(&cfg.deviceID, "device-id", "", "installation UUID sent to OpenAI (default: created once in the user config directory)")
	fs.DurationVar(&cfg.timeout, "timeout", 0, "bound on the whole browser sign-in (default 5m)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return cfg, errHelp
		}
		return cfg, errUsage
	}
	return cfg, trailingProvider(fs, &cfg.provider, stderr)
}

func parseLogoutFlags(args []string, stderr io.Writer) (logoutConfig, error) {
	var cfg logoutConfig
	cfg.provider, args = splitProvider(args)
	fs := flag.NewFlagSet("llm-cli logout", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { logoutUsage(fs) }
	fs.StringVar(&cfg.authFile, "auth-file", "", "pi-compatible auth.json (default $LLM_AUTH_FILE, else the user config dir's llm-cli/auth.json)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return cfg, errHelp
		}
		return cfg, errUsage
	}
	return cfg, trailingProvider(fs, &cfg.provider, stderr)
}

func parseAuthStatusFlags(args []string, stderr io.Writer) (authStatusConfig, error) {
	var cfg authStatusConfig
	fs := flag.NewFlagSet("llm-cli auth", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { authStatusUsage(fs) }
	fs.StringVar(&cfg.authFile, "auth-file", "", "pi-compatible auth.json (default $LLM_AUTH_FILE, else the user config dir's llm-cli/auth.json)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return cfg, errHelp
		}
		return cfg, errUsage
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "llm-cli auth: unexpected argument %q\n", fs.Arg(0))
		fmt.Fprintln(stderr, "Run 'llm-cli auth -h' for usage.")
		return cfg, errUsage
	}
	return cfg, nil
}

func loginUsage(fs *flag.FlagSet) {
	w := fs.Output()
	fmt.Fprint(w, `Usage: llm-cli login <provider> [--auth oauth|api-key] [flags]

Store a credential for a provider in the auth file, as pi's /login does.
openai signs in with ChatGPT by default (--auth oauth), using your ChatGPT
plan; --auth api-key stores an API key instead. The other providers take an
API key. A key is read at a hidden prompt, or from stdin when piped; a pi
reference such as "$OPENAI_API_KEY" is stored as is and resolved when used.

The ChatGPT sign-in prints a URL: open it in a browser. It completes through
a loopback callback on 127.0.0.1:1455; when the browser runs on another
machine (over SSH, say), paste the final redirect URL, as shown in its
address bar, into this terminal. Credentials are never printed.

Providers: `+loginProviderIDs()+`

Flags:
`)
	fs.PrintDefaults()
	fmt.Fprint(w, `
Examples:
  llm-cli login openai                     # Sign in with ChatGPT
  llm-cli login openai --auth api-key      # store an OpenAI API key
  echo "$KEY" | llm-cli login anthropic    # store a piped API key
  llm-cli login openai --auth-file ~/.pi/agent/auth.json   # share with pi
`)
}

func logoutUsage(fs *flag.FlagSet) {
	w := fs.Output()
	fmt.Fprint(w, `Usage: llm-cli logout <provider> [flags]

Remove the provider's stored credential from the auth file, as pi's /logout
does. Environment variables are unchanged.

Flags:
`)
	fs.PrintDefaults()
}

func authStatusUsage(fs *flag.FlagSet) {
	w := fs.Output()
	fmt.Fprint(w, `Usage: llm-cli auth [flags]

List each provider's sign-in methods and what is configured: a stored
credential, an environment variable, or nothing. Key references are not
resolved.

Flags:
`)
	fs.PrintDefaults()
}

func chatUsage(fs *flag.FlagSet) {
	w := fs.Output()
	fmt.Fprint(w, `Usage: llm-cli [flags] [prompt]
       llm-cli models [flags]
       llm-cli classify [flags]
       llm-cli login <provider> [--auth oauth|api-key]
       llm-cli logout <provider>
       llm-cli auth

Single-shot chat against an LLM provider. The prompt comes from the
positional argument, piped stdin, or both. The response streams to stdout
by default; --schema (like --json and --no-stream) forces the
non-streaming path.

Credentials, as in pi: a credential stored in the auth file wins; without
one, providers read their usual environment variable (ANTHROPIC_API_KEY,
OPENAI_API_KEY, OPENROUTER_API_KEY). The auth file is --auth-file, else
$LLM_AUTH_FILE, else llm-cli/auth.json in the user config directory; use
~/.pi/agent/auth.json to share credentials with pi. 'llm-cli login <provider>'
stores one (Sign in with ChatGPT for openai, or an API key), 'llm-cli logout'
removes it, and 'llm-cli auth' shows what is configured. OAuth refreshes are
written back under the file's lock. An api_key value may reference
environment variables ("$OPENAI_API_KEY", "${VAR}"); a "!command" value runs
only with --auth-file-commands. --api-key bypasses the auth file; its value is
exposed through shell history and process argv.

Self-hosted servers: -p vllm or -p chatcompletions (Ollama, llama.cpp, any
OpenAI Chat Completions server) with --base-url pointing at the server's /v1
root. Their API key is optional and never read from the environment; it is
the auth file's "api_key" entry named after the provider, if any.

Flags:
`)
	fs.PrintDefaults()
	fmt.Fprint(w, `
Examples:
  llm-cli -p anthropic -m claude-opus-4-8 "explain me this error: ..."
  echo "long doc" | llm-cli -p openai -m gpt-5.5 -s "summarize stdin"
  llm-cli -p openrouter -m openai/gpt-5.5 --effort high --json "..."
  llm-cli models -p openrouter
  llm-cli -p vllm --base-url http://localhost:8000/v1 -m Qwen/Qwen3-8B "hi"
  llm-cli -p chatcompletions --base-url http://localhost:11434/v1 -m llama3.2 "hi"
`)
}

func modelsUsage(fs *flag.FlagSet) {
	w := fs.Output()
	fmt.Fprint(w, `Usage: llm-cli models [flags]

List the models a provider exposes as a table (--json for machine output).

Credentials work as for chat: --auth-file (pi-compatible auth.json, API
keys, Sign in with ChatGPT OAuth for openai, or stored OAuth for anthropic), else the provider's
environment variable. An --api-key value is exposed through process argv.

Flags:
`)
	fs.PrintDefaults()
	fmt.Fprint(w, `
Example:
  llm-cli models -p openrouter
`)
}

func bindCommonFlags(fs *flag.FlagSet, provider, apiKey, authFile, baseURL *string, timeout *time.Duration, version, authFileCommands *bool) {
	fs.StringVar(provider, "p", "", "provider: anthropic|openai|openrouter|vllm|chatcompletions; classify also typesafe|cloudflare-workers-ai|vercel-ai-gateway|opencode|systemone")
	fs.StringVar(provider, "provider", "", "provider (alias of -p)")
	fs.StringVar(authFile, "auth-file", "", "pi-compatible auth.json credential store (API keys; Sign in with ChatGPT OAuth for openai; stored OAuth for anthropic)")
	fs.BoolVar(authFileCommands, "auth-file-commands", false, `let --auth-file api_key values run shell commands ("!op read ..."), as pi does`)
	fs.StringVar(apiKey, "api-key", "", "provider API key (exposed via argv; prefer --auth-file or the environment)")
	fs.StringVar(baseURL, "base-url", "", "provider base URL (required for vllm, chatcompletions and systemone: the server's /v1 root)")
	fs.DurationVar(timeout, "timeout", 0, "provider call timeout")
	fs.BoolVar(version, "version", false, "print version")
}

type effortValue llm.Effort

func (v *effortValue) String() string {
	if v == nil {
		return ""
	}
	return string(*v)
}

func (v *effortValue) Set(s string) error {
	switch s {
	case "", "none", "minimal", "low", "medium", "high", "xhigh", "max":
		*v = effortValue(s)
		return nil
	default:
		return fmt.Errorf("invalid effort %q", s)
	}
}

func parseFloat(s string) (float64, error) {
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid float %q: %w", s, err)
	}
	return f, nil
}

func formatFloat(f float64) string {
	return strconv.FormatFloat(f, 'f', -1, 64)
}

func shouldReadStdin(r io.Reader) bool {
	file, ok := r.(*os.File)
	if !ok {
		return true
	}
	info, err := file.Stat()
	if err != nil {
		return false
	}
	return info.Mode()&os.ModeCharDevice == 0
}
