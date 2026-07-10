package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
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
	bindCommonFlags(fs, &cfg.provider, &cfg.apiKey, &cfg.authFile, &cfg.baseURL, &cfg.timeout, &cfg.version)
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
	bindCommonFlags(fs, &cfg.provider, &cfg.apiKey, &cfg.authFile, &cfg.baseURL, &cfg.timeout, &cfg.version)
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

func chatUsage(fs *flag.FlagSet) {
	w := fs.Output()
	fmt.Fprint(w, `Usage: llm-cli [flags] [prompt]
       llm-cli models [flags]

Single-shot chat against an LLM provider. The prompt comes from the
positional argument, piped stdin, or both. The response streams to stdout
by default; --schema (like --json and --no-stream) forces the
non-streaming path.

OpenAI Codex auth precedence is --auth-file, then
OPENAI_CODEX_ACCESS_TOKEN, then the compatibility --api-key flag.
The --api-key value is exposed through shell history and process argv;
prefer --auth-file or OPENAI_CODEX_ACCESS_TOKEN on shared systems.

Flags:
`)
	fs.PrintDefaults()
	fmt.Fprint(w, `
Examples:
  llm-cli -p anthropic -m claude-opus-4-8 "explain me this error: ..."
  echo "long doc" | llm-cli -p openai -m gpt-5.5 -s "summarize stdin"
  llm-cli -p openrouter -m openai/gpt-5.5 --effort high --json "..."
  llm-cli models -p openrouter
`)
}

func modelsUsage(fs *flag.FlagSet) {
	w := fs.Output()
	fmt.Fprint(w, `Usage: llm-cli models [flags]

List the models a provider exposes as a table (--json for machine output).

OpenAI Codex auth precedence is --auth-file, then
OPENAI_CODEX_ACCESS_TOKEN, then the compatibility --api-key flag.
The --api-key value is exposed through shell history and process argv.

Flags:
`)
	fs.PrintDefaults()
	fmt.Fprint(w, `
Example:
  llm-cli models -p openrouter
`)
}

func bindCommonFlags(fs *flag.FlagSet, provider, apiKey, authFile, baseURL *string, timeout *time.Duration, version *bool) {
	fs.StringVar(provider, "p", "", "provider: anthropic|openai|openai-codex|openrouter")
	fs.StringVar(provider, "provider", "", "provider: anthropic|openai|openai-codex|openrouter")
	fs.StringVar(authFile, "auth-file", "", "explicit credential file (openai-codex; highest precedence)")
	fs.StringVar(apiKey, "api-key", "", "provider API key (openai-codex compatibility: OAuth access token exposed via argv)")
	fs.StringVar(baseURL, "base-url", "", "provider base URL")
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
