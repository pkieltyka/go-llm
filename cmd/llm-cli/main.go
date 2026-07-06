package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime/debug"
	"strings"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	go func() {
		// Release signal capture as soon as the first SIGINT cancels ctx, so
		// a second SIGINT force-kills via the default disposition instead of
		// being swallowed while shutdown drains.
		<-ctx.Done()
		stop()
	}()

	a := app{
		stdin:           os.Stdin,
		stdout:          os.Stdout,
		stderr:          os.Stderr,
		providerFactory: newProvider,
	}
	if err := a.run(ctx, os.Args[1:]); err != nil {
		switch {
		case errors.Is(err, errUsage):
		case ctx.Err() != nil && errors.Is(err, context.Canceled):
			// The run was cut short by SIGINT: say so instead of surfacing a
			// raw "context canceled".
			fmt.Fprintln(os.Stderr, "interrupted")
		default:
			fmt.Fprintln(os.Stderr, err)
		}
		os.Exit(1)
	}
}

func (a app) run(ctx context.Context, args []string) error {
	if len(args) > 0 && args[0] == "models" {
		cfg, err := parseModelsFlags(args[1:], a.stderr)
		if err != nil {
			if errors.Is(err, errHelp) {
				return nil
			}
			return err
		}
		if cfg.version {
			printVersion(a.stdout)
			return nil
		}
		return a.runModels(ctx, cfg)
	}

	cfg, err := parseChatFlags(args, a.stderr)
	if err != nil {
		if errors.Is(err, errHelp) {
			return nil
		}
		return err
	}
	if cfg.version {
		printVersion(a.stdout)
		return nil
	}
	if shouldReadStdin(a.stdin) {
		text, err := readAllContext(ctx, a.stdin)
		if err != nil {
			return err
		}
		cfg.stdinText = text
	}
	return a.runChat(ctx, cfg)
}

// readAllContext reads r to EOF but abandons the read as soon as ctx is
// canceled, so a SIGINT during a blocked stdin read (held-open pipe,
// interactive terminal) exits promptly instead of hanging until SIGKILL.
// The reader goroutine may outlive the call; the process exits right after.
func readAllContext(ctx context.Context, r io.Reader) (string, error) {
	type readResult struct {
		text string
		err  error
	}
	results := make(chan readResult, 1)
	go func() {
		text, err := io.ReadAll(r)
		if err != nil {
			results <- readResult{err: fmt.Errorf("read stdin: %w", err)}
			return
		}
		results <- readResult{text: string(text)}
	}()
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case res := <-results:
		return res.text, res.err
	}
}

func printVersion(w io.Writer) {
	version := "(devel)"
	if info, ok := debug.ReadBuildInfo(); ok {
		if info.Main.Version != "" {
			version = info.Main.Version
		}
	}
	fmt.Fprintf(w, "llm-cli %s\n", version)
}

func promptFromArgs(args []string) string {
	return strings.TrimSpace(strings.Join(args, " "))
}
