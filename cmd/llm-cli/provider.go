package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"os"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
	"github.com/pkieltyka/go-llm/providers/anthropic"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
	"github.com/pkieltyka/go-llm/providers/cloudflare"
	"github.com/pkieltyka/go-llm/providers/openai"
	"github.com/pkieltyka/go-llm/providers/openrouter"
	"github.com/pkieltyka/go-llm/providers/systemone"
	"github.com/pkieltyka/go-llm/providers/vllm"
)

func newProvider(_ context.Context, cfg providerConfig) (llm.Provider, error) {
	if cfg.authFile != "" && cfg.apiKey != "" {
		return nil, fmt.Errorf("%w: use either --auth-file or --api-key, not both", llm.ErrBadRequest)
	}
	// --auth-file is a pi-compatible credential store: every request reads it
	// and OAuth refreshes are written back under its lock.
	if cfg.authFileCommands && cfg.authFile == "" {
		return nil, fmt.Errorf("%w: --auth-file-commands applies to the auth file, which --api-key bypasses", llm.ErrBadRequest)
	}
	var store llm.CredentialStore
	if cfg.authFile != "" {
		var opts []authstore.Option
		if cfg.authFileCommands {
			opts = append(opts, authstore.WithKeyCommands())
		}
		store = authstore.NewFile(cfg.authFile, opts...)
	}
	switch cfg.name {
	case "anthropic":
		opts := []anthropic.Option{}
		if store != nil {
			opts = append(opts, anthropic.WithCredentials(store))
		} else if cfg.apiKey != "" {
			opts = append(opts, anthropic.WithAPIKey(cfg.apiKey))
		}
		if cfg.baseURL != "" {
			opts = append(opts, anthropic.WithBaseURL(cfg.baseURL))
		}
		if cfg.timeout > 0 {
			opts = append(opts, anthropic.WithTimeout(cfg.timeout))
		}
		if logger := debugLogger(cfg); logger != nil {
			opts = append(opts, anthropic.WithLogger(logger), anthropic.WithWireCapture(llm.WireCaptureToLogger(logger)))
		}
		return anthropic.New(opts...)
	case "openai":
		opts := []openai.Option{}
		if store != nil {
			opts = append(opts, openai.WithCredentials(store))
		} else if cfg.apiKey != "" {
			opts = append(opts, openai.WithAPIKey(cfg.apiKey))
		}
		if cfg.baseURL != "" {
			opts = append(opts, openai.WithBaseURL(cfg.baseURL))
		}
		if cfg.timeout > 0 {
			opts = append(opts, openai.WithTimeout(cfg.timeout))
		}
		if logger := debugLogger(cfg); logger != nil {
			opts = append(opts, openai.WithLogger(logger), openai.WithWireCapture(llm.WireCaptureToLogger(logger)))
		}
		return openai.New(opts...)
	case "openrouter":
		opts := []openrouter.Option{}
		if store != nil {
			opts = append(opts, openrouter.WithCredentials(store))
		} else if cfg.apiKey != "" {
			opts = append(opts, openrouter.WithAPIKey(cfg.apiKey))
		}
		if cfg.baseURL != "" {
			opts = append(opts, openrouter.WithBaseURL(cfg.baseURL))
		}
		if cfg.timeout > 0 {
			opts = append(opts, openrouter.WithTimeout(cfg.timeout))
		}
		if logger := debugLogger(cfg); logger != nil {
			opts = append(opts, openrouter.WithLogger(logger), openrouter.WithWireCapture(llm.WireCaptureToLogger(logger)))
		}
		return openrouter.New(opts...)
	case "vllm":
		if err := checkServerURL(cfg.name, cfg.baseURL, "http://localhost:8000/v1"); err != nil {
			return nil, err
		}
		opts := []vllm.Option{}
		if store != nil {
			opts = append(opts, vllm.WithAPIKeyFunc(storedAPIKey(store, cfg.name)))
		} else if cfg.apiKey != "" {
			opts = append(opts, vllm.WithAPIKey(cfg.apiKey))
		}
		if cfg.timeout > 0 {
			opts = append(opts, vllm.WithTimeout(cfg.timeout))
		}
		if logger := debugLogger(cfg); logger != nil {
			opts = append(opts, vllm.WithLogger(logger), vllm.WithWireCapture(llm.WireCaptureToLogger(logger)))
		}
		return vllm.New(cfg.baseURL, opts...)
	case "chatcompletions":
		if err := checkServerURL(cfg.name, cfg.baseURL, "http://localhost:11434/v1"); err != nil {
			return nil, err
		}
		opts := []chatcompletions.Option{}
		if store != nil {
			opts = append(opts, chatcompletions.WithAPIKeyFunc(storedAPIKey(store, cfg.name)))
		} else if cfg.apiKey != "" {
			opts = append(opts, chatcompletions.WithAPIKey(cfg.apiKey))
		}
		if cfg.timeout > 0 {
			opts = append(opts, chatcompletions.WithTimeout(cfg.timeout))
		}
		if logger := debugLogger(cfg); logger != nil {
			opts = append(opts, chatcompletions.WithLogger(logger), chatcompletions.WithWireCapture(llm.WireCaptureToLogger(logger)))
		}
		return chatcompletions.New(cfg.baseURL, opts...)
	case "typesafe", "vercel-ai-gateway", "opencode", "systemone":
		return newSystemOneProvider(cfg, store)
	case "cloudflare-workers-ai":
		opts := []cloudflare.Option{}
		if store != nil {
			opts = append(opts, cloudflare.WithCredentials(store))
		} else if cfg.apiKey != "" {
			opts = append(opts, cloudflare.WithAPIKey(cfg.apiKey))
		}
		if cfg.baseURL != "" {
			opts = append(opts, cloudflare.WithBaseURL(cfg.baseURL))
		}
		if cfg.timeout > 0 {
			opts = append(opts, cloudflare.WithTimeout(cfg.timeout))
		}
		if logger := debugLogger(cfg); logger != nil {
			opts = append(opts, cloudflare.WithLogger(logger), cloudflare.WithWireCapture(llm.WireCaptureToLogger(logger)))
		}
		return cloudflare.New(opts...)
	case "":
		return nil, fmt.Errorf("%w: missing provider; set -p/--provider", llm.ErrBadRequest)
	default:
		return nil, fmt.Errorf("%w: unknown provider %q", llm.ErrBadRequest, cfg.name)
	}
}

// newSystemOneProvider builds the classifier-only System One services:
// TypeSafe, Vercel AI Gateway, OpenCode Zen, or any System One server
// (-p systemone with --base-url; its key is optional, as for vllm).
func newSystemOneProvider(cfg providerConfig, store llm.CredentialStore) (llm.Provider, error) {
	opts := []systemone.Option{}
	switch {
	case cfg.name == "systemone" && store != nil:
		opts = append(opts, systemone.WithAPIKeyFunc(storedAPIKey(store, cfg.name)))
	case store != nil:
		opts = append(opts, systemone.WithCredentials(store))
	case cfg.apiKey != "":
		opts = append(opts, systemone.WithAPIKey(cfg.apiKey))
	}
	if cfg.baseURL != "" && cfg.name != "systemone" {
		opts = append(opts, systemone.WithBaseURL(cfg.baseURL))
	}
	if cfg.timeout > 0 {
		opts = append(opts, systemone.WithTimeout(cfg.timeout))
	}
	if logger := debugLogger(cfg); logger != nil {
		opts = append(opts, systemone.WithLogger(logger), systemone.WithWireCapture(llm.WireCaptureToLogger(logger)))
	}
	switch cfg.name {
	case "typesafe":
		return systemone.NewTypeSafe(opts...)
	case "vercel-ai-gateway":
		return systemone.NewVercelAIGateway(opts...)
	case "opencode":
		return systemone.NewOpenCode(opts...)
	}
	if err := checkServerURL(cfg.name, cfg.baseURL, "http://localhost:8080/v1"); err != nil {
		return nil, err
	}
	return systemone.New(cfg.baseURL, opts...)
}

// checkServerURL requires a self-hosted server's --base-url to be an
// absolute HTTP(S) URL, so a missing scheme fails here rather than inside
// the SDK.
func checkServerURL(provider, raw, example string) error {
	parsed, err := url.Parse(raw)
	if raw == "" || err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return fmt.Errorf("%w: -p %s needs --base-url set to the server's /v1 root as an http(s) URL (e.g. %s)", llm.ErrBadRequest, provider, example)
	}
	return nil
}

// storedAPIKey reads a self-hosted server's optional key from the auth file:
// the "api_key" entry named after the provider, or no key when absent.
func storedAPIKey(store llm.CredentialStore, providerID string) func(context.Context) (string, error) {
	return func(ctx context.Context) (string, error) {
		cred, ok, err := store.Read(ctx, providerID)
		if err != nil || !ok {
			return "", err
		}
		if cred.Type != "api_key" {
			return "", fmt.Errorf("%w: auth file entry %q is a %q credential; self-hosted servers take an api_key", llm.ErrAuth, providerID, cred.Type)
		}
		return cred.Key, nil
	}
}

func debugLogger(cfg providerConfig) *slog.Logger {
	if !cfg.debug {
		return nil
	}
	w := cfg.stderr
	if w == nil {
		w = os.Stderr
	}
	return slog.New(slog.NewTextHandler(w, &slog.HandlerOptions{Level: slog.LevelDebug}))
}

func providerConfigFromChat(cfg chatConfig, stderr io.Writer) providerConfig {
	return providerConfig{
		name:     cfg.provider,
		apiKey:   cfg.apiKey,
		authFile: cfg.authFile,

		authFileCommands: cfg.authFileCommands,
		baseURL:          cfg.baseURL,
		timeout:          cfg.timeout,
		debug:            cfg.debug,
		stderr:           stderr,
	}
}

func providerConfigFromModels(cfg modelsConfig, stderr io.Writer) providerConfig {
	return providerConfig{
		name:     cfg.provider,
		apiKey:   cfg.apiKey,
		authFile: cfg.authFile,

		authFileCommands: cfg.authFileCommands,
		baseURL:          cfg.baseURL,
		timeout:          cfg.timeout,
		debug:            cfg.debug,
		stderr:           stderr,
	}
}

func providerConfigFromClassify(cfg classifyConfig, stderr io.Writer) providerConfig {
	return providerConfig{
		name:     cfg.provider,
		apiKey:   cfg.apiKey,
		authFile: cfg.authFile,

		authFileCommands: cfg.authFileCommands,
		baseURL:          cfg.baseURL,
		timeout:          cfg.timeout,
		debug:            cfg.debug,
		stderr:           stderr,
	}
}
