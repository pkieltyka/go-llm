package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/anthropic"
	"github.com/pkieltyka/go-llm/providers/openai"
	"github.com/pkieltyka/go-llm/providers/openaicodex"
	"github.com/pkieltyka/go-llm/providers/openrouter"
)

func newProvider(_ context.Context, cfg providerConfig) (llm.Provider, error) {
	switch cfg.name {
	case "anthropic":
		opts := []anthropic.Option{}
		if cfg.apiKey != "" {
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
		if cfg.apiKey != "" {
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
	case "openai-codex":
		cred := llm.AuthCredential{Type: "oauth", Access: cfg.apiKey}
		opts := []openaicodex.Option{openaicodex.WithOAuth(cred, nil)}
		if cfg.baseURL != "" {
			opts = append(opts, openaicodex.WithBaseURL(cfg.baseURL))
		}
		if cfg.timeout > 0 {
			opts = append(opts, openaicodex.WithTimeout(cfg.timeout))
		}
		if logger := debugLogger(cfg); logger != nil {
			opts = append(opts, openaicodex.WithLogger(logger), openaicodex.WithWireCapture(llm.WireCaptureToLogger(logger)))
		}
		return openaicodex.New(opts...)
	case "openrouter":
		opts := []openrouter.Option{}
		if cfg.apiKey != "" {
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
	case "zai":
		return nil, fmt.Errorf("%w: provider zai is deferred in this module", llm.ErrUnsupported)
	case "":
		return nil, fmt.Errorf("%w: missing provider; set -p/--provider", llm.ErrBadRequest)
	default:
		return nil, fmt.Errorf("%w: unknown provider %q", llm.ErrBadRequest, cfg.name)
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
		name:    cfg.provider,
		apiKey:  cfg.apiKey,
		baseURL: cfg.baseURL,
		timeout: cfg.timeout,
		debug:   cfg.debug,
		stderr:  stderr,
	}
}

func providerConfigFromModels(cfg modelsConfig, stderr io.Writer) providerConfig {
	return providerConfig{
		name:    cfg.provider,
		apiKey:  cfg.apiKey,
		baseURL: cfg.baseURL,
		timeout: cfg.timeout,
		debug:   cfg.debug,
		stderr:  stderr,
	}
}
