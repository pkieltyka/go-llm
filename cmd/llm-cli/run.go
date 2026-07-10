package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"os"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/schema"
)

func (a app) runChat(ctx context.Context, cfg chatConfig) error {
	bundle, err := buildRequest(cfg)
	if err != nil {
		return err
	}
	provider, err := a.providerFactory(ctx, providerConfigFromChat(cfg, a.stderr))
	if err != nil {
		return err
	}

	var resp *llm.Response
	if cfg.jsonOutput || cfg.noStream || cfg.schemaPath != "" {
		resp, err = provider.Chat(ctx, bundle.request)
		if err != nil {
			return err
		}
		validatedJSON, err := validateStructuredOutput(bundle.request.ResponseFormat, resp.Text())
		if err != nil {
			return err
		}
		if cfg.jsonOutput {
			data, err := llm.MarshalResponse(resp)
			if err != nil {
				return err
			}
			if _, err := fmt.Fprintln(a.stdout, string(data)); err != nil {
				return fmt.Errorf("write response: %w", err)
			}
		} else if validatedJSON != nil {
			if _, err := fmt.Fprintln(a.stdout, string(validatedJSON)); err != nil {
				return fmt.Errorf("write response: %w", err)
			}
		} else {
			if _, err := fmt.Fprint(a.stdout, resp.Text()); err != nil {
				return fmt.Errorf("write response: %w", err)
			}
		}
	} else {
		resp, err = a.runStreaming(ctx, provider, bundle.request, cfg)
		if err != nil {
			return err
		}
	}

	if len(resp.ToolCalls()) > 0 && !cfg.jsonOutput {
		if err := printToolCalls(a.stdout, resp.ToolCalls()); err != nil {
			return err
		}
	}
	if cfg.usage {
		if err := printUsage(a.stderr, resp.Usage); err != nil {
			return err
		}
	}
	if cfg.savePath != "" {
		data, err := llm.MarshalMessages(historyMessages(bundle, resp))
		if err != nil {
			return err
		}
		if err := os.WriteFile(cfg.savePath, data, 0o600); err != nil {
			return fmt.Errorf("save conversation: %w", err)
		}
	}
	return nil
}

func validateStructuredOutput(format *llm.ResponseFormat, text string) ([]byte, error) {
	if format == nil {
		return nil, nil
	}
	raw := json.RawMessage(text)
	if !json.Valid(raw) {
		return nil, fmt.Errorf("%w: structured output is not valid JSON", llm.ErrBadRequest)
	}
	if err := schema.ValidateArgs(llm.Tool{Name: format.Name, InputSchema: format.Schema}, raw); err != nil {
		return nil, fmt.Errorf("structured output validation failed: %w", err)
	}
	var compact bytes.Buffer
	if err := json.Compact(&compact, raw); err != nil {
		return nil, err
	}
	return compact.Bytes(), nil
}

func (a app) runStreaming(ctx context.Context, provider llm.Provider, req *llm.Request, cfg chatConfig) (*llm.Response, error) {
	events := streamOutput(provider.ChatStream(ctx, req), a.stdout, a.stderr, cfg.reasoning)
	return llm.Collect(events)
}

func streamOutput(events iter.Seq2[llm.Event, error], stdout, stderr io.Writer, reasoning bool) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		for event, err := range events {
			if err != nil {
				yield(event, err)
				return
			}
			if err := printStreamEvent(stdout, stderr, reasoning, event); err != nil {
				yield(nil, err)
				return
			}
			if !yield(event, nil) {
				return
			}
		}
	}
}

func printStreamEvent(stdout, stderr io.Writer, reasoning bool, event llm.Event) error {
	switch e := event.(type) {
	case llm.TextDelta:
		if _, err := fmt.Fprint(stdout, e.Text); err != nil {
			return fmt.Errorf("write response event: %w", err)
		}
	case llm.ReasoningDelta:
		if reasoning && e.Text != "" {
			if _, err := fmt.Fprint(stderr, e.Text); err != nil {
				return fmt.Errorf("write reasoning event: %w", err)
			}
		}
	}
	return nil
}

func printToolCalls(w io.Writer, calls []llm.ToolCallPart) error {
	data, err := json.MarshalIndent(calls, "", "  ")
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintln(w); err != nil {
		return fmt.Errorf("write tool calls: %w", err)
	}
	if _, err := fmt.Fprintln(w, string(data)); err != nil {
		return fmt.Errorf("write tool calls: %w", err)
	}
	return nil
}

func printUsage(w io.Writer, u llm.Usage) error {
	var line bytes.Buffer
	fmt.Fprintf(&line, "usage input=%d output=%d total=%d cache_read=%d cache_write=%d reasoning=%d",
		u.InputTokens,
		u.OutputTokens,
		u.TotalTokens,
		u.CacheReadTokens,
		u.CacheWriteTokens,
		u.ReasoningTokens,
	)
	if u.CostUSD != nil {
		fmt.Fprintf(&line, " cost_usd=%.8f", *u.CostUSD)
	}
	if _, err := fmt.Fprintln(w, line.String()); err != nil {
		return fmt.Errorf("write usage: %w", err)
	}
	return nil
}
