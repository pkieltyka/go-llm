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
			fmt.Fprintln(a.stdout, string(data))
		} else if validatedJSON != nil {
			fmt.Fprintln(a.stdout, string(validatedJSON))
		} else {
			fmt.Fprint(a.stdout, resp.Text())
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
		printUsage(a.stderr, resp.Usage)
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
	var events []llm.Event
	for event, err := range provider.ChatStream(ctx, req) {
		if err != nil {
			// The caller returns immediately on error, so a partial response
			// collected from the buffered events would never be used.
			return nil, err
		}
		events = append(events, event)
		switch e := event.(type) {
		case llm.TextDelta:
			fmt.Fprint(a.stdout, e.Text)
		case llm.ReasoningDelta:
			if cfg.reasoning && e.Text != "" {
				fmt.Fprint(a.stderr, e.Text)
			}
		}
	}
	return llm.Collect(eventsSeq(events))
}

func eventsSeq(events []llm.Event) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		for _, event := range events {
			if !yield(event, nil) {
				return
			}
		}
	}
}

func printToolCalls(w io.Writer, calls []llm.ToolCallPart) error {
	data, err := json.MarshalIndent(calls, "", "  ")
	if err != nil {
		return err
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, string(data))
	return nil
}

func printUsage(w io.Writer, u llm.Usage) {
	fmt.Fprintf(w, "usage input=%d output=%d total=%d cache_read=%d cache_write=%d reasoning=%d",
		u.InputTokens,
		u.OutputTokens,
		u.TotalTokens,
		u.CacheReadTokens,
		u.CacheWriteTokens,
		u.ReasoningTokens,
	)
	if u.CostUSD != nil {
		fmt.Fprintf(w, " cost_usd=%.8f", *u.CostUSD)
	}
	fmt.Fprintln(w)
}
