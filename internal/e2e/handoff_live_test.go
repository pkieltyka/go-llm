//go:build live

package e2e

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/anthropic"
	"github.com/pkieltyka/go-llm/providers/openai"
	"github.com/pkieltyka/go-llm/providers/openaicodex"
	"github.com/pkieltyka/go-llm/providers/openrouter"
	"github.com/pkieltyka/go-llm/providers/vllm"
)

func recordingSecrets(ctx context.Context) *SecretSet {
	secrets, _ := ctx.Value(recordingSecretsContextKey{}).(*SecretSet)
	return secrets
}

// liveCrossProviderHandoffScenario implements ARCH §9's cross-provider
// handoff check: a tool-using conversation started on the scenario's provider
// is round-tripped through the canonical persistence envelope
// (MarshalMessages/UnmarshalMessages) and continued on a DIFFERENT configured
// provider. Foreign reasoning parts in the history must be dropped silently
// (FS §18) — the core assertion is that the continuation succeeds at all
// (no 400) and produces a sane completion.
func liveCrossProviderHandoffScenario(ctx context.Context, t *testing.T, source llm.Provider, model string) {
	t.Helper()
	secrets := recordingSecrets(ctx)
	if secrets == nil {
		t.Fatal("cross-provider handoff requires a recording secret set in context")
	}
	target, targetModel := handoffContinuationProvider(t, source.Name(), secrets)
	t.Logf("handoff: %s(%s) -> %s(%s)", source.Name(), model, target.Name(), targetModel)

	tools := []llm.Tool{{
		Name:        "lookup",
		Description: "Look up a short value.",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}`),
		Strict:      true,
	}}
	temperature := 0.0
	first := &llm.Request{
		Model:       model,
		MaxTokens:   128,
		Temperature: &temperature,
		Messages:    []llm.Message{llm.UserText(`Use the lookup tool with q exactly "go".`)},
		Tools:       tools,
		ToolChoice:  llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "lookup"},
	}
	resp, err := source.Chat(ctx, first)
	if err != nil {
		t.Fatalf("handoff source Chat returned error: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) == 0 || calls[0].ID == "" || calls[0].Name != "lookup" {
		t.Fatalf("handoff source tool calls = %+v", calls)
	}

	history := []llm.Message{
		first.Messages[0],
		{
			Role:     llm.RoleAssistant,
			Parts:    resp.Parts,
			Provider: resp.Provider,
			Model:    resp.Model,
		},
		{
			Role:  llm.RoleTool,
			Parts: []llm.Part{llm.ToolResultPart{ToolCallID: calls[0].ID, Name: calls[0].Name, Content: []llm.Part{llm.Text(`{"answer":"ok"}`)}}},
		},
		llm.UserText("Given the tool result, answer exactly: handoff ok"),
	}

	// Round-trip through the canonical envelope — the point of the scenario:
	// what one provider produced must continue on another after persistence.
	data, err := llm.MarshalMessages(history)
	if err != nil {
		t.Fatalf("MarshalMessages returned error: %v", err)
	}
	restored, err := llm.UnmarshalMessages(data)
	if err != nil {
		t.Fatalf("UnmarshalMessages returned error: %v", err)
	}

	continuation, err := target.Chat(ctx, &llm.Request{
		Model:     targetModel,
		MaxTokens: 256,
		Effort:    llm.EffortNone,
		Messages:  restored,
		Tools:     tools,
	})
	if err != nil {
		t.Fatalf("handoff continuation on %s returned error: %v", target.Name(), err)
	}
	if strings.TrimSpace(continuation.Text()) == "" && len(continuation.ToolCalls()) == 0 {
		t.Fatalf("handoff continuation produced no content: stop_reason=%s parts=%+v", continuation.StopReason, continuation.Parts)
	}
	if continuation.Usage.OutputTokens == 0 {
		t.Fatalf("handoff continuation usage missing output tokens: %+v", continuation.Usage)
	}
}

// handoffContinuationProvider builds the first configured provider other than
// sourceName, in a fixed priority order. It skips the scenario visibly when
// fewer than two providers are configured.
func handoffContinuationProvider(t *testing.T, sourceName string, secrets *SecretSet) (llm.Provider, string) {
	t.Helper()
	root, err := RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("LoadConfig returned error: %v", err)
	}
	for _, name := range []string{"anthropic", "openrouter", "openai", "openai-codex", "vllm"} {
		if name == sourceName {
			continue
		}
		provider, model, ok := handoffProvider(t, root, cfg, name, secrets)
		if !ok {
			continue
		}
		return provider, model
	}
	t.Skipf("cross-provider handoff needs a second configured provider besides %s in gollm-test.json", sourceName)
	return nil, ""
}

func handoffProvider(t *testing.T, root string, cfg Config, name string, secrets *SecretSet) (llm.Provider, string, bool) {
	t.Helper()
	switch name {
	case "anthropic":
		providerCfg := cfg.Provider("anthropic", "ANTHROPIC_API_KEY")
		opts := []anthropic.Option{anthropic.WithMaxRetries(0)}
		switch {
		case providerCfg.Auth.Type == "oauth" && providerCfg.Auth.Access != "":
			onRefresh := PersistOnRefresh(filepath.Join(root, "gollm-test.json"), "anthropic", t.Logf, secrets)
			opts = append(opts, anthropic.WithOAuth(providerCfg.Auth, onRefresh))
		case providerCfg.Auth.Key != "":
			opts = append(opts, anthropic.WithAPIKey(providerCfg.Auth.Key))
		default:
			return nil, "", false
		}
		if providerCfg.BaseURL != "" {
			opts = append(opts, anthropic.WithBaseURL(providerCfg.BaseURL))
		}
		p, err := anthropic.New(opts...)
		if err != nil {
			t.Fatalf("anthropic.New returned error: %v", err)
		}
		return p, orDefault(providerCfg.Model, anthropicCheapModel), true
	case "openrouter":
		providerCfg := cfg.Provider("openrouter", "OPENROUTER_API_KEY")
		if providerCfg.Auth.Key == "" {
			return nil, "", false
		}
		opts := []openrouter.Option{
			openrouter.WithAPIKey(providerCfg.Auth.Key),
			openrouter.WithMaxRetries(0),
			openrouter.WithAttribution("https://github.com/pkieltyka/go-llm", "go-llm live tests"),
		}
		if providerCfg.BaseURL != "" {
			opts = append(opts, openrouter.WithBaseURL(providerCfg.BaseURL))
		}
		p, err := openrouter.New(opts...)
		if err != nil {
			t.Fatalf("openrouter.New returned error: %v", err)
		}
		return p, orDefault(providerCfg.Model, openRouterCheapModel), true
	case "openai":
		providerCfg := cfg.Provider("openai", "OPENAI_API_KEY")
		if providerCfg.Auth.Key == "" {
			return nil, "", false
		}
		opts := []openai.Option{
			openai.WithAPIKey(providerCfg.Auth.Key),
			openai.WithMaxRetries(0),
		}
		if providerCfg.BaseURL != "" {
			opts = append(opts, openai.WithBaseURL(providerCfg.BaseURL))
		}
		p, err := openai.New(opts...)
		if err != nil {
			t.Fatalf("openai.New returned error: %v", err)
		}
		return p, orDefault(providerCfg.Model, openAICheapModel), true
	case "openai-codex":
		providerCfg := cfg.Provider("openai-codex", "")
		if providerCfg.Auth.Type != "oauth" || providerCfg.Auth.Access == "" {
			return nil, "", false
		}
		onRefresh := PersistOnRefresh(filepath.Join(root, "gollm-test.json"), "openai-codex", t.Logf, secrets)
		opts := []openaicodex.Option{
			openaicodex.WithOAuth(providerCfg.Auth, onRefresh),
			openaicodex.WithMaxRetries(0),
		}
		if providerCfg.BaseURL != "" {
			opts = append(opts, openaicodex.WithBaseURL(providerCfg.BaseURL))
		}
		p, err := openaicodex.New(opts...)
		if err != nil {
			t.Fatalf("openaicodex.New returned error: %v", err)
		}
		return p, orDefault(providerCfg.Model, openAICodexCheapModel), true
	case "vllm":
		providerCfg := cfg.Provider("vllm", "")
		// Keyless self-hosted entries are configured by base_url presence;
		// there is no universal default model, so one must be configured.
		if providerCfg.BaseURL == "" || providerCfg.Model == "" {
			return nil, "", false
		}
		opts := []vllm.Option{vllm.WithMaxRetries(0)}
		if providerCfg.Auth.Key != "" {
			opts = append(opts, vllm.WithAPIKey(providerCfg.Auth.Key))
		}
		p, err := vllm.New(providerCfg.BaseURL, opts...)
		if err != nil {
			t.Fatalf("vllm.New returned error: %v", err)
		}
		return p, providerCfg.Model, true
	default:
		return nil, "", false
	}
}

func orDefault(value, fallback string) string {
	if value != "" {
		return value
	}
	return fallback
}
