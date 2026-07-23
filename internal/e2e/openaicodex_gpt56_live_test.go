//go:build live

package e2e

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/openaicodex"
)

// TestLiveOpenAICodexGPT56 is the plan-2 phase-3b spike: verify the gpt-5.6
// "Responses Lite" contract live before trusting the hardcoded shape. Checks
// run in the plan's order — models-endpoint probe, 5.6 without tools, 5.6
// with tools and system instructions, stream decode + cache-write usage,
// then a pre-5.6 control. Failures here are OBSERVATIONS for the spike
// record; do not "fix" the contract from guesses.
func TestLiveOpenAICodexGPT56(t *testing.T) {
	root, err := RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("LoadConfig returned error: %v", err)
	}
	providerCfg := cfg.Provider("openai-codex", "")
	requireLiveProviderConfig(t, "openai-codex", providerCfg)

	secrets := NewSecretSet(providerCfg.Auth.Access, providerCfg.Auth.Refresh, providerCfg.Auth.AccountID)
	persist := AuthFilePersistence(filepath.Join(root, "gollm-test.json"), "openai-codex", t.Logf, secrets)
	opts := []openaicodex.Option{
		openaicodex.WithOAuth(providerCfg.Auth, persist),
		openaicodex.WithMaxRetries(1),
		openaicodex.WithTimeout(120 * time.Second),
	}
	baseURL := "https://chatgpt.com/backend-api/codex"
	if providerCfg.BaseURL != "" {
		baseURL = strings.TrimSuffix(providerCfg.BaseURL, "/")
		opts = append(opts, openaicodex.WithBaseURL(providerCfg.BaseURL))
	}
	p, err := openaicodex.New(opts...)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	ctx := context.Background()

	// Spike item 4: does the authenticated models endpoint exist, and does
	// it list a 5.6 family id? Informational — never fails the spike.
	model56 := os.Getenv("GOLLM_CODEX_GPT56_MODEL")
	t.Run("models_endpoint_probe", func(t *testing.T) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/models?client_version=0.144.0", nil)
		if err != nil {
			t.Fatalf("build request: %v", err)
		}
		req.Header.Set("Authorization", "Bearer "+providerCfg.Auth.Access)
		req.Header.Set("originator", "codex_cli_rs")
		req.Header.Set("User-Agent", "codex_cli_rs/0.144.0")
		req.Header.Set("Accept", "application/json")
		if providerCfg.Auth.AccountID != "" {
			req.Header.Set("chatgpt-account-id", providerCfg.Auth.AccountID)
		}
		resp, err := llm.DefaultHTTPClient().Do(req)
		if err != nil {
			t.Logf("models endpoint unreachable: %v", err)
			return
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		t.Logf("GET /models status=%d bytes=%d content-type=%s", resp.StatusCode, len(body), resp.Header.Get("Content-Type"))
		if resp.StatusCode != http.StatusOK {
			t.Logf("models endpoint body (non-200): %.500s", body)
			return
		}
		ids := extractModelIDs(body)
		t.Logf("models endpoint ids: %v", ids)
		if entry := rawModelEntry(body, "gpt-5.6-sol"); entry != nil {
			t.Logf("gpt-5.6-sol entry: %.3000s", entry)
		}
		for _, id := range ids {
			if model56 == "" && (id == "gpt-5.6" || strings.HasPrefix(id, "gpt-5.6-")) {
				model56 = id
			}
		}
	})
	if model56 == "" {
		model56 = "gpt-5.6"
	}
	t.Logf("using 5.6 model id: %s", model56)

	t.Run("no_tools", func(t *testing.T) {
		resp, err := p.Chat(ctx, &llm.Request{
			Model:    model56,
			Messages: []llm.Message{llm.UserText("Reply with exactly: ok")},
		})
		if err != nil {
			t.Fatalf("5.6 without tools failed (STOP condition — record, do not improvise): %v", err)
		}
		t.Logf("model=%s stop=%s text=%q usage=%+v", resp.Model, resp.StopReason, resp.Text(), resp.Usage)
	})

	t.Run("tools_and_system", func(t *testing.T) {
		resp, err := p.Chat(ctx, &llm.Request{
			Model:  model56,
			System: "You are a terse test probe. Always use tools when asked.",
			Messages: []llm.Message{
				llm.UserText("Call the lookup tool with q set to \"go\"."),
			},
			Tools: []llm.Tool{{
				Name:        "lookup",
				Description: "Look something up.",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`),
			}},
		})
		if err != nil {
			t.Fatalf("5.6 with tools+system failed (STOP condition — record, do not improvise): %v", err)
		}
		calls := resp.ToolCalls()
		t.Logf("model=%s stop=%s toolCalls=%d text=%q usage=%+v", resp.Model, resp.StopReason, len(calls), resp.Text(), resp.Usage)
		for _, call := range calls {
			t.Logf("tool call: name=%s args=%s", call.Name, call.Args)
		}
		if len(calls) == 0 && resp.Text() == "" {
			t.Fatal("no tool call and no text — decode problem?")
		}
	})

	t.Run("stream_and_cache_write_usage", func(t *testing.T) {
		resp, err := llm.Collect(p.ChatStream(ctx, &llm.Request{
			Model:    model56,
			Messages: []llm.Message{llm.UserText("Reply with exactly: ok")},
		}))
		if err != nil {
			t.Fatalf("5.6 stream failed (STOP condition — record, do not improvise): %v", err)
		}
		u := resp.Usage
		t.Logf("stream usage: input=%d cacheRead=%d cacheWrite=%d output=%d reasoning=%d total=%d",
			u.InputTokens, u.CacheReadTokens, u.CacheWriteTokens, u.OutputTokens, u.ReasoningTokens, u.TotalTokens)
		if got := u.InputTokens + u.CacheReadTokens + u.CacheWriteTokens + u.OutputTokens; u.TotalTokens != 0 && got != u.TotalTokens {
			t.Errorf("usage not additive: components sum %d, total %d", got, u.TotalTokens)
		}
	})

	t.Run("pre56_control", func(t *testing.T) {
		resp, err := p.Chat(ctx, &llm.Request{
			Model:    openAICodexCheapModel,
			Messages: []llm.Message{llm.UserText("Reply with exactly: ok")},
		})
		if err != nil {
			t.Fatalf("pre-5.6 control failed — the global User-Agent bump may have broken existing models: %v", err)
		}
		t.Logf("control model=%s stop=%s text=%q", resp.Model, resp.StopReason, resp.Text())
	})
}

// extractModelIDs pulls model identifiers out of an unknown-schema model
// list without logging the raw body: it looks for "id" or "slug" string
// fields on objects in any top-level array field.
func extractModelIDs(body []byte) []string {
	var ids []string
	var root map[string]json.RawMessage
	if err := json.Unmarshal(body, &root); err != nil {
		var list []map[string]any
		if err := json.Unmarshal(body, &list); err == nil {
			return idsFromObjects(list)
		}
		return nil
	}
	for _, raw := range root {
		var list []map[string]any
		if err := json.Unmarshal(raw, &list); err == nil {
			ids = append(ids, idsFromObjects(list)...)
		}
	}
	return ids
}

// rawModelEntry returns the JSON object whose id/slug equals wanted, from
// any top-level array field.
func rawModelEntry(body []byte, wanted string) []byte {
	var root map[string]json.RawMessage
	if err := json.Unmarshal(body, &root); err != nil {
		return nil
	}
	for _, raw := range root {
		var list []json.RawMessage
		if err := json.Unmarshal(raw, &list); err != nil {
			continue
		}
		for _, item := range list {
			var probe struct {
				ID   string `json:"id"`
				Slug string `json:"slug"`
			}
			if err := json.Unmarshal(item, &probe); err == nil && (probe.ID == wanted || probe.Slug == wanted) {
				return item
			}
		}
	}
	return nil
}

func idsFromObjects(list []map[string]any) []string {
	var ids []string
	for _, item := range list {
		if id, ok := item["id"].(string); ok && id != "" {
			ids = append(ids, id)
		} else if slug, ok := item["slug"].(string); ok && slug != "" {
			ids = append(ids, slug)
		}
	}
	return ids
}
