package e2e

import (
	"context"
	"errors"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestResolveListedModelAcceptsAliasesAndCanonicalIDs(t *testing.T) {
	tests := []struct {
		name       string
		provider   string
		configured string
		models     []llm.ModelInfo
		want       string
	}{
		{
			name:       "exact",
			provider:   "openai",
			configured: "gpt-5-mini",
			models:     []llm.ModelInfo{{ID: "gpt-5-mini"}},
			want:       "gpt-5-mini",
		},
		{
			name:       "dated alias",
			provider:   "anthropic",
			configured: "claude-haiku-4-5",
			models:     []llm.ModelInfo{{ID: "claude-haiku-4-5-20251001"}},
			want:       "claude-haiku-4-5-20251001",
		},
		{
			name:       "provider qualified canonical",
			provider:   "openrouter",
			configured: "claude-haiku-4.5",
			models:     []llm.ModelInfo{{ID: "anthropic/claude-haiku-4.5", CanonicalID: "anthropic/claude-haiku-4.5-20251001"}},
			want:       "anthropic/claude-haiku-4.5",
		},
		{
			name:       "same provider qualified",
			provider:   "openrouter",
			configured: "anthropic/claude-haiku-4.5",
			models:     []llm.ModelInfo{{ID: "anthropic/claude-haiku-4.5-20251001"}},
			want:       "anthropic/claude-haiku-4.5-20251001",
		},
		{
			name:       "explicitly unqualified alias",
			provider:   "openrouter",
			configured: "claude-haiku-4.5",
			models:     []llm.ModelInfo{{ID: "anthropic/claude-haiku-4.5"}},
			want:       "anthropic/claude-haiku-4.5",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model, ok := resolveListedModel(test.provider, test.configured, test.models)
			if !ok || model.ID != test.want {
				t.Fatalf("resolveListedModel = (%+v, %v), want %q", model, ok, test.want)
			}
		})
	}
	if _, ok := resolveListedModel("openai", "gpt-5-mini", []llm.ModelInfo{{ID: "gpt-52-mini"}}); ok {
		t.Fatal("boundary-less prefix must not match")
	}
	if _, ok := resolveListedModel("openai", "gpt-5", []llm.ModelInfo{{ID: "gpt-5.1"}}); ok {
		t.Fatal("different model versions must not be treated as aliases")
	}
	if _, ok := resolveListedModel("openrouter", "anthropic/foo", []llm.ModelInfo{{ID: "openai/foo"}}); ok {
		t.Fatal("different qualified providers must not match")
	}
	if _, ok := resolveListedModel("openrouter", "anthropic/foo", []llm.ModelInfo{{ID: "alias", CanonicalID: "openai/foo"}}); ok {
		t.Fatal("different qualified canonical providers must not match")
	}
}

func TestResolveConfiguredModelUsesVLLMFuzzyResolver(t *testing.T) {
	provider := &resolvingProvider{
		coverageProvider: coverageProvider{name: "vllm"},
		wantPreference:   "qwen",
		resolved:         llm.ModelInfo{ID: "nvidia/Qwen-3.6-27B-NVFP4"},
	}
	model, err := ResolveConfiguredModel(context.Background(), provider, "qwen")
	if err != nil {
		t.Fatalf("ResolveConfiguredModel returned error: %v", err)
	}
	if model != provider.resolved.ID || provider.calls != 1 {
		t.Fatalf("model = %q calls=%d", model, provider.calls)
	}
}

func TestResolveConfiguredModelRejectsEmptyResolution(t *testing.T) {
	provider := &resolvingProvider{coverageProvider: coverageProvider{name: "vllm"}}
	if _, err := ResolveConfiguredModel(context.Background(), provider, "qwen"); !errors.Is(err, llm.ErrNotFound) {
		t.Fatalf("error = %v, want ErrNotFound", err)
	}
}

type resolvingProvider struct {
	coverageProvider
	wantPreference string
	resolved       llm.ModelInfo
	calls          int
}

func (p *resolvingProvider) ResolveModel(_ context.Context, preference string) (llm.ModelInfo, error) {
	p.calls++
	if p.wantPreference != "" && preference != p.wantPreference {
		return llm.ModelInfo{}, errors.New("unexpected preference")
	}
	return p.resolved, nil
}

var _ llm.Provider = (*resolvingProvider)(nil)
var _ modelPreferenceResolver = (*resolvingProvider)(nil)
