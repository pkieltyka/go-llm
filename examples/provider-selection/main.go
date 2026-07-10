// Provider-selection constructs a provider from whatever credentials are
// available, in priority order:
//
//  1. LLM_AUTH_FILE — a pi-compatible auth file loaded with llm.LoadAuthFile,
//     covering both API-key providers and subscription OAuth (openai-codex).
//  2. Provider env keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY.
//  3. The scripted llmtest fallback, so the example still runs offline.
//
// Every provider satisfies the same llm.Provider interface, so the calling
// code below is identical no matter which one is selected.
package main

import (
	"context"
	"fmt"
	"os"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/providers/anthropic"
	"github.com/pkieltyka/go-llm/providers/openai"
	"github.com/pkieltyka/go-llm/providers/openaicodex"
	"github.com/pkieltyka/go-llm/providers/openrouter"
)

func main() {
	p, model, err := selectProvider()
	if err != nil {
		panic(err)
	}
	fmt.Printf("selected: %s/%s\n", p.Name(), model)

	ctx := context.Background()
	resp, err := p.Chat(ctx, &llm.Request{
		Model:    model,
		Messages: []llm.Message{llm.UserText("Say hello in one short sentence.")},
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Text())
}

// selectProvider returns the first provider with usable credentials.
func selectProvider() (llm.Provider, string, error) {
	if path := os.Getenv("LLM_AUTH_FILE"); path != "" {
		return fromAuthFile(path)
	}
	switch {
	case os.Getenv("ANTHROPIC_API_KEY") != "":
		p, err := anthropic.New() // reads ANTHROPIC_API_KEY
		return p, "claude-haiku-4-5", err
	case os.Getenv("OPENAI_API_KEY") != "":
		p, err := openai.New() // reads OPENAI_API_KEY
		return p, "gpt-4.1-mini", err
	case os.Getenv("OPENROUTER_API_KEY") != "":
		p, err := openrouter.New() // reads OPENROUTER_API_KEY
		return p, "qwen/qwen3.6-27b", err
	}
	return offlineProvider()
}

// fromAuthFile builds the first configured provider found in a pi-compatible
// credential file (either bare or wrapped in {"providers": ...}).
func fromAuthFile(path string) (llm.Provider, string, error) {
	auth, err := llm.LoadAuthFile(path)
	if err != nil {
		return nil, "", err
	}
	if cred, ok := auth["anthropic"]; ok && cred.Key != "" {
		p, err := anthropic.New(anthropic.WithAPIKey(cred.Key))
		return p, modelOr(cred.Model, "claude-haiku-4-5"), err
	}
	if cred, ok := auth["openai"]; ok && cred.Key != "" {
		p, err := openai.New(openai.WithAPIKey(cred.Key))
		return p, modelOr(cred.Model, "gpt-4.1-mini"), err
	}
	if cred, ok := auth["openrouter"]; ok && cred.Key != "" {
		p, err := openrouter.New(openrouter.WithAPIKey(cred.Key))
		return p, modelOr(cred.Model, "qwen/qwen3.6-27b"), err
	}
	if cred, ok := auth["openai-codex"]; ok && cred.Access != "" {
		var persist llm.OAuthPersistenceFunc
		if cred.Refresh != "" {
			// This demo deliberately keeps rotations in memory. Production code
			// should durably update path instead; this no-op can leave the stored
			// refresh token stale after restart.
			fmt.Fprintln(os.Stderr, "warning: OAuth rotations are not persisted by this demo")
			persist = func(ctx context.Context, _ llm.AuthCredential) error {
				return ctx.Err()
			}
		}
		p, err := openaicodex.New(openaicodex.WithOAuth(cred, persist))
		return p, modelOr(cred.Model, "gpt-5.4-mini"), err
	}
	return nil, "", fmt.Errorf("no usable credentials in %s", path)
}

func modelOr(model, fallback string) string {
	if model != "" {
		return model
	}
	return fallback
}

// offlineProvider is the no-credentials fallback: a scripted llmtest fake.
func offlineProvider() (llm.Provider, string, error) {
	fmt.Println("(offline demo — set LLM_AUTH_FILE or a provider API key env var to run against a real API)")
	p := llmtest.New()
	p.EnqueueResponse(&llm.Response{
		Provider: "llmtest",
		Model:    "demo-model",
		Parts:    []llm.Part{llm.Text("Hello from go-llm.")},
	})
	return p, "demo-model", nil
}
