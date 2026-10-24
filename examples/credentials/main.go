// Credentials keeps provider credentials in a pi-compatible auth.json and
// lets providers read them on every request.
//
// authstore.NewFile is an llm.CredentialStore over pi's auth.json format,
// with pi's file lock, so pi and go-llm can share one file. An api_key may
// be a pi reference instead of the secret itself: "$OPENROUTER_API_KEY" (or
// "${VAR}") reads the environment when the key is used, and "!op read ..."
// runs a command (only with authstore.WithKeyCommands). The file keeps the
// reference; the resolved key is never written back. `llm-cli login` adds a
// Sign in with ChatGPT credential to the same file.
//
// With OPENROUTER_API_KEY set it calls OpenRouter; without it, it serves the
// same request from an in-process OpenRouter-compatible server, so the
// example runs offline over real HTTP.
package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
	"github.com/pkieltyka/go-llm/providers/openrouter"
)

func main() {
	ctx := context.Background()
	dir, err := os.MkdirTemp("", "go-llm-credentials-*")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)

	// The file names an environment variable rather than holding the key.
	// Offline, the variable is one only this demo sets, so the fake server
	// accepting the bearer proves the key came through the store.
	reference := "$OPENROUTER_API_KEY"
	opts := []openrouter.Option{}
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		fmt.Println("(offline demo against an in-process server — set OPENROUTER_API_KEY to call OpenRouter)")
		server := httptest.NewServer(fakeOpenRouter("offline-demo-key"))
		defer server.Close()
		reference = "$GO_LLM_DEMO_KEY"
		os.Setenv("GO_LLM_DEMO_KEY", "offline-demo-key")
		opts = append(opts, openrouter.WithBaseURL(server.URL+"/api/v1"))
	}
	path := filepath.Join(dir, "auth.json")
	if err := os.WriteFile(path, []byte(`{
  "openrouter": {"type": "api_key", "key": "`+reference+`", "model": "qwen/qwen3.6-27b"}
}`), 0o600); err != nil {
		panic(err)
	}

	store := authstore.NewFile(path)
	cred, ok, err := store.Read(ctx, "openrouter")
	if err != nil || !ok {
		panic(fmt.Sprintf("read credential: %v (found %v)", err, ok))
	}
	fmt.Printf("stored credential: type %s, resolved key of %d bytes, model %s\n", cred.Type, len(cred.Key), cred.Model)

	// The provider reads the store on every request: rotate the variable or
	// edit the file and the next call uses the new key, with no rebuild.
	p, err := openrouter.New(append(opts, openrouter.WithCredentials(store))...)
	if err != nil {
		panic(err)
	}
	resp, err := p.Chat(ctx, &llm.Request{
		Model:     cred.Model,
		Messages:  []llm.Message{llm.UserText("Say hello in one short sentence.")},
		MaxTokens: 100,
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Text())

	raw, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	fmt.Printf("auth.json still holds the reference %s: %v\n", reference, strings.Contains(string(raw), `"`+reference+`"`))
}

// fakeOpenRouter answers chat completions for one expected bearer token.
func fakeOpenRouter(key string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+key {
			http.Error(w, `{"error":{"message":"bad key","code":401}}`, http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"gen-1","object":"chat.completion","model":"qwen/qwen3.6-27b",
			"choices":[{"index":0,"message":{"role":"assistant","content":"Hello from a key the file only referenced."},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":12,"completion_tokens":9,"total_tokens":21}}`)
	})
}
