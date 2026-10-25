//go:build live

package e2e

import (
	"context"
	"errors"
	"flag"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/anthropic"
)

// TestAnthropicOAuthNamedProfileLive is a single explicitly selected manual
// compatibility check. It never records traffic or inspects other profiles.
func TestAnthropicOAuthNamedProfileLive(t *testing.T) {
	if os.Getenv("GOLLM_LIVE_ANTHROPIC_OAUTH") != "1" {
		t.Skip("requires explicit named-profile opt-in")
	}
	if flag.Lookup("test.run").Value.String() != "^TestAnthropicOAuthNamedProfileLive$" {
		t.Fatal("requires exact anchored named-profile test selection")
	}
	if *record || *recordAllowIncomplete {
		t.Fatal("recording is forbidden for this manual check")
	}
	root, err := RepoRoot(".")
	if err != nil {
		t.Fatal("repository root unavailable")
	}
	if exec.Command("git", "-C", root, "check-ignore", "--quiet", "gollm-test.json").Run() != nil {
		t.Fatal("manual credential file must be ignored")
	}
	if exec.Command("git", "-C", root, "ls-files", "--error-unmatch", "gollm-test.json").Run() == nil {
		t.Fatal("manual credential file must be untracked")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	store, err := newNamedAnthropicStore(ctx, filepath.Join(root, "gollm-test.json"))
	if err != nil {
		t.Fatal("named OAuth profile unavailable")
	}
	client := &http.Client{Timeout: 20 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	p, err := anthropic.New(anthropic.WithCredentials(store), anthropic.WithHTTPClient(client), anthropic.WithMaxRetries(0))
	if err != nil {
		t.Fatal("OAuth provider construction failed")
	}
	response, err := p.Chat(ctx, &llm.Request{Model: anthropicCheapModel, MaxTokens: 8, Messages: []llm.Message{llm.UserText("Reply with the word ok.")}})
	if err != nil {
		category := "provider failure"
		switch {
		case errors.Is(err, llm.ErrAuth):
			category = "authentication rejected"
		case errors.Is(err, llm.ErrRateLimited):
			category = "rate limited"
		case errors.Is(err, context.Canceled):
			category = "canceled"
		case errors.Is(err, context.DeadlineExceeded), errors.Is(err, llm.ErrTimeout):
			category = "timed out"
		case errors.Is(err, llm.ErrBadRequest):
			category = "request rejected"
		}
		t.Fatal(category)
	}
	if response == nil {
		t.Fatal("provider returned no response")
	}
	t.Log("named Anthropic OAuth request succeeded")
}
