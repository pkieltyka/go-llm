package openaicodex

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/provideroauth"
)

func TestOAuthRefreshRefusesRedirects(t *testing.T) {
	var trapHits atomic.Int32
	trap := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		trapHits.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer trap.Close()

	for _, status := range []int{http.StatusTemporaryRedirect, http.StatusPermanentRedirect} {
		t.Run(fmt.Sprintf("status_%d", status), func(t *testing.T) {
			redirector := httptest.NewServer(http.RedirectHandler(trap.URL, status))
			defer redirector.Close()

			client := &http.Client{}
			_, err := refreshCodexOAuth(context.Background(), client, redirector.URL,
				llm.AuthCredential{Type: "oauth", Refresh: "refresh-token"})
			if !errors.Is(err, provideroauth.ErrUnsafeRedirect) {
				t.Fatalf("refresh error = %v, want ErrUnsafeRedirect", err)
			}
			if client.CheckRedirect != nil {
				t.Fatal("caller client CheckRedirect was mutated")
			}
		})
	}
	if got := trapHits.Load(); got != 0 {
		t.Fatalf("redirect target hits = %d, want 0", got)
	}
}
