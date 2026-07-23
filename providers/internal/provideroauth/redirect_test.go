package provideroauth

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestNoRedirectClientCopiesWithoutMutating(t *testing.T) {
	orig := &http.Client{Timeout: 1234 * time.Millisecond}
	c := NoRedirectClient(orig)
	if c == orig {
		t.Fatal("NoRedirectClient returned the original client")
	}
	if c.Timeout != orig.Timeout {
		t.Fatalf("copy Timeout = %v, want %v", c.Timeout, orig.Timeout)
	}
	if orig.CheckRedirect != nil {
		t.Fatal("original client CheckRedirect was mutated")
	}
	if c.CheckRedirect == nil {
		t.Fatal("copy CheckRedirect not set")
	}
	if err := c.CheckRedirect(nil, nil); !errors.Is(err, ErrUnsafeRedirect) {
		t.Fatalf("CheckRedirect error = %v, want ErrUnsafeRedirect", err)
	}
	if nilBased := NoRedirectClient(nil); nilBased == nil || nilBased.CheckRedirect == nil {
		t.Fatal("NoRedirectClient(nil) did not return a guarded client")
	}
}

func TestNoRedirectClientRefusesRedirects(t *testing.T) {
	var trapHits atomic.Int32
	trap := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		trapHits.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer trap.Close()

	for _, status := range []int{http.StatusTemporaryRedirect, http.StatusPermanentRedirect} {
		redirector := httptest.NewServer(http.RedirectHandler(trap.URL, status))
		resp, err := NoRedirectClient(redirector.Client()).Post(redirector.URL, "application/json", strings.NewReader("credential"))
		if resp != nil {
			resp.Body.Close()
		}
		if !errors.Is(err, ErrUnsafeRedirect) {
			t.Fatalf("status %d: err = %v, want ErrUnsafeRedirect", status, err)
		}
		redirector.Close()
	}
	if got := trapHits.Load(); got != 0 {
		t.Fatalf("redirect target hits = %d, want 0", got)
	}
}
