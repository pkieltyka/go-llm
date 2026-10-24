package provideroauth

import (
	"context"
	"errors"
	"net/http"
	"reflect"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
)

func TestTokenErrorClassifiesStatus(t *testing.T) {
	for status, want := range map[int]error{
		http.StatusTooManyRequests:     llm.ErrRateLimited,
		http.StatusRequestTimeout:      llm.ErrTimeout,
		http.StatusBadRequest:          llm.ErrAuth,
		http.StatusUnauthorized:        llm.ErrAuth,
		http.StatusForbidden:           llm.ErrAuth,
		http.StatusInternalServerError: llm.ErrServer,
		http.StatusBadGateway:          llm.ErrServer,
	} {
		err := TokenError("p", "refresh", status)
		var providerErr *llm.ProviderError
		if !errors.Is(err, want) || !errors.As(err, &providerErr) || providerErr.HTTPStatus != status || providerErr.Message != "OAuth refresh failed" {
			t.Errorf("TokenError(%d) = %v, want %v", status, err, want)
		}
	}
}

func TestExpiresAtSubtractsPiMargin(t *testing.T) {
	before := time.Now()
	got := time.UnixMilli(ExpiresAt(3600))
	want := before.Add(time.Hour - ExpiryMargin)
	if got.Before(want.Add(-time.Second)) || got.After(want.Add(time.Second)) {
		t.Fatalf("ExpiresAt(3600) = %v, want about %v", got, want)
	}
}

// pi refreshes a stored OAuth credential without an expiry before use.
func TestResolverRefreshesCredentialWithoutExpiry(t *testing.T) {
	var calls int
	resolver := newTestResolver(t, llm.AuthCredential{Type: "oauth", Access: "old", Refresh: "r"},
		func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
			calls++
			return llm.AuthCredential{Access: "new", Expires: inAnHour()}, nil
		})
	cred, err := resolver.Credential(context.Background())
	if err != nil || cred.Access != "new" || calls != 1 {
		t.Fatalf("Credential = %+v, %v after %d refreshes; want one refresh", cred, err, calls)
	}
}

func TestRefreshKeepsFieldsTheResponseOmits(t *testing.T) {
	store := authstore.NewMemory(llm.AuthFile{testProvider: {
		Type: "oauth", Access: "old", Refresh: "r1", AccountID: "acct", ClientID: "client", Model: "m", BaseURL: "u",
		Env: map[string]string{"A": "b"},
	}})
	resolver := &Resolver{Store: store, ProviderID: testProvider, Refresh: func(context.Context, llm.AuthCredential) (llm.AuthCredential, error) {
		return llm.AuthCredential{Access: "new", Expires: inAnHour()}, nil
	}}
	if _, err := resolver.Credential(context.Background()); err != nil {
		t.Fatal(err)
	}
	got, _, err := store.Read(context.Background(), testProvider)
	if err != nil {
		t.Fatal(err)
	}
	want := llm.AuthCredential{Type: "oauth", Access: "new", Refresh: "r1", AccountID: "acct", ClientID: "client", Model: "m", BaseURL: "u", Env: map[string]string{"A": "b"}, Expires: got.Expires}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("stored = %+v, want %+v", got, want)
	}
}
