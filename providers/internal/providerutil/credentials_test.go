package providerutil

import (
	"context"
	"errors"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
)

func TestStoredAPIKey(t *testing.T) {
	ctx := context.Background()
	stored := StoredAPIKey(authstore.NewMemory(llm.AuthFile{"p": {Type: "api_key", Key: "stored"}}), "p", "ambient")
	if key, err := stored(ctx); err != nil || key != "stored" {
		t.Fatalf("stored key = %q, %v", key, err)
	}
	ambient := StoredAPIKey(authstore.NewMemory(nil), "p", "ambient")
	if key, err := ambient(ctx); err != nil || key != "ambient" {
		t.Fatalf("ambient fallback = %q, %v", key, err)
	}
	for name, store := range map[string]llm.CredentialStore{
		"oauth owns the provider": authstore.NewMemory(llm.AuthFile{"p": {Type: "oauth", Access: "a"}}),
		"empty stored key":        authstore.NewMemory(llm.AuthFile{"p": {Type: "api_key"}}),
	} {
		if _, err := StoredAPIKey(store, "p", "ambient")(ctx); !errors.Is(err, llm.ErrAuth) {
			t.Errorf("%s: err = %v, want ErrAuth without falling back", name, err)
		}
	}
	if _, err := StoredAPIKey(authstore.NewMemory(nil), "p", "")(ctx); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("nothing configured = %v, want ErrAuth", err)
	}
}
