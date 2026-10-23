package providerutil

import (
	"context"
	"fmt"

	llm "github.com/pkieltyka/go-llm"
)

// StoredAPIKey returns a per-request key resolver for an API-key provider
// backed by a CredentialStore; see APIKeyFromCredential for the rules.
func StoredAPIKey(store llm.CredentialStore, providerID, ambient string) func(context.Context) (string, error) {
	return func(ctx context.Context) (string, error) {
		cred, ok, err := store.Read(ctx, providerID)
		if err != nil {
			return "", fmt.Errorf("%w: read %s credential: %w", llm.ErrAuth, providerID, err)
		}
		return APIKeyFromCredential(providerID, cred, ok, ambient)
	}
}

// APIKeyFromCredential applies pi's resolution rules to one store read: a
// stored "api_key" credential's key is used; one without a key resolves
// like nothing stored (pi keeps such entries for their other fields). Any
// other stored credential type fails, because a stored credential owns the
// provider and never silently falls back to the environment. With nothing
// stored (ok false), ambient (the environment value captured at
// construction) is used when non-empty.
func APIKeyFromCredential(providerID string, cred llm.AuthCredential, ok bool, ambient string) (string, error) {
	if ok {
		if cred.Type != "" && cred.Type != "api_key" {
			return "", fmt.Errorf("%w: stored %s credential has type %q; this provider accepts API keys only", llm.ErrAuth, providerID, cred.Type)
		}
		if cred.Key != "" {
			return cred.Key, nil
		}
	}
	if ambient != "" {
		return ambient, nil
	}
	return "", fmt.Errorf("%w: no %s API key is stored or configured", llm.ErrAuth, providerID)
}
