package e2e

import (
	"context"
	"fmt"
	"strings"

	llm "github.com/pkieltyka/go-llm"
)

type modelPreferenceResolver interface {
	ResolveModel(context.Context, string) (llm.ModelInfo, error)
}

// ResolveConfiguredModel uses a provider-specific resolver when available.
// vLLM implements this interface and performs its fuzzy Qwen-aware selection.
// Other providers retain the request alias; their Models scenario validates
// that alias against IDs and canonical IDs returned by the listing endpoint.
func ResolveConfiguredModel(ctx context.Context, provider llm.Provider, preference string) (string, error) {
	if resolver, ok := provider.(modelPreferenceResolver); ok {
		model, err := resolver.ResolveModel(ctx, preference)
		if err != nil {
			return "", err
		}
		if strings.TrimSpace(model.ID) == "" {
			return "", fmt.Errorf("%w: provider %s resolved an empty model ID", llm.ErrNotFound, provider.Name())
		}
		return model.ID, nil
	}
	return preference, nil
}

func resolveListedModel(providerID, configured string, models []llm.ModelInfo) (llm.ModelInfo, bool) {
	configured = strings.TrimSpace(configured)
	if configured == "" {
		return llm.ModelInfo{}, false
	}
	configuredIDs := []modelIdentity{parseModelIdentity(configured)}
	if embedded, ok := llm.LookupModelInfo(providerID, configured); ok {
		configuredIDs = appendModelIdentities(configuredIDs, embedded.ID, embedded.CanonicalID)
	}
	for _, model := range models {
		listedIDs := appendModelIdentities(nil, model.ID, model.CanonicalID)
		if modelIdentityListsMatch(configuredIDs, listedIDs) {
			return model, true
		}
	}
	return llm.ModelInfo{}, false
}

type modelIdentity struct {
	provider  string
	model     string
	qualified bool
}

func parseModelIdentity(value string) modelIdentity {
	value = strings.ToLower(strings.TrimSpace(value))
	provider, model, qualified := strings.Cut(value, "/")
	if !qualified {
		return modelIdentity{model: value}
	}
	return modelIdentity{provider: provider, model: model, qualified: true}
}

func appendModelIdentities(dst []modelIdentity, values ...string) []modelIdentity {
	for _, value := range values {
		identity := parseModelIdentity(value)
		if identity.model != "" {
			dst = append(dst, identity)
		}
	}
	return dst
}

func modelIdentityListsMatch(left, right []modelIdentity) bool {
	for _, leftIdentity := range left {
		for _, rightIdentity := range right {
			if modelIdentitiesMatch(leftIdentity, rightIdentity) {
				return true
			}
		}
	}
	return false
}

func modelIdentitiesMatch(left, right modelIdentity) bool {
	if left.qualified && right.qualified && left.provider != right.provider {
		return false
	}
	return left.model == right.model || datedModelAlias(left.model, right.model) || datedModelAlias(right.model, left.model)
}

func datedModelAlias(value, prefix string) bool {
	if prefix == "" || len(value) <= len(prefix)+1 || !strings.HasPrefix(value, prefix) || value[len(prefix)] != '-' {
		return false
	}
	suffix := value[len(prefix)+1:]
	digits := 0
	for digits < len(suffix) && suffix[digits] >= '0' && suffix[digits] <= '9' {
		digits++
	}
	return digits >= 6
}
