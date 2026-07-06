package vllm

import (
	"context"
	"fmt"
	"strings"
	"unicode"

	llm "github.com/pkieltyka/go-llm"
)

const defaultModelPreference = "qwen"

// ResolveModel lists the served vLLM models and returns the best match for
// preference. Matching is intentionally soft: exact match wins, then
// case-insensitive substring/normalized-substring match, then a token overlap
// score. If preference is empty or has no useful match, ResolveModel prefers
// the first served Qwen model and finally the first served model.
//
// This is useful for self-hosted servers whose model IDs include deployment
// prefixes or quantization suffixes, e.g. preference "qwen-3.6-27b" matching
// "nvidia/Qwen-3.6-27B-NVFP4".
func (p *Provider) ResolveModel(ctx context.Context, preference string) (llm.ModelInfo, error) {
	models, err := p.Models(ctx)
	if err != nil {
		return llm.ModelInfo{}, err
	}
	if len(models) == 0 {
		return llm.ModelInfo{}, fmt.Errorf("%w: vllm models list is empty", llm.ErrNotFound)
	}
	return selectModel(models, preference), nil
}

func selectModel(models []llm.ModelInfo, preference string) llm.ModelInfo {
	preference = strings.TrimSpace(preference)
	if preference != "" {
		if model, ok := exactModel(models, preference); ok {
			return model
		}
		if model, ok := substringModel(models, preference); ok {
			return model
		}
		if model, ok := scoredModel(models, preference); ok {
			return model
		}
	}
	if model, ok := substringModel(models, defaultModelPreference); ok {
		return model
	}
	return models[0]
}

func exactModel(models []llm.ModelInfo, preference string) (llm.ModelInfo, bool) {
	for _, model := range models {
		if strings.EqualFold(model.ID, preference) {
			return model, true
		}
	}
	return llm.ModelInfo{}, false
}

func substringModel(models []llm.ModelInfo, preference string) (llm.ModelInfo, bool) {
	needle := strings.ToLower(preference)
	normalizedNeedle := normalizeModelID(preference)
	for _, model := range models {
		id := strings.ToLower(model.ID)
		if strings.Contains(id, needle) {
			return model, true
		}
		if normalizedNeedle != "" && strings.Contains(normalizeModelID(model.ID), normalizedNeedle) {
			return model, true
		}
	}
	return llm.ModelInfo{}, false
}

func scoredModel(models []llm.ModelInfo, preference string) (llm.ModelInfo, bool) {
	needles := modelTokenSet(preference)
	var best llm.ModelInfo
	bestScore := 0
	for _, model := range models {
		score := 0
		haystack := modelTokenSet(model.ID)
		for token := range needles {
			if haystack[token] {
				score += len(token)
			}
		}
		if score > bestScore {
			best = model
			bestScore = score
		}
	}
	return best, bestScore > 0
}

func normalizeModelID(value string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(value) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func modelTokenSet(value string) map[string]bool {
	out := map[string]bool{}
	segments := strings.FieldsFunc(strings.ToLower(value), func(r rune) bool {
		return !(unicode.IsLetter(r) || unicode.IsDigit(r))
	})
	for _, segment := range segments {
		addModelToken(out, segment)
		for _, part := range splitAlphaNumeric(segment) {
			addModelToken(out, part)
		}
	}
	return out
}

func addModelToken(tokens map[string]bool, token string) {
	if len(token) >= 2 {
		tokens[token] = true
	}
}

func splitAlphaNumeric(value string) []string {
	if value == "" {
		return nil
	}
	var parts []string
	start := 0
	lastClass := runeClass(rune(value[0]))
	for i, r := range value {
		if i == 0 {
			continue
		}
		class := runeClass(r)
		if class != lastClass {
			parts = append(parts, value[start:i])
			start = i
			lastClass = class
		}
	}
	parts = append(parts, value[start:])
	return parts
}

func runeClass(r rune) int {
	switch {
	case unicode.IsDigit(r):
		return 1
	case unicode.IsLetter(r):
		return 2
	default:
		return 0
	}
}
