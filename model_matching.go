package llm

import (
	"strings"
	"unicode"
)

// MatchModel returns the best fuzzy match for query from models.
//
// Model IDs, canonical IDs, and display names are compared case-insensitively
// after dashes and dots are removed. A normalized exact match wins, followed by
// prefix and substring matches. Equal-rank fuzzy matches prefer the highest
// numeric model generation regardless of catalog order, then a "latest" alias,
// a shorter base model ID, and finally lexical ID order. Surrounding whitespace
// is ignored; queries without a letter or digit and unrelated queries do not
// match. MatchModel scans models once and never reorders or mutates the slice.
func MatchModel(models []ModelInfo, query string) (ModelInfo, bool) {
	normalizedQuery := normalizeModelSearch(strings.TrimSpace(query))
	if !containsModelSearchText(normalizedQuery) {
		return ModelInfo{}, false
	}

	bestRank := 3
	bestIndex := -1
	for i, model := range models {
		rank, ok := modelMatchRank(normalizedQuery, model)
		if ok && (rank < bestRank || (rank == bestRank && (bestIndex < 0 || newerModelMatch(model, models[bestIndex])))) {
			bestRank = rank
			bestIndex = i
		}
	}
	if bestIndex < 0 {
		return ModelInfo{}, false
	}
	return models[bestIndex], true
}

func newerModelMatch(candidate, current ModelInfo) bool {
	candidateVersion := modelVersionParts(candidate)
	currentVersion := modelVersionParts(current)
	if len(candidateVersion) > 0 && len(currentVersion) > 0 {
		if comparison := compareModelVersions(candidateVersion, currentVersion); comparison != 0 {
			return comparison > 0
		}
	}

	candidateLatest := modelHasAlias(candidate, "latest")
	currentLatest := modelHasAlias(current, "latest")
	if candidateLatest != currentLatest {
		return candidateLatest
	}
	if len(candidateVersion) != len(currentVersion) {
		return len(candidateVersion) > len(currentVersion)
	}

	candidateID := modelSortIdentity(candidate)
	currentID := modelSortIdentity(current)
	if len(candidateID) != len(currentID) {
		return len(candidateID) < len(currentID)
	}
	return candidateID > currentID
}

// modelVersionParts returns the major/minor generation encoded in the first
// two numeric runs of a model identity. Later numeric runs commonly describe
// parameter counts, quantization, or dated snapshots rather than a newer model
// family, so they do not participate in fuzzy freshness ordering.
func modelVersionParts(model ModelInfo) []string {
	for _, value := range []string{model.ID, model.CanonicalID, model.DisplayName} {
		var parts []string
		for start, i := -1, 0; i <= len(value); i++ {
			isDigit := i < len(value) && value[i] >= '0' && value[i] <= '9'
			switch {
			case isDigit && start < 0:
				start = i
			case !isDigit && start >= 0:
				parts = append(parts, value[start:i])
				start = -1
				if len(parts) == 2 {
					return parts
				}
			}
		}
		if len(parts) > 0 {
			return parts
		}
	}
	return nil
}

func compareModelVersions(left, right []string) int {
	common := min(len(left), len(right))
	for i := range common {
		if comparison := compareModelVersionPart(left[i], right[i]); comparison != 0 {
			return comparison
		}
	}
	switch {
	case len(left) < len(right):
		return -1
	case len(left) > len(right):
		return 1
	default:
		return 0
	}
}

func compareModelVersionPart(left, right string) int {
	left = strings.TrimLeft(left, "0")
	right = strings.TrimLeft(right, "0")
	if left == "" {
		left = "0"
	}
	if right == "" {
		right = "0"
	}
	if len(left) < len(right) {
		return -1
	}
	if len(left) > len(right) {
		return 1
	}
	return strings.Compare(left, right)
}

func modelHasAlias(model ModelInfo, alias string) bool {
	for _, value := range []string{model.ID, model.CanonicalID, model.DisplayName} {
		for _, token := range strings.FieldsFunc(strings.ToLower(value), func(r rune) bool {
			return !(unicode.IsLetter(r) || unicode.IsDigit(r))
		}) {
			if token == alias {
				return true
			}
		}
	}
	return false
}

func modelSortIdentity(model ModelInfo) string {
	for _, value := range []string{model.ID, model.CanonicalID, model.DisplayName} {
		if value = strings.ToLower(strings.TrimSpace(value)); value != "" {
			return value
		}
	}
	return ""
}

func containsModelSearchText(value string) bool {
	for _, r := range value {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return true
		}
	}
	return false
}

func modelMatchRank(normalizedQuery string, model ModelInfo) (int, bool) {
	bestRank := 3
	for _, value := range []string{model.ID, model.CanonicalID, model.DisplayName} {
		normalizedValue := normalizeModelSearch(value)
		var rank int
		switch {
		case normalizedValue == "":
			continue
		case normalizedValue == normalizedQuery:
			rank = 0
		case strings.HasPrefix(normalizedValue, normalizedQuery):
			rank = 1
		case strings.Contains(normalizedValue, normalizedQuery):
			rank = 2
		default:
			continue
		}
		if rank < bestRank {
			bestRank = rank
		}
	}
	return bestRank, bestRank < 3
}

func normalizeModelSearch(value string) string {
	value = strings.ToLower(value)
	value = strings.ReplaceAll(value, "-", "")
	return strings.ReplaceAll(value, ".", "")
}
