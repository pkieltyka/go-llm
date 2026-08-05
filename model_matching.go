package llm

import (
	"strings"
	"unicode"
)

// MatchModel returns the best fuzzy match for query from models.
//
// Model IDs, canonical IDs, and display names are compared case-insensitively
// after dashes and dots are removed. A normalized exact match wins, followed by
// prefix and substring matches. Ties prefer the later model, which lets sorted
// catalogs naturally select the newest matching version. Surrounding whitespace
// is ignored; queries without a letter or digit and unrelated queries do not
// match.
func MatchModel(models []ModelInfo, query string) (ModelInfo, bool) {
	normalizedQuery := normalizeModelSearch(strings.TrimSpace(query))
	if !containsModelSearchText(normalizedQuery) {
		return ModelInfo{}, false
	}

	bestRank := 3
	bestIndex := -1
	for i, model := range models {
		rank, ok := modelMatchRank(normalizedQuery, model)
		if ok && rank <= bestRank {
			bestRank = rank
			bestIndex = i
		}
	}
	if bestIndex < 0 {
		return ModelInfo{}, false
	}
	return models[bestIndex], true
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
