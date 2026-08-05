package llm

import (
	"strings"
	"unicode"
)

// MatchModel returns the best fuzzy match for query from models.
//
// Model IDs, canonical IDs, and display names are compared case-insensitively
// after dashes and dots are removed. A normalized exact match wins, except that
// a major-only family query such as "gpt-5" may resolve to a newer minor version.
// Prefix matches follow, then substring matches. Equal-rank fuzzy matches prefer
// the highest numeric model generation regardless of catalog order, then a
// "latest" alias, a shorter base model ID, and finally lexical ID order.
// Surrounding whitespace is ignored; queries without a letter or digit and
// unrelated queries do not match. MatchModel scans models once and never
// reorders or mutates the slice.
func MatchModel(models []ModelInfo, query string) (ModelInfo, bool) {
	query = strings.TrimSpace(query)
	normalizedQuery := normalizeModelSearch(query)
	if !containsModelSearchText(normalizedQuery) {
		return ModelInfo{}, false
	}
	// A query containing only a major generation (for example, "gpt-5")
	// names a model family even when an unsuffixed model with that exact ID is
	// present. Treat its exact match like a prefix match so a newer minor
	// generation can win the freshness tie-breaker. Explicit minor versions
	// such as "gpt-5.1" retain normal exact-match priority.
	familyQuery := len(modelVersionPartsFromIdentity(query)) == 1

	bestRank := 3
	bestIndex := -1
	for i, model := range models {
		rank, ok := modelMatchRank(normalizedQuery, model, familyQuery)
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

// modelVersionParts returns the major/minor generation encoded in a model
// identity. CanonicalID is authoritative for aggregator aliases. Provider
// prefixes, parameter counts, quantization levels, and dated snapshots do not
// participate in fuzzy freshness ordering.
func modelVersionParts(model ModelInfo) []string {
	for _, value := range []string{model.CanonicalID, model.ID, model.DisplayName} {
		if parts := modelVersionPartsFromIdentity(value); len(parts) > 0 {
			return parts
		}
	}
	return nil
}

func modelVersionPartsFromIdentity(value string) []string {
	value = strings.ToLower(value)
	if slash := strings.LastIndexByte(value, '/'); slash >= 0 {
		value = value[slash+1:]
	}

	var parts []string
	for i := 0; i < len(value) && len(parts) < 2; {
		if !isASCIIDigit(value[i]) {
			i++
			continue
		}
		start := i
		for i < len(value) && isASCIIDigit(value[i]) {
			i++
		}
		end := i

		if dateEnd := modelDateEnd(value, start, end, len(parts) > 0); dateEnd > 0 {
			i = dateEnd
			continue
		}
		if modelNumericRunIsMetadata(value, start, end, len(parts) > 0) {
			continue
		}
		parts = append(parts, value[start:end])
	}
	return parts
}

func modelNumericRunIsMetadata(value string, start, end int, haveVersion bool) bool {
	if end < len(value) {
		switch value[end] {
		case 'b', 'k', 'm', 't':
			return true
		}
	}

	wordStart := start
	for wordStart > 0 && value[wordStart-1] >= 'a' && value[wordStart-1] <= 'z' {
		wordStart--
	}
	switch value[wordStart:start] {
	case "bf", "fp", "int", "iq", "nf", "nvfp", "q", "uint":
		return true
	}

	run := value[start:end]
	return haveVersion && compactModelDate(run)
}

func modelDateEnd(value string, start, end int, haveVersion bool) int {
	run := value[start:end]
	if len(run) == 8 && plausibleModelDate(run[0:4], run[4:6], run[6:8]) {
		return end
	}
	if len(run) == 6 && haveVersion && plausibleModelDate("20"+run[0:2], run[2:4], run[4:6]) {
		return end
	}
	if len(run) != 4 || end+6 > len(value) || (value[end] != '-' && value[end] != '_') {
		return 0
	}
	separator := value[end]
	if value[end+3] != separator ||
		!isASCIIDigit(value[end+1]) || !isASCIIDigit(value[end+2]) ||
		!isASCIIDigit(value[end+4]) || !isASCIIDigit(value[end+5]) {
		return 0
	}
	if plausibleModelDate(run, value[end+1:end+3], value[end+4:end+6]) {
		return end + 6
	}
	return 0
}

func compactModelDate(value string) bool {
	if len(value) != 4 {
		return false
	}
	first := twoDigitNumber(value[0:2])
	second := twoDigitNumber(value[2:4])
	return (first >= 1 && first <= 12 && second >= 1 && second <= 31) ||
		(second >= 1 && second <= 12)
}

func plausibleModelDate(year, month, day string) bool {
	if len(year) != 4 || year < "1900" || year > "2199" {
		return false
	}
	monthNumber := twoDigitNumber(month)
	dayNumber := twoDigitNumber(day)
	return monthNumber >= 1 && monthNumber <= 12 && dayNumber >= 1 && dayNumber <= 31
}

func twoDigitNumber(value string) int {
	if len(value) != 2 || !isASCIIDigit(value[0]) || !isASCIIDigit(value[1]) {
		return -1
	}
	return int(value[0]-'0')*10 + int(value[1]-'0')
}

func isASCIIDigit(value byte) bool {
	return value >= '0' && value <= '9'
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

func modelMatchRank(normalizedQuery string, model ModelInfo, familyQuery bool) (int, bool) {
	bestRank := 3
	for _, value := range []string{model.ID, model.CanonicalID, model.DisplayName} {
		normalizedValue := normalizeModelSearch(value)
		var rank int
		switch {
		case normalizedValue == "":
			continue
		case normalizedValue == normalizedQuery:
			rank = 0
			if familyQuery {
				rank = 1
			}
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
