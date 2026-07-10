package schemajson

import (
	"math/big"
	"strings"
)

// normalizedJSONNumber keeps a decimal coefficient and power of ten instead
// of expanding the exponent. That makes equality exact even for exponents far
// larger than available memory.
type normalizedJSONNumber struct {
	negative    bool
	coefficient string
	exponent    big.Int
}

func canonicalJSONNumber(value string) (normalizedJSONNumber, bool) {
	var normalized normalizedJSONNumber
	if value == "" {
		return normalized, false
	}
	if value[0] == '-' {
		normalized.negative = true
		value = value[1:]
		if value == "" {
			return normalizedJSONNumber{}, false
		}
	}

	mantissa := value
	exponentText := "0"
	if index := strings.IndexAny(value, "eE"); index >= 0 {
		if strings.IndexAny(value[index+1:], "eE") >= 0 {
			return normalizedJSONNumber{}, false
		}
		mantissa = value[:index]
		exponentText = value[index+1:]
		if mantissa == "" || !validExponent(exponentText) {
			return normalizedJSONNumber{}, false
		}
	}
	if _, ok := normalized.exponent.SetString(exponentText, 10); !ok {
		return normalizedJSONNumber{}, false
	}

	integer, fraction := mantissa, ""
	if index := strings.IndexByte(mantissa, '.'); index >= 0 {
		if strings.IndexByte(mantissa[index+1:], '.') >= 0 {
			return normalizedJSONNumber{}, false
		}
		integer, fraction = mantissa[:index], mantissa[index+1:]
		if fraction == "" {
			return normalizedJSONNumber{}, false
		}
	}
	if !validIntegerPart(integer) || !allDigits(fraction) {
		return normalizedJSONNumber{}, false
	}

	digits := strings.TrimLeft(integer+fraction, "0")
	if digits == "" {
		normalized.negative = false
		normalized.coefficient = "0"
		normalized.exponent.SetInt64(0)
		return normalized, true
	}

	var adjustment big.Int
	adjustment.SetInt64(int64(len(fraction)))
	normalized.exponent.Sub(&normalized.exponent, &adjustment)
	trimmed := strings.TrimRight(digits, "0")
	adjustment.SetInt64(int64(len(digits) - len(trimmed)))
	normalized.exponent.Add(&normalized.exponent, &adjustment)
	normalized.coefficient = trimmed
	return normalized, true
}

func validIntegerPart(value string) bool {
	if value == "0" {
		return true
	}
	return len(value) > 0 && value[0] >= '1' && value[0] <= '9' && allDigits(value[1:])
}

func validExponent(value string) bool {
	if value == "" {
		return false
	}
	if value[0] == '+' || value[0] == '-' {
		value = value[1:]
	}
	return value != "" && allDigits(value)
}

func allDigits(value string) bool {
	for i := range len(value) {
		if value[i] < '0' || value[i] > '9' {
			return false
		}
	}
	return true
}

func (n normalizedJSONNumber) equal(other normalizedJSONNumber) bool {
	return n.negative == other.negative &&
		n.coefficient == other.coefficient &&
		n.exponent.Cmp(&other.exponent) == 0
}

func (n normalizedJSONNumber) isInteger() bool {
	return n.coefficient == "0" || n.exponent.Sign() >= 0
}
