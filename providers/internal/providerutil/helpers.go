package providerutil

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"math/big"
	"strings"
	"sync"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

// SingleUse enforces the Provider contract's single-use stream semantics:
// the first range delegates to seq; every later range yields exactly one
// ErrBadRequest ("<name> stream already consumed") instead of silently
// producing an empty stream.
func SingleUse(name string, seq iter.Seq2[llm.Event, error]) iter.Seq2[llm.Event, error] {
	var mu sync.Mutex
	consumed := false
	return func(yield func(llm.Event, error) bool) {
		mu.Lock()
		if consumed {
			mu.Unlock()
			yield(nil, fmt.Errorf("%w: %s stream already consumed", llm.ErrBadRequest, name))
			return
		}
		consumed = true
		mu.Unlock()
		seq(yield)
	}
}

// ContextWithTimeout applies the provider-level call timeout. A nil ctx
// becomes context.Background(); a non-positive timeout leaves ctx unchanged.
func ContextWithTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if ctx == nil {
		ctx = context.Background()
	}
	if timeout <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, timeout)
}

// OptionsOf extracts typed request options after ValidateProviderOptions has
// established the provider identity. It accepts both value and pointer forms
// and deliberately does not call ForProvider again.
func OptionsOf[T any](req *llm.Request) (T, bool, error) {
	var zero T
	if req == nil || req.ProviderOptions == nil {
		return zero, false, nil
	}
	// Assert through any: *T is a pointer to a type parameter, which type
	// switches over the ProviderOptions interface cannot name directly.
	switch options := any(req.ProviderOptions).(type) {
	case T:
		return options, true, nil
	case *T:
		if options == nil {
			return zero, false, nil
		}
		return *options, true, nil
	}
	return zero, false, fmt.Errorf("%w: provider options are %T, want %T", llm.ErrBadRequest, req.ProviderOptions, zero)
}

// UniqueSyntheticToolCallID mints a deterministic synthetic tool-call ID
// ("call_<n>", seeded by block index) that does not collide with any ID in
// seenIDs. Callers record the returned ID in seenIDs themselves.
func UniqueSyntheticToolCallID(index int, seenIDs map[string]struct{}) string {
	for suffix := index; ; suffix++ {
		id := fmt.Sprintf("call_%d", suffix)
		if _, exists := seenIDs[id]; !exists {
			return id
		}
	}
}

// SchemaAsMap normalizes a caller-supplied JSON schema (json.RawMessage,
// []byte, or any marshalable value) into a map with json.Number numbers.
func SchemaAsMap(value any) (map[string]any, error) {
	if value == nil {
		return nil, fmt.Errorf("missing schema")
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var out map[string]any
	if err := dec.Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

// JSONEqual reports whether two valid JSON documents represent the same
// structured value. Object key order and insignificant number formatting are
// ignored; numbers are compared exactly, without conversion through float64.
func JSONEqual(left, right []byte) bool {
	if !json.Valid(left) || !json.Valid(right) {
		return false
	}
	if bytes.Equal(left, right) {
		return true
	}
	decode := func(raw []byte) any {
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.UseNumber()
		var value any
		if err := dec.Decode(&value); err != nil {
			return nil
		}
		return value
	}
	return jsonValueEqual(decode(left), decode(right))
}

func jsonValueEqual(left, right any) bool {
	switch left := left.(type) {
	case nil:
		return right == nil
	case bool:
		right, ok := right.(bool)
		return ok && left == right
	case string:
		right, ok := right.(string)
		return ok && left == right
	case json.Number:
		right, ok := right.(json.Number)
		if !ok {
			return false
		}
		leftCanonical, leftOK := canonicalJSONNumber(left.String())
		rightCanonical, rightOK := canonicalJSONNumber(right.String())
		return leftOK && rightOK && leftCanonical.equal(rightCanonical)
	case []any:
		right, ok := right.([]any)
		if !ok || len(left) != len(right) {
			return false
		}
		for index := range left {
			if !jsonValueEqual(left[index], right[index]) {
				return false
			}
		}
		return true
	case map[string]any:
		right, ok := right.(map[string]any)
		if !ok || len(left) != len(right) {
			return false
		}
		for key, leftValue := range left {
			rightValue, ok := right[key]
			if !ok || !jsonValueEqual(leftValue, rightValue) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

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
		mantissa = value[:index]
		exponentText = value[index+1:]
		if mantissa == "" || exponentText == "" {
			return normalizedJSONNumber{}, false
		}
	}
	if _, ok := normalized.exponent.SetString(exponentText, 10); !ok {
		return normalizedJSONNumber{}, false
	}

	integer, fraction := mantissa, ""
	if index := strings.IndexByte(mantissa, '.'); index >= 0 {
		integer, fraction = mantissa[:index], mantissa[index+1:]
		if integer == "" || fraction == "" || strings.IndexByte(fraction, '.') >= 0 {
			return normalizedJSONNumber{}, false
		}
	}
	digits := integer + fraction
	if digits == "" {
		return normalizedJSONNumber{}, false
	}
	for _, digit := range digits {
		if digit < '0' || digit > '9' {
			return normalizedJSONNumber{}, false
		}
	}
	digits = strings.TrimLeft(digits, "0")
	if digits == "" {
		normalized.negative = false
		normalized.coefficient = "0"
		normalized.exponent.SetInt64(0)
		return normalized, true
	}

	normalized.exponent.Sub(&normalized.exponent, big.NewInt(int64(len(fraction))))
	trimmed := strings.TrimRight(digits, "0")
	normalized.exponent.Add(&normalized.exponent, big.NewInt(int64(len(digits)-len(trimmed))))
	normalized.coefficient = trimmed
	return normalized, true
}

func (n normalizedJSONNumber) equal(other normalizedJSONNumber) bool {
	return n.negative == other.negative &&
		n.coefficient == other.coefficient &&
		n.exponent.Cmp(&other.exponent) == 0
}

// ToolResultText flattens text-only tool-result content into a single
// string. Non-text parts (other than UnknownPart, which is skipped) are
// ErrUnsupported: callers that can express richer tool-result content on
// their wire handle those parts before falling back to this helper.
func ToolResultText(part llm.ToolResultPart, providerName string) (string, error) {
	var b strings.Builder
	for _, nested := range part.Content {
		switch p := DerefPart(nested).(type) {
		case llm.TextPart:
			b.WriteString(p.Text)
		case llm.UnknownPart:
			continue
		default:
			return "", fmt.Errorf("%w: %s tool result cannot send part %T", llm.ErrUnsupported, providerName, nested)
		}
	}
	return b.String(), nil
}
