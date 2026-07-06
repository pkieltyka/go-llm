package providerutil

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
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

// OptionsOf extracts a provider's typed request options from
// req.ProviderOptions, accepting both the value and pointer form. Options
// belonging to a different provider are ignored (ok=false, nil error) so a
// request built for one provider can be routed to another; a value that
// claims this provider's name but has an unexpected concrete type is an
// ErrBadRequest.
func OptionsOf[T llm.ProviderOptions](req *llm.Request) (T, bool, error) {
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
	if req.ProviderOptions.ForProvider() == zero.ForProvider() {
		return zero, false, fmt.Errorf("%w: provider options for %q are %T, want %T", llm.ErrBadRequest, zero.ForProvider(), req.ProviderOptions, zero)
	}
	return zero, false, nil
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
