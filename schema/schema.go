// Package schema generates the strict JSON Schema subset used for tool inputs
// and structured output.
package schema

import (
	"encoding/json"
	"reflect"

	"github.com/pkieltyka/go-llm/internal/schemajson"
)

// JSONSchemaer lets a type provide its own schema instead of using reflection.
type JSONSchemaer = schemajson.JSONSchemaer

// Option configures schema generation.
type Option = schemajson.Option

// WithModifier applies fn to each generated field schema before it is attached
// to the parent object.
func WithModifier(fn func(field reflect.StructField, s map[string]any)) Option {
	return schemajson.WithModifier(fn)
}

// For generates a JSON Schema for T.
func For[T any](opts ...Option) (json.RawMessage, error) {
	return schemajson.For[T](opts...)
}

// MustFor is For but panics on generation errors.
func MustFor[T any](opts ...Option) json.RawMessage {
	return schemajson.MustFor[T](opts...)
}
