// Package schema generates the strict JSON Schema subset used for tool inputs
// and structured output.
package schema

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
)

const draft2020Schema = "https://json-schema.org/draft/2020-12/schema"

var jsonSchemaerType = reflect.TypeOf((*JSONSchemaer)(nil)).Elem()

// JSONSchemaer lets a type provide its own schema instead of using reflection.
type JSONSchemaer interface {
	JSONSchema() json.RawMessage
}

// Option configures schema generation.
type Option func(*options)

type options struct {
	modifier func(field reflect.StructField, s map[string]any)
}

// WithModifier applies fn to each generated field schema before it is attached
// to the parent object.
func WithModifier(fn func(field reflect.StructField, s map[string]any)) Option {
	return func(opts *options) {
		opts.modifier = fn
	}
}

// For generates a JSON Schema for T.
func For[T any](opts ...Option) (json.RawMessage, error) {
	var cfg options
	for _, opt := range opts {
		opt(&cfg)
	}

	typ := reflect.TypeOf((*T)(nil)).Elem()
	if raw, ok, err := schemaFromSelfDescriber(typ); ok || err != nil {
		return raw, err
	}

	builder := schemaBuilder{
		options: cfg,
		stack:   make(map[reflect.Type]bool),
	}
	s, err := builder.schemaForType(typ)
	if err != nil {
		return nil, err
	}
	s["$schema"] = draft2020Schema

	raw, err := json.Marshal(s)
	if err != nil {
		return nil, err
	}
	return raw, nil
}

// MustFor is For but panics on generation errors.
func MustFor[T any](opts ...Option) json.RawMessage {
	raw, err := For[T](opts...)
	if err != nil {
		panic(err)
	}
	return raw
}

type schemaBuilder struct {
	options options
	stack   map[reflect.Type]bool
}

func (b *schemaBuilder) schemaForType(typ reflect.Type) (map[string]any, error) {
	for typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}

	if raw, ok, err := schemaFromSelfDescriber(typ); ok || err != nil {
		if err != nil {
			return nil, err
		}
		return schemaObjectFromRaw(raw)
	}

	switch typ.Kind() {
	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil
	case reflect.String:
		return map[string]any{"type": "string"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}, nil
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil
	case reflect.Struct:
		return b.schemaForStruct(typ)
	case reflect.Slice, reflect.Array:
		itemSchema, err := b.schemaForType(typ.Elem())
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "array", "items": itemSchema}, nil
	case reflect.Map:
		if typ.Key().Kind() != reflect.String {
			return nil, fmt.Errorf("schema: unsupported map key type %s", typ.Key())
		}
		valueSchema, err := b.schemaForType(typ.Elem())
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "object", "additionalProperties": valueSchema}, nil
	default:
		return nil, fmt.Errorf("schema: unsupported Go type %s", typ)
	}
}

func (b *schemaBuilder) schemaForStruct(typ reflect.Type) (map[string]any, error) {
	if b.stack[typ] {
		return nil, fmt.Errorf("schema: recursive type %s is unsupported", typ)
	}
	b.stack[typ] = true
	defer delete(b.stack, typ)

	properties := make(map[string]any)
	var required []string

	for i := range typ.NumField() {
		field := typ.Field(i)
		if field.PkgPath != "" {
			continue
		}

		name, omitEmpty, skip := jsonFieldName(field)
		if skip {
			continue
		}

		fieldSchema, err := b.schemaForType(field.Type)
		if err != nil {
			return nil, fmt.Errorf("schema: field %s: %w", field.Name, err)
		}

		tagOptions, err := applyJSONSchemaTag(field.Tag.Get("jsonschema"), fieldSchema)
		if err != nil {
			return nil, fmt.Errorf("schema: field %s: %w", field.Name, err)
		}
		if b.options.modifier != nil {
			b.options.modifier(field, fieldSchema)
		}

		properties[name] = fieldSchema
		if fieldRequired(field.Type, omitEmpty, tagOptions) {
			required = append(required, name)
		}
	}

	sort.Strings(required)
	out := map[string]any{
		"type":                 "object",
		"properties":           properties,
		"additionalProperties": false,
	}
	if len(required) > 0 {
		out["required"] = required
	}
	return out, nil
}

func schemaFromSelfDescriber(typ reflect.Type) (json.RawMessage, bool, error) {
	if typ == nil {
		return nil, false, nil
	}

	if typ.Kind() == reflect.Pointer && typ.Implements(jsonSchemaerType) {
		raw := reflect.New(typ.Elem()).Interface().(JSONSchemaer).JSONSchema()
		return compactRaw(raw)
	}
	if typ.Implements(jsonSchemaerType) {
		raw := reflect.Zero(typ).Interface().(JSONSchemaer).JSONSchema()
		return compactRaw(raw)
	}
	if typ.Kind() != reflect.Pointer && reflect.PointerTo(typ).Implements(jsonSchemaerType) {
		raw := reflect.New(typ).Interface().(JSONSchemaer).JSONSchema()
		return compactRaw(raw)
	}
	return nil, false, nil
}

func schemaObjectFromRaw(raw json.RawMessage) (map[string]any, error) {
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	if out == nil {
		return nil, fmt.Errorf("schema: self-described schema must be an object")
	}
	return out, nil
}

func compactRaw(raw json.RawMessage) (json.RawMessage, bool, error) {
	if len(raw) == 0 {
		return nil, true, fmt.Errorf("schema: empty self-described schema")
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return nil, true, err
	}
	return append(json.RawMessage(nil), buf.Bytes()...), true, nil
}

type fieldTagOptions struct {
	required bool
	optional bool
}

func applyJSONSchemaTag(tag string, schema map[string]any) (fieldTagOptions, error) {
	var opts fieldTagOptions
	if tag == "" {
		return opts, nil
	}

	for _, raw := range strings.Split(tag, ",") {
		item := strings.TrimSpace(raw)
		if item == "" {
			continue
		}

		key, value, hasValue := strings.Cut(item, "=")
		key = strings.TrimSpace(key)
		if !hasValue {
			switch key {
			case "required":
				opts.required = true
			case "optional":
				opts.optional = true
			default:
				return opts, fmt.Errorf("unsupported jsonschema tag %q", key)
			}
			continue
		}

		switch key {
		case "description":
			schema["description"] = value
		case "enum":
			values := strings.Split(value, "|")
			enum := make([]any, len(values))
			for i, v := range values {
				enum[i] = v
			}
			schema["enum"] = enum
		case "format":
			schema["format"] = value
		default:
			return opts, fmt.Errorf("unsupported jsonschema tag %q", key)
		}
	}
	if opts.required && opts.optional {
		return opts, fmt.Errorf("field cannot be both required and optional")
	}
	return opts, nil
}

func fieldRequired(typ reflect.Type, omitEmpty bool, tag fieldTagOptions) bool {
	if tag.required {
		return true
	}
	if tag.optional {
		return false
	}
	return typ.Kind() != reflect.Pointer && !omitEmpty
}

func jsonFieldName(field reflect.StructField) (name string, omitEmpty bool, skip bool) {
	tag := field.Tag.Get("json")
	if tag == "-" {
		return "", false, true
	}
	if tag == "" {
		return field.Name, false, false
	}

	parts := strings.Split(tag, ",")
	if parts[0] == "-" {
		return "", false, true
	}
	name = parts[0]
	if name == "" {
		name = field.Name
	}
	for _, opt := range parts[1:] {
		if opt == "omitempty" {
			omitEmpty = true
			break
		}
	}
	return name, omitEmpty, false
}
