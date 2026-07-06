// Package schemajson generates the strict JSON Schema subset used for tool
// inputs and structured output.
package schemajson

import (
	"bytes"
	"encoding"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"
)

const Draft2020Schema = "https://json-schema.org/draft/2020-12/schema"

var jsonSchemaerType = reflect.TypeOf((*JSONSchemaer)(nil)).Elem()
var jsonMarshalerType = reflect.TypeOf((*json.Marshaler)(nil)).Elem()
var textMarshalerType = reflect.TypeOf((*encoding.TextMarshaler)(nil)).Elem()
var rawMessageType = reflect.TypeOf(json.RawMessage{})

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
	return forType(typ, cfg)
}

// ForType generates a JSON Schema for typ. It exists for internal callers
// that already have a reflect.Type; public callers should normally use For.
func ForType(typ reflect.Type, opts ...Option) (json.RawMessage, error) {
	var cfg options
	for _, opt := range opts {
		opt(&cfg)
	}
	return forType(typ, cfg)
}

func forType(typ reflect.Type, cfg options) (json.RawMessage, error) {
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
	s["$schema"] = Draft2020Schema

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
	if typ == rawMessageType {
		return nil, fmt.Errorf("schema: json.RawMessage is unsupported; implement JSONSchemaer for an explicit schema")
	}
	if implementsJSONMarshaler(typ) {
		return nil, fmt.Errorf("schema: %s implements json.Marshaler; implement JSONSchemaer for an explicit schema", typ)
	}
	if implementsTextMarshaler(typ) {
		return nil, fmt.Errorf("schema: %s implements encoding.TextMarshaler; implement JSONSchemaer for an explicit schema", typ)
	}
	if isByteSlice(typ) {
		return map[string]any{"type": "string"}, nil
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
	requiredSet := make(map[string]struct{})

	if err := b.addStructFields(typ, true, properties, requiredSet); err != nil {
		return nil, err
	}

	required := make([]string, 0, len(requiredSet))
	for name := range requiredSet {
		required = append(required, name)
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

func (b *schemaBuilder) addStructFields(typ reflect.Type, requiredParent bool, properties map[string]any, requiredSet map[string]struct{}) error {
	for i := range typ.NumField() {
		field := typ.Field(i)
		if !fieldVisibleToJSON(field) {
			continue
		}

		name, omitEmpty, stringOpt, skip, tagged := jsonFieldName(field)
		if skip {
			continue
		}
		if stringOpt {
			return fmt.Errorf("schema: field %s uses json string option; implement JSONSchemaer for an explicit schema", field.Name)
		}
		if field.Anonymous && !tagged && implementsJSONSchemaer(indirectType(field.Type)) {
			return fmt.Errorf("schema: field %s embeds JSONSchemaer; implement JSONSchemaer for the parent or give the embedded field an explicit json name", field.Name)
		}

		if shouldFlattenAnonymousField(field, tagged) {
			tagOptions, err := parseJSONSchemaTag(field.Tag.Get("jsonschema"))
			if err != nil {
				return fmt.Errorf("schema: field %s: %w", field.Name, err)
			}
			if tagOptions.hasFieldAnnotations() {
				return fmt.Errorf("schema: field %s: jsonschema annotations are unsupported on flattened embedded fields", field.Name)
			}

			embeddedType := indirectType(field.Type)
			if b.stack[embeddedType] {
				return fmt.Errorf("schema: recursive type %s is unsupported", embeddedType)
			}
			b.stack[embeddedType] = true
			embeddedRequired := requiredParent && fieldRequired(field.Type, omitEmpty, tagOptions)
			err = b.addStructFields(embeddedType, embeddedRequired, properties, requiredSet)
			delete(b.stack, embeddedType)
			if err != nil {
				return err
			}
			continue
		}

		fieldSchema, err := b.schemaForType(field.Type)
		if err != nil {
			return fmt.Errorf("schema: field %s: %w", field.Name, err)
		}

		tagOptions, err := applyJSONSchemaTag(field.Tag.Get("jsonschema"), fieldSchema)
		if err != nil {
			return fmt.Errorf("schema: field %s: %w", field.Name, err)
		}
		if b.options.modifier != nil {
			b.options.modifier(field, fieldSchema)
		}

		if _, exists := properties[name]; exists {
			return fmt.Errorf("schema: duplicate JSON field %q", name)
		}
		properties[name] = fieldSchema
		if requiredParent && fieldRequired(field.Type, omitEmpty, tagOptions) {
			requiredSet[name] = struct{}{}
		}
	}
	return nil
}

func schemaFromSelfDescriber(typ reflect.Type) (json.RawMessage, bool, error) {
	if typ == nil {
		return nil, false, nil
	}
	if hasAnonymousJSONSchemaer(typ) {
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
	required    bool
	optional    bool
	description *string
	enum        []string
	format      *string
	enumSet     bool
}

func applyJSONSchemaTag(tag string, schema map[string]any) (fieldTagOptions, error) {
	opts, err := parseJSONSchemaTag(tag)
	if err != nil {
		return opts, err
	}
	if opts.description != nil {
		schema["description"] = *opts.description
	}
	if opts.enumSet {
		enum, err := enumValuesForSchema(schema, opts.enum)
		if err != nil {
			return opts, err
		}
		schema["enum"] = enum
	}
	if opts.format != nil {
		schema["format"] = *opts.format
	}
	return opts, nil
}

func parseJSONSchemaTag(tag string) (fieldTagOptions, error) {
	var opts fieldTagOptions
	if tag == "" {
		return opts, nil
	}

	for _, raw := range splitEscaped(tag, ',') {
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
			description := unescapeTagValue(value)
			opts.description = &description
		case "enum":
			values := splitEscaped(value, '|')
			opts.enum = opts.enum[:0]
			for _, v := range values {
				opts.enum = append(opts.enum, unescapeTagValue(v))
			}
			opts.enumSet = true
		case "format":
			format := unescapeTagValue(value)
			opts.format = &format
		default:
			return opts, fmt.Errorf("unsupported jsonschema tag %q", key)
		}
	}
	if opts.required && opts.optional {
		return opts, fmt.Errorf("field cannot be both required and optional")
	}
	return opts, nil
}

func (opts fieldTagOptions) hasFieldAnnotations() bool {
	return opts.description != nil || opts.enumSet || opts.format != nil
}

func enumValuesForSchema(schema map[string]any, values []string) ([]any, error) {
	typ, _ := schema["type"].(string)
	enum := make([]any, len(values))
	switch typ {
	case "string":
		for i, v := range values {
			enum[i] = v
		}
	case "integer":
		for i, v := range values {
			n, err := strconv.ParseInt(v, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("enum value %q is not an integer", v)
			}
			enum[i] = n
		}
	case "number":
		for i, v := range values {
			n, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return nil, fmt.Errorf("enum value %q is not a number", v)
			}
			enum[i] = n
		}
	case "boolean":
		for i, v := range values {
			b, err := strconv.ParseBool(v)
			if err != nil {
				return nil, fmt.Errorf("enum value %q is not a boolean", v)
			}
			enum[i] = b
		}
	default:
		return nil, fmt.Errorf("enum is unsupported for schema type %q", typ)
	}
	return enum, nil
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

func jsonFieldName(field reflect.StructField) (name string, omitEmpty bool, stringOpt bool, skip bool, tagged bool) {
	tag := field.Tag.Get("json")
	if tag == "-" {
		return "", false, false, true, false
	}
	if tag == "" {
		return field.Name, false, false, false, false
	}

	parts := strings.Split(tag, ",")
	name = parts[0]
	if name == "" {
		name = field.Name
	} else {
		tagged = true
	}
	for _, opt := range parts[1:] {
		switch opt {
		case "omitempty":
			omitEmpty = true
		case "string":
			stringOpt = true
		}
	}
	return name, omitEmpty, stringOpt, false, tagged
}

func fieldVisibleToJSON(field reflect.StructField) bool {
	if field.PkgPath == "" {
		return true
	}
	if !field.Anonymous {
		return false
	}
	return indirectType(field.Type).Kind() == reflect.Struct
}

func shouldFlattenAnonymousField(field reflect.StructField, tagged bool) bool {
	if !field.Anonymous || tagged {
		return false
	}
	typ := indirectType(field.Type)
	return typ.Kind() == reflect.Struct &&
		!implementsJSONSchemaer(typ) &&
		!implementsJSONMarshaler(typ) &&
		!implementsTextMarshaler(typ)
}

func indirectType(typ reflect.Type) reflect.Type {
	for typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	return typ
}

func implementsJSONMarshaler(typ reflect.Type) bool {
	return typ.Implements(jsonMarshalerType) || reflect.PointerTo(typ).Implements(jsonMarshalerType)
}

func implementsTextMarshaler(typ reflect.Type) bool {
	return typ.Implements(textMarshalerType) || reflect.PointerTo(typ).Implements(textMarshalerType)
}

func implementsJSONSchemaer(typ reflect.Type) bool {
	return typ.Implements(jsonSchemaerType) || reflect.PointerTo(typ).Implements(jsonSchemaerType)
}

func isByteSlice(typ reflect.Type) bool {
	if typ.Kind() != reflect.Slice || typ.Elem().Kind() != reflect.Uint8 {
		return false
	}
	return !implementsJSONMarshaler(typ.Elem()) && !implementsTextMarshaler(typ.Elem())
}

func hasAnonymousJSONSchemaer(typ reflect.Type) bool {
	for typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	if typ.Kind() != reflect.Struct {
		return false
	}
	for i := range typ.NumField() {
		field := typ.Field(i)
		if !field.Anonymous {
			continue
		}
		fieldType := indirectType(field.Type)
		if fieldType.Implements(jsonSchemaerType) || reflect.PointerTo(fieldType).Implements(jsonSchemaerType) {
			return true
		}
	}
	return false
}

func splitEscaped(s string, delim byte) []string {
	var parts []string
	var b strings.Builder
	escaped := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if escaped {
			b.WriteByte('\\')
			b.WriteByte(c)
			escaped = false
			continue
		}
		if c == '\\' {
			escaped = true
			continue
		}
		if c == delim {
			parts = append(parts, b.String())
			b.Reset()
			continue
		}
		b.WriteByte(c)
	}
	if escaped {
		b.WriteByte('\\')
	}
	parts = append(parts, b.String())
	return parts
}

func unescapeTagValue(s string) string {
	var b strings.Builder
	escaped := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if escaped {
			b.WriteByte(c)
			escaped = false
			continue
		}
		if c == '\\' {
			escaped = true
			continue
		}
		b.WriteByte(c)
	}
	if escaped {
		b.WriteByte('\\')
	}
	return b.String()
}
