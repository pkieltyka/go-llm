package schemajson

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// ErrBadRequest marks invalid schemas or arguments. The public schema package
// maps this sentinel onto llm.ErrBadRequest.
var ErrBadRequest = errors.New("schemajson: bad request")

// BadRequestDetail returns err without the internal ErrBadRequest prefix,
// so public packages can wrap their own sentinel without duplicating labels.
func BadRequestDetail(err error) string {
	if err == nil {
		return ""
	}
	message := err.Error()
	prefix := ErrBadRequest.Error()
	if message == prefix {
		return prefix
	}
	return strings.TrimPrefix(message, prefix+": ")
}

// ValidateArgs checks model-emitted tool arguments against the supported
// strict-mode JSON Schema subset: type, required, properties,
// additionalProperties, items, and enum. Annotation keywords such as
// description and format are accepted in schemas but not enforced here.
//
// Policy choices: additionalProperties:false is enforced because model-emitted
// arguments are untrusted, while annotation keywords such as format and
// description are deliberately accepted but not enforced.
func ValidateArgs(toolName string, inputSchema any, args json.RawMessage) error {
	if inputSchema == nil {
		return fmt.Errorf("%w: tool %q has no input schema", ErrBadRequest, toolName)
	}

	s, err := schemaFromAny(inputSchema)
	if err != nil {
		return err
	}
	if err := validateSchemaShape("$", s); err != nil {
		return err
	}

	var value any
	if err := decodeJSON(args, &value); err != nil {
		return fmt.Errorf("%w: invalid tool args: %v", ErrBadRequest, err)
	}
	return validateValue("$", s, value)
}

// ValidateSchema checks that inputSchema is inside the supported strict-mode
// JSON Schema subset (the SHAPE only), without validating any arguments against
// it. It fails closed on schemas outside the focused subset: a root missing
// "type", a union/nullable type, or an array without "items". Schemas produced
// by For are conformant by construction and always pass. Callers use this to
// reject an unusable caller-supplied schema before spending provider calls.
func ValidateSchema(inputSchema any) error {
	if inputSchema == nil {
		return fmt.Errorf("%w: schema is nil", ErrBadRequest)
	}
	s, err := schemaFromAny(inputSchema)
	if err != nil {
		return err
	}
	return validateSchemaShape("$", s)
}

func schemaFromAny(input any) (map[string]any, error) {
	var raw []byte
	switch s := input.(type) {
	case json.RawMessage:
		raw = append([]byte(nil), s...)
	case []byte:
		raw = append([]byte(nil), s...)
	default:
		var err error
		raw, err = json.Marshal(input)
		if err != nil {
			return nil, fmt.Errorf("%w: invalid tool schema: %v", ErrBadRequest, err)
		}
	}

	var out map[string]any
	if err := decodeJSON(raw, &out); err != nil {
		return nil, fmt.Errorf("%w: invalid tool schema: %v", ErrBadRequest, err)
	}
	if out == nil {
		return nil, fmt.Errorf("%w: tool schema must be an object", ErrBadRequest)
	}
	return out, nil
}

func decodeJSON(data []byte, out any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	if err := dec.Decode(out); err != nil {
		return err
	}
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		return fmt.Errorf("unexpected trailing JSON")
	}
	return nil
}

func validateSchemaShape(path string, s map[string]any) error {
	rawType, ok := s["type"]
	if !ok {
		return fmt.Errorf("%w: %s schema is missing type", ErrBadRequest, path)
	}
	typ, ok := rawType.(string)
	if !ok {
		return fmt.Errorf("%w: %s schema type must be a string", ErrBadRequest, path)
	}
	switch typ {
	case "object", "array", "string", "boolean", "integer", "number":
	default:
		return fmt.Errorf("%w: unsupported schema type %q at %s", ErrBadRequest, typ, path)
	}

	if raw, exists := s["enum"]; exists {
		values, ok := raw.([]any)
		if !ok || len(values) == 0 {
			return fmt.Errorf("%w: %s enum must be a non-empty array", ErrBadRequest, path)
		}
	}

	if _, err := requiredFields(path, s); err != nil {
		return err
	}

	props, err := schemaProperties(path, s)
	if err != nil {
		return err
	}
	for name, property := range props {
		if err := validateSchemaShape(path+".properties."+name, property); err != nil {
			return err
		}
	}

	if raw, exists := s["items"]; exists {
		items, ok := raw.(map[string]any)
		if !ok {
			return fmt.Errorf("%w: %s items must be an object", ErrBadRequest, path)
		}
		if err := validateSchemaShape(path+".items", items); err != nil {
			return err
		}
	} else if typ == "array" {
		return fmt.Errorf("%w: %s array schema is missing items", ErrBadRequest, path)
	}

	if raw, exists := s["additionalProperties"]; exists {
		switch additional := raw.(type) {
		case bool:
		case map[string]any:
			if err := validateSchemaShape(path+".additionalProperties", additional); err != nil {
				return err
			}
		default:
			return fmt.Errorf("%w: %s additionalProperties must be a boolean or object", ErrBadRequest, path)
		}
	}
	return nil
}

func validateValue(path string, s map[string]any, value any) error {
	if err := validateType(path, s, value); err != nil {
		return err
	}
	if err := validateEnum(path, s, value); err != nil {
		return err
	}
	return nil
}

func validateType(path string, s map[string]any, value any) error {
	typ, _ := s["type"].(string)
	switch typ {
	case "object":
		obj, ok := value.(map[string]any)
		if !ok {
			return fmt.Errorf("%w: %s must be an object", ErrBadRequest, path)
		}
		return validateObject(path, s, obj)
	case "array":
		values, ok := value.([]any)
		if !ok {
			return fmt.Errorf("%w: %s must be an array", ErrBadRequest, path)
		}
		return validateArray(path, s, values)
	case "string":
		if _, ok := value.(string); !ok {
			return fmt.Errorf("%w: %s must be a string", ErrBadRequest, path)
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("%w: %s must be a boolean", ErrBadRequest, path)
		}
	case "integer":
		number, ok := value.(json.Number)
		if !ok {
			return fmt.Errorf("%w: %s must be an integer", ErrBadRequest, path)
		}
		normalized, ok := canonicalJSONNumber(number.String())
		if !ok || !normalized.isInteger() {
			return fmt.Errorf("%w: %s must be an integer", ErrBadRequest, path)
		}
	case "number":
		number, ok := value.(json.Number)
		if !ok {
			return fmt.Errorf("%w: %s must be a number", ErrBadRequest, path)
		}
		if _, ok := canonicalJSONNumber(number.String()); !ok {
			return fmt.Errorf("%w: %s must be a number", ErrBadRequest, path)
		}
	case "":
		return fmt.Errorf("%w: %s schema is missing type", ErrBadRequest, path)
	default:
		return fmt.Errorf("%w: unsupported schema type %q at %s", ErrBadRequest, typ, path)
	}
	return nil
}

func validateObject(path string, s map[string]any, value map[string]any) error {
	props, err := schemaProperties(path, s)
	if err != nil {
		return err
	}

	required, err := requiredFields(path, s)
	if err != nil {
		return err
	}
	for _, name := range required {
		if _, ok := value[name]; !ok {
			return fmt.Errorf("%w: %s.%s is required", ErrBadRequest, path, name)
		}
	}

	additional := s["additionalProperties"]
	for name, fieldValue := range value {
		fieldSchema, known := props[name]
		if known {
			if err := validateValue(path+"."+name, fieldSchema, fieldValue); err != nil {
				return err
			}
			continue
		}

		switch extra := additional.(type) {
		case bool:
			if !extra {
				return fmt.Errorf("%w: %s.%s is not allowed", ErrBadRequest, path, name)
			}
		case map[string]any:
			if err := validateValue(path+"."+name, extra, fieldValue); err != nil {
				return err
			}
		case nil:
			continue
		default:
			return fmt.Errorf("%w: %s additionalProperties is unsupported", ErrBadRequest, path)
		}
	}
	return nil
}

func validateArray(path string, s map[string]any, values []any) error {
	itemSchema, ok := s["items"].(map[string]any)
	if !ok {
		return fmt.Errorf("%w: %s array schema is missing items", ErrBadRequest, path)
	}
	for i, value := range values {
		if err := validateValue(fmt.Sprintf("%s[%d]", path, i), itemSchema, value); err != nil {
			return err
		}
	}
	return nil
}

func validateEnum(path string, s map[string]any, value any) error {
	raw, ok := s["enum"]
	if !ok {
		return nil
	}
	values, ok := raw.([]any)
	if !ok {
		return fmt.Errorf("%w: %s enum must be an array", ErrBadRequest, path)
	}
	for _, allowed := range values {
		if jsonValuesEqual(allowed, value) {
			return nil
		}
	}
	return fmt.Errorf("%w: %s must be one of %s", ErrBadRequest, path, enumValues(values))
}

func schemaProperties(path string, s map[string]any) (map[string]map[string]any, error) {
	raw, ok := s["properties"]
	if !ok {
		return nil, nil
	}
	props, ok := raw.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%w: %s properties must be an object", ErrBadRequest, path)
	}
	out := make(map[string]map[string]any, len(props))
	for name, value := range props {
		fieldSchema, ok := value.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("%w: %s property %q schema must be an object", ErrBadRequest, path, name)
		}
		out[name] = fieldSchema
	}
	return out, nil
}

func requiredFields(path string, s map[string]any) ([]string, error) {
	rawValue, exists := s["required"]
	if !exists {
		return nil, nil
	}
	raw, ok := rawValue.([]any)
	if !ok {
		return nil, fmt.Errorf("%w: %s required must be an array", ErrBadRequest, path)
	}
	out := make([]string, 0, len(raw))
	seen := make(map[string]struct{}, len(raw))
	for _, value := range raw {
		name, ok := value.(string)
		if !ok {
			return nil, fmt.Errorf("%w: %s required members must be strings", ErrBadRequest, path)
		}
		if _, duplicate := seen[name]; duplicate {
			return nil, fmt.Errorf("%w: %s required member %q is duplicated", ErrBadRequest, path, name)
		}
		seen[name] = struct{}{}
		out = append(out, name)
	}
	return out, nil
}

func jsonValuesEqual(a, b any) bool {
	switch a := a.(type) {
	case nil:
		return b == nil
	case bool:
		value, ok := b.(bool)
		return ok && a == value
	case string:
		value, ok := b.(string)
		return ok && a == value
	case json.Number:
		value, ok := b.(json.Number)
		if !ok {
			return false
		}
		left, leftOK := canonicalJSONNumber(a.String())
		right, rightOK := canonicalJSONNumber(value.String())
		return leftOK && rightOK && left.equal(right)
	case []any:
		value, ok := b.([]any)
		if !ok || len(a) != len(value) {
			return false
		}
		for i := range a {
			if !jsonValuesEqual(a[i], value[i]) {
				return false
			}
		}
		return true
	case map[string]any:
		value, ok := b.(map[string]any)
		if !ok || len(a) != len(value) {
			return false
		}
		for key, member := range a {
			other, exists := value[key]
			if !exists || !jsonValuesEqual(member, other) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

func enumValues(values []any) string {
	parts := make([]string, 0, len(values))
	for _, value := range values {
		parts = append(parts, fmt.Sprint(value))
	}
	return strings.Join(parts, ", ")
}
