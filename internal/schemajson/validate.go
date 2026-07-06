package schemajson

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
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

	var value any
	if err := decodeJSON(args, &value); err != nil {
		return fmt.Errorf("%w: invalid tool args: %v", ErrBadRequest, err)
	}
	return validateValue("$", s, value)
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
		if _, err := number.Int64(); err != nil {
			return fmt.Errorf("%w: %s must be an integer", ErrBadRequest, path)
		}
	case "number":
		number, ok := value.(json.Number)
		if !ok {
			return fmt.Errorf("%w: %s must be a number", ErrBadRequest, path)
		}
		if _, err := number.Float64(); err != nil {
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
	props, err := schemaProperties(s)
	if err != nil {
		return err
	}

	for _, name := range requiredFields(s) {
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

func schemaProperties(s map[string]any) (map[string]map[string]any, error) {
	raw, ok := s["properties"]
	if !ok {
		return nil, nil
	}
	props, ok := raw.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%w: properties must be an object", ErrBadRequest)
	}
	out := make(map[string]map[string]any, len(props))
	for name, value := range props {
		fieldSchema, ok := value.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("%w: property %q schema must be an object", ErrBadRequest, name)
		}
		out[name] = fieldSchema
	}
	return out, nil
}

func requiredFields(s map[string]any) []string {
	raw, ok := s["required"].([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(raw))
	for _, value := range raw {
		if name, ok := value.(string); ok {
			out = append(out, name)
		}
	}
	return out
}

func jsonValuesEqual(a, b any) bool {
	a = normalizeJSONNumber(a)
	b = normalizeJSONNumber(b)
	return reflect.DeepEqual(a, b)
}

func normalizeJSONNumber(v any) any {
	number, ok := v.(json.Number)
	if !ok {
		return v
	}
	if i, err := number.Int64(); err == nil {
		return i
	}
	if f, err := number.Float64(); err == nil {
		return f
	}
	return number.String()
}

func enumValues(values []any) string {
	parts := make([]string, 0, len(values))
	for _, value := range values {
		parts = append(parts, fmt.Sprint(value))
	}
	return strings.Join(parts, ", ")
}
