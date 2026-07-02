package schema_test

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/schema"
)

type goldenAddress struct {
	City    string  `json:"city" jsonschema:"description=City name,enum=Toronto|Montreal"`
	Country *string `json:"country,omitempty"`
}

type goldenArgs struct {
	Name    string        `json:"name" jsonschema:"description=Full name"`
	Age     int           `json:"age,omitempty" jsonschema:"optional"`
	Email   string        `json:"email" jsonschema:"format=email"`
	Tags    []string      `json:"tags,omitempty"`
	Address goldenAddress `json:"address"`
}

func TestForGeneratesGoldenSchema(t *testing.T) {
	raw, err := schema.For[goldenArgs]()
	if err != nil {
		t.Fatalf("For returned error: %v", err)
	}

	const want = `{"$schema":"https://json-schema.org/draft/2020-12/schema","additionalProperties":false,"properties":{"address":{"additionalProperties":false,"properties":{"city":{"description":"City name","enum":["Toronto","Montreal"],"type":"string"},"country":{"type":"string"}},"required":["city"],"type":"object"},"age":{"type":"integer"},"email":{"format":"email","type":"string"},"name":{"description":"Full name","type":"string"},"tags":{"items":{"type":"string"},"type":"array"}},"required":["address","email","name"],"type":"object"}`
	if string(raw) != want {
		t.Fatalf("schema mismatch:\n got: %s\nwant: %s", raw, want)
	}
}

func TestForRejectsUnsupportedTypesAndTags(t *testing.T) {
	type unsupportedType struct {
		Ch chan int `json:"ch"`
	}
	if _, err := schema.For[unsupportedType](); err == nil {
		t.Fatalf("For unsupported type returned nil error")
	}

	type unsupportedTag struct {
		Count int `json:"count" jsonschema:"minimum=1"`
	}
	if _, err := schema.For[unsupportedTag](); err == nil {
		t.Fatalf("For unsupported tag returned nil error")
	}

	type node struct {
		Next *node `json:"next"`
	}
	if _, err := schema.For[node](); err == nil {
		t.Fatalf("For recursive type returned nil error")
	}
}

type modifierArgs struct {
	Query string `json:"query"`
}

type selfDescribed struct{}

func (selfDescribed) JSONSchema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"id":{"type":"string"}},"additionalProperties":false}`)
}

func TestForSupportsModifierAndJSONSchemaer(t *testing.T) {
	raw, err := schema.For[modifierArgs](schema.WithModifier(func(field reflect.StructField, s map[string]any) {
		if field.Name == "Query" {
			s["description"] = "Search query"
		}
	}))
	if err != nil {
		t.Fatalf("For with modifier returned error: %v", err)
	}
	if !strings.Contains(string(raw), `"description":"Search query"`) {
		t.Fatalf("modified schema missing description: %s", raw)
	}

	raw, err = schema.For[selfDescribed]()
	if err != nil {
		t.Fatalf("For self-described returned error: %v", err)
	}
	const want = `{"type":"object","properties":{"id":{"type":"string"}},"additionalProperties":false}`
	if string(raw) != want {
		t.Fatalf("self-described schema = %s, want %s", raw, want)
	}
}

type weatherArgs struct {
	City  string         `json:"city" jsonschema:"enum=Toronto|Montreal"`
	Count int            `json:"count"`
	Units string         `json:"units,omitempty" jsonschema:"enum=c|f,optional"`
	Tags  []string       `json:"tags,omitempty"`
	Meta  map[string]int `json:"meta,omitempty"`
}

func TestValidateArgs(t *testing.T) {
	tool := llm.Tool{
		Name:        "weather",
		InputSchema: schema.MustFor[weatherArgs](),
	}

	tests := []struct {
		name    string
		args    json.RawMessage
		wantErr bool
	}{
		{
			name: "valid",
			args: json.RawMessage(`{"city":"Toronto","count":2,"units":"c","tags":["now"],"meta":{"priority":1}}`),
		},
		{
			name:    "missing required",
			args:    json.RawMessage(`{"city":"Toronto"}`),
			wantErr: true,
		},
		{
			name:    "wrong type",
			args:    json.RawMessage(`{"city":"Toronto","count":"two"}`),
			wantErr: true,
		},
		{
			name:    "bad enum",
			args:    json.RawMessage(`{"city":"Ottawa","count":2}`),
			wantErr: true,
		},
		{
			name:    "unknown property",
			args:    json.RawMessage(`{"city":"Toronto","count":2,"extra":true}`),
			wantErr: true,
		},
		{
			name:    "bad map value",
			args:    json.RawMessage(`{"city":"Toronto","count":2,"meta":{"priority":"high"}}`),
			wantErr: true,
		},
		{
			name:    "invalid json",
			args:    json.RawMessage(`{"city":`),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := schema.ValidateArgs(tool, tt.args)
			if tt.wantErr {
				if !errors.Is(err, llm.ErrBadRequest) {
					t.Fatalf("error = %v, want ErrBadRequest", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("ValidateArgs returned error: %v", err)
			}
		})
	}
}
