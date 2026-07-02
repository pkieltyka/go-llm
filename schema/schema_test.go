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

	type stringifiedField struct {
		Count int `json:"count,string"`
	}
	if _, err := schema.For[stringifiedField](); err == nil {
		t.Fatalf("For stringifiedField returned nil error")
	}

	type textMarshalerField struct {
		Value textEncoded `json:"value"`
	}
	if _, err := schema.For[textMarshalerField](); err == nil {
		t.Fatalf("For textMarshalerField returned nil error")
	}
}

type textEncoded string

func (textEncoded) MarshalText() ([]byte, error) {
	return []byte("encoded"), nil
}

type embeddedBase struct {
	ID string `json:"id"`
}

type embeddedArgs struct {
	embeddedBase
	Label string `json:"label"`
}

func TestForFlattensEmbeddedStructsLikeEncodingJSON(t *testing.T) {
	raw, err := schema.For[embeddedArgs]()
	if err != nil {
		t.Fatalf("For returned error: %v", err)
	}
	if strings.Contains(string(raw), "embeddedBase") || strings.Contains(string(raw), "Base") {
		t.Fatalf("embedded field was emitted as a named property: %s", raw)
	}
	for _, want := range []string{`"id":{"type":"string"}`, `"label":{"type":"string"}`, `"required":["id","label"]`} {
		if !strings.Contains(string(raw), want) {
			t.Fatalf("schema missing %s: %s", want, raw)
		}
	}

	args, err := json.Marshal(embeddedArgs{
		embeddedBase: embeddedBase{ID: "123"},
		Label:        "ready",
	})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}
	tool := llm.Tool{Name: "embedded", InputSchema: raw}
	if err := schema.ValidateArgs(tool, args); err != nil {
		t.Fatalf("ValidateArgs rejected encoding/json output %s: %v", args, err)
	}
}

func TestForHandlesByteSlicesAndRejectsRawMessage(t *testing.T) {
	type bytesArgs struct {
		Data []byte `json:"data"`
	}
	raw, err := schema.For[bytesArgs]()
	if err != nil {
		t.Fatalf("For bytesArgs returned error: %v", err)
	}
	if !strings.Contains(string(raw), `"data":{"type":"string"}`) {
		t.Fatalf("[]byte schema should be a string, got: %s", raw)
	}

	args, err := json.Marshal(bytesArgs{Data: []byte("go")})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}
	tool := llm.Tool{Name: "bytes", InputSchema: raw}
	if err := schema.ValidateArgs(tool, args); err != nil {
		t.Fatalf("ValidateArgs rejected encoding/json []byte output %s: %v", args, err)
	}

	type rawArgs struct {
		Raw json.RawMessage `json:"raw"`
	}
	if _, err := schema.For[rawArgs](); err == nil {
		t.Fatalf("For rawArgs returned nil error")
	}
}

func TestForCoercesEnumValuesToFieldType(t *testing.T) {
	type enumArgs struct {
		Count int `json:"count" jsonschema:"enum=1|2"`
	}
	raw, err := schema.For[enumArgs]()
	if err != nil {
		t.Fatalf("For enumArgs returned error: %v", err)
	}
	if strings.Contains(string(raw), `"enum":["1","2"]`) {
		t.Fatalf("integer enum values were emitted as strings: %s", raw)
	}
	tool := llm.Tool{Name: "enum", InputSchema: raw}
	if err := schema.ValidateArgs(tool, json.RawMessage(`{"count":1}`)); err != nil {
		t.Fatalf("ValidateArgs rejected valid integer enum: %v", err)
	}
	if err := schema.ValidateArgs(tool, json.RawMessage(`{"count":3}`)); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("ValidateArgs error = %v, want ErrBadRequest", err)
	}

	type badEnumArgs struct {
		Count int `json:"count" jsonschema:"enum=one|two"`
	}
	if _, err := schema.For[badEnumArgs](); err == nil {
		t.Fatalf("For badEnumArgs returned nil error")
	}
}

func TestForAllowsEscapedCommaInDescription(t *testing.T) {
	type describedArgs struct {
		Place string `json:"place" jsonschema:"description=City\\, Province"`
	}
	raw, err := schema.For[describedArgs]()
	if err != nil {
		t.Fatalf("For describedArgs returned error: %v", err)
	}
	if !strings.Contains(string(raw), `"description":"City, Province"`) {
		t.Fatalf("description did not preserve escaped comma: %s", raw)
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

func TestValidateArgsDocumentsAnnotationSubset(t *testing.T) {
	tool := llm.Tool{
		Name: "email",
		InputSchema: json.RawMessage(`{
			"type":"object",
			"properties":{"email":{"type":"string","format":"email","description":"Email address"}},
			"required":["email"],
			"additionalProperties":false
		}`),
	}
	if err := schema.ValidateArgs(tool, json.RawMessage(`{"email":"not an email"}`)); err != nil {
		t.Fatalf("ValidateArgs enforced annotation keywords: %v", err)
	}
}
