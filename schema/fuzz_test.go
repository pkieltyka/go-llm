package schema

import (
	"encoding/json"
	"reflect"
	"strconv"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/schemajson"
)

type fuzzValidateArgs struct {
	City  string         `json:"city" jsonschema:"enum=Toronto|Montreal"`
	Count int            `json:"count" jsonschema:"enum=1|2|3"`
	Blob  []byte         `json:"blob,omitempty"`
	Meta  map[string]int `json:"meta,omitempty"`
}

type FuzzEmbedded struct {
	ID string `json:"id" jsonschema:"description=Identifier"`
}

var fuzzValidateTool = llm.Tool{
	Name:        "fuzz",
	InputSchema: MustFor[fuzzValidateArgs](),
}

func FuzzValidateArgs(f *testing.F) {
	seeds := [][]byte{
		[]byte(`{"city":"Toronto","count":1}`),
		[]byte(`{"city":"Montreal","count":2,"blob":"Z28=","meta":{"priority":1}}`),
		[]byte(`{"city":"Ottawa","count":1}`),
		[]byte(`{"city":"Toronto","count":"one"}`),
		[]byte(`{"city":"Toronto","count":4}`),
		[]byte(`{"city":"Toronto","count":1,"extra":true}`),
		[]byte(`{"city":`),
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		err := ValidateArgs(fuzzValidateTool, json.RawMessage(data))
		if err == nil && !json.Valid(data) {
			t.Fatalf("ValidateArgs accepted invalid JSON: %q", data)
		}
	})
}

func FuzzFor(f *testing.F) {
	seeds := [][]byte{
		{0, 1, 2, 3, 4},
		{1, 2, 3, 4, 5, 6},
		{2, 8, 13, 21, 34},
		[]byte("embedded,bytes,enum"),
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		typ, ok := fuzzStructType(data)
		if !ok {
			return
		}
		raw, err := schemajson.ForType(typ)
		if err != nil {
			return
		}
		if !json.Valid(raw) {
			t.Fatalf("For generated invalid JSON: %s", raw)
		}
		var decoded map[string]any
		if err := json.Unmarshal(raw, &decoded); err != nil {
			t.Fatalf("generated schema did not decode: %v", err)
		}
		if decoded["$schema"] != schemajson.Draft2020Schema {
			t.Fatalf("generated schema missing draft marker: %s", raw)
		}
	})
}

func fuzzStructType(data []byte) (reflect.Type, bool) {
	if len(data) == 0 {
		return nil, false
	}

	fieldTypes := []reflect.Type{
		reflect.TypeOf(""),
		reflect.TypeOf(0),
		reflect.TypeOf(false),
		reflect.TypeOf([]byte{}),
		reflect.TypeOf([]string{}),
		reflect.TypeOf(map[string]int{}),
		reflect.TypeOf(struct {
			Nested string `json:"nested"`
		}{}),
	}

	fieldCount := int(data[0]%5) + 1
	fields := make([]reflect.StructField, 0, fieldCount+1)
	if data[0]&0x80 != 0 {
		fields = append(fields, reflect.StructField{
			Name:      "FuzzEmbedded",
			Type:      reflect.TypeOf(FuzzEmbedded{}),
			Anonymous: true,
		})
	}

	for i := 0; i < fieldCount; i++ {
		b := data[(i+1)%len(data)]
		fieldType := fieldTypes[int(b)%len(fieldTypes)]
		jsonName := "field_" + string(rune('a'+i))
		jsonTag := jsonName
		if b&0x20 != 0 {
			jsonTag += ",omitempty"
		}
		tag := structTag(jsonTag, fuzzSchemaTag(fieldType, b))
		if b&0x40 != 0 {
			fieldType = reflect.PointerTo(fieldType)
		}
		fields = append(fields, reflect.StructField{
			Name: "Field" + string(rune('A'+i)),
			Type: fieldType,
			Tag:  reflect.StructTag(tag),
		})
	}

	return reflect.StructOf(fields), true
}

func structTag(jsonTag, schemaTag string) reflect.StructTag {
	var b strings.Builder
	b.WriteString("json:")
	b.WriteString(strconv.Quote(jsonTag))
	if schemaTag != "" {
		b.WriteString(" jsonschema:")
		b.WriteString(strconv.Quote(schemaTag))
	}
	return reflect.StructTag(b.String())
}

func fuzzSchemaTag(typ reflect.Type, b byte) string {
	for typ.Kind() == reflect.Pointer {
		typ = typ.Elem()
	}
	switch typ.Kind() {
	case reflect.String:
		if b&1 == 0 {
			return `description=City\, Province,enum=Toronto|Montreal`
		}
		return `format=email`
	case reflect.Int:
		return `enum=1|2`
	case reflect.Bool:
		return `enum=true|false`
	case reflect.Slice:
		if typ.Elem().Kind() == reflect.Uint8 {
			return `description=Base64 payload`
		}
	}
	if b&1 == 0 {
		return "optional"
	}
	return ""
}
