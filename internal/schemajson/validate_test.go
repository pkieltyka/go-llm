package schemajson

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestCanonicalJSONNumber(t *testing.T) {
	equivalent := [][2]string{
		{"1", "1.0"},
		{"1", "1e0"},
		{"-1200", "-12e2"},
		{"0", "-0.000e999999999999999999"},
		{"1e100000000000000000000", "10e99999999999999999999"},
	}
	for _, pair := range equivalent {
		left, leftOK := canonicalJSONNumber(pair[0])
		right, rightOK := canonicalJSONNumber(pair[1])
		if !leftOK || !rightOK || !left.equal(right) {
			t.Errorf("canonicalJSONNumber(%q) != canonicalJSONNumber(%q)", pair[0], pair[1])
		}
	}

	distinct := [][2]string{
		{"9007199254740992", "9007199254740993"},
		{"18446744073709551615", "18446744073709551614"},
		{"1e100000000000000000000", "1e99999999999999999999"},
	}
	for _, pair := range distinct {
		left, leftOK := canonicalJSONNumber(pair[0])
		right, rightOK := canonicalJSONNumber(pair[1])
		if !leftOK || !rightOK || left.equal(right) {
			t.Errorf("canonicalJSONNumber(%q) == canonicalJSONNumber(%q)", pair[0], pair[1])
		}
	}

	for _, invalid := range []string{"", "+1", "01", "1.", ".1", "1e", "1e+", "1e2x", "1 2", "--1"} {
		if _, ok := canonicalJSONNumber(invalid); ok {
			t.Errorf("canonicalJSONNumber(%q) accepted invalid JSON number", invalid)
		}
	}
}

func TestValidateArgsUsesExactJSONNumbers(t *testing.T) {
	tests := []struct {
		name   string
		schema string
		args   string
		ok     bool
	}{
		{name: "integer above MaxInt64", schema: `{"type":"integer"}`, args: `18446744073709551615`, ok: true},
		{name: "arbitrary negative integer", schema: `{"type":"integer"}`, args: `-184467440737095516160000000000000000000`, ok: true},
		{name: "mathematical integer fraction", schema: `{"type":"integer"}`, args: `1.0`, ok: true},
		{name: "mathematical integer exponent", schema: `{"type":"integer"}`, args: `1e100000000000000000000`, ok: true},
		{name: "non-integer exponent", schema: `{"type":"integer"}`, args: `1e-100000000000000000000`, ok: false},
		{name: "equivalent enum decimal", schema: `{"type":"number","enum":[1.0]}`, args: `1`, ok: true},
		{name: "equivalent enum exponent", schema: `{"type":"number","enum":[1e0]}`, args: `1.00`, ok: true},
		{name: "distinct huge enum", schema: `{"type":"integer","enum":[9007199254740992]}`, args: `9007199254740993`, ok: false},
		{name: "huge exponent enum", schema: `{"type":"number","enum":[1e100000000000000000000]}`, args: `10e99999999999999999999`, ok: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateArgs("number", json.RawMessage(tt.schema), json.RawMessage(tt.args))
			if tt.ok && err != nil {
				t.Fatalf("ValidateArgs returned error: %v", err)
			}
			if !tt.ok && !errors.Is(err, ErrBadRequest) {
				t.Fatalf("ValidateArgs error = %v, want ErrBadRequest", err)
			}
		})
	}

	if err := ValidateArgs("number", json.RawMessage(`{"type":"integer"} trailing`), json.RawMessage(`1`)); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("trailing schema error = %v, want ErrBadRequest", err)
	}
	if err := ValidateArgs("number", json.RawMessage(`{"type":"integer"}`), json.RawMessage(`1 2`)); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("trailing args error = %v, want ErrBadRequest", err)
	}
}

func TestValidateArgsRejectsMalformedSchemaBeforeArguments(t *testing.T) {
	tests := []struct {
		name   string
		schema string
		want   string
	}{
		{name: "required is string", schema: `{"type":"object","required":"name"}`, want: "required must be an array"},
		{name: "required member is number", schema: `{"type":"object","required":["name",1]}`, want: "required members must be strings"},
		{name: "duplicate required", schema: `{"type":"object","required":["name","name"]}`, want: "is duplicated"},
		{name: "nested malformed required", schema: `{"type":"object","properties":{"unused":{"type":"object","required":[false]}}}`, want: "required members must be strings"},
		{name: "property is not schema", schema: `{"type":"object","properties":{"name":true}}`, want: "schema must be an object"},
		{name: "items is not schema", schema: `{"type":"array","items":false}`, want: "items must be an object"},
		{name: "enum is not array", schema: `{"type":"string","enum":"x"}`, want: "enum must be a non-empty array"},
		{name: "additionalProperties malformed", schema: `{"type":"object","additionalProperties":1}`, want: "additionalProperties must be a boolean or object"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateArgs("broken", json.RawMessage(tt.schema), json.RawMessage(`{`))
			if !errors.Is(err, ErrBadRequest) || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("ValidateArgs error = %v, want ErrBadRequest containing %q", err, tt.want)
			}
			if strings.Contains(err.Error(), "invalid tool args") {
				t.Fatalf("arguments were decoded before schema preflight: %v", err)
			}
		})
	}
}
