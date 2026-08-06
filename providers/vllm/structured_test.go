package vllm

import (
	"encoding/json"
	"errors"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
)

func buildStructuredRequest(t *testing.T, p *Provider, req *llm.Request) json.RawMessage {
	t.Helper()
	params, err := p.inner.BuildParams(req, false)
	if err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	wire, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	var body struct {
		StructuredOutputs json.RawMessage `json:"structured_outputs"`
	}
	if err := json.Unmarshal(wire, &body); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	return body.StructuredOutputs
}

// TestStructuredOutputsGolden pins the structured_outputs extras shape for
// each native mode.
func TestStructuredOutputsGolden(t *testing.T) {
	cases := map[string]struct {
		so   StructuredOutputs
		want string
	}{
		"regex": {
			so:   StructuredOutputs{Regex: "^[0-9]{4}$"},
			want: `{"regex":"^[0-9]{4}$"}`,
		},
		"choice": {
			so:   StructuredOutputs{Choice: []string{"red", "green", "blue"}},
			want: `{"choice":["red","green","blue"]}`,
		},
		"grammar": {
			so:   StructuredOutputs{Grammar: "root ::= \"yes\" | \"no\""},
			want: `{"grammar":"root ::= \"yes\" | \"no\""}`,
		},
		"structural_tag": {
			so:   StructuredOutputs{StructuralTag: json.RawMessage(`{"type":"structural_tag","format":{"begin":"<x>","end":"</x>"}}`)},
			want: `{"structural_tag":{"type":"structural_tag","format":{"begin":"<x>","end":"</x>"}}}`,
		},
		"regex_with_whitespace_pattern": {
			so:   StructuredOutputs{Regex: "[0-9]+", WhitespacePattern: `[\n\t ]*`},
			want: `{"regex":"[0-9]+","whitespace_pattern":"[\\n\\t ]*"}`,
		},
	}
	p, err := New("http://vllm.test/v1")
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			got := buildStructuredRequest(t, p, &llm.Request{
				Model:           "m",
				Messages:        []llm.Message{llm.UserText("hi")},
				ProviderOptions: Options{StructuredOutputs: &tc.so},
			})
			testutil.AssertJSONEqual(t, string(got), tc.want)
		})
	}
}

// TestStructuredOutputsConflicts is the conflict-rule table: one constraint
// system per request, exactly one mode, valid raw JSON — all fail loud at
// build (ErrBadRequest) before any network call.
func TestStructuredOutputsConflicts(t *testing.T) {
	cases := map[string]struct {
		req llm.Request
	}{
		"with_response_format_json_schema": {
			req: llm.Request{
				ResponseFormat: &llm.ResponseFormat{
					Type:   llm.FormatJSONSchema,
					Schema: json.RawMessage(`{"type":"object"}`),
				},
				ProviderOptions: Options{StructuredOutputs: &StructuredOutputs{Choice: []string{"a"}}},
			},
		},
		"with_response_format_json_mode": {
			req: llm.Request{
				ResponseFormat:  &llm.ResponseFormat{Type: llm.FormatJSONMode},
				ProviderOptions: Options{StructuredOutputs: &StructuredOutputs{Regex: "[0-9]+"}},
			},
		},
		"zero_modes": {
			req: llm.Request{ProviderOptions: Options{StructuredOutputs: &StructuredOutputs{}}},
		},
		"whitespace_pattern_without_mode": {
			req: llm.Request{ProviderOptions: Options{StructuredOutputs: &StructuredOutputs{WhitespacePattern: `\s*`}}},
		},
		"two_modes": {
			req: llm.Request{ProviderOptions: Options{StructuredOutputs: &StructuredOutputs{
				Regex:  "[0-9]+",
				Choice: []string{"a", "b"},
			}}},
		},
		"invalid_structural_tag_json": {
			req: llm.Request{ProviderOptions: Options{StructuredOutputs: &StructuredOutputs{
				StructuralTag: json.RawMessage(`{not json`),
			}}},
		},
	}
	p, err := New("http://vllm.test/v1")
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			req := tc.req
			req.Model = "m"
			req.Messages = []llm.Message{llm.UserText("hi")}
			if _, err := p.inner.BuildParams(&req, false); !errors.Is(err, llm.ErrBadRequest) {
				t.Fatalf("BuildParams error = %v, want ErrBadRequest", err)
			}
		})
	}
}
