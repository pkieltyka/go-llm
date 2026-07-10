package providerutil

import (
	"errors"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

type testOptions struct{ Label string }

func (testOptions) ForProvider() string { return "testprov" }

type spoofOptions struct{}

func (spoofOptions) ForProvider() string { return "testprov" }

type foreignOptions struct{}

func (foreignOptions) ForProvider() string { return "otherprov" }

func TestOptionsOf(t *testing.T) {
	value := testOptions{Label: "v"}
	pointer := &testOptions{Label: "p"}

	cases := []struct {
		name    string
		req     *llm.Request
		want    string
		wantOK  bool
		wantErr bool
	}{
		{name: "nil request", req: nil},
		{name: "no options", req: &llm.Request{}},
		{name: "value form", req: &llm.Request{ProviderOptions: value}, want: "v", wantOK: true},
		{name: "pointer form", req: &llm.Request{ProviderOptions: pointer}, want: "p", wantOK: true},
		{name: "nil typed pointer", req: &llm.Request{ProviderOptions: (*testOptions)(nil)}},
		{name: "foreign concrete type", req: &llm.Request{ProviderOptions: foreignOptions{}}, wantErr: true},
		{name: "same name wrong type", req: &llm.Request{ProviderOptions: spoofOptions{}}, wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok, err := OptionsOf[testOptions](tc.req)
			if tc.wantErr {
				if !errors.Is(err, llm.ErrBadRequest) {
					t.Fatalf("want ErrBadRequest, got %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if ok && got.Label != tc.want {
				t.Fatalf("Label = %q, want %q", got.Label, tc.want)
			}
		})
	}
}

func TestJSONEqualIsStructuredAndLossless(t *testing.T) {
	tests := []struct {
		name        string
		left, right string
		want        bool
	}{
		{
			name:  "formatting and key order",
			left:  `{"a":[1,true],"b":{"x":"y"}}`,
			right: "{\n  \"b\": {\"x\": \"y\"}, \"a\": [1.0, true]\n}",
			want:  true,
		},
		{
			name:  "numeric equivalent exponent",
			left:  `{"n":1e3}`,
			right: `{"n":1000.0}`,
			want:  true,
		},
		{
			name:  "large integers remain distinct",
			left:  `{"n":9007199254740992}`,
			right: `{"n":9007199254740993}`,
			want:  false,
		},
		{
			name:  "arbitrarily large exponent equivalent",
			left:  `{"n":1e1000000000}`,
			right: `{"n":10e999999999}`,
			want:  true,
		},
		{
			name:  "arbitrarily large exponent unequal",
			left:  `{"n":1e1000000000}`,
			right: `{"n":1e999999999}`,
			want:  false,
		},
		{
			name:  "signed decimal equivalent",
			left:  `{"n":-1.2300e5}`,
			right: `{"n":-123000}`,
			want:  true,
		},
		{
			name:  "all zero spellings equal",
			left:  `{"n":-0.000e1000000000}`,
			right: `{"n":0e-1000000000}`,
			want:  true,
		},
		{
			name:  "invalid JSON",
			left:  `not-json`,
			right: `not-json`,
			want:  false,
		},
		{
			name:  "trailing document rejected",
			left:  `{"n":1} trailing`,
			right: `{"n":1}`,
			want:  false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := JSONEqual([]byte(tt.left), []byte(tt.right)); got != tt.want {
				t.Fatalf("JSONEqual(%s, %s) = %v, want %v", tt.left, tt.right, got, tt.want)
			}
		})
	}
}

// TestStatusErrorKindCanonicalTable pins the FS §16 status→sentinel table in
// one place; each provider package additionally asserts parity through its
// own error-mapping tables.
func TestStatusErrorKindCanonicalTable(t *testing.T) {
	cases := []struct {
		status int
		want   error
	}{
		{400, llm.ErrBadRequest},
		{401, llm.ErrAuth},
		{402, llm.ErrInsufficientCredits},
		{403, llm.ErrPermission},
		{404, llm.ErrNotFound},
		{408, llm.ErrTimeout},
		{429, llm.ErrRateLimited},
		{500, llm.ErrServer},
		{502, llm.ErrServer},
		{503, llm.ErrOverloaded},
		{529, llm.ErrOverloaded},
	}
	for _, tc := range cases {
		if got := StatusErrorKind(tc.status); !errors.Is(got, tc.want) {
			t.Errorf("StatusErrorKind(%d) = %v, want %v", tc.status, got, tc.want)
		}
	}
}
