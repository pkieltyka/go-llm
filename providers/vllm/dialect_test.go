package vllm

import (
	"errors"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

// The numeric in-stream code fallback moved into the shared classifier;
// this pins that the vLLM dialect still classifies status-less numeric
// codes (and rejects fractional ones) through the delegation.
func TestDialectMapErrorStatusNumericCodes(t *testing.T) {
	cases := []struct {
		status  int
		code    string
		message string
		want    error
	}{
		{0, "503", "", llm.ErrOverloaded},
		{0, "429", "", llm.ErrRateLimited},
		{0, "429.5", "", llm.ErrServer},
		{0, "abort", "", llm.ErrServer},
		{429, "", "", llm.ErrRateLimited},
	}
	for _, tc := range cases {
		if got := (dialect{}).MapErrorStatus(tc.status, tc.code, tc.message); !errors.Is(got, tc.want) {
			t.Fatalf("MapErrorStatus(%d, %q, %q) = %v, want %v", tc.status, tc.code, tc.message, got, tc.want)
		}
	}
}
