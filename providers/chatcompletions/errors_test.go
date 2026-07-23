package chatcompletions

import (
	"testing"
)

func TestStringifyCodeForms(t *testing.T) {
	cases := []struct {
		code any
		want string
	}{
		{nil, ""},
		{"429", "429"},
		{float64(429), "429"},     // JSON 429 and 429.0 both decode to this
		{float64(429.5), "429.5"}, // faithful, never rounded to a valid status
		{429, "429"},
		{"rate_limit_exceeded", "rate_limit_exceeded"},
	}
	for _, tc := range cases {
		if got := stringifyCode(tc.code); got != tc.want {
			t.Fatalf("stringifyCode(%v) = %q, want %q", tc.code, got, tc.want)
		}
	}
}

func TestParseProviderErrorCodeForms(t *testing.T) {
	cases := []struct {
		body     string
		wantCode string
	}{
		{`{"error":{"code":"429","message":"m"}}`, "429"},
		{`{"error":{"code":429,"message":"m"}}`, "429"},
		{`{"error":{"code":429.5,"message":"m"}}`, "429.5"},
		{`{"object":"error","code":503,"message":"m"}`, "503"},
	}
	for _, tc := range cases {
		code, message, _ := parseProviderError([]byte(tc.body))
		if code != tc.wantCode {
			t.Fatalf("parseProviderError(%s) code = %q, want %q", tc.body, code, tc.wantCode)
		}
		if message != "m" {
			t.Fatalf("parseProviderError(%s) message = %q, want %q", tc.body, message, "m")
		}
	}
}
