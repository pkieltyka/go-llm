package e2e

import (
	"encoding/json"
	"flag"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

var updateFixtureRedactions = flag.Bool("update-fixture-redactions", false, "rewrite recorded fixtures through the current redactor")

func TestRedactBytesRemovesAnthropicSecrets(t *testing.T) {
	secret := "sk" + "-ant-test_secret123456789"
	input := []byte(`{"api_key":"` + secret + `","authorization":"Bearer ` + secret + `","metadata":{"x-api-key":"` + secret + `"}}`)
	redacted := string(RedactBytes(input, secret))
	if strings.Contains(redacted, secret) || strings.Contains(redacted, "sk-ant-") {
		t.Fatalf("redacted bytes leaked secret: %s", redacted)
	}
	for _, want := range []string{`"api_key":"[REDACTED]"`, `"authorization":"[REDACTED]"`, `"x-api-key":"[REDACTED]"`} {
		if !strings.Contains(redacted, want) {
			t.Fatalf("redacted bytes missing %s in %s", want, redacted)
		}
	}
}

func TestRedactCaptureRedactsHeadersBodiesAndURL(t *testing.T) {
	secret := "sk" + "-ant-test_secret123456789"
	capture := llm.WireCapture{
		Provider:       "anthropic",
		Method:         http.MethodPost,
		URL:            "https://api.example.test/v1/messages?api_key=" + secret,
		RequestHeaders: http.Header{"X-Api-Key": []string{secret}, "Authorization": []string{"Bearer " + secret}},
		RequestBody:    []byte(`{"metadata":{"api_key":"` + secret + `"}}`),
		ResponseHeaders: http.Header{
			"Set-Cookie": []string{"session=" + secret},
		},
		ResponseBody: []byte(`{"echo":"` + secret + `"}`),
	}
	redacted := RedactCapture(capture, secret)
	joined := redacted.URL + redacted.RequestHeaders.Get("X-Api-Key") + redacted.RequestHeaders.Get("Authorization") + redacted.RequestBody + redacted.ResponseHeaders.Get("Set-Cookie") + redacted.ResponseBody
	if strings.Contains(joined, secret) || strings.Contains(joined, "sk-ant-") {
		t.Fatalf("redacted capture leaked secret: %+v", redacted)
	}
	if redacted.RequestHeaders.Get("X-Api-Key") != "[REDACTED]" || redacted.ResponseHeaders.Get("Set-Cookie") != "[REDACTED]" {
		t.Fatalf("headers not redacted: %+v %+v", redacted.RequestHeaders, redacted.ResponseHeaders)
	}
}

func TestRedactCaptureRedactsStableProviderIdentifiers(t *testing.T) {
	capture := llm.WireCapture{
		Provider: "openai-codex",
		Method:   http.MethodPost,
		URL:      "https://api.example.test/v1/messages?request_id=req_live_123&prompt_cache_key=cache_live_123",
		ResponseHeaders: http.Header{
			"Anthropic-Ratelimit-Unified-Overage-Status": []string{"allowed"},
			"Anthropic-Organization-Id":                  []string{"00000000-0000-4000-8000-000000000001"},
			"Cf-Ray":                                     []string{"mockray1234567890-YYZ"},
			"Nel":                                        []string{`{"report_to":"cf-nel","success_fraction":0.01,"max_age":604800}`},
			"Report-To":                                  []string{`{"group":"cf-nel","endpoints":[{"url":"https://a.nel.cloudflare.com/report/v4?s=token"}]}`},
			"Request-Id":                                 []string{"req_mocklive1234567890"},
			"Traceresponse":                              []string{"00-00000000000000000000000000000001-0000000000000001-01"},
			"X-Codex-Active-Limit":                       []string{"premium"},
			"X-Codex-Credits-Unlimited":                  []string{"False"},
			"X-Codex-Plan-Type":                          []string{"pro"},
			"X-Generation-Id":                            []string{"gen-0000000000-mocklive"},
			"X-Oai-Request-Id":                           []string{"00000000-0000-4000-8000-000000000002"},
		},
		ResponseBody: []byte(`{"id":"resp_mocklive1234567890","item_id":"msg_mocklive1234567890","request_id":"req_mocklive1234567890","tool_call_id":"chatcmpl-tool-mocklive1234567890","tool_use_id":"toolu_mocklive1234567890","prompt_cache_key":"00000000-0000-4000-8000-000000000003","safety_identifier":"user-mocklive1234567890","encrypted_content":"abc123","signature":"sig123","obfuscation":"noise"}`),
	}
	redacted := RedactCapture(capture)
	for _, header := range []string{
		"Anthropic-Ratelimit-Unified-Overage-Status",
		"X-Codex-Active-Limit",
		"X-Codex-Credits-Unlimited",
		"X-Codex-Plan-Type",
	} {
		if got := redacted.ResponseHeaders.Get(header); got != "[REDACTED]" {
			t.Fatalf("header %s = %q, want [REDACTED]", header, got)
		}
	}
	if !strings.Contains(redacted.URL, "request_id=MOCK-REQ-") || !strings.Contains(redacted.URL, "prompt_cache_key=MOCK-CACHE-") {
		t.Fatalf("url identifiers not mocked: %s", redacted.URL)
	}
	for _, want := range []string{
		`"id":"MOCK-RESP-`,
		`"item_id":"MOCK-MSG-`,
		`"request_id":"MOCK-REQ-`,
		`"tool_call_id":"MOCK-TOOL-CALL-`,
		`"tool_use_id":"MOCK-TOOL-CALL-`,
		`"prompt_cache_key":"MOCK-CACHE-`,
		`"safety_identifier":"MOCK-USER-`,
		`"encrypted_content":"MOCK-ENCRYPTED-`,
		`"signature":"MOCK-SIGNATURE-`,
		`"obfuscation":"MOCK-OBFUSCATION-`,
	} {
		if !strings.Contains(redacted.ResponseBody, want) {
			t.Fatalf("response body missing %s in %s", want, redacted.ResponseBody)
		}
	}

	joined := redacted.URL +
		redacted.ResponseHeaders.Get("Anthropic-Ratelimit-Unified-Overage-Status") +
		redacted.ResponseHeaders.Get("Anthropic-Organization-Id") +
		redacted.ResponseHeaders.Get("Cf-Ray") +
		redacted.ResponseHeaders.Get("Nel") +
		redacted.ResponseHeaders.Get("Report-To") +
		redacted.ResponseHeaders.Get("Request-Id") +
		redacted.ResponseHeaders.Get("Traceresponse") +
		redacted.ResponseHeaders.Get("X-Codex-Active-Limit") +
		redacted.ResponseHeaders.Get("X-Codex-Credits-Unlimited") +
		redacted.ResponseHeaders.Get("X-Codex-Plan-Type") +
		redacted.ResponseHeaders.Get("X-Generation-Id") +
		redacted.ResponseHeaders.Get("X-Oai-Request-Id") +
		redacted.ResponseBody
	for _, leaked := range []string{
		"00000000-0000-4000-8000-000000000001",
		"mockray1234567890-YYZ",
		"cf-nel",
		"a.nel.cloudflare.com",
		"req_mocklive1234567890",
		"00-00000000000000000000000000000001-0000000000000001-01",
		"gen-0000000000-mocklive",
		"00000000-0000-4000-8000-000000000002",
		"resp_mocklive1234567890",
		"msg_mocklive1234567890",
		"chatcmpl-tool-mocklive1234567890",
		"toolu_mocklive1234567890",
		"00000000-0000-4000-8000-000000000003",
		"user-mocklive1234567890",
		"abc123",
		"sig123",
		"noise",
	} {
		if strings.Contains(joined, leaked) {
			t.Fatalf("redacted capture leaked %q: %+v", leaked, redacted)
		}
	}
	if got := strings.Count(joined, "[REDACTED]"); got < 8 {
		t.Fatalf("redacted capture did not redact expected fields: %+v", redacted)
	}
}

func TestRecordedFixturesAreRedacted(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join("fixtures", "*", "live.json"))
	if err != nil {
		t.Fatalf("Glob returned error: %v", err)
	}
	if len(paths) == 0 {
		t.Skip("no recorded fixtures present")
	}
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("ReadFile returned error: %v", err)
			}
			if *updateFixtureRedactions {
				data = rewriteFixtureRedactions(t, path, data)
			}
			assertFixtureRedacted(t, path, data)
		})
	}
}

func rewriteFixtureRedactions(t *testing.T, path string, data []byte) []byte {
	t.Helper()
	var exchanges []RecordedExchange
	if err := json.Unmarshal(data, &exchanges); err != nil {
		t.Fatalf("Unmarshal fixture returned error: %v", err)
	}
	for i := range exchanges {
		exchanges[i] = RedactRecordedExchange(exchanges[i])
	}
	raw, err := json.MarshalIndent(exchanges, "", "  ")
	if err != nil {
		t.Fatalf("MarshalIndent returned error: %v", err)
	}
	raw = append(RedactBytes(raw), '\n')
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	return raw
}

func assertFixtureRedacted(t *testing.T, path string, data []byte) {
	t.Helper()
	var exchanges []RecordedExchange
	if err := json.Unmarshal(data, &exchanges); err != nil {
		t.Fatalf("Unmarshal fixture returned error: %v", err)
	}
	for i, exchange := range exchanges {
		for name, values := range exchange.RequestHeaders {
			assertFixtureHeaderRedacted(t, path, i, name, values)
		}
		for name, values := range exchange.ResponseHeaders {
			assertFixtureHeaderRedacted(t, path, i, name, values)
		}
		joined := exchange.URL + "\n" + exchange.RequestBody + "\n" + exchange.ResponseBody + "\n" + exchange.Err
		for _, pattern := range unredactedFixturePatterns {
			for _, match := range pattern.FindAllString(joined, -1) {
				if strings.Contains(match, "[REDACTED]") {
					continue
				}
				if strings.Contains(match, "MOCK-") {
					continue
				}
				t.Fatalf("%s exchange %d leaks %s via %q", path, i, pattern, match)
			}
		}
	}
}

func assertFixtureHeaderRedacted(t *testing.T, path string, index int, name string, values []string) {
	t.Helper()
	if !isSensitiveHeaderName(name) {
		return
	}
	for _, value := range values {
		if value != "[REDACTED]" {
			t.Fatalf("%s exchange %d header %s leaked %q", path, index, name, value)
		}
	}
}

var unredactedFixturePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)"(?:safety_identifier|prompt_cache_key|encrypted_content|signature|obfuscation|api_key|access|refresh|access_token|refresh_token|accountId|chatgpt-account-id)"\s*:\s*"[^"]+`),
	regexp.MustCompile(`(?i)"(?:id|item_id|response_id|previous_response_id|request_id|call_id|tool_call_id|tool_use_id)"\s*:\s*"(?:resp|msg|req|rs|fc|call|toolu|gen|chatcmpl)[A-Za-z0-9._:-]+`),
	regexp.MustCompile(`(?i)bearer\s+[A-Za-z0-9._~+/=-]{16,}`),
	regexp.MustCompile(`(?i)user-[A-Za-z0-9_-]{8,}`),
}
