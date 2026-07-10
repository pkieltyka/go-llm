package e2e

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
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

func TestWriteFixtureStructurallySanitizesCapture(t *testing.T) {
	redactionInput := "fixture-value-to-remove"
	path := filepath.Join(t.TempDir(), "live.json")
	capture := llm.WireCapture{
		Provider: "test",
		Method:   http.MethodPost,
		URL:      "http://private-user:private-password@pax.local:8000/v1/chat?api_key=" + redactionInput + "&request_id=req_live_one",
		RequestHeaders: http.Header{
			"Authorization": []string{"Bearer " + redactionInput},
		},
		RequestBody: []byte(`{"metadata":{"access_token":"` + redactionInput + `","callback":"http://10.0.0.7/hook","request_id":"req_live_one"}}`),
		Status:      http.StatusOK,
		ResponseHeaders: http.Header{
			"Set-Cookie": []string{"session=" + redactionInput},
		},
		ResponseBody: []byte("event: response\ndata: {\"id\":\"req_live_one\",\"request_id\":\"req_live_two\",\"origin_host\":\"pax.local\",\"access\":\"" + redactionInput + "\"}\n\n"),
	}

	result, err := WriteFixtureChecked(path, []llm.WireCapture{capture}, FixtureWriteOptions{Secrets: []string{redactionInput}})
	if err != nil {
		t.Fatalf("WriteFixtureChecked returned error: %v", err)
	}
	if !result.Replaced {
		t.Fatalf("fixture was not replaced: %+v", result)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	got := string(data)
	for _, leaked := range []string{redactionInput, "pax.local", "10.0.0.7", "private-user", "private-password", "req_live_one", "req_live_two"} {
		if strings.Contains(got, leaked) {
			t.Fatalf("fixture leaked %q:\n%s", leaked, got)
		}
	}
	for _, want := range []string{fixtureHostname, "[REDACTED]", "MOCK-REQ-1", "MOCK-REQ-2"} {
		if !strings.Contains(got, want) {
			t.Fatalf("fixture missing %q:\n%s", want, got)
		}
	}
	if strings.Count(got, "MOCK-REQ-1") < 3 {
		t.Fatalf("correlated request id was not stable:\n%s", got)
	}
}

func TestWriteFixtureRequiresIncompleteAcknowledgement(t *testing.T) {
	path := filepath.Join(t.TempDir(), "live.json")
	original := []byte("[\n  {\"provider\":\"old\",\"method\":\"POST\",\"url\":\"https://api.example.test/one\",\"status\":200},\n  {\"provider\":\"old\",\"method\":\"POST\",\"url\":\"https://api.example.test/two\",\"status\":200}\n]\n")
	if err := os.WriteFile(path, original, 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	captures := []llm.WireCapture{{
		Provider:     "new",
		Method:       http.MethodPost,
		URL:          "https://api.example.test/chat",
		Status:       http.StatusOK,
		ResponseBody: []byte(`{"ok":true}`),
	}}
	var warnings []string
	options := FixtureWriteOptions{
		ExpectedScenarios:  []string{"chat", "stream"},
		CompletedScenarios: []string{"chat"},
		Warnf: func(format string, args ...any) {
			warnings = append(warnings, fmt.Sprintf(format, args...))
		},
	}

	result, err := WriteFixtureChecked(path, captures, options)
	if err != nil {
		t.Fatalf("WriteFixtureChecked returned error: %v", err)
	}
	if result.Replaced || len(result.Incomplete) == 0 || len(warnings) != 1 {
		t.Fatalf("unacknowledged result = %+v warnings=%v", result, warnings)
	}
	unchanged, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	if !bytes.Equal(unchanged, original) {
		t.Fatalf("partial recording changed fixture:\n%s", unchanged)
	}

	options.AllowIncomplete = true
	result, err = WriteFixtureChecked(path, captures, options)
	if err != nil {
		t.Fatalf("acknowledged WriteFixtureChecked returned error: %v", err)
	}
	if !result.Replaced || len(result.Incomplete) == 0 {
		t.Fatalf("acknowledged result = %+v", result)
	}
	replaced, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	if bytes.Equal(replaced, original) {
		t.Fatal("acknowledged partial recording did not replace fixture")
	}
}

func TestWriteFixtureCaptureErrorRequiresIncompleteAcknowledgement(t *testing.T) {
	path := filepath.Join(t.TempDir(), "live.json")
	original := []byte(`[{"provider":"old","method":"POST","url":"https://api.example.test","status":200}]`)
	if err := os.WriteFile(path, original, 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	captures := []llm.WireCapture{{
		Provider: "new",
		Method:   http.MethodPost,
		URL:      "https://api.example.test/chat",
		Err:      errors.New("response body close failed"),
	}}

	result, err := WriteFixtureChecked(path, captures, FixtureWriteOptions{})
	if err != nil {
		t.Fatalf("WriteFixtureChecked returned error: %v", err)
	}
	if result.Replaced || !containsReason(result.Incomplete, "capture error") {
		t.Fatalf("result = %+v, want capture error to require acknowledgement", result)
	}
	unchanged, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	if !bytes.Equal(unchanged, original) {
		t.Fatalf("capture error changed fixture:\n%s", unchanged)
	}
}

func TestWriteFixtureUnsafeStageLeavesFixtureUnchanged(t *testing.T) {
	path := filepath.Join(t.TempDir(), "live.json")
	original := []byte(`[{"provider":"old","method":"POST","url":"https://api.example.test","status":200}]`)
	if err := os.WriteFile(path, original, 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	highEntropy := "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
	capture := llm.WireCapture{
		Provider:     "new",
		Method:       http.MethodPost,
		URL:          "https://api.example.test/chat",
		Status:       http.StatusOK,
		ResponseBody: []byte(`{"novel_provider_metadata":"` + highEntropy + `"}`),
	}
	result, err := WriteFixtureChecked(path, []llm.WireCapture{capture}, FixtureWriteOptions{AllowIncomplete: true})
	if err == nil {
		t.Fatalf("WriteFixtureChecked result = %+v, want entropy error", result)
	}
	unchanged, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatalf("ReadFile returned error: %v", readErr)
	}
	if !bytes.Equal(unchanged, original) {
		t.Fatalf("unsafe stage changed fixture:\n%s", unchanged)
	}
	temps, globErr := filepath.Glob(filepath.Join(filepath.Dir(path), ".fixture-*.tmp"))
	if globErr != nil {
		t.Fatalf("Glob returned error: %v", globErr)
	}
	if len(temps) != 0 {
		t.Fatalf("staged files were not removed: %v", temps)
	}
}

func TestFixtureEntropyAllowsOnlyExactRedPixel(t *testing.T) {
	fixture := func(payload string) []byte {
		exchanges := []RecordedExchange{{
			Provider:    "test",
			Method:      http.MethodPost,
			URL:         "https://api.example.test/chat",
			Status:      http.StatusOK,
			RequestBody: `{"image":"` + payload + `"}`,
		}}
		data, err := json.Marshal(exchanges)
		if err != nil {
			t.Fatalf("Marshal returned error: %v", err)
		}
		return data
	}
	if err := ValidateFixtureBytes("red-pixel", fixture(redPixelPNGBase64)); err != nil {
		t.Fatalf("exact red pixel rejected: %v", err)
	}
	if err := ValidateFixtureBytes("altered-red-pixel", fixture(redPixelPNGBase64+"A")); err == nil {
		t.Fatal("altered red pixel bypassed entropy guard")
	}
	otherBase64 := "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
	if err := ValidateFixtureBytes("other-image", fixture(otherBase64)); err == nil {
		t.Fatal("general base64 token bypassed entropy guard")
	}
}

func TestFixtureEntropyRejectsBase64URLAndRandomHex(t *testing.T) {
	fixture := func(payload string) []byte {
		exchanges := []RecordedExchange{{
			Provider:     "test",
			Method:       http.MethodPost,
			URL:          "https://api.example.test/chat",
			Status:       http.StatusOK,
			ResponseBody: `{"novel_metadata":"` + payload + `"}`,
		}}
		data, err := json.Marshal(exchanges)
		if err != nil {
			t.Fatalf("Marshal returned error: %v", err)
		}
		return data
	}
	for name, payload := range map[string]string{
		"base64url":             "AbCdEfGhIjKlMnOp" + "QrStUvWxYz012345" + "6789_-ABCD",
		"hex":                   "4f8c2e9a7b1d6f0c" + "3e5a8b2d9f1c7e4a" + "6b0d3f9c2a5e8b1d" + "7f4c0a6e3b9d2f5",
		"low_entropy_base64":    "AAAAAAAAAAAAAAAA" + "AAAAAAAAAAAA/w==",
		"standard_base64_slash": "YYpekL8jKkdZ0i" + "/czlVBLhiRUBt5bx" + "/Q",
	} {
		t.Run(name, func(t *testing.T) {
			if err := ValidateFixtureBytes(name, fixture(payload)); err == nil {
				t.Fatalf("%s payload bypassed entropy guard", name)
			}
		})
	}
	if err := ValidateFixtureBytes("model-id", fixture("NVIDIA-Nemotron-3-Super-120B-A12B-FP8")); err != nil {
		t.Fatalf("ordinary model identifier was rejected: %v", err)
	}
}

func TestFixtureEntropyScansDecodedURLValuesWithoutTreatingRoutesAsTokens(t *testing.T) {
	const token = "AAAAAAAAAAAAAAAAAAAAAAAAAAAA/w=="
	fixture := func(rawURL string) []byte {
		data, err := json.Marshal([]RecordedExchange{{
			Provider: "test",
			Method:   http.MethodGet,
			URL:      rawURL,
			Status:   http.StatusOK,
		}})
		if err != nil {
			t.Fatalf("Marshal returned error: %v", err)
		}
		return data
	}
	for name, rawURL := range map[string]string{
		"encoded_path_segment": "https://api.example.test/v1/" + url.PathEscape(token),
		"query_value":          "https://api.example.test/v1/chat?cursor=" + url.QueryEscape(token),
	} {
		t.Run(name, func(t *testing.T) {
			if err := ValidateFixtureBytes(name, fixture(rawURL)); err == nil {
				t.Fatalf("URL token bypassed entropy guard: %s", rawURL)
			}
		})
	}
	if err := ValidateFixtureBytes("ordinary-route", fixture("https://api.example.test/api/v1/models/perceptron/perceptron")); err != nil {
		t.Fatalf("ordinary URL route was rejected: %v", err)
	}
}

func TestFixtureEntropyDistinguishesSlashDelimitedPathsFromBase64Tokens(t *testing.T) {
	fixture := func(payload string) []byte {
		data, err := json.Marshal([]RecordedExchange{{
			Provider:     "test",
			Method:       http.MethodGet,
			URL:          "https://api.example.test/models",
			Status:       http.StatusOK,
			ResponseBody: payload,
		}})
		if err != nil {
			t.Fatalf("Marshal returned error: %v", err)
		}
		return data
	}
	const openRouterPath = `/api/v1/models/perceptron/perceptron`
	if err := ValidateFixtureBytes("openrouter-path", fixture(`{"details":"`+openRouterPath+`"}`)); err != nil {
		t.Fatalf("ordinary slash-delimited path was rejected: %v", err)
	}
	const slashBearingBase64 = "AAAAAAAAAAAAAAAAAAAAAAAAAAAA/w=="
	if err := ValidateFixtureBytes("slash-bearing-base64", fixture(`{"value":"`+slashBearingBase64+`"}`)); err == nil {
		t.Fatal("standalone decode-valid slash-bearing Base64 token passed validation")
	}
}

func TestRedactCaptureHandlesBarePrivateHostFieldsStructurally(t *testing.T) {
	capture := llm.WireCapture{
		Provider:     "test",
		Method:       http.MethodPost,
		URL:          "https://api.example.test/chat",
		RequestBody:  []byte(`{"origin_host":"pax","message":"pax","public_host":"api.example.test"}`),
		Status:       http.StatusOK,
		ResponseBody: []byte("data: {\"backend_host\":\ndata: \"worker\",\"text\":\"worker\"}\n\n"),
	}
	redacted := RedactCapture(capture)
	var request map[string]any
	if err := json.Unmarshal([]byte(redacted.RequestBody), &request); err != nil {
		t.Fatalf("Unmarshal request body returned error: %v", err)
	}
	if request["origin_host"] != fixtureHostname || request["message"] != "pax" || request["public_host"] != "api.example.test" {
		t.Fatalf("structurally redacted request = %+v", request)
	}
	if !strings.Contains(redacted.ResponseBody, `"backend_host":"fixture.invalid"`) || !strings.Contains(redacted.ResponseBody, `"text":"worker"`) {
		t.Fatalf("structurally redacted SSE = %s", redacted.ResponseBody)
	}

	fixture := func(body string) []byte {
		data, err := json.Marshal([]RecordedExchange{{
			Provider:    "test",
			Method:      http.MethodPost,
			URL:         "https://api.example.test/chat",
			Status:      http.StatusOK,
			RequestBody: body,
		}})
		if err != nil {
			t.Fatalf("Marshal returned error: %v", err)
		}
		return data
	}
	if err := ValidateFixtureBytes("private-host", fixture(`{"origin_host":"pax"}`)); err == nil {
		t.Fatal("bare private hostname in host-shaped field passed validation")
	}
	multiline, err := json.Marshal([]RecordedExchange{{
		Provider:     "test",
		Method:       http.MethodPost,
		URL:          "https://api.example.test/chat",
		Status:       http.StatusOK,
		ResponseBody: "data: {\"origin_host\":\ndata: \"pax\"}\n\n",
	}})
	if err != nil {
		t.Fatalf("Marshal multiline fixture returned error: %v", err)
	}
	if err := ValidateFixtureBytes("multiline-private-host", multiline); err == nil {
		t.Fatal("multiline SSE private hostname passed validation")
	}
	if err := ValidateFixtureBytes("ordinary-word", fixture(`{"message":"pax","origin_host":"api.example.test"}`)); err != nil {
		t.Fatalf("ordinary word or public host was rejected: %v", err)
	}
}

func TestFixtureRejectsForgedMockTokens(t *testing.T) {
	fixture := func(body string) []byte {
		data, err := json.Marshal([]RecordedExchange{{
			Provider:     "test",
			Method:       http.MethodPost,
			URL:          "https://api.example.test/chat",
			Status:       http.StatusOK,
			ResponseBody: body,
		}})
		if err != nil {
			t.Fatalf("Marshal returned error: %v", err)
		}
		return data
	}
	for _, forged := range []string{
		"MOCK-random-token-with-hyphens",
		"MOCK-REQ-0",
		"MOCK-REQ-0001",
	} {
		if err := ValidateFixtureBytes("forged-mock", fixture(`{"request_id":"`+forged+`"}`)); err == nil {
			t.Fatalf("forged placeholder %q passed validation", forged)
		}
	}
	if err := ValidateFixtureBytes("sequential-mock", fixture(`{"request_id":"MOCK-REQ-1"}`)); err != nil {
		t.Fatalf("canonical sequential placeholder was rejected: %v", err)
	}
	redacted := RedactCapture(llm.WireCapture{
		Provider:     "test",
		Method:       http.MethodPost,
		URL:          "https://api.example.test/chat",
		Status:       http.StatusOK,
		ResponseBody: []byte(`{"request_id":"MOCK-random-token-with-hyphens"}`),
	})
	if strings.Contains(redacted.ResponseBody, "random-token") || !strings.Contains(redacted.ResponseBody, "MOCK-REQUEST-ID-1") {
		t.Fatalf("forged mock was not migrated: %s", redacted.ResponseBody)
	}
}

func TestRedactCaptureRedactsPercentEncodedURLPathSecret(t *testing.T) {
	knownValue := "path-secret"
	rawURL := "https://api.example.test/v1/path%2Dsecret/keep%2Fslash"
	redacted := RedactCapture(llm.WireCapture{
		Provider: "test",
		Method:   http.MethodGet,
		URL:      rawURL,
		Status:   http.StatusOK,
	}, knownValue)
	if strings.Contains(redacted.URL, "path%2Dsecret") || !strings.Contains(redacted.URL, "/v1/%5BREDACTED%5D/keep%2Fslash") {
		t.Fatalf("redacted URL path = %q", redacted.URL)
	}
	data, err := json.Marshal([]RecordedExchange{{
		Provider: "test",
		Method:   http.MethodGet,
		URL:      rawURL,
		Status:   http.StatusOK,
	}})
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}
	if err := ValidateFixtureBytes("encoded-path", data, knownValue); err == nil {
		t.Fatal("percent-encoded URL path secret passed validation")
	}
}

func TestRedactCaptureAssignsMocksDeterministicallyAcrossMaps(t *testing.T) {
	newCapture := func(reverse bool) llm.WireCapture {
		headers := make(http.Header)
		if reverse {
			headers.Set("X-B", "https://api.example.test/callback?request_id=req_header_b")
			headers.Set("X-A", "https://api.example.test/callback?request_id=req_header_a")
		} else {
			headers.Set("X-A", "https://api.example.test/callback?request_id=req_header_a")
			headers.Set("X-B", "https://api.example.test/callback?request_id=req_header_b")
		}
		return llm.WireCapture{
			Provider:       "test",
			Method:         http.MethodGet,
			URL:            "https://api.example.test/chat?response_id=req_query_b&request_id=req_query_a",
			RequestHeaders: headers,
			Status:         http.StatusOK,
		}
	}
	first := RedactCapture(newCapture(false))
	second := RedactCapture(newCapture(true))
	if first.URL != second.URL || first.RequestHeaders.Get("X-A") != second.RequestHeaders.Get("X-A") || first.RequestHeaders.Get("X-B") != second.RequestHeaders.Get("X-B") {
		t.Fatalf("redaction depends on map insertion order:\nfirst=%+v\nsecond=%+v", first, second)
	}
	for got, want := range map[string]string{
		first.URL:                       "request_id=MOCK-REQ-1&response_id=MOCK-REQ-2",
		first.RequestHeaders.Get("X-A"): "request_id=MOCK-REQ-3",
		first.RequestHeaders.Get("X-B"): "request_id=MOCK-REQ-4",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("deterministic redaction %q missing %q", got, want)
		}
	}
}

func TestRedactRecordedExchangeMigratesLegacyMockIDsSequentially(t *testing.T) {
	redactor := newFixtureRedactor()
	exchange := RecordedExchange{
		Provider: "test",
		Method:   http.MethodPost,
		URL:      "https://api.example.test/chat?request_id=MOCK-REQ-ABCDEF012345",
		Status:   http.StatusOK,
		ResponseBody: `{"id":"MOCK-CHATCMPL-111111AAAAAA","response_id":"MOCK-CHATCMPL-111111AAAAAA",` +
			`"item_id":"MOCK-CHATCMPL-222222BBBBBB"}`,
	}
	got := redactor.redactExchange(exchange)
	joined := got.URL + "\n" + got.ResponseBody
	for _, want := range []string{"MOCK-REQ-1", "MOCK-CHATCMPL-1", "MOCK-CHATCMPL-2"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("migrated exchange missing %q: %s", want, joined)
		}
	}
	if strings.Contains(joined, "ABCDEF012345") || strings.Contains(joined, "111111AAAAAA") || strings.Contains(joined, "222222BBBBBB") {
		t.Fatalf("legacy mock hash survived migration: %s", joined)
	}
	if strings.Count(joined, "MOCK-CHATCMPL-1") != 2 {
		t.Fatalf("legacy correlation was not preserved: %s", joined)
	}
	idempotent := newFixtureRedactor().redactExchange(got)
	if idempotent.URL != got.URL || idempotent.ResponseBody != got.ResponseBody {
		t.Fatalf("sequential migration was not idempotent:\nfirst: %+v\nsecond: %+v", got, idempotent)
	}
}

func TestWireTapOutstandingResponseBlocksFixtureReplacement(t *testing.T) {
	path := filepath.Join(t.TempDir(), "live.json")
	original := []byte(`[{"provider":"old","method":"POST","url":"https://api.example.test","status":200}]`)
	if err := os.WriteFile(path, original, 0o644); err != nil {
		t.Fatalf("WriteFile returned error: %v", err)
	}
	captures := &CaptureLog{}
	transport := roundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusOK,
			ContentLength: -1,
			Body:          io.NopCloser(strings.NewReader(`{"ok":true}`)),
		}, nil
	})
	firstClient := &http.Client{Transport: llm.NewWireTap(transport, "test-sdk", captures.Capture)}
	secondClient := &http.Client{Transport: llm.NewWireTap(transport, "test-stream", captures.Capture)}
	ctx := RecordingContext(context.Background(), captures, NewSecretSet())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://api.example.test/first", nil)
	if err != nil {
		t.Fatalf("NewRequestWithContext returned error: %v", err)
	}
	firstResponse, err := firstClient.Do(req)
	if err != nil {
		t.Fatalf("first Do returned error: %v", err)
	}
	secondReq, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://api.example.test/second", nil)
	if err != nil {
		t.Fatalf("second NewRequestWithContext returned error: %v", err)
	}
	secondResponse, err := secondClient.Do(secondReq)
	if err != nil {
		t.Fatalf("second Do returned error: %v", err)
	}
	t.Cleanup(func() {
		_ = firstResponse.Body.Close()
		_ = secondResponse.Body.Close()
	})

	snapshot := captures.Snapshot()
	if len(snapshot.Captures) != 0 || snapshot.OutstandingResponseBodies != 2 {
		t.Fatalf("snapshot = %+v, want both transports' first responses visible as outstanding", snapshot)
	}
	result, err := WriteFixtureChecked(path, snapshot.Captures, FixtureWriteOptions{
		OutstandingResponseBodies: snapshot.OutstandingResponseBodies,
	})
	if err != nil {
		t.Fatalf("WriteFixtureChecked returned error: %v", err)
	}
	if result.Replaced || !containsReason(result.Incomplete, "outstanding") {
		t.Fatalf("result = %+v, want outstanding response warning", result)
	}
	unchanged, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile returned error: %v", err)
	}
	if !bytes.Equal(unchanged, original) {
		t.Fatalf("outstanding response changed fixture:\n%s", unchanged)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func containsReason(reasons []string, want string) bool {
	for _, reason := range reasons {
		if strings.Contains(reason, want) {
			return true
		}
	}
	return false
}

func rewriteFixtureRedactions(t *testing.T, path string, data []byte) []byte {
	t.Helper()
	var exchanges []RecordedExchange
	if err := json.Unmarshal(data, &exchanges); err != nil {
		t.Fatalf("Unmarshal fixture returned error: %v", err)
	}
	redactor := newFixtureRedactor()
	for i := range exchanges {
		exchanges[i] = redactor.redactExchange(exchanges[i])
	}
	raw, err := json.MarshalIndent(exchanges, "", "  ")
	if err != nil {
		t.Fatalf("MarshalIndent returned error: %v", err)
	}
	raw = append(raw, '\n')
	if err := stageAndReplaceFixture(path, raw, nil); err != nil {
		t.Fatalf("stageAndReplaceFixture returned error: %v", err)
	}
	return raw
}

func assertFixtureRedacted(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := ValidateFixtureBytes(path, data); err != nil {
		t.Fatal(err)
	}
}
