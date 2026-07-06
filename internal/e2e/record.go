package e2e

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

var tokenRedactionPatterns = []*regexp.Regexp{
	regexp.MustCompile(`sk-ant-[A-Za-z0-9._-]+`),
	regexp.MustCompile(`sk-[A-Za-z0-9._-]{16,}`),
}

var bearerPattern = regexp.MustCompile(`(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{16,}`)

var credentialJSONPattern = regexp.MustCompile(`(?i)("(api_key|key|authorization|x-api-key|access|refresh|access_token|refresh_token|accountId|chatgpt-account-id)"\s*:\s*")([^"]+)(")`)

var mockJSONPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)("(safety_identifier|prompt_cache_key|encrypted_content|signature|obfuscation|user|user_id|end_user_id)"\s*:\s*")([^"]+)(")`),
	regexp.MustCompile(`(?i)("(id|item_id|response_id|previous_response_id|request_id|call_id|tool_call_id|tool_use_id)"\s*:\s*")((?:resp|msg|req|rs|fc|call|toolu|gen|chatcmpl)[A-Za-z0-9._:-]+)(")`),
}

var credentialQueryPattern = regexp.MustCompile(`(?i)([?&](api_key|key|access_token|refresh_token|access|refresh|authorization)=)([^&\s"']+)`)
var mockQueryPattern = regexp.MustCompile(`(?i)([?&](prompt_cache_key|safety_identifier|user|user_id|end_user_id|request_id|response_id|previous_response_id|call_id|tool_call_id|tool_use_id)=)([^&\s"']+)`)
var userIdentifierPattern = regexp.MustCompile(`(?i)(^|[^A-Za-z0-9-])(user-[A-Za-z0-9_-]{8,})`)

var sensitiveHeaderNames = map[string]struct{}{
	"authorization":             {},
	"proxy-authorization":       {},
	"x-api-key":                 {},
	"api-key":                   {},
	"cookie":                    {},
	"set-cookie":                {},
	"chatgpt-account-id":        {},
	"anthropic-organization-id": {},
	"cf-ray":                    {},
	"nel":                       {},
	"openai-organization":       {},
	"openai-project":            {},
	"report-to":                 {},
	"request-id":                {},
	"traceresponse":             {},
	"x-generation-id":           {},
	"x-oai-request-id":          {},
	"x-request-id":              {},
	"x-stainless-retry-count":   {},
}

var sensitiveHeaderPrefixes = []string{
	"anthropic-ratelimit-",
	"x-codex-",
}

// RecordedExchange is a redacted wire capture safe to commit.
type RecordedExchange struct {
	Provider        string      `json:"provider"`
	Method          string      `json:"method"`
	URL             string      `json:"url"`
	RequestHeaders  http.Header `json:"request_headers,omitempty"`
	RequestBody     string      `json:"request_body,omitempty"`
	Status          int         `json:"status"`
	ResponseHeaders http.Header `json:"response_headers,omitempty"`
	ResponseBody    string      `json:"response_body,omitempty"`
	StartedAt       time.Time   `json:"started_at"`
	DurationMS      int64       `json:"duration_ms"`
	Err             string      `json:"err,omitempty"`
}

// RedactBytes removes known secrets and secret-looking values from b.
func RedactBytes(b []byte, secrets ...string) []byte {
	out := append([]byte(nil), b...)
	for _, secret := range secrets {
		if secret == "" {
			continue
		}
		out = bytes.ReplaceAll(out, []byte(secret), []byte("[REDACTED]"))
	}
	for _, pattern := range tokenRedactionPatterns {
		out = pattern.ReplaceAll(out, []byte("[REDACTED]"))
	}
	out = bearerPattern.ReplaceAll(out, []byte("${1}[REDACTED]"))
	out = replaceJSONValue(out, credentialJSONPattern, func(_, value string) string {
		if isPlaceholder(value) {
			return value
		}
		return "[REDACTED]"
	})
	for _, pattern := range mockJSONPatterns {
		out = replaceJSONValue(out, pattern, mockValue)
	}
	out = replaceQueryValue(out, credentialQueryPattern, func(_, value string) string {
		if isPlaceholder(value) {
			return value
		}
		return "[REDACTED]"
	})
	out = replaceQueryValue(out, mockQueryPattern, mockValue)
	out = userIdentifierPattern.ReplaceAllFunc(out, func(match []byte) []byte {
		parts := userIdentifierPattern.FindSubmatch(match)
		if len(parts) != 3 {
			return match
		}
		value := string(parts[2])
		if isPlaceholder(value) {
			return match
		}
		out := append([]byte(nil), parts[1]...)
		out = append(out, mockValue("user", value)...)
		return out
	})
	return out
}

func replaceJSONValue(b []byte, pattern *regexp.Regexp, replace func(field, value string) string) []byte {
	return pattern.ReplaceAllFunc(b, func(match []byte) []byte {
		parts := pattern.FindSubmatch(match)
		if len(parts) != 5 {
			return match
		}
		out := append([]byte(nil), parts[1]...)
		out = append(out, replace(string(parts[2]), string(parts[3]))...)
		out = append(out, parts[4]...)
		return out
	})
}

func replaceQueryValue(b []byte, pattern *regexp.Regexp, replace func(field, value string) string) []byte {
	return pattern.ReplaceAllFunc(b, func(match []byte) []byte {
		parts := pattern.FindSubmatch(match)
		if len(parts) != 4 {
			return match
		}
		out := append([]byte(nil), parts[1]...)
		out = append(out, replace(string(parts[2]), string(parts[3]))...)
		return out
	})
}

func mockValue(field, value string) string {
	if isPlaceholder(value) {
		return value
	}
	sum := sha256.Sum256([]byte(value))
	encoded := strings.ToUpper(hex.EncodeToString(sum[:]))
	return "MOCK-" + mockKind(field, value) + "-" + encoded[:12]
}

func isPlaceholder(value string) bool {
	return value == "[REDACTED]" || strings.HasPrefix(value, "MOCK-")
}

func mockKind(field, value string) string {
	lowerValue := strings.ToLower(value)
	switch {
	case strings.HasPrefix(lowerValue, "resp"):
		return "RESP"
	case strings.HasPrefix(lowerValue, "msg"):
		return "MSG"
	case strings.HasPrefix(lowerValue, "req"):
		return "REQ"
	case strings.HasPrefix(lowerValue, "rs"):
		return "REASONING"
	case strings.HasPrefix(lowerValue, "fc"),
		strings.HasPrefix(lowerValue, "call"),
		strings.HasPrefix(lowerValue, "toolu"),
		strings.HasPrefix(lowerValue, "chatcmpl-tool"):
		return "TOOL-CALL"
	case strings.HasPrefix(lowerValue, "gen"):
		return "GEN"
	case strings.HasPrefix(lowerValue, "chatcmpl"):
		return "CHATCMPL"
	case strings.HasPrefix(lowerValue, "user-"):
		return "USER"
	}

	field = strings.ToLower(field)
	switch {
	case strings.Contains(field, "cache"):
		return "CACHE"
	case strings.Contains(field, "safety") || field == "user" || strings.HasSuffix(field, "user_id"):
		return "USER"
	case strings.Contains(field, "encrypted"):
		return "ENCRYPTED"
	case strings.Contains(field, "signature"):
		return "SIGNATURE"
	case strings.Contains(field, "obfuscation"):
		return "OBFUSCATION"
	default:
		return sanitizeMockKind(field)
	}
}

func sanitizeMockKind(field string) string {
	var b strings.Builder
	for _, r := range strings.ToUpper(field) {
		switch {
		case r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '_' || r == '-':
			b.WriteByte('-')
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "VALUE"
	}
	return out
}

// RedactHeaders returns a copy of h with credential headers redacted.
func RedactHeaders(h http.Header, secrets ...string) http.Header {
	out := make(http.Header, len(h))
	for name, values := range h {
		copied := append([]string(nil), values...)
		if isSensitiveHeaderName(name) {
			for i := range copied {
				copied[i] = "[REDACTED]"
			}
		} else {
			for i, value := range copied {
				copied[i] = string(RedactBytes([]byte(value), secrets...))
			}
		}
		out[name] = copied
	}
	return out
}

func isSensitiveHeaderName(name string) bool {
	name = strings.ToLower(name)
	if _, ok := sensitiveHeaderNames[name]; ok {
		return true
	}
	for _, prefix := range sensitiveHeaderPrefixes {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

// RedactRecordedExchange redacts a previously recorded exchange. It is used
// when strengthening redaction rules for an existing fixture corpus.
func RedactRecordedExchange(e RecordedExchange, secrets ...string) RecordedExchange {
	e.URL = string(RedactBytes([]byte(e.URL), secrets...))
	e.RequestHeaders = RedactHeaders(e.RequestHeaders, secrets...)
	e.RequestBody = string(RedactBytes([]byte(e.RequestBody), secrets...))
	e.ResponseHeaders = RedactHeaders(e.ResponseHeaders, secrets...)
	e.ResponseBody = string(RedactBytes([]byte(e.ResponseBody), secrets...))
	e.Err = string(RedactBytes([]byte(e.Err), secrets...))
	return e
}

// RedactCapture converts a WireCapture to a commit-safe recorded exchange.
func RedactCapture(c llm.WireCapture, secrets ...string) RecordedExchange {
	out := RecordedExchange{
		Provider:        c.Provider,
		Method:          c.Method,
		URL:             string(RedactBytes([]byte(c.URL), secrets...)),
		RequestHeaders:  RedactHeaders(c.RequestHeaders, secrets...),
		RequestBody:     string(RedactBytes(c.RequestBody, secrets...)),
		Status:          c.Status,
		ResponseHeaders: RedactHeaders(c.ResponseHeaders, secrets...),
		ResponseBody:    string(RedactBytes(c.ResponseBody, secrets...)),
		StartedAt:       c.StartedAt,
		DurationMS:      c.Duration.Milliseconds(),
	}
	if c.Err != nil {
		out.Err = string(RedactBytes([]byte(c.Err.Error()), secrets...))
	}
	return out
}

// WriteFixture writes redacted captures under internal/e2e/fixtures.
func WriteFixture(path string, captures []llm.WireCapture, secrets ...string) error {
	exchanges := make([]RecordedExchange, len(captures))
	for i, capture := range captures {
		exchanges[i] = RedactCapture(capture, secrets...)
	}
	raw, err := json.MarshalIndent(exchanges, "", "  ")
	if err != nil {
		return err
	}
	raw = RedactBytes(raw, secrets...)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, append(raw, '\n'), 0o644)
}
