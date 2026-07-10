package e2e

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

const fixtureHostname = "fixture.invalid"

var ErrIncompleteFixture = errors.New("incomplete fixture recording")

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
var privateURLPattern = regexp.MustCompile(`(?i)https?://[^\s"'<>]+`)
var privateAddressPattern = regexp.MustCompile(`(?i)\b(?:localhost|[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\.(?:local|internal)|(?:10|127)(?:\.[0-9]{1,3}){3}|192\.168(?:\.[0-9]{1,3}){2}|172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2})\b`)
var entropyTokenPattern = regexp.MustCompile(`[A-Za-z0-9_+=]{32,}`)
var standardBase64TokenPattern = regexp.MustCompile(`[A-Za-z0-9+/]{30,}={0,2}`)
var base64URLTokenPattern = regexp.MustCompile(`[A-Za-z0-9_-]{32,}`)
var hexSecretPattern = regexp.MustCompile(`(?i)\b[0-9a-f]{32,}\b`)
var legacyMockPattern = regexp.MustCompile(`^MOCK-([A-Z0-9]+(?:-[A-Z0-9]+)*)-([A-F0-9]{12})$`)
var sequentialMockPattern = regexp.MustCompile(`^MOCK-([A-Z0-9]+(?:-[A-Z0-9]+)*)-([1-9][0-9]*)$`)
var legacyMockFixturePattern = regexp.MustCompile(`MOCK-[A-Z0-9]+(?:-[A-Z0-9]+)*-[A-F0-9]{12}`)
var mockTokenPattern = regexp.MustCompile(`MOCK-[A-Za-z0-9_-]+`)

var credentialFieldNames = map[string]struct{}{
	"api_key":            {},
	"key":                {},
	"authorization":      {},
	"x-api-key":          {},
	"access":             {},
	"refresh":            {},
	"access_token":       {},
	"refresh_token":      {},
	"accountid":          {},
	"chatgpt-account-id": {},
}

var mockFieldNames = map[string]struct{}{
	"safety_identifier":    {},
	"prompt_cache_key":     {},
	"encrypted_content":    {},
	"signature":            {},
	"obfuscation":          {},
	"user":                 {},
	"user_id":              {},
	"end_user_id":          {},
	"id":                   {},
	"item_id":              {},
	"response_id":          {},
	"previous_response_id": {},
	"request_id":           {},
	"call_id":              {},
	"tool_call_id":         {},
	"tool_use_id":          {},
}

var sensitiveHeaderNames = map[string]struct{}{
	"authorization":             {},
	"proxy-authorization":       {},
	"x-api-key":                 {},
	"api-key":                   {},
	"cookie":                    {},
	"etag":                      {},
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
	"x-models-etag":             {},
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
	Incomplete      string      `json:"incomplete,omitempty"`
}

// FixtureWriteOptions controls completeness checks for a recording. Safety
// checks are unconditional and cannot be bypassed.
type FixtureWriteOptions struct {
	Secrets                   []string
	ExpectedScenarios         []string
	CompletedScenarios        []string
	OutstandingResponseBodies int64
	AllowIncomplete           bool
	Warnf                     func(format string, args ...any)
}

// FixtureWriteResult reports whether the tracked fixture was replaced.
type FixtureWriteResult struct {
	Replaced   bool
	Incomplete []string
}

// CaptureLog collects captures safely when requests finish concurrently.
type CaptureLog struct {
	mu       sync.Mutex
	captures []llm.WireCapture
	trackers map[llm.WireCaptureTracker]struct{}
}

func (l *CaptureLog) Capture(c llm.WireCapture) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.captures = append(l.captures, c)
}

func (l *CaptureLog) ObserveWireCaptureTracker(tracker llm.WireCaptureTracker) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.trackers == nil {
		l.trackers = make(map[llm.WireCaptureTracker]struct{})
	}
	l.trackers[tracker] = struct{}{}
}

type CaptureSnapshot struct {
	Captures                  []llm.WireCapture
	OutstandingResponseBodies int64
}

func (l *CaptureLog) Snapshot() CaptureSnapshot {
	l.mu.Lock()
	defer l.mu.Unlock()
	snapshot := CaptureSnapshot{Captures: append([]llm.WireCapture(nil), l.captures...)}
	for tracker := range l.trackers {
		snapshot.OutstandingResponseBodies += tracker.OutstandingResponseBodies()
	}
	return snapshot
}

// SecretSet holds both initial and rotated credentials for final redaction.
type SecretSet struct {
	mu     sync.RWMutex
	values map[string]struct{}
}

func NewSecretSet(values ...string) *SecretSet {
	s := &SecretSet{values: make(map[string]struct{})}
	s.Add(values...)
	return s
}

func (s *SecretSet) Add(values ...string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, value := range values {
		if value != "" {
			s.values[value] = struct{}{}
		}
	}
}

func (s *SecretSet) AddCredential(cred llm.AuthCredential) {
	s.Add(cred.Key, cred.Access, cred.Refresh, cred.AccountID)
}

func (s *SecretSet) Values() []string {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	values := make([]string, 0, len(s.values))
	for value := range s.values {
		values = append(values, value)
	}
	sort.Slice(values, func(i, j int) bool {
		if len(values[i]) != len(values[j]) {
			return len(values[i]) > len(values[j])
		}
		return values[i] < values[j]
	})
	return values
}

type fixtureRedactor struct {
	secrets  []string
	mocks    map[string]string
	counters map[string]int
}

func newFixtureRedactor(secrets ...string) *fixtureRedactor {
	set := NewSecretSet(secrets...)
	return &fixtureRedactor{
		secrets:  set.Values(),
		mocks:    make(map[string]string),
		counters: make(map[string]int),
	}
}

// RedactBytes removes known secrets and secret-looking values from b.
func RedactBytes(b []byte, secrets ...string) []byte {
	return newFixtureRedactor(secrets...).redactBody(b)
}

func (r *fixtureRedactor) redactBody(body []byte) []byte {
	if len(bytes.TrimSpace(body)) == 0 {
		return append([]byte(nil), body...)
	}
	if redacted, ok := r.redactJSON(body); ok {
		return redacted
	}
	if bytes.Contains(body, []byte("data:")) {
		return r.redactSSE(body)
	}
	return r.redactText(body)
}

func (r *fixtureRedactor) redactJSON(raw []byte) ([]byte, bool) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, false
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return nil, false
	}
	value = r.redactJSONValue("", value)
	out, err := json.Marshal(value)
	return out, err == nil
}

func (r *fixtureRedactor) redactJSONValue(field string, value any) any {
	switch typed := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			if strings.EqualFold(key, "id") && jsonObjectType(typed) == "model" {
				if id, ok := typed[key].(string); ok {
					typed[key] = string(r.redactText([]byte(id)))
					continue
				}
			}
			typed[key] = r.redactJSONValue(key, typed[key])
		}
		return typed
	case []any:
		for i, child := range typed {
			typed[i] = r.redactJSONValue(field, child)
		}
		return typed
	case string:
		lowerField := strings.ToLower(field)
		if _, ok := credentialFieldNames[lowerField]; ok && !isPlaceholder(typed) {
			return "[REDACTED]"
		}
		if _, ok := mockFieldNames[lowerField]; ok {
			return r.mockValue(field, typed)
		}
		if isHostShapedField(lowerField) {
			return r.redactHostValue(typed)
		}
		return string(r.redactText([]byte(typed)))
	default:
		return value
	}
}

func jsonObjectType(value map[string]any) string {
	for key, child := range value {
		if strings.EqualFold(key, "object") {
			objectType, _ := child.(string)
			return strings.ToLower(objectType)
		}
	}
	return ""
}

func (r *fixtureRedactor) redactSSE(raw []byte) []byte {
	lines := strings.Split(strings.ReplaceAll(string(raw), "\r\n", "\n"), "\n")
	out := make([]string, 0, len(lines))
	event := make([]string, 0, 4)
	flush := func() {
		if len(event) == 0 {
			return
		}
		payload, hasData := joinedSSEData(event)
		redactedJSON, isJSON := r.redactJSON([]byte(payload))
		emittedJSON := false
		for _, line := range event {
			value, isData := sseDataLineValue(line)
			if !isData {
				out = append(out, string(r.redactText([]byte(line))))
				continue
			}
			if hasData && isJSON {
				if !emittedJSON {
					out = append(out, "data: "+string(redactedJSON))
					emittedJSON = true
				}
				continue
			}
			redacted := string(r.redactText([]byte(value)))
			if redacted == "" {
				out = append(out, "data:")
			} else {
				out = append(out, "data: "+redacted)
			}
		}
		event = event[:0]
	}
	for _, line := range lines {
		if line == "" {
			flush()
			out = append(out, "")
			continue
		}
		event = append(event, line)
	}
	flush()
	return []byte(strings.Join(out, "\n"))
}

func sseDataLineValue(line string) (string, bool) {
	if line == "data" {
		return "", true
	}
	if !strings.HasPrefix(line, "data:") {
		return "", false
	}
	value := strings.TrimPrefix(line, "data:")
	if strings.HasPrefix(value, " ") {
		value = strings.TrimPrefix(value, " ")
	}
	return value, true
}

func joinedSSEData(lines []string) (string, bool) {
	values := make([]string, 0, len(lines))
	for _, line := range lines {
		if value, ok := sseDataLineValue(line); ok {
			values = append(values, value)
		}
	}
	return strings.Join(values, "\n"), len(values) > 0
}

func sseDataEvents(body []byte) []string {
	lines := strings.Split(strings.ReplaceAll(string(body), "\r\n", "\n"), "\n")
	events := make([]string, 0)
	event := make([]string, 0, 4)
	flush := func() {
		if payload, ok := joinedSSEData(event); ok {
			events = append(events, payload)
		}
		event = event[:0]
	}
	for _, line := range lines {
		if line == "" {
			flush()
			continue
		}
		event = append(event, line)
	}
	flush()
	return events
}

func (r *fixtureRedactor) redactText(b []byte) []byte {
	return r.redactTextWithURLs(b, true)
}

func (r *fixtureRedactor) redactTextWithURLs(b []byte, redactURLs bool) []byte {
	out := append([]byte(nil), b...)
	for _, secret := range r.secrets {
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
		out = replaceJSONValue(out, pattern, r.mockValue)
	}
	out = replaceQueryValue(out, credentialQueryPattern, func(_, value string) string {
		if isPlaceholder(value) {
			return value
		}
		return "[REDACTED]"
	})
	out = replaceQueryValue(out, mockQueryPattern, r.mockValue)
	out = userIdentifierPattern.ReplaceAllFunc(out, func(match []byte) []byte {
		parts := userIdentifierPattern.FindSubmatch(match)
		if len(parts) != 3 || isPlaceholder(string(parts[2])) {
			return match
		}
		replacement := append([]byte(nil), parts[1]...)
		return append(replacement, r.mockValue("user", string(parts[2]))...)
	})
	if redactURLs {
		out = privateURLPattern.ReplaceAllFunc(out, func(match []byte) []byte {
			return []byte(r.redactURL(string(match)))
		})
		out = privateAddressPattern.ReplaceAllFunc(out, func(match []byte) []byte {
			if isPrivateHostname(string(match)) {
				return []byte(fixtureHostname)
			}
			return match
		})
	}
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
		return append(out, parts[4]...)
	})
}

func replaceQueryValue(b []byte, pattern *regexp.Regexp, replace func(field, value string) string) []byte {
	return pattern.ReplaceAllFunc(b, func(match []byte) []byte {
		parts := pattern.FindSubmatch(match)
		if len(parts) != 4 {
			return match
		}
		out := append([]byte(nil), parts[1]...)
		return append(out, replace(string(parts[2]), string(parts[3]))...)
	})
}

func (r *fixtureRedactor) mockValue(field, value string) string {
	if value == "[REDACTED]" {
		return value
	}
	if parts := legacyMockPattern.FindStringSubmatch(value); len(parts) == 3 {
		return r.sequentialMock(strings.ToUpper(parts[1]), "legacy\x00"+value)
	}
	if parts := sequentialMockPattern.FindStringSubmatch(value); len(parts) == 3 {
		kind := strings.ToUpper(parts[1])
		sequence, err := strconv.Atoi(parts[2])
		if err == nil && sequence > r.counters[kind] {
			r.counters[kind] = sequence
		}
		return value
	}
	if strings.HasPrefix(value, "MOCK-") {
		kind := mockKind(field, "")
		return r.sequentialMock(kind, kind+"\x00"+value)
	}
	kind := mockKind(field, value)
	return r.sequentialMock(kind, kind+"\x00"+value)
}

func (r *fixtureRedactor) sequentialMock(kind, key string) string {
	if mock, ok := r.mocks[key]; ok {
		return mock
	}
	r.counters[kind]++
	mock := fmt.Sprintf("MOCK-%s-%d", kind, r.counters[kind])
	r.mocks[key] = mock
	return mock
}

func isPlaceholder(value string) bool {
	return value == "[REDACTED]" || sequentialMockPattern.MatchString(value)
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
	case strings.HasPrefix(lowerValue, "fc"), strings.HasPrefix(lowerValue, "call"), strings.HasPrefix(lowerValue, "toolu"), strings.HasPrefix(lowerValue, "chatcmpl-tool"):
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

func (r *fixtureRedactor) redactURL(raw string) string {
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return string(r.redactTextWithoutURLs([]byte(raw)))
	}
	if u.User != nil {
		u.User = url.User("REDACTED")
	}
	hostname := u.Hostname()
	if isPrivateHostname(hostname) {
		if port := u.Port(); port != "" {
			u.Host = net.JoinHostPort(fixtureHostname, port)
		} else {
			u.Host = fixtureHostname
		}
	}
	r.redactURLPath(u)
	query := u.Query()
	queryKeys := make([]string, 0, len(query))
	for key := range query {
		queryKeys = append(queryKeys, key)
	}
	sort.Strings(queryKeys)
	for _, key := range queryKeys {
		values := query[key]
		lowerKey := strings.ToLower(key)
		for i, value := range values {
			switch {
			case isCredentialQueryKey(lowerKey):
				if !isPlaceholder(value) {
					values[i] = "[REDACTED]"
				}
			case isMockQueryKey(lowerKey):
				values[i] = r.mockValue(key, value)
			default:
				values[i] = string(r.redactTextWithoutURLs([]byte(value)))
			}
		}
		query[key] = values
	}
	u.RawQuery = query.Encode()
	u.Fragment = string(r.redactTextWithoutURLs([]byte(u.Fragment)))
	return u.String()
}

func (r *fixtureRedactor) redactURLPath(u *url.URL) {
	escapedPath := u.EscapedPath()
	if escapedPath == "" {
		return
	}
	segments := strings.Split(escapedPath, "/")
	changed := false
	for i, segment := range segments {
		decoded, err := url.PathUnescape(segment)
		if err != nil {
			continue
		}
		redacted := string(r.redactTextWithoutURLs([]byte(decoded)))
		if redacted == decoded {
			continue
		}
		segments[i] = url.PathEscape(redacted)
		changed = true
	}
	if !changed {
		return
	}
	rawPath := strings.Join(segments, "/")
	path, err := url.PathUnescape(rawPath)
	if err != nil {
		return
	}
	u.Path = path
	u.RawPath = rawPath
}

func (r *fixtureRedactor) redactTextWithoutURLs(b []byte) []byte {
	return r.redactTextWithURLs(b, false)
}

func isCredentialQueryKey(key string) bool {
	switch key {
	case "api_key", "key", "access_token", "refresh_token", "access", "refresh", "authorization":
		return true
	default:
		return false
	}
}

func isMockQueryKey(key string) bool {
	switch key {
	case "prompt_cache_key", "safety_identifier", "user", "user_id", "end_user_id", "request_id", "response_id", "previous_response_id", "call_id", "tool_call_id", "tool_use_id":
		return true
	default:
		return false
	}
}

func isPrivateHostname(hostname string) bool {
	hostname = strings.TrimSuffix(strings.ToLower(hostname), ".")
	if hostname == "" || hostname == fixtureHostname {
		return false
	}
	if ip := net.ParseIP(hostname); ip != nil {
		return ip.IsPrivate() || ip.IsLoopback() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsUnspecified()
	}
	return hostname == "localhost" || strings.HasSuffix(hostname, ".localhost") || strings.HasSuffix(hostname, ".local") || strings.HasSuffix(hostname, ".internal") || !strings.Contains(hostname, ".")
}

func isHostShapedField(field string) bool {
	field = strings.ToLower(field)
	return field == "host" || field == "hostname" || strings.HasSuffix(field, "_host") || strings.HasSuffix(field, "-host") || strings.HasSuffix(field, "_hostname") || strings.HasSuffix(field, "-hostname")
}

func (r *fixtureRedactor) redactHostValue(value string) string {
	redacted := string(r.redactText([]byte(value)))
	hostname, port, ok := splitHostValue(redacted)
	if !ok || !isPrivateHostname(hostname) {
		return redacted
	}
	if port != "" {
		return net.JoinHostPort(fixtureHostname, port)
	}
	return fixtureHostname
}

func splitHostValue(value string) (hostname, port string, ok bool) {
	value = strings.TrimSpace(value)
	if value == "" || strings.ContainsAny(value, "/?#@ ") {
		return "", "", false
	}
	if host, parsedPort, err := net.SplitHostPort(value); err == nil {
		return strings.Trim(host, "[]"), parsedPort, true
	}
	if strings.Count(value, ":") > 1 {
		return strings.Trim(value, "[]"), "", net.ParseIP(strings.Trim(value, "[]")) != nil
	}
	return value, "", true
}

// RedactHeaders returns a copy of h with credential headers redacted.
func RedactHeaders(h http.Header, secrets ...string) http.Header {
	return newFixtureRedactor(secrets...).redactHeaders(h)
}

func (r *fixtureRedactor) redactHeaders(h http.Header) http.Header {
	out := make(http.Header, len(h))
	names := make([]string, 0, len(h))
	for name := range h {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		values := h[name]
		copied := append([]string(nil), values...)
		if isSensitiveHeaderName(name) {
			for i := range copied {
				copied[i] = "[REDACTED]"
			}
		} else {
			for i, value := range copied {
				copied[i] = string(r.redactText([]byte(value)))
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
	return newFixtureRedactor(secrets...).redactExchange(e)
}

func (r *fixtureRedactor) redactExchange(e RecordedExchange) RecordedExchange {
	e.URL = r.redactURL(e.URL)
	e.RequestHeaders = r.redactHeaders(e.RequestHeaders)
	e.RequestBody = string(r.redactBody([]byte(e.RequestBody)))
	e.ResponseHeaders = r.redactHeaders(e.ResponseHeaders)
	e.ResponseBody = string(r.redactBody([]byte(e.ResponseBody)))
	e.Err = string(r.redactText([]byte(e.Err)))
	return e
}

// RedactCapture converts a WireCapture to a commit-safe recorded exchange.
func RedactCapture(c llm.WireCapture, secrets ...string) RecordedExchange {
	return newFixtureRedactor(secrets...).redactCapture(c)
}

func (r *fixtureRedactor) redactCapture(c llm.WireCapture) RecordedExchange {
	out := RecordedExchange{
		Provider:        c.Provider,
		Method:          c.Method,
		URL:             r.redactURL(c.URL),
		RequestHeaders:  r.redactHeaders(c.RequestHeaders),
		RequestBody:     string(r.redactBody(c.RequestBody)),
		Status:          c.Status,
		ResponseHeaders: r.redactHeaders(c.ResponseHeaders),
		ResponseBody:    string(r.redactBody(c.ResponseBody)),
		StartedAt:       c.StartedAt,
		DurationMS:      c.Duration.Milliseconds(),
	}
	if c.Err != nil {
		out.Err = string(r.redactText([]byte(c.Err.Error())))
	}
	if c.ResponseIncomplete {
		out.Incomplete = "response body closed before EOF"
	}
	return out
}

// WriteFixture writes a complete, redacted fixture atomically. Callers that
// intentionally record only selected scenarios should use WriteFixtureChecked.
func WriteFixture(path string, captures []llm.WireCapture, secrets ...string) error {
	result, err := WriteFixtureChecked(path, captures, FixtureWriteOptions{Secrets: secrets})
	if err != nil {
		return err
	}
	if !result.Replaced {
		return fmt.Errorf("%w: %s", ErrIncompleteFixture, strings.Join(result.Incomplete, "; "))
	}
	return nil
}

// WriteFixtureChecked stages, validates, and atomically replaces a fixture.
// An incomplete recording is retained only with explicit acknowledgement.
func WriteFixtureChecked(path string, captures []llm.WireCapture, options FixtureWriteOptions) (FixtureWriteResult, error) {
	incomplete, err := fixtureIncompleteness(path, captures, options)
	if err != nil {
		return FixtureWriteResult{}, err
	}
	result := FixtureWriteResult{Incomplete: incomplete}
	if len(incomplete) > 0 {
		if options.Warnf != nil {
			if options.AllowIncomplete {
				options.Warnf("WARNING: incomplete fixture recording for %s: %s; replacing because -record-allow-incomplete was explicitly set", path, strings.Join(incomplete, "; "))
			} else {
				options.Warnf("WARNING: incomplete fixture recording for %s: %s; fixture left unchanged (rerun with -record-allow-incomplete to acknowledge an intentional partial replacement)", path, strings.Join(incomplete, "; "))
			}
		}
		if !options.AllowIncomplete {
			return result, nil
		}
	}

	redactor := newFixtureRedactor(options.Secrets...)
	exchanges := make([]RecordedExchange, len(captures))
	for i, capture := range captures {
		exchanges[i] = redactor.redactCapture(capture)
	}
	raw, err := json.MarshalIndent(exchanges, "", "  ")
	if err != nil {
		return result, err
	}
	raw = append(raw, '\n')
	if err := stageAndReplaceFixture(path, raw, options.Secrets); err != nil {
		return result, err
	}
	result.Replaced = true
	return result, nil
}

func fixtureIncompleteness(path string, captures []llm.WireCapture, options FixtureWriteOptions) ([]string, error) {
	var reasons []string
	completed := make(map[string]struct{}, len(options.CompletedScenarios))
	for _, name := range options.CompletedScenarios {
		completed[name] = struct{}{}
	}
	for _, name := range options.ExpectedScenarios {
		if _, ok := completed[name]; !ok {
			reasons = append(reasons, "scenario "+name+" did not complete")
		}
	}
	if len(captures) == 0 {
		reasons = append(reasons, "no exchanges were captured")
	}
	for i, capture := range captures {
		if capture.Err != nil {
			reasons = append(reasons, fmt.Sprintf("exchange %d ended with a capture error: %v", i, capture.Err))
		}
		if capture.ResponseIncomplete {
			reasons = append(reasons, fmt.Sprintf("exchange %d has an incomplete response body", i))
		}
	}
	if options.OutstandingResponseBodies > 0 {
		reasons = append(reasons, fmt.Sprintf("%d response bodies are still outstanding", options.OutstandingResponseBodies))
	}

	existing, err := os.ReadFile(path)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	if err == nil {
		var exchanges []RecordedExchange
		existingComplete := json.Unmarshal(existing, &exchanges) == nil
		for _, exchange := range exchanges {
			existingComplete = existingComplete && exchange.Incomplete == ""
		}
		if existingComplete && len(captures) < len(exchanges) {
			reasons = append(reasons, fmt.Sprintf("captured %d exchanges; existing complete fixture has %d", len(captures), len(exchanges)))
		}
	}
	return reasons, nil
}

func stageAndReplaceFixture(path string, raw []byte, secrets []string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".fixture-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o644); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(raw); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	staged, err := os.ReadFile(tmpPath)
	if err != nil {
		return err
	}
	if err := ValidateFixtureBytes(path, staged, secrets...); err != nil {
		return fmt.Errorf("validate staged fixture: %w", err)
	}
	return os.Rename(tmpPath, path)
}

// ValidateFixtureBytes applies the same safety boundary used before atomic
// fixture replacement.
func ValidateFixtureBytes(path string, data []byte, secrets ...string) error {
	for _, secret := range secrets {
		if secret != "" && bytes.Contains(data, []byte(secret)) {
			return fmt.Errorf("%s contains a known secret", path)
		}
	}
	if legacy := legacyMockFixturePattern.Find(data); legacy != nil {
		return fmt.Errorf("%s contains legacy hashed mock identifier %q", path, legacy)
	}
	for _, mock := range mockTokenPattern.FindAll(data, -1) {
		if !sequentialMockPattern.Match(mock) {
			return fmt.Errorf("%s contains untrusted mock identifier %q", path, mock)
		}
	}
	var exchanges []RecordedExchange
	if err := json.Unmarshal(data, &exchanges); err != nil {
		return fmt.Errorf("decode fixture: %w", err)
	}
	for i, exchange := range exchanges {
		if secret := knownSecretInURLPath(exchange.URL, secrets); secret != "" {
			return fmt.Errorf("%s exchange %d URL path contains a percent-encoded known secret", path, i)
		}
		for name, values := range exchange.RequestHeaders {
			if err := validateFixtureHeader(i, name, values); err != nil {
				return fmt.Errorf("%s: %w", path, err)
			}
		}
		for name, values := range exchange.ResponseHeaders {
			if err := validateFixtureHeader(i, name, values); err != nil {
				return fmt.Errorf("%s: %w", path, err)
			}
		}
		joined := exchange.URL + "\n" + exchange.RequestBody + "\n" + exchange.ResponseBody + "\n" + exchange.Err
		for label, body := range map[string]string{"request body": exchange.RequestBody, "response body": exchange.ResponseBody} {
			if host, ok := privateHostInStructuredBody([]byte(body)); ok {
				return fmt.Errorf("exchange %d %s contains private host %q in a host-shaped field", i, label, host)
			}
		}
		for _, pattern := range unredactedFixturePatterns {
			for _, match := range pattern.FindAllString(joined, -1) {
				if strings.Contains(match, "[REDACTED]") || strings.Contains(match, "MOCK-") {
					continue
				}
				return fmt.Errorf("exchange %d leaks %s via %q", i, pattern, match)
			}
		}
		for _, match := range privateURLPattern.FindAllString(joined, -1) {
			u, err := url.Parse(match)
			if err == nil && isPrivateHostname(u.Hostname()) {
				return fmt.Errorf("exchange %d contains private host %q", i, u.Hostname())
			}
		}
		for _, match := range privateAddressPattern.FindAllString(joined, -1) {
			if isPrivateHostname(match) {
				return fmt.Errorf("exchange %d contains private host %q", i, match)
			}
		}
	}
	if err := validateFixtureEntropy(exchanges); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

func knownSecretInURLPath(rawURL string, secrets []string) string {
	u, err := url.Parse(rawURL)
	if err != nil {
		return ""
	}
	path, err := url.PathUnescape(u.EscapedPath())
	if err != nil {
		return ""
	}
	for _, secret := range secrets {
		if secret != "" && strings.Contains(path, secret) {
			return secret
		}
	}
	return ""
}

func validateFixtureHeader(index int, name string, values []string) error {
	if !isSensitiveHeaderName(name) {
		return nil
	}
	for _, value := range values {
		if value != "[REDACTED]" {
			return fmt.Errorf("exchange %d header %s leaked %q", index, name, value)
		}
	}
	return nil
}

func privateHostInStructuredBody(body []byte) (string, bool) {
	if host, ok := privateHostInJSON(body); ok {
		return host, true
	}
	for _, payload := range sseDataEvents(body) {
		payload = strings.TrimSpace(payload)
		if payload == "" || payload == "[DONE]" {
			continue
		}
		if host, ok := privateHostInJSON([]byte(payload)); ok {
			return host, true
		}
	}
	return "", false
}

func privateHostInJSON(raw []byte) (string, bool) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return "", false
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return "", false
	}
	return privateHostInValue(value)
}

func privateHostInValue(value any) (string, bool) {
	switch typed := value.(type) {
	case map[string]any:
		for field, child := range typed {
			if isHostShapedField(field) {
				if hostValue, ok := child.(string); ok {
					hostname, _, valid := splitHostValue(hostValue)
					if valid && isPrivateHostname(hostname) {
						return hostname, true
					}
				}
			}
			if host, ok := privateHostInValue(child); ok {
				return host, true
			}
		}
	case []any:
		for _, child := range typed {
			if host, ok := privateHostInValue(child); ok {
				return host, true
			}
		}
	}
	return "", false
}

func validateFixtureEntropy(exchanges []RecordedExchange) error {
	for _, exchange := range exchanges {
		if err := validateURLValueEntropy(exchange.URL); err != nil {
			return err
		}
		values := []string{
			exchange.Provider,
			exchange.Method,
			exchange.RequestBody,
			exchange.ResponseBody,
			exchange.Err,
			exchange.Incomplete,
		}
		for _, headers := range []http.Header{exchange.RequestHeaders, exchange.ResponseHeaders} {
			names := make([]string, 0, len(headers))
			for name := range headers {
				names = append(names, name)
			}
			sort.Strings(names)
			for _, name := range names {
				values = append(values, name)
				values = append(values, headers[name]...)
			}
		}
		for _, value := range values {
			if err := validateEntropyValue([]byte(value)); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateURLValueEntropy(rawURL string) error {
	u, err := url.Parse(rawURL)
	if err != nil || u.Host == "" {
		return validateEntropyValue([]byte(rawURL))
	}
	values := []string{u.Scheme, u.Hostname(), u.Port(), u.Fragment}
	if u.User != nil {
		values = append(values, u.User.Username())
		if password, ok := u.User.Password(); ok {
			values = append(values, password)
		}
	}
	for _, segment := range strings.Split(u.EscapedPath(), "/") {
		decoded, decodeErr := url.PathUnescape(segment)
		if decodeErr != nil {
			decoded = segment
		}
		values = append(values, decoded)
	}
	query := u.Query()
	keys := make([]string, 0, len(query))
	for key := range query {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		values = append(values, key)
		values = append(values, query[key]...)
	}
	for _, value := range values {
		if err := validateEntropyValue([]byte(value)); err != nil {
			return err
		}
	}
	return nil
}

func validateEntropyValue(data []byte) error {
	for _, match := range standardBase64TokenPattern.FindAllIndex(data, -1) {
		if !hasStandardBase64Boundaries(data, match[0], match[1]) {
			continue
		}
		token := data[match[0]:match[1]]
		candidates := [][]byte{token}
		if isAbsoluteSlashDelimitedPath(token) {
			candidates = bytes.Split(bytes.TrimPrefix(token, []byte{'/'}), []byte{'/'})
		}
		for _, candidate := range candidates {
			if len(candidate) < 32 {
				continue
			}
			if bytes.Contains(candidate, []byte(redPixelPNGBase64)) && string(candidate) != redPixelPNGBase64 {
				return fmt.Errorf("only the exact red-pixel token is allowlisted: %q", candidate)
			}
			if !isStandardBase64(candidate) {
				continue
			}
			if string(candidate) == redPixelPNGBase64 {
				continue
			}
			return fmt.Errorf("standard base64 token is not allowlisted: %q", candidate)
		}
	}
	scanData, err := maskExactRedPixel(data)
	if err != nil {
		return err
	}
	if token := hexSecretPattern.Find(scanData); token != nil {
		return fmt.Errorf("random hex token is not allowlisted: %q", token)
	}
	for _, token := range entropyTokenPattern.FindAll(scanData, -1) {
		if bytes.HasPrefix(token, []byte("MOCK")) || bytes.Contains(token, []byte("REDACTED")) {
			continue
		}
		if shannonEntropy(token) >= 4.0 {
			return fmt.Errorf("high-entropy token is not allowlisted: %q", token)
		}
	}
	for _, token := range base64URLTokenPattern.FindAll(scanData, -1) {
		if sequentialMockPattern.Match(token) || bytes.Contains(token, []byte("REDACTED")) {
			continue
		}
		if shannonEntropy(token) >= 4.4 {
			return fmt.Errorf("high-entropy base64url token is not allowlisted: %q", token)
		}
	}
	return nil
}

func hasStandardBase64Boundaries(data []byte, start, end int) bool {
	if start > 0 && isStandardBase64CoreByte(data[start-1]) {
		return false
	}
	return end == len(data) || !isStandardBase64CoreByte(data[end]) && data[end] != '='
}

func isStandardBase64CoreByte(value byte) bool {
	return value >= 'a' && value <= 'z' || value >= 'A' && value <= 'Z' || value >= '0' && value <= '9' || value == '+' || value == '/'
}

// A leading slash establishes absolute-path lexical context. In that context,
// slashes delimit segments rather than joining the entire route into one token;
// each segment is still scanned independently for encoded secrets.
func isAbsoluteSlashDelimitedPath(token []byte) bool {
	if len(token) == 0 || token[0] != '/' || bytes.ContainsAny(token, "+=") {
		return false
	}
	segments := bytes.Split(token[1:], []byte{'/'})
	if len(segments) < 2 {
		return false
	}
	for _, segment := range segments {
		if len(segment) == 0 {
			return false
		}
	}
	return true
}

func maskExactRedPixel(data []byte) ([]byte, error) {
	redPixel := []byte(redPixelPNGBase64)
	masked := append([]byte(nil), data...)
	for offset := 0; ; {
		relative := bytes.Index(data[offset:], redPixel)
		if relative < 0 {
			return masked, nil
		}
		start := offset + relative
		end := start + len(redPixel)
		if (start > 0 && isEncodedTokenByte(data[start-1])) || (end < len(data) && isEncodedTokenByte(data[end])) {
			return nil, errors.New("only the exact red-pixel token is allowlisted")
		}
		copy(masked[start:end], bytes.Repeat([]byte{'Z'}, len(redPixel)))
		offset = end
	}
}

func isEncodedTokenByte(value byte) bool {
	return value >= 'a' && value <= 'z' || value >= 'A' && value <= 'Z' || value >= '0' && value <= '9' || strings.ContainsRune("+/_=-", rune(value))
}

func isStandardBase64(token []byte) bool {
	if len(token)%4 != 0 {
		return false
	}
	_, err := base64.StdEncoding.DecodeString(string(token))
	return err == nil
}

func shannonEntropy(value []byte) float64 {
	counts := make(map[byte]int)
	for _, b := range value {
		counts[b]++
	}
	length := float64(len(value))
	var entropy float64
	for _, count := range counts {
		p := float64(count) / length
		entropy -= p * math.Log2(p)
	}
	return entropy
}

var unredactedFixturePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)"(?:safety_identifier|prompt_cache_key|encrypted_content|signature|obfuscation|api_key|access|refresh|access_token|refresh_token|accountId|chatgpt-account-id)"\s*:\s*"[^"]+`),
	regexp.MustCompile(`(?i)"(?:id|item_id|response_id|previous_response_id|request_id|call_id|tool_call_id|tool_use_id)"\s*:\s*"(?:resp|msg|req|rs|fc|call|toolu|gen|chatcmpl)[A-Za-z0-9._:-]+`),
	regexp.MustCompile(`(?i)bearer\s+[A-Za-z0-9._~+/=-]{16,}`),
	regexp.MustCompile(`(?i)user-[A-Za-z0-9_-]{8,}`),
}
