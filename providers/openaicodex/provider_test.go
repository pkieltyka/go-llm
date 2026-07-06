package openaicodex

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
)

func TestOpenAICodexBuildRequestGolden(t *testing.T) {
	rawReasoning := json.RawMessage(`{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"because"}],"encrypted_content":"enc","status":"completed"}`)
	params, err := (&Provider{}).adapter().BuildParams(&llm.Request{
		Model:     "gpt-5.4-mini",
		System:    "You are terse.",
		MaxTokens: 64,
		Messages: []llm.Message{
			llm.UserText("hello"),
			llm.AssistantParts(
				llm.ReasoningPart{Provider: providerName, Raw: rawReasoning},
				llm.ReasoningPart{Provider: "openai", Raw: json.RawMessage(`{"id":"rs_other","type":"reasoning","encrypted_content":"foreign"}`)},
			),
		},
		Effort: llm.EffortLow,
	}, false)
	if err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	got := testutil.MustCompactJSON(t, params)
	want := `{
	"instructions": "You are terse.",
	"max_output_tokens": 64,
	"store": false,
	"include": [
		"reasoning.encrypted_content"
	],
	"input": [
		{
			"content": [
				{
					"text": "hello",
					"type": "input_text"
				}
			],
			"role": "user"
		},
		{
			"id": "rs_1",
			"type": "reasoning",
			"summary": [
				{
					"type": "summary_text",
					"text": "because"
				}
			],
			"encrypted_content": "enc",
			"status": "completed"
		}
	],
	"model": "gpt-5.4-mini",
	"reasoning": {
		"effort": "low",
		"summary": "auto"
	}
}`
	testutil.AssertJSONEqual(t, got, want)
	if strings.Contains(got, "foreign") {
		t.Fatalf("foreign OpenAI reasoning was replayed: %s", got)
	}
}

func TestOpenAICodexStreamingBodyGolden(t *testing.T) {
	temp := 0.2
	topP := 0.9
	params, err := (&Provider{}).adapter().BuildParams(&llm.Request{
		Model:       "gpt-5.4-mini",
		MaxTokens:   123,
		Temperature: &temp,
		TopP:        &topP,
		Messages:    []llm.Message{llm.UserText("hello")},
	}, true)
	if err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	raw, err := codexStreamingBody(params)
	if err != nil {
		t.Fatalf("codexStreamingBody returned error: %v", err)
	}
	want := `{"include":["reasoning.encrypted_content"],"input":[{"content":[{"text":"hello","type":"input_text"}],"role":"user"}],"model":"gpt-5.4-mini","store":false,"stream":true}`
	testutil.AssertJSONEqual(t, string(raw), want)

	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("streaming body is invalid JSON: %v\n%s", err, raw)
	}
	if body["stream"] != true {
		t.Fatalf("stream = %#v, want true", body["stream"])
	}
	// The codex backend rejects these knobs (live-verified 2026-07-03:
	// temperature → 400 "Unsupported parameter"), so all three are dropped.
	for _, knob := range []string{"temperature", "top_p", "max_output_tokens"} {
		if _, ok := body[knob]; ok {
			t.Fatalf("%s should be omitted for Codex backend: %s", knob, raw)
		}
	}
}

func TestOpenAICodexStreamStartKeepsServerModel(t *testing.T) {
	event := normalizeCodexStreamEvent(llm.MessageStart{
		ID:       "resp_1",
		Provider: providerName,
		Model:    "gpt-5.4-mini-2026-03-17",
	}, "gpt-5.4-mini")
	start, ok := event.(llm.MessageStart)
	if !ok {
		t.Fatalf("event type = %T, want MessageStart", event)
	}
	if start.Model != "gpt-5.4-mini-2026-03-17" {
		t.Fatalf("model = %q, want server-reported model", start.Model)
	}

	// Only when the backend omits the model does the request model fill in.
	event = normalizeCodexStreamEvent(llm.MessageStart{ID: "resp_1", Provider: providerName}, "gpt-5.4-mini")
	start = event.(llm.MessageStart)
	if start.Model != "gpt-5.4-mini" {
		t.Fatalf("fallback model = %q, want request model", start.Model)
	}
}

func TestOpenAICodexAssistantTextReplayUsesOutputText(t *testing.T) {
	input, err := (&Provider{}).adapter().BuildInput([]llm.Message{
		{Role: llm.RoleAssistant, Parts: []llm.Part{llm.Text("previous answer")}},
	})
	if err != nil {
		t.Fatalf("BuildInput returned error: %v", err)
	}
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}
	if !bytes.Contains(raw, []byte(`"type":"output_text"`)) {
		t.Fatalf("assistant replay did not use output_text: %s", raw)
	}
	if bytes.Contains(raw, []byte(`"type":"input_text"`)) {
		t.Fatalf("assistant replay used input_text: %s", raw)
	}
}

func TestOpenAICodexReasoningReplayIsolation(t *testing.T) {
	codexRaw := json.RawMessage(`{"id":"rs_codex","type":"reasoning","encrypted_content":"codex"}`)
	openAIRaw := json.RawMessage(`{"id":"rs_openai","type":"reasoning","encrypted_content":"openai"}`)
	input, err := (&Provider{}).adapter().BuildInput([]llm.Message{llm.AssistantParts(
		llm.ReasoningPart{Provider: providerName, Raw: codexRaw},
		llm.ReasoningPart{Provider: "openai", Raw: openAIRaw},
	)})
	if err != nil {
		t.Fatalf("BuildInput returned error: %v", err)
	}
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatalf("Marshal returned error: %v", err)
	}
	if !bytes.Contains(raw, codexRaw) {
		t.Fatalf("codex reasoning missing: %s", raw)
	}
	if bytes.Contains(raw, openAIRaw) || bytes.Contains(raw, []byte("openai")) {
		t.Fatalf("openai reasoning replayed into codex: %s", raw)
	}
}

func TestOpenAICodexHeadersAndRetry(t *testing.T) {
	oldAccess := fakeCodexJWT(t, "acct-old")
	newAccess := fakeCodexJWT(t, "acct-new")
	var authHeaders []string
	var accountHeaders []string
	var originators []string
	var accepts []string
	var betas []string
	var userAgents []string
	var refreshForm string
	var refreshed llm.AuthCredential
	var requestBodies []string
	responseCalls := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/responses":
			responseCalls++
			body, _ := io.ReadAll(r.Body)
			requestBodies = append(requestBodies, string(body))
			authHeaders = append(authHeaders, r.Header.Get("Authorization"))
			accountHeaders = append(accountHeaders, r.Header.Get(accountIDHeader))
			originators = append(originators, r.Header.Get(originatorHeader))
			accepts = append(accepts, r.Header.Get("Accept"))
			betas = append(betas, r.Header.Get("OpenAI-Beta"))
			userAgents = append(userAgents, r.Header.Get("User-Agent"))
			if responseCalls == 1 {
				http.Error(w, `{"error":{"code":"invalid_token","message":"expired","type":"authentication_error"}}`, http.StatusUnauthorized)
				return
			}
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"pong"}`+"\n\n")
			_, _ = io.WriteString(w, `data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.4-mini","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"pong","annotations":[]}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}}`+"\n\n")
		case "/oauth/token":
			body, _ := io.ReadAll(r.Body)
			refreshForm = string(body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"access_token":`+strconvQuote(newAccess)+`,"refresh_token":"new-refresh","expires_in":3600}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	p, err := New(
		WithOAuth(llm.AuthCredential{Type: "oauth", Access: oldAccess, Refresh: "old-refresh"}, func(cred llm.AuthCredential) {
			refreshed = cred
		}),
		WithBaseURL(server.URL),
		withOAuthTokenURL(server.URL+"/oauth/token"),
		WithHTTPClient(server.Client()),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	resp, err := p.Chat(context.Background(), &llm.Request{
		Model:     "gpt-5.4-mini",
		MaxTokens: 8,
		Messages:  []llm.Message{llm.UserText("ping")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Text() != "pong" {
		t.Fatalf("response text = %q", resp.Text())
	}
	if !reflect.DeepEqual(authHeaders, []string{"Bearer " + oldAccess, "Bearer " + newAccess}) {
		t.Fatalf("Authorization headers = %+v", authHeaders)
	}
	if !reflect.DeepEqual(accountHeaders, []string{"acct-old", "acct-new"}) {
		t.Fatalf("account headers = %+v", accountHeaders)
	}
	if !reflect.DeepEqual(originators, []string{defaultOriginator, defaultOriginator}) {
		t.Fatalf("originators = %+v", originators)
	}
	if !reflect.DeepEqual(accepts, []string{"text/event-stream", "text/event-stream"}) {
		t.Fatalf("Accept headers = %+v", accepts)
	}
	if !reflect.DeepEqual(betas, []string{"responses=experimental", "responses=experimental"}) {
		t.Fatalf("OpenAI-Beta headers = %+v", betas)
	}
	if !reflect.DeepEqual(userAgents, []string{defaultCodexUserAgent, defaultCodexUserAgent}) {
		t.Fatalf("User-Agent headers = %+v", userAgents)
	}
	for _, body := range requestBodies {
		if !jsonFieldBool(t, body, "stream") {
			t.Fatalf("request body did not force stream=true: %s", body)
		}
	}
	if !strings.Contains(refreshForm, "refresh_token=old-refresh") || !strings.Contains(refreshForm, "client_id="+openAICodexOAuthClientID) {
		t.Fatalf("refresh form missing expected non-secret fields: %s", refreshForm)
	}
	if refreshed.Access != newAccess || refreshed.Refresh != "new-refresh" || refreshed.AccountID != "acct-new" {
		t.Fatalf("refreshed credential = %+v", refreshed)
	}
}

func writeCodexSSESuccess(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	_, _ = io.WriteString(w, `data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"pong"}`+"\n\n")
	_, _ = io.WriteString(w, `data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.4-mini","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"pong","annotations":[]}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}}`+"\n\n")
}

func newCodexRetryTestProvider(t *testing.T, server *httptest.Server, delays *[]time.Duration, opts ...Option) *Provider {
	t.Helper()
	access := fakeCodexJWT(t, "acct-1")
	base := []Option{
		WithOAuth(llm.AuthCredential{Type: "oauth", Access: access, Refresh: "refresh"}, nil),
		WithBaseURL(server.URL),
		withOAuthTokenURL(server.URL + "/oauth/token"),
		WithHTTPClient(server.Client()),
	}
	p, err := New(append(base, opts...)...)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	p.transport.sleep = func(ctx context.Context, d time.Duration) error {
		*delays = append(*delays, d)
		return nil
	}
	return p
}

func TestOpenAICodexTransportRetries429ThenSucceeds(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("Retry-After", "3")
			http.Error(w, `{"error":{"code":"rate_limit_exceeded","message":"slow down","type":"rate_limit_error"}}`, http.StatusTooManyRequests)
			return
		}
		writeCodexSSESuccess(w)
	}))
	defer server.Close()

	var delays []time.Duration
	p := newCodexRetryTestProvider(t, server, &delays)
	resp, err := p.Chat(context.Background(), &llm.Request{
		Model:    "gpt-5.4-mini",
		Messages: []llm.Message{llm.UserText("ping")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Text() != "pong" || calls != 2 {
		t.Fatalf("text/calls = %q/%d, want pong/2", resp.Text(), calls)
	}
	// Retry-After from the 429 must be honored verbatim.
	if len(delays) != 1 || delays[0] != 3*time.Second {
		t.Fatalf("retry delays = %+v, want [3s]", delays)
	}
}

func TestOpenAICodexTransportNoRetryOn400(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		http.Error(w, `{"error":{"code":"invalid_request","message":"bad","type":"invalid_request_error"}}`, http.StatusBadRequest)
	}))
	defer server.Close()

	var delays []time.Duration
	p := newCodexRetryTestProvider(t, server, &delays)
	_, err := p.Chat(context.Background(), &llm.Request{
		Model:    "gpt-5.4-mini",
		Messages: []llm.Message{llm.UserText("ping")},
	})
	if !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("error = %v, want ErrBadRequest", err)
	}
	if calls != 1 || len(delays) != 0 {
		t.Fatalf("calls/delays = %d/%+v, want single attempt without backoff", calls, delays)
	}
}

func TestOpenAICodexTransportMaxRetriesBounds(t *testing.T) {
	tests := []struct {
		name       string
		maxRetries int
		wantCalls  int
	}{
		{name: "zero disables retries", maxRetries: 0, wantCalls: 1},
		{name: "one extra attempt", maxRetries: 1, wantCalls: 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				http.Error(w, `{"error":{"code":"rate_limit_exceeded","message":"slow down","type":"rate_limit_error"}}`, http.StatusTooManyRequests)
			}))
			defer server.Close()

			var delays []time.Duration
			p := newCodexRetryTestProvider(t, server, &delays, WithMaxRetries(tt.maxRetries))
			_, err := p.Chat(context.Background(), &llm.Request{
				Model:    "gpt-5.4-mini",
				Messages: []llm.Message{llm.UserText("ping")},
			})
			if !errors.Is(err, llm.ErrRateLimited) {
				t.Fatalf("error = %v, want ErrRateLimited", err)
			}
			if calls != tt.wantCalls {
				t.Fatalf("calls = %d, want %d", calls, tt.wantCalls)
			}
			if len(delays) != tt.wantCalls-1 {
				t.Fatalf("delays = %+v, want %d backoff waits", delays, tt.wantCalls-1)
			}
		})
	}
}

func TestOpenAICodexRefreshErrorMapping(t *testing.T) {
	tests := []struct {
		status int
		want   error
	}{
		{status: http.StatusBadRequest, want: llm.ErrAuth},
		{status: http.StatusUnauthorized, want: llm.ErrAuth},
		{status: http.StatusForbidden, want: llm.ErrAuth},
		{status: http.StatusRequestTimeout, want: llm.ErrTimeout},
		{status: http.StatusTooManyRequests, want: llm.ErrRateLimited},
		{status: http.StatusInternalServerError, want: llm.ErrServer},
	}
	for _, tt := range tests {
		t.Run(http.StatusText(tt.status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, `{"error":"invalid_grant"}`, tt.status)
			}))
			defer server.Close()
			_, err := refreshCodexOAuth(context.Background(), server.Client(), server.URL, llm.AuthCredential{Refresh: "stale"})
			if !errors.Is(err, tt.want) {
				t.Fatalf("refresh error = %v, want %v", err, tt.want)
			}
		})
	}
}

func jsonFieldBool(t *testing.T, raw, name string) bool {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal([]byte(raw), &body); err != nil {
		t.Fatalf("request body is invalid JSON: %v\n%s", err, raw)
	}
	value, _ := body[name].(bool)
	return value
}

func fakeCodexJWT(t *testing.T, accountID string) string {
	t.Helper()
	payload := map[string]any{
		codexAccountClaimPath: map[string]any{"chatgpt_account_id": accountID},
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("Marshal payload returned error: %v", err)
	}
	return "header." + base64.RawURLEncoding.EncodeToString(raw) + ".sig"
}

func strconvQuote(s string) string {
	raw, _ := json.Marshal(s)
	return string(raw)
}
