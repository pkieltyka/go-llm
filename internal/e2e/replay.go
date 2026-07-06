package e2e

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

// This file is the offline replay harness over the recorded live fixture
// corpus (internal/e2e/fixtures/*/live.json). Provider packages construct a
// provider whose HTTP traffic is answered by one recorded exchange and run
// their full mapping paths against it, asserting normalized-response
// invariants (parts present, FS §11 usage math, mapped stop reasons,
// preserved reasoning raw payloads, parsed tool calls) instead of brittle
// full goldens. The outbound request is asserted too — invariant-level, not
// byte goldens: the body must be valid JSON, echo the recorded model, carry
// tools when the recorded scenario had tools, and carry a non-empty
// message/input list.

// LoadRecordedExchanges reads a recorded fixture file written by WriteFixture.
func LoadRecordedExchanges(path string) ([]RecordedExchange, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var exchanges []RecordedExchange
	if err := json.Unmarshal(raw, &exchanges); err != nil {
		return nil, err
	}
	return exchanges, nil
}

// NewReplayClient returns an HTTP client that answers every request with the
// recorded exchange's response. Each replayed exchange gets its own provider
// instance, so serving one canned response unconditionally is sufficient.
func NewReplayClient(exchange RecordedExchange) *http.Client {
	client, _ := newCapturingReplayClient(exchange)
	return client
}

// newCapturingReplayClient additionally records every outbound request so
// the harness can assert request-side invariants after the replay.
func newCapturingReplayClient(exchange RecordedExchange) (*http.Client, *requestLog) {
	log := &requestLog{}
	return &http.Client{Transport: replayTransport{exchange: exchange, log: log}}, log
}

// capturedRequest is one outbound HTTP request observed during a replay.
type capturedRequest struct {
	Method string
	URL    string
	Body   []byte
}

type requestLog struct {
	mu       sync.Mutex
	requests []capturedRequest
}

func (l *requestLog) add(req capturedRequest) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.requests = append(l.requests, req)
}

func (l *requestLog) all() []capturedRequest {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]capturedRequest(nil), l.requests...)
}

type replayTransport struct {
	exchange RecordedExchange
	log      *requestLog
}

func (t replayTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	var body []byte
	if req.Body != nil {
		body, _ = io.ReadAll(req.Body)
		_ = req.Body.Close()
	}
	if t.log != nil {
		t.log.add(capturedRequest{Method: req.Method, URL: req.URL.String(), Body: body})
	}
	header := make(http.Header, len(t.exchange.ResponseHeaders))
	for name, values := range t.exchange.ResponseHeaders {
		switch strings.ToLower(name) {
		// The recorded body is decoded, dechunked text; replaying the wire
		// transfer headers would make clients try to re-decode it.
		case "content-encoding", "content-length", "transfer-encoding":
			continue
		}
		header[http.CanonicalHeaderKey(name)] = append([]string(nil), values...)
	}
	status := t.exchange.Status
	if status == 0 {
		status = http.StatusOK
	}
	return &http.Response{
		StatusCode:    status,
		Status:        fmt.Sprintf("%d %s", status, http.StatusText(status)),
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        header,
		Body:          io.NopCloser(strings.NewReader(t.exchange.ResponseBody)),
		ContentLength: -1,
		Request:       req,
	}, nil
}

// ReplayProfile describes how one provider replays its recorded corpus.
type ReplayProfile struct {
	// Provider is the expected llm.Response.Provider / ReasoningPart.Provider.
	Provider string
	// New constructs a provider whose HTTP traffic is served by client.
	New func(t *testing.T, client *http.Client) llm.Provider
	// ToolCallMarkers are substrings whose presence in a recorded response
	// body means the mapped response must contain parsed tool calls.
	ToolCallMarkers []string
	// ReasoningMarkers are substrings whose presence in a recorded response
	// body means the mapped response must preserve raw reasoning payloads.
	ReasoningMarkers []string
	// ReasoningTextMarkers are substrings whose presence in a recorded
	// response body means the mapped response must carry plain-text
	// reasoning (ReasoningPart.Text) — for providers whose reasoning has no
	// raw replay payload (vLLM).
	ReasoningTextMarkers []string
}

// ReplayExchanges replays every recorded exchange in the fixture file through
// the profile's provider and asserts normalized mapping invariants.
func ReplayExchanges(t *testing.T, fixturePath string, profile ReplayProfile) {
	t.Helper()
	exchanges, err := LoadRecordedExchanges(fixturePath)
	if err != nil {
		t.Fatalf("LoadRecordedExchanges(%s) returned error: %v", fixturePath, err)
	}
	if len(exchanges) == 0 {
		t.Fatalf("fixture %s contains no exchanges", fixturePath)
	}
	for i, exchange := range exchanges {
		t.Run(fmt.Sprintf("%02d_%s", i, classifyExchange(exchange)), func(t *testing.T) {
			replayOne(t, exchange, profile)
		})
	}
}

func classifyExchange(exchange RecordedExchange) string {
	switch {
	case exchange.Err != "" || exchange.Status == 0:
		return "transport_error"
	case isModelsExchange(exchange):
		return "models"
	case exchange.Status >= 400:
		return fmt.Sprintf("error_%d", exchange.Status)
	case isStreamExchange(exchange):
		return "stream"
	default:
		return "chat"
	}
}

func isModelsExchange(exchange RecordedExchange) bool {
	return exchange.Method == http.MethodGet && strings.Contains(exchange.URL, "/models")
}

func isStreamExchange(exchange RecordedExchange) bool {
	if strings.Contains(exchange.RequestBody, `"stream":true`) {
		return true
	}
	if strings.Contains(strings.ToLower(exchange.ResponseHeaders.Get("Content-Type")), "text/event-stream") {
		return true
	}
	body := strings.TrimSpace(exchange.ResponseBody)
	return strings.HasPrefix(body, "event:") || strings.HasPrefix(body, "data:")
}

func replayOne(t *testing.T, exchange RecordedExchange, profile ReplayProfile) {
	t.Helper()
	if exchange.Err != "" || exchange.Status == 0 {
		t.Skipf("exchange recorded a transport error: %s", exchange.Err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	client, requests := newCapturingReplayClient(exchange)
	provider := profile.New(t, client)

	switch {
	case isModelsExchange(exchange):
		models, err := provider.Models(ctx)
		if err != nil {
			t.Fatalf("Models replay returned error: %v", err)
		}
		if len(models) == 0 {
			t.Fatalf("Models replay returned empty list")
		}
		for _, info := range models {
			if info.ID == "" {
				t.Fatalf("Models replay returned model with empty ID: %+v", info)
			}
		}
		if len(requests.all()) == 0 {
			t.Fatalf("Models replay sent no request")
		}
	case exchange.Status >= 400:
		_, err := replayCompletion(ctx, provider, exchange)
		assertReplayError(t, exchange, err)
		assertReplayRequest(t, exchange, requests.all())
	default:
		resp, err := replayCompletion(ctx, provider, exchange)
		if err != nil {
			t.Fatalf("replay returned error: %v", err)
		}
		AssertReplayResponse(t, exchange, profile, resp)
		assertReplayRequest(t, exchange, requests.all())
	}
}

// assertReplayRequest checks invariant-level request-side properties of the
// first outbound completions request (later requests, if any, belong to
// auxiliary flows such as OAuth refresh and are not part of the recorded
// scenario): valid JSON body, recorded model echoed, tools present when the
// recorded scenario carried tools, and a non-empty message/input list.
func assertReplayRequest(t *testing.T, exchange RecordedExchange, requests []capturedRequest) {
	t.Helper()
	if len(requests) == 0 {
		t.Fatalf("replay sent no outbound request")
	}
	sent := requests[0]
	if len(sent.Body) == 0 {
		t.Fatalf("replayed completions request has no body (method=%s url=%s)", sent.Method, sent.URL)
	}
	var body map[string]any
	if err := json.Unmarshal(sent.Body, &body); err != nil {
		t.Fatalf("outbound request body is invalid JSON: %v\n%s", err, sent.Body)
	}

	var recorded struct {
		Model string `json:"model"`
	}
	_ = json.Unmarshal([]byte(exchange.RequestBody), &recorded)
	if recorded.Model != "" {
		if got, _ := body["model"].(string); got != recorded.Model {
			t.Fatalf("outbound request model = %q, want recorded model %q", got, recorded.Model)
		}
	}

	if strings.Contains(exchange.RequestBody, `"tools"`) {
		tools, ok := body["tools"].([]any)
		if !ok || len(tools) == 0 {
			t.Fatalf("recorded scenario had tools but outbound request has none: %s", sent.Body)
		}
	}

	messages := requestMessageList(body)
	if len(messages) == 0 {
		t.Fatalf("outbound request carries no messages/input: %s", sent.Body)
	}
}

// requestMessageList extracts the conversation list from a wire request
// body: "messages" (Messages/chat-completions shapes) or "input" (Responses
// shape).
func requestMessageList(body map[string]any) []any {
	if messages, ok := body["messages"].([]any); ok {
		return messages
	}
	if input, ok := body["input"].([]any); ok {
		return input
	}
	return nil
}

func replayCompletion(ctx context.Context, provider llm.Provider, exchange RecordedExchange) (*llm.Response, error) {
	req := ReplayRequest(exchange)
	if isStreamExchange(exchange) {
		return llm.Collect(provider.ChatStream(ctx, req))
	}
	return provider.Chat(ctx, req)
}

// ReplayRequest reconstructs a request that matches the recorded exchange's
// salient features (model, tools, sampling); response mapping does not depend
// on prompt content, so the messages are generic.
func ReplayRequest(exchange RecordedExchange) *llm.Request {
	var wire struct {
		Model string `json:"model"`
	}
	_ = json.Unmarshal([]byte(exchange.RequestBody), &wire)
	if wire.Model == "" {
		wire.Model = "replay-model"
	}
	req := &llm.Request{
		Model:     wire.Model,
		MaxTokens: 64,
		Messages:  []llm.Message{llm.UserText("replay")},
	}
	if strings.Contains(exchange.RequestBody, `"temperature"`) {
		temperature := 0.0
		req.Temperature = &temperature
	}
	if strings.Contains(exchange.RequestBody, `"tools"`) {
		req.Tools = []llm.Tool{{
			Name:        "lookup",
			Description: "Look up a short value.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}`),
			Strict:      true,
		}}
	}
	return req
}

// AssertReplayResponse checks the normalized mapping invariants for one
// successfully replayed exchange.
func AssertReplayResponse(t *testing.T, exchange RecordedExchange, profile ReplayProfile, resp *llm.Response) {
	t.Helper()
	if resp == nil {
		t.Fatalf("replay returned nil response")
	}
	if profile.Provider != "" && resp.Provider != profile.Provider {
		t.Fatalf("replay response provider = %q, want %q", resp.Provider, profile.Provider)
	}
	if resp.Model == "" {
		t.Fatalf("replay response missing model")
	}
	if len(resp.Parts) == 0 {
		t.Fatalf("replay response has no parts: %+v", resp)
	}
	assertReplayStopReason(t, resp)
	assertReplayUsage(t, resp.Usage)
	if bodyContainsAny(exchange.ResponseBody, profile.ToolCallMarkers) {
		assertReplayToolCalls(t, resp)
	}
	if bodyContainsAny(exchange.ResponseBody, profile.ReasoningMarkers) {
		assertReplayReasoningRaw(t, profile.Provider, resp)
	}
	if bodyContainsAny(exchange.ResponseBody, profile.ReasoningTextMarkers) {
		assertReplayReasoningText(t, profile.Provider, resp)
	}
}

var knownStopReasons = map[llm.StopReason]struct{}{
	llm.StopReasonEndTurn:         {},
	llm.StopReasonMaxTokens:       {},
	llm.StopReasonStopSequence:    {},
	llm.StopReasonToolUse:         {},
	llm.StopReasonContentFilter:   {},
	llm.StopReasonRefusal:         {},
	llm.StopReasonContextOverflow: {},
	llm.StopReasonPaused:          {},
	llm.StopReasonError:           {},
	llm.StopReasonOther:           {},
}

func assertReplayStopReason(t *testing.T, resp *llm.Response) {
	t.Helper()
	if resp.StopReason == "" {
		t.Fatalf("replay response missing stop reason (raw=%q)", resp.StopReasonRaw)
	}
	if _, ok := knownStopReasons[resp.StopReason]; !ok {
		t.Fatalf("replay response stop reason %q is not a known StopReason constant", resp.StopReason)
	}
	if resp.StopReasonRaw == "" {
		t.Fatalf("replay response missing raw stop reason (mapped=%q)", resp.StopReason)
	}
}

// assertReplayUsage enforces the FS §11 normalization invariant: InputTokens
// excludes cache tokens, ReasoningTokens is an informational subset of
// OutputTokens, and TotalTokens equals prompt occupancy plus output.
func assertReplayUsage(t *testing.T, u llm.Usage) {
	t.Helper()
	if u.InputTokens < 0 || u.OutputTokens < 0 || u.TotalTokens < 0 || u.CacheReadTokens < 0 || u.CacheWriteTokens < 0 || u.ReasoningTokens < 0 {
		t.Fatalf("replay usage has negative counts: %+v", u)
	}
	if u.OutputTokens == 0 {
		t.Fatalf("replay usage missing output tokens: %+v", u)
	}
	if u.InputTokens+u.CacheReadTokens+u.CacheWriteTokens == 0 {
		t.Fatalf("replay usage missing prompt occupancy: %+v", u)
	}
	if total := u.InputTokens + u.CacheReadTokens + u.CacheWriteTokens + u.OutputTokens; u.TotalTokens != total {
		t.Fatalf("replay usage violates total invariant: total=%d want %d (usage=%+v)", u.TotalTokens, total, u)
	}
	// No ReasoningTokens <= OutputTokens check: adapters pass native counts
	// through without clamping (FS §18), and the recorded OpenRouter corpus
	// shows upstream accounting where reasoning_tokens slightly exceeds
	// completion_tokens.
}

func assertReplayToolCalls(t *testing.T, resp *llm.Response) {
	t.Helper()
	calls := resp.ToolCalls()
	if len(calls) == 0 {
		t.Fatalf("recorded response carries tool calls but mapped response has none: %+v", resp.Parts)
	}
	for _, call := range calls {
		if call.ID == "" || call.Name == "" {
			t.Fatalf("mapped tool call missing id or name: %+v", call)
		}
		if len(call.Args) > 0 && !json.Valid(call.Args) {
			t.Fatalf("mapped tool call args are invalid JSON: %s", call.Args)
		}
	}
	if resp.StopReason != llm.StopReasonToolUse {
		t.Fatalf("tool-call replay stop reason = %q, want %q", resp.StopReason, llm.StopReasonToolUse)
	}
}

func assertReplayReasoningRaw(t *testing.T, provider string, resp *llm.Response) {
	t.Helper()
	for _, part := range resp.Parts {
		reasoning, ok := reasoningPart(part)
		if !ok {
			continue
		}
		if provider != "" && reasoning.Provider != provider {
			t.Fatalf("reasoning part provider = %q, want %q", reasoning.Provider, provider)
		}
		if len(reasoning.Raw) == 0 {
			t.Fatalf("reasoning part missing raw replay payload: %+v", reasoning)
		}
		if !json.Valid(reasoning.Raw) {
			t.Fatalf("reasoning part raw payload is invalid JSON: %s", reasoning.Raw)
		}
		return
	}
	t.Fatalf("recorded response carries reasoning but mapped response has no reasoning part: %+v", resp.Parts)
}

// assertReplayReasoningText asserts plain-text reasoning preservation for
// providers without raw replay payloads (vLLM: `reasoning` is a plain
// string, so ReasoningPart.Raw stays empty by design).
func assertReplayReasoningText(t *testing.T, provider string, resp *llm.Response) {
	t.Helper()
	for _, part := range resp.Parts {
		reasoning, ok := reasoningPart(part)
		if !ok {
			continue
		}
		if provider != "" && reasoning.Provider != provider {
			t.Fatalf("reasoning part provider = %q, want %q", reasoning.Provider, provider)
		}
		if strings.TrimSpace(reasoning.Text) == "" {
			t.Fatalf("reasoning part missing text: %+v", reasoning)
		}
		return
	}
	t.Fatalf("recorded response carries reasoning but mapped response has no reasoning part: %+v", resp.Parts)
}

func reasoningPart(part llm.Part) (llm.ReasoningPart, bool) {
	// Parts are value types (adapters never emit pointer parts).
	reasoning, ok := part.(llm.ReasoningPart)
	return reasoning, ok
}

func assertReplayError(t *testing.T, exchange RecordedExchange, err error) {
	t.Helper()
	if err == nil {
		t.Fatalf("replay of %d exchange returned no error", exchange.Status)
	}
	allowed := allowedErrorSentinels(exchange.Status)
	matched := false
	for _, sentinel := range allowed {
		if errors.Is(err, sentinel) {
			matched = true
			break
		}
	}
	if !matched {
		t.Fatalf("replayed %d error = %v, want one of %v", exchange.Status, err, allowed)
	}
	var providerErr *llm.ProviderError
	if errors.As(err, &providerErr) && providerErr.HTTPStatus != 0 && providerErr.HTTPStatus != exchange.Status {
		t.Fatalf("replayed error HTTP status = %d, want %d", providerErr.HTTPStatus, exchange.Status)
	}
}

func allowedErrorSentinels(status int) []error {
	switch {
	case status == http.StatusUnauthorized:
		return []error{llm.ErrAuth}
	case status == http.StatusPaymentRequired:
		return []error{llm.ErrInsufficientCredits}
	case status == http.StatusForbidden:
		return []error{llm.ErrPermission, llm.ErrAuth}
	case status == http.StatusNotFound:
		return []error{llm.ErrNotFound, llm.ErrBadRequest}
	case status == http.StatusRequestTimeout:
		return []error{llm.ErrTimeout}
	case status == http.StatusTooManyRequests:
		return []error{llm.ErrRateLimited}
	case status >= 500:
		return []error{llm.ErrServer, llm.ErrOverloaded}
	default:
		return []error{llm.ErrBadRequest, llm.ErrNotFound}
	}
}

func bodyContainsAny(body string, markers []string) bool {
	for _, marker := range markers {
		if marker != "" && strings.Contains(body, marker) {
			return true
		}
	}
	return false
}
