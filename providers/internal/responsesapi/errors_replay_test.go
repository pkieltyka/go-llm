package responsesapi_test

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
)

func TestMapStreamErrorKinds(t *testing.T) {
	adapter := replayAdapter()
	cases := []struct {
		code    string
		message string
		want    error
	}{
		{"rate_limit_exceeded", "", llm.ErrRateLimited},
		{"invalid_api_key", "", llm.ErrAuth},
		{"insufficient_quota", "", llm.ErrInsufficientCredits},
		{"", "maximum context length exceeded", llm.ErrContextTooLong},
		{"content_filter", "", llm.ErrContentFiltered},
		{"", "policy violation", llm.ErrContentFiltered},
		{"timeout", "", llm.ErrTimeout},
		{"server_error", "", llm.ErrServer},
		{"invalid_request", "", llm.ErrBadRequest},
		// A code merely containing "context" must NOT classify as
		// context-too-long (the bare-substring false positive removed in
		// v0.2); with no other signal it falls back to ErrServer.
		{"context_switch_failure", "", llm.ErrServer},
	}
	for _, tc := range cases {
		if got := adapter.MapStreamError(tc.code, tc.message, "param"); !errors.Is(got, tc.want) {
			t.Fatalf("MapStreamError(%q, %q) = %v, want %v", tc.code, tc.message, got, tc.want)
		}
	}
	if got := adapter.MapResponseError(responses.ResponseError{Code: "rate_limit_exceeded", Message: "slow down"}); !errors.Is(got, llm.ErrRateLimited) {
		t.Fatalf("MapResponseError = %v, want ErrRateLimited", got)
	}
}

func TestMapHTTPResponseErrorStatuses(t *testing.T) {
	adapter := replayAdapter()
	cases := []struct {
		status int
		want   error
	}{
		// The FS §16 canonical status table — identical across responsesapi,
		// chatcompletions, and anthropic (each engine asserts the same rows).
		{401, llm.ErrAuth},
		{402, llm.ErrInsufficientCredits},
		{403, llm.ErrPermission},
		{404, llm.ErrNotFound},
		{408, llm.ErrTimeout},
		{429, llm.ErrRateLimited},
		{503, llm.ErrOverloaded},
		{529, llm.ErrOverloaded},
		{500, llm.ErrServer},
		{502, llm.ErrServer},
		{400, llm.ErrBadRequest},
	}
	for _, tc := range cases {
		resp := &http.Response{
			StatusCode: tc.status,
			Header:     http.Header{},
			Body:       io.NopCloser(strings.NewReader(`{"error":{"message":"nope"}}`)),
		}
		if got := adapter.MapHTTPResponseError(resp); !errors.Is(got, tc.want) {
			t.Fatalf("MapHTTPResponseError(%d) = %v, want %v", tc.status, got, tc.want)
		}
	}
	if got := adapter.MapHTTPResponseError(nil); !errors.Is(got, llm.ErrServer) {
		t.Fatalf("MapHTTPResponseError(nil) = %v, want ErrServer", got)
	}
}

func streamEvent(t *testing.T, raw string) responses.ResponseStreamEventUnion {
	t.Helper()
	var event responses.ResponseStreamEventUnion
	if err := json.Unmarshal([]byte(raw), &event); err != nil {
		t.Fatalf("stream event unmarshal returned error: %v: %s", err, raw)
	}
	return event
}

// TestStreamRefusalMapsToRefusalStop covers the FS §18 refusal edge case on
// the stream path: refusal deltas surface as text and the terminal response
// maps to StopReasonRefusal.
func TestStreamRefusalMapsToRefusalStop(t *testing.T) {
	adapter := replayAdapter()
	state := adapter.NewStreamState()
	var events []llm.Event
	for _, raw := range []string{
		`{"type":"response.created","response":{"id":"resp_refusal","model":"replay-model","status":"in_progress"}}`,
		`{"type":"response.refusal.delta","output_index":0,"delta":"I cannot "}`,
		`{"type":"response.refusal.delta","output_index":0,"delta":"help with that."}`,
		`{"type":"response.refusal.done","output_index":0,"refusal":"I cannot help with that."}`,
		`{"type":"response.completed","response":{"id":"resp_refusal","model":"replay-model","status":"completed","output":[{"type":"message","content":[{"type":"refusal","refusal":"I cannot help with that."}]}],"usage":{"input_tokens":10,"output_tokens":8,"total_tokens":18}}}`,
	} {
		mapped, err := state.MapEvent(streamEvent(t, raw))
		if err != nil {
			t.Fatalf("MapEvent returned error: %v", err)
		}
		events = append(events, mapped...)
	}
	if got := state.Model(); got != "replay-model" {
		t.Fatalf("stream model = %q, want replay-model", got)
	}
	resp, err := llm.Collect(testutil.EventSeq(events...))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if resp.Text() != "I cannot help with that." {
		t.Fatalf("refusal text = %q", resp.Text())
	}
	if resp.StopReason != llm.StopReasonRefusal {
		t.Fatalf("refusal stop reason = %q, want %q", resp.StopReason, llm.StopReasonRefusal)
	}
}

// TestStreamRefusalDoneWithoutDeltas covers the refusal.done-only path where
// the full refusal text arrives in the terminal refusal event.
func TestStreamRefusalDoneWithoutDeltas(t *testing.T) {
	adapter := replayAdapter()
	state := adapter.NewStreamState()
	var events []llm.Event
	for _, raw := range []string{
		`{"type":"response.created","response":{"id":"resp_refusal","model":"replay-model","status":"in_progress"}}`,
		`{"type":"response.refusal.done","output_index":0,"refusal":"No."}`,
		`{"type":"response.completed","response":{"id":"resp_refusal","model":"replay-model","status":"completed","output":[{"type":"message","content":[{"type":"refusal","refusal":"No."}]}],"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}}`,
	} {
		mapped, err := state.MapEvent(streamEvent(t, raw))
		if err != nil {
			t.Fatalf("MapEvent returned error: %v", err)
		}
		events = append(events, mapped...)
	}
	resp, err := llm.Collect(testutil.EventSeq(events...))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if resp.Text() != "No." || resp.StopReason != llm.StopReasonRefusal {
		t.Fatalf("refusal response = text %q stop %q", resp.Text(), resp.StopReason)
	}
}
