package openai

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestOpenAIConformance machine-checks the llm.Provider contract
// (single-use streams, cancellation, concurrency, panic-freedom, Collect
// partial shape) against a fixture server speaking the Responses wire shape.
func TestOpenAIConformance(t *testing.T) {
	llmtest.RunConformance(t, func(t *testing.T) llm.Provider {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			scenario := llmtest.ConformanceScenarioFromRequest(r)
			body, _ := io.ReadAll(r.Body)
			if strings.Contains(string(body), `"stream":true`) {
				w.Header().Set("Content-Type", "text/event-stream")
				if scenario == llmtest.ConformanceEmpty {
					return
				}
				_, _ = io.WriteString(w, `event: response.created`+"\n")
				_, _ = io.WriteString(w, `data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}`+"\n\n")
				switch scenario {
				case llmtest.ConformanceCancel:
					w.(http.Flusher).Flush()
					<-r.Context().Done()
					return
				case llmtest.ConformanceTruncated:
					return
				case llmtest.ConformanceTools:
					_, _ = io.WriteString(w, `event: response.output_item.added`+"\n")
					_, _ = io.WriteString(w, `data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_conformance","name":"conformance_echo","arguments":"","status":"in_progress"}}`+"\n\n")
					_, _ = io.WriteString(w, `event: response.function_call_arguments.delta`+"\n")
					_, _ = io.WriteString(w, `data: {"type":"response.function_call_arguments.delta","sequence_number":2,"item_id":"fc_1","output_index":0,"delta":"{\"value\":\"pong\"}"}`+"\n\n")
					_, _ = io.WriteString(w, `event: response.output_item.done`+"\n")
					_, _ = io.WriteString(w, `data: {"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_conformance","name":"conformance_echo","arguments":"{\"value\":\"pong\"}","status":"completed"}}`+"\n\n")
					_, _ = io.WriteString(w, `event: response.completed`+"\n")
					_, _ = io.WriteString(w, `data: {"type":"response.completed","sequence_number":4,"response":{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"fc_1","type":"function_call","call_id":"call_conformance","name":"conformance_echo","arguments":"{\"value\":\"pong\"}","status":"completed"}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}}`+"\n\n")
					return
				}
				_, _ = io.WriteString(w, `event: response.output_text.delta`+"\n")
				_, _ = io.WriteString(w, `data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"pong","logprobs":[]}`+"\n\n")
				_, _ = io.WriteString(w, `event: response.completed`+"\n")
				_, _ = io.WriteString(w, `data: {"type":"response.completed","sequence_number":2,"response":{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"pong","annotations":[]}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}}`+"\n\n")
				return
			}
			if scenario == llmtest.ConformanceCancel {
				<-r.Context().Done()
				return
			}
			w.Header().Set("Content-Type", "application/json")
			if scenario == llmtest.ConformanceTools {
				_, _ = io.WriteString(w, `{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"fc_1","type":"function_call","call_id":"call_conformance","name":"conformance_echo","arguments":"{\"value\":\"pong\"}","status":"completed"}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}`)
				return
			}
			_, _ = io.WriteString(w, `{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"pong","annotations":[]}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}`)
		}))
		t.Cleanup(server.Close)

		p, err := New(
			WithAPIKey("test-key"),
			WithBaseURL(server.URL),
			WithHTTPClient(server.Client()),
			WithMaxRetries(0),
		)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		return p
	})
}

// TestProviderIdentitySurface pins the trivial identity accessors.
func TestProviderIdentitySurface(t *testing.T) {
	p, err := New(WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if p.Name() != "openai" {
		t.Fatalf("Name = %q, want openai", p.Name())
	}
	caps := p.Capabilities()
	if len(caps) == 0 || caps[0] != llm.CapabilityStreaming {
		t.Fatalf("capabilities = %+v", caps)
	}
}
