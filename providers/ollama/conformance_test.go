package ollama_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
	"github.com/pkieltyka/go-llm/providers/ollama"
)

// TestOllamaConformance machine-checks the llm.Provider contract through the
// data-only Ollama preset, independently of the shared engine's own suite.
func TestOllamaConformance(t *testing.T) {
	llmtest.RunConformance(t, func(t *testing.T) llm.Provider {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			scenario := llmtest.ConformanceScenarioFromRequest(r)
			if r.Header.Get("Accept") == "text/event-stream" {
				w.Header().Set("Content-Type", "text/event-stream")
				if scenario == llmtest.ConformanceEmpty {
					return
				}
				_, _ = io.WriteString(w, `data: {"id":"c1","model":"qwen3:8b","choices":[{"index":0,"delta":{"role":"assistant"}}]}`+"\n\n")
				switch scenario {
				case llmtest.ConformanceCancel:
					w.(http.Flusher).Flush()
					<-r.Context().Done()
					return
				case llmtest.ConformanceTruncated:
					return
				case llmtest.ConformanceTools:
					_, _ = io.WriteString(w, `data: {"id":"c1","model":"qwen3:8b","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_conformance","type":"function","function":{"name":"conformance_echo","arguments":"{\"value\":\"pong\"}"}}]}}]}`+"\n\n")
					_, _ = io.WriteString(w, `data: {"id":"c1","model":"qwen3:8b","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
					_, _ = io.WriteString(w, `data: {"id":"c1","model":"qwen3:8b","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\n")
					_, _ = io.WriteString(w, "data: [DONE]\n\n")
					return
				}
				_, _ = io.WriteString(w, `data: {"id":"c1","model":"qwen3:8b","choices":[{"index":0,"delta":{"content":"po"}}]}`+"\n\n")
				_, _ = io.WriteString(w, `data: {"id":"c1","model":"qwen3:8b","choices":[{"index":0,"delta":{"content":"ng"},"finish_reason":"stop"}]}`+"\n\n")
				_, _ = io.WriteString(w, `data: {"id":"c1","model":"qwen3:8b","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\n")
				_, _ = io.WriteString(w, "data: [DONE]\n\n")
				return
			}
			if scenario == llmtest.ConformanceCancel {
				<-r.Context().Done()
				return
			}
			w.Header().Set("Content-Type", "application/json")
			if scenario == llmtest.ConformanceTools {
				_, _ = io.WriteString(w, `{"id":"c1","model":"qwen3:8b","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_conformance","type":"function","function":{"name":"conformance_echo","arguments":"{\"value\":\"pong\"}"}}]}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
				return
			}
			_, _ = io.WriteString(w, `{"id":"c1","model":"qwen3:8b","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"pong"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
		}))
		t.Cleanup(server.Close)

		p, err := ollama.New(server.URL,
			chatcompletions.WithHTTPClient(server.Client()),
			chatcompletions.WithMaxRetries(0),
		)
		if err != nil {
			t.Fatalf("ollama.New returned error: %v", err)
		}
		return p
	})
}

// TestOllamaCapabilityConformance applies the generic compatible-engine
// activation profile to the data-only Ollama preset.
func TestOllamaCapabilityConformance(t *testing.T) {
	const model = "qwen3:8b"
	llmtest.RunCapabilityConformance(t, func(t *testing.T, invocation llmtest.CapabilityInvocation) llm.Provider {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode activation request: %v", err)
				http.Error(w, "invalid JSON", http.StatusBadRequest)
				return
			}
			if err := testutil.AssertCompatibleActivationRequest(body, invocation, model); err != nil {
				t.Errorf("%s native request: %v; body=%#v", invocation.CaseName, err, body)
			}
			if invocation.Capability == llm.CapabilityReasoning && body["reasoning_effort"] != "high" {
				t.Errorf("reasoning_effort = %#v, want high", body["reasoning_effort"])
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, testutil.CompatibleActivationResponse(invocation, model))
		}))
		t.Cleanup(server.Close)
		provider, err := ollama.New(server.URL,
			chatcompletions.WithHTTPClient(server.Client()),
			chatcompletions.WithMaxRetries(0),
		)
		if err != nil {
			t.Fatalf("ollama.New returned error: %v", err)
		}
		return provider
	}, testutil.CompatibleCapabilityProfile(model))
}
