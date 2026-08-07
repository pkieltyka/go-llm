package vllm

import (
	"net/http"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestVLLMConformance machine-checks the llm.Provider contract (single-use
// streams, cancellation, concurrency, panic-freedom, Collect partial shape)
// against a fixture handler speaking the vLLM chat-completions wire shape.
func TestVLLMConformance(t *testing.T) {
	llmtest.RunConformance(t, func(t *testing.T) llm.Provider {
		return newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
			scenario := llmtest.ConformanceScenarioFromRequest(r)
			if r.Header.Get("Accept") == "text/event-stream" {
				w.Header().Set("Content-Type", "text/event-stream")
				if scenario == llmtest.ConformanceEmpty {
					return
				}
				mustWrite(t, w, `data: {"id":"c1","model":"Qwen/Qwen3.6-27B-FP8","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`+"\n\n")
				switch scenario {
				case llmtest.ConformanceCancel:
					w.(http.Flusher).Flush()
					<-r.Context().Done()
					return
				case llmtest.ConformanceTruncated:
					return
				case llmtest.ConformanceTools:
					mustWrite(t, w, `data: {"id":"c1","model":"Qwen/Qwen3.6-27B-FP8","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_conformance","type":"function","function":{"name":"conformance_echo","arguments":"{\"value\":\"pong\"}"}}]}}]}`+"\n\n")
					mustWrite(t, w, `data: {"id":"c1","model":"Qwen/Qwen3.6-27B-FP8","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
					mustWrite(t, w, `data: {"id":"c1","model":"Qwen/Qwen3.6-27B-FP8","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\n")
					mustWrite(t, w, "data: [DONE]\n\n")
					return
				}
				mustWrite(t, w, `data: {"id":"c1","model":"Qwen/Qwen3.6-27B-FP8","choices":[{"index":0,"delta":{"content":"po"},"finish_reason":null}]}`+"\n\n")
				mustWrite(t, w, `data: {"id":"c1","model":"Qwen/Qwen3.6-27B-FP8","choices":[{"index":0,"delta":{"content":"ng"},"finish_reason":"stop","stop_reason":null}]}`+"\n\n")
				mustWrite(t, w, `data: {"id":"c1","model":"Qwen/Qwen3.6-27B-FP8","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\n")
				mustWrite(t, w, "data: [DONE]\n\n")
				return
			}
			if scenario == llmtest.ConformanceCancel {
				<-r.Context().Done()
				return
			}
			w.Header().Set("Content-Type", "application/json")
			if scenario == llmtest.ConformanceTools {
				mustWrite(t, w, `{"id":"c1","model":"Qwen/Qwen3.6-27B-FP8","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_conformance","type":"function","function":{"name":"conformance_echo","arguments":"{\"value\":\"pong\"}"}}]}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
				return
			}
			mustWrite(t, w, `{"id":"c1","model":"Qwen/Qwen3.6-27B-FP8","choices":[{"index":0,"finish_reason":"stop","stop_reason":null,"message":{"role":"assistant","content":"pong","reasoning":null}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"prompt_tokens_details":null}}`)
		})
	})
}
