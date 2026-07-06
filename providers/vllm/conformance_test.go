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
			if r.Header.Get("Accept") == "text/event-stream" {
				w.Header().Set("Content-Type", "text/event-stream")
				mustWrite(t, w, `data: {"id":"c1","model":"Qwen/Qwen3.6-27B-FP8","choices":[{"index":0,"delta":{"role":"assistant","content":"po"},"finish_reason":null}]}`+"\n\n")
				mustWrite(t, w, `data: {"id":"c1","model":"Qwen/Qwen3.6-27B-FP8","choices":[{"index":0,"delta":{"content":"ng"},"finish_reason":"stop","stop_reason":null}]}`+"\n\n")
				mustWrite(t, w, `data: {"id":"c1","model":"Qwen/Qwen3.6-27B-FP8","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\n")
				mustWrite(t, w, "data: [DONE]\n\n")
				return
			}
			w.Header().Set("Content-Type", "application/json")
			mustWrite(t, w, `{"id":"c1","model":"Qwen/Qwen3.6-27B-FP8","choices":[{"index":0,"finish_reason":"stop","stop_reason":null,"message":{"role":"assistant","content":"pong","reasoning":null}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"prompt_tokens_details":null}}`)
		})
	})
}
