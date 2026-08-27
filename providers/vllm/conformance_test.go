package vllm

import (
	"encoding/json"
	"io"
	"net/http"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
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

// TestVLLMCapabilityConformance proves every reviewed capability advertised by
// the preset. Activation of tools, reasoning, and multimodal input still
// depends on the documented vLLM server flags/model; this fixture proves the
// configured request shape deterministically without claiming live admission.
func TestVLLMCapabilityConformance(t *testing.T) {
	const model = "Qwen/Qwen3.6-27B-FP8"
	profile := testutil.CompatibleCapabilityProfile(model)
	for index := range profile.Cases {
		if profile.Cases[index].Capability != llm.CapabilityReasoning {
			continue
		}
		profile.Cases[index].Request = func() *llm.Request {
			budget := 2048
			enabled := true
			return &llm.Request{
				Model:    model,
				Messages: []llm.Message{llm.UserText("reason")},
				Effort:   llm.EffortHigh,
				ProviderOptions: Options{
					ThinkingTokenBudget: &budget,
					EnableThinking:      &enabled,
				},
			}
		}
	}

	llmtest.RunCapabilityConformance(t, func(t *testing.T, invocation llmtest.CapabilityInvocation) llm.Provider {
		return newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Errorf("decode activation request: %v", err)
				http.Error(w, "invalid JSON", http.StatusBadRequest)
				return
			}
			if err := testutil.AssertCompatibleActivationRequest(body, invocation, model); err != nil {
				t.Errorf("%s native request: %v; body=%#v", invocation.CaseName, err, body)
			}
			if invocation.Capability == llm.CapabilityReasoning {
				kwargs, _ := body["chat_template_kwargs"].(map[string]any)
				if body["reasoning_effort"] != "high" || body["thinking_token_budget"] != float64(2048) || kwargs["enable_thinking"] != true {
					t.Errorf("configured reasoning fields = effort %#v budget %#v kwargs %#v", body["reasoning_effort"], body["thinking_token_budget"], kwargs)
				}
			}
			if invocation.Path == llmtest.ConformanceStream {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, testutil.CompatibleActivationStream(invocation, model))
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, testutil.CompatibleActivationResponse(invocation, model))
		})
	}, profile)
}
