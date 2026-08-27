package openrouter

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
	"github.com/pkieltyka/go-llm/llmtest"
)

// TestOpenRouterConformance machine-checks the llm.Provider contract
// (single-use streams, cancellation, concurrency, panic-freedom, Collect
// partial shape) against a fixture handler speaking the chat-completions
// wire shape through the shared engine.
func TestOpenRouterConformance(t *testing.T) {
	llmtest.RunConformance(t, func(t *testing.T) llm.Provider {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			scenario := llmtest.ConformanceScenarioFromRequest(r)
			if r.Header.Get("Accept") == "text/event-stream" {
				w.Header().Set("Content-Type", "text/event-stream")
				if scenario == llmtest.ConformanceEmpty {
					return
				}
				_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"role":"assistant"}}]}`+"\n\n")
				switch scenario {
				case llmtest.ConformanceCancel:
					w.(http.Flusher).Flush()
					<-r.Context().Done()
					return
				case llmtest.ConformanceTruncated:
					return
				case llmtest.ConformanceTools:
					_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_conformance","type":"function","function":{"name":"conformance_echo","arguments":"{\"value\":\"pong\"}"}}]}}]}`+"\n\n")
					_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`+"\n\n")
					_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"cost":0.00001}}`+"\n\n")
					_, _ = io.WriteString(w, "data: [DONE]\n\n")
					return
				}
				_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"content":"po"}}]}`+"\n\n")
				_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"delta":{"content":"ng"},"finish_reason":"stop"}]}`+"\n\n")
				_, _ = io.WriteString(w, `data: {"id":"gen_1","model":"openai/gpt-test","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"cost":0.00001}}`+"\n\n")
				_, _ = io.WriteString(w, "data: [DONE]\n\n")
				return
			}
			if scenario == llmtest.ConformanceCancel {
				<-r.Context().Done()
				return
			}
			w.Header().Set("Content-Type", "application/json")
			if scenario == llmtest.ConformanceTools {
				_, _ = io.WriteString(w, `{"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_conformance","type":"function","function":{"name":"conformance_echo","arguments":"{\"value\":\"pong\"}"}}]}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"cost":0.00001}}`)
				return
			}
			_, _ = io.WriteString(w, `{"id":"gen_1","model":"openai/gpt-test","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"pong"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"cost":0.00001}}`)
		}))
		t.Cleanup(server.Close)
		p, err := New(
			WithAPIKey("test"),
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

// TestOpenRouterCapabilityConformance proves compatible fields plus
// OpenRouter's nested reasoning and Anthropic-style cache-control passthrough.
// Positive backend cache admission remains live evidence, not a fixture claim.
func TestOpenRouterCapabilityConformance(t *testing.T) {
	const model = "anthropic/claude-test"
	profile := testutil.CompatibleCapabilityProfile(model)
	profile.Cases = append(profile.Cases,
		llmtest.CapabilityCase{
			Name:       "prompt-cache-content",
			Capability: llm.CapabilityPromptCaching,
			Paths:      []llmtest.ConformancePath{llmtest.ConformanceChat, llmtest.ConformanceStream},
			Request: func() *llm.Request {
				return &llm.Request{Model: model, Messages: []llm.Message{llm.UserParts(llm.TextPart{Text: "cache me", Cache: &llm.CacheHint{TTL: time.Hour}})}}
			},
			Assert: assertOpenRouterCacheResponse,
		},
		llmtest.CapabilityCase{
			Name:       "prompt-cache-tool-result",
			Capability: llm.CapabilityPromptCaching,
			Paths:      []llmtest.ConformancePath{llmtest.ConformanceChat, llmtest.ConformanceStream},
			Request: func() *llm.Request {
				return &llm.Request{Model: model, Messages: []llm.Message{{Role: llm.RoleTool, Parts: []llm.Part{
					llm.ToolResultPart{ToolCallID: "call_activation", Content: []llm.Part{llm.TextPart{Text: "cached result", Cache: &llm.CacheHint{}}}},
				}}}}
			},
			Assert: assertOpenRouterCacheResponse,
		},
	)

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
			switch invocation.CaseName {
			case "reasoning":
				reasoning, _ := body["reasoning"].(map[string]any)
				if reasoning["effort"] != "high" {
					t.Errorf("reasoning = %#v, want effort high", reasoning)
				}
			case "prompt-cache-content":
				if err := assertOpenRouterCacheBlock(body, false, "1h"); err != nil {
					t.Errorf("content cache request: %v; body=%#v", err, body)
				}
			case "prompt-cache-tool-result":
				if err := assertOpenRouterCacheBlock(body, true, ""); err != nil {
					t.Errorf("tool-result cache request: %v; body=%#v", err, body)
				}
			}
			if invocation.Path == llmtest.ConformanceStream {
				w.Header().Set("Content-Type", "text/event-stream")
				if invocation.Capability == llm.CapabilityPromptCaching {
					_, _ = io.WriteString(w, openRouterCacheActivationStream(model))
				} else {
					_, _ = io.WriteString(w, testutil.CompatibleActivationStream(invocation, model))
				}
				return
			}
			w.Header().Set("Content-Type", "application/json")
			if invocation.Capability == llm.CapabilityPromptCaching {
				_, _ = io.WriteString(w, `{"id":"activation_response","model":"anthropic/claude-test","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"activated"}}],"usage":{"prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":2},"completion_tokens":1,"total_tokens":4}}`)
				return
			}
			_, _ = io.WriteString(w, testutil.CompatibleActivationResponse(invocation, model))
		}))
		t.Cleanup(server.Close)
		provider, err := New(WithAPIKey("test"), WithBaseURL(server.URL), WithHTTPClient(server.Client()), WithMaxRetries(0))
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		return provider
	}, profile)
}

func openRouterCacheActivationStream(model string) string {
	return `data: {"id":"activation_response","model":"` + model + `","choices":[{"index":0,"delta":{"role":"assistant","content":"activated"},"finish_reason":"stop"}]}` + "\n\n" +
		`data: {"id":"activation_response","model":"` + model + `","choices":[],"usage":{"prompt_tokens":3,"prompt_tokens_details":{"cached_tokens":2},"completion_tokens":1,"total_tokens":4}}` + "\n\n" +
		"data: [DONE]\n\n"
}

func assertOpenRouterCacheBlock(body map[string]any, toolResult bool, ttl string) error {
	messages, _ := body["messages"].([]any)
	if len(messages) != 1 {
		return fmt.Errorf("messages = %#v", body["messages"])
	}
	message, _ := messages[0].(map[string]any)
	content, _ := message["content"].([]any)
	if len(content) != 1 {
		return fmt.Errorf("content = %#v", message["content"])
	}
	if toolResult && message["role"] != "tool" {
		return fmt.Errorf("role = %#v, want tool", message["role"])
	}
	block, _ := content[0].(map[string]any)
	cache, _ := block["cache_control"].(map[string]any)
	if cache["type"] != "ephemeral" {
		return fmt.Errorf("cache_control = %#v", cache)
	}
	if ttl == "" {
		if _, present := cache["ttl"]; present {
			return fmt.Errorf("default cache_control unexpectedly contains ttl: %#v", cache)
		}
	} else if cache["ttl"] != ttl {
		return fmt.Errorf("cache_control ttl = %#v, want %q", cache["ttl"], ttl)
	}
	return nil
}

func assertOpenRouterCacheResponse(response *llm.Response) error {
	if err := testutil.AssertActivationText(response); err != nil {
		return err
	}
	if response.Usage.InputTokens != 1 || response.Usage.CacheReadTokens != 2 {
		return fmt.Errorf("cache usage = input %d read %d", response.Usage.InputTokens, response.Usage.CacheReadTokens)
	}
	return nil
}

// TestProviderIdentitySurface pins the trivial identity accessors.
func TestProviderIdentitySurface(t *testing.T) {
	p, err := New(WithAPIKey("test"))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if p.Name() != providerName {
		t.Fatalf("Name = %q, want %q", p.Name(), providerName)
	}
	if p.Client() == nil {
		t.Fatalf("Client() = nil")
	}
	if (*Provider)(nil).Client() != nil {
		t.Fatalf("nil provider Client() should be nil")
	}
	caps := p.Capabilities()
	if len(caps) == 0 || caps[0] != llm.CapabilityStreaming {
		t.Fatalf("capabilities = %+v", caps)
	}
}

var _ llm.Provider = (*Provider)(nil)
