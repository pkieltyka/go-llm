package llmtest_test

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
)

// CapabilityProfile describes the request and normalized response sides of a
// provider-owned activation fixture.
func ExampleCapabilityProfile() {
	profile := llmtest.CapabilityProfile{Cases: []llmtest.CapabilityCase{{
		Name:       "required-tool-choice",
		Capability: llm.CapabilityToolChoiceRequired,
		Paths:      []llmtest.ConformancePath{llmtest.ConformanceChat},
		Request: func() *llm.Request {
			return &llm.Request{
				Model:      "real-provider-model",
				Messages:   []llm.Message{llm.UserText("look up the value")},
				Tools:      []llm.Tool{{Name: "lookup", InputSchema: map[string]any{"type": "object"}}},
				ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceRequired},
			}
		},
		Assert: func(*llm.Response) error { return nil },
	}}}

	fmt.Println(profile.Cases[0].Name, profile.Cases[0].Request().Model)
	// Output: required-tool-choice real-provider-model
}

// TestCapabilityConformanceRunnerExample demonstrates the complete runner
// contract: zero-invocation probing, out-of-band case identity, native wire
// assertion, and normalized response assertion.
func TestCapabilityConformanceRunnerExample(t *testing.T) {
	const model = "real-provider-model"
	profile := llmtest.CapabilityProfile{Cases: []llmtest.CapabilityCase{{
		Name: "tools", Capability: llm.CapabilityTools, Paths: []llmtest.ConformancePath{llmtest.ConformanceChat},
		Request: func() *llm.Request {
			return &llm.Request{
				Model: model, Messages: []llm.Message{llm.UserText("call the tool")},
				Tools: []llm.Tool{{Name: "lookup", InputSchema: map[string]any{"type": "object", "properties": map[string]any{}}}},
			}
		},
		Assert: func(response *llm.Response) error {
			if response.Text() != "activated" || response.StopReason != llm.StopReasonEndTurn {
				return fmt.Errorf("response text=%q stop=%q", response.Text(), response.StopReason)
			}
			return nil
		},
	}}}
	probeSeen := false
	llmtest.RunCapabilityConformance(t, func(t *testing.T, invocation llmtest.CapabilityInvocation) llm.Provider {
		if invocation == (llmtest.CapabilityInvocation{}) {
			probeSeen = true
		}
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if !probeSeen || invocation.CaseName != "tools" || invocation.Path != llmtest.ConformanceChat {
				t.Errorf("fixture invocation = %#v after probe=%v", invocation, probeSeen)
			}
			var body struct {
				Tools []struct {
					Type     string `json:"type"`
					Function struct {
						Name string `json:"name"`
					} `json:"function"`
				} `json:"tools"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body.Tools) != 1 || body.Tools[0].Type != "function" || body.Tools[0].Function.Name != "lookup" {
				t.Errorf("native tools request = %#v, decode error %v", body.Tools, err)
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"response","model":"real-provider-model","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"activated"}}]}`)
		}))
		t.Cleanup(server.Close)
		provider, err := chatcompletions.New(server.URL, chatcompletions.WithHTTPClient(server.Client()), chatcompletions.WithMaxRetries(0))
		if err != nil {
			t.Fatalf("chatcompletions.New: %v", err)
		}
		return toolsOnlyProvider{Provider: provider}
	}, profile)
}

type toolsOnlyProvider struct{ llm.Provider }

func (toolsOnlyProvider) Capabilities() []llm.Capability {
	return []llm.Capability{llm.CapabilityTools}
}
