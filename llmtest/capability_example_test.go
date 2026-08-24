package llmtest_test

import (
	"fmt"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

// Provider packages pass this profile to RunCapabilityConformance from a
// normal Test function. Its factory receives the case identity out of band,
// constructs an isolated httptest fixture, and asserts the native wire body.
func ExampleCapabilityProfile() {
	profile := llmtest.CapabilityProfile{Cases: []llmtest.CapabilityCase{{
		Name:       "required-tool-choice",
		Capability: llm.CapabilityToolChoiceRequired,
		Paths:      []llmtest.ConformancePath{llmtest.ConformanceChat},
		Request: func() *llm.Request {
			return &llm.Request{
				Model:      "real-provider-model",
				Tools:      []llm.Tool{{Name: "lookup", InputSchema: map[string]any{"type": "object"}}},
				ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceRequired},
			}
		},
		Assert: func(*llm.Response) error { return nil },
	}}}

	fmt.Println(profile.Cases[0].Name, profile.Cases[0].Request().Model)
	// Output: required-tool-choice real-provider-model
}
