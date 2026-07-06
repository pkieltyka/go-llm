package e2e

import (
	"context"
	"encoding/base64"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

// redPixelPNGBase64 is the small red-square PNG used across live scenarios
// (multimodal input and tool-result image content).
const redPixelPNGBase64 = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQAQMAAAAlPW0iAAAAA1BMVEX/AAAZ4gk3AAAADElEQVQI12NgIA0AAAAwAAHHqoWOAAAAAElFTkSuQmCC"

// RedPixelPNG decodes the shared red-square PNG fixture.
func RedPixelPNG(t *testing.T) []byte {
	t.Helper()
	image, err := base64.StdEncoding.DecodeString(redPixelPNGBase64)
	if err != nil {
		t.Fatalf("decode fixture image: %v", err)
	}
	return image
}

// Scenario is a provider-neutral live e2e check.
type Scenario struct {
	Name       string
	Capability llm.Capability
	Run        func(context.Context, *testing.T, llm.Provider, string)
}

// RunScenarios executes scenarios whose required capability is declared by p.
func RunScenarios(ctx context.Context, t *testing.T, p llm.Provider, model string, scenarios []Scenario) {
	t.Helper()
	caps := map[llm.Capability]struct{}{}
	for _, cap := range p.Capabilities() {
		caps[cap] = struct{}{}
	}
	for _, scenario := range scenarios {
		scenario := scenario
		t.Run(scenario.Name, func(t *testing.T) {
			if scenario.Capability != "" {
				if _, ok := caps[scenario.Capability]; !ok {
					t.Skipf("provider %s lacks capability %s", p.Name(), scenario.Capability)
				}
			}
			scenario.Run(ctx, t, p, model)
		})
	}
}
