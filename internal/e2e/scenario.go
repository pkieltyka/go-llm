package e2e

import (
	"context"
	"encoding/base64"
	"path/filepath"
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
	Name string
	Run  ScenarioRun
}

// ScenarioReport records the capability-applicable scenarios and the subset
// that actually ran to completion. Test filtering and scenario-local skips
// therefore remain visible to fixture recording.
type ScenarioReport struct {
	Expected  []string
	Completed []string
}

type recordingSecretsContextKey struct{}

func RecordingContext(ctx context.Context, captures *CaptureLog, secrets *SecretSet) context.Context {
	ctx = llm.WithWireCaptureObserver(ctx, captures)
	return context.WithValue(ctx, recordingSecretsContextKey{}, secrets)
}

func ScheduleFixtureRecording(t *testing.T, path string, captures *CaptureLog, secrets *SecretSet, report *ScenarioReport, allowIncomplete bool) {
	t.Helper()
	t.Cleanup(func() {
		if t.Failed() || t.Skipped() {
			t.Logf("WARNING: fixture recording for %s was not written because the live test failed or skipped", filepath.ToSlash(path))
			return
		}
		snapshot := captures.Snapshot()
		result, err := WriteFixtureChecked(path, snapshot.Captures, FixtureWriteOptions{
			Secrets:                   secrets.Values(),
			ExpectedScenarios:         report.Expected,
			CompletedScenarios:        report.Completed,
			OutstandingResponseBodies: snapshot.OutstandingResponseBodies,
			AllowIncomplete:           allowIncomplete,
			Warnf:                     t.Logf,
		})
		if err != nil {
			t.Errorf("write fixture %s: %v", filepath.ToSlash(path), err)
			return
		}
		if result.Replaced {
			t.Logf("recorded fixture %s", filepath.ToSlash(path))
		}
	})
}

// RunCapabilityScenarios derives the provider's live suite from its advertised
// capabilities and the explicit provider profile.
func RunCapabilityScenarios(ctx context.Context, t *testing.T, providerID string, p llm.Provider, model string, runners map[string]ScenarioRun) ScenarioReport {
	t.Helper()
	scenarios, exemptions, err := liveScenarios(providerID, p, runners)
	if err != nil {
		t.Fatalf("build %s live scenarios: %v", providerID, err)
	}
	for _, exemption := range exemptions {
		t.Logf("live capability exemption: provider=%s capability=%s reason=%s", exemption.Provider, exemption.Capability, exemption.Reason)
	}
	return RunScenarios(ctx, t, p, model, scenarios)
}

// RunScenarios executes the already capability-derived scenario list.
func RunScenarios(ctx context.Context, t *testing.T, p llm.Provider, model string, scenarios []Scenario) ScenarioReport {
	t.Helper()
	report := ScenarioReport{}
	for _, scenario := range scenarios {
		scenario := scenario
		report.Expected = append(report.Expected, scenario.Name)
		completed := false
		t.Run(scenario.Name, func(t *testing.T) {
			defer func() {
				completed = !t.Failed() && !t.Skipped()
			}()
			scenario.Run(ctx, t, p, model)
		})
		if completed {
			report.Completed = append(report.Completed, scenario.Name)
		}
	}
	return report
}
