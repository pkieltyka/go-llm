package llmtest

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

// ConformancePath identifies the provider entry point exercised by a
// capability activation case.
type ConformancePath string

const (
	// ConformanceChat exercises Provider.Chat.
	ConformanceChat ConformancePath = "chat"
	// ConformanceStream exercises Provider.ChatStream and fully collects it.
	ConformanceStream ConformancePath = "stream"
)

// CapabilityInvocation is fixture control data passed out of band to a
// CapabilityProviderFactory. RunCapabilityConformance never copies it into or
// derives it from an llm.Request.
type CapabilityInvocation struct {
	CaseName   string
	Capability llm.Capability
	Path       ConformancePath
}

// CapabilityCase proves that a standard capability activates the expected
// native provider request and produces an expected normalized response.
type CapabilityCase struct {
	Name       string
	Capability llm.Capability
	Paths      []ConformancePath
	Request    func() *llm.Request
	Assert     func(*llm.Response) error
}

// CapabilityExemption records a reviewed gap in offline activation evidence.
// It is evidence metadata, not a denial of the provider's capability claim.
type CapabilityExemption struct {
	Capability llm.Capability
	Reason     string
}

// CapabilityProfile is the complete offline activation evidence for the
// reviewed capabilities advertised by a provider.
type CapabilityProfile struct {
	Cases      []CapabilityCase
	Exemptions []CapabilityExemption
}

// CapabilityProviderFactory constructs an isolated provider fixture. The
// runner first calls it with a zero invocation to probe identity and claims,
// then once per case/path with the exact invocation being executed. Each
// populated factory call happens before that path's CapabilityCase.Request
// callback; all providers and requests are prepared before dispatch begins.
type CapabilityProviderFactory func(t *testing.T, invocation CapabilityInvocation) llm.Provider

var reviewedCapabilityActivations = []llm.Capability{
	llm.CapabilityTools,
	llm.CapabilityToolChoiceRequired,
	llm.CapabilityJSONSchema,
	llm.CapabilityImageInput,
	llm.CapabilityPromptCaching,
	llm.CapabilityStopSequences,
	llm.CapabilityReasoning,
}

var capabilityConformanceStreamTimeout = conformanceStreamTimeout

type preparedCapabilityInvocation struct {
	caseDef    CapabilityCase
	invocation CapabilityInvocation
	provider   llm.Provider
	request    *llm.Request
}

// RunCapabilityConformance runs a provider's offline native-activation
// profile. Unlike RunConformance, it preserves real request models and passes
// case identity only to the fixture factory. Fixture handlers own native wire
// assertions; this runner owns profile integrity, isolated execution, complete
// stream collection, and normalized result assertions.
func RunCapabilityConformance(t *testing.T, newProvider CapabilityProviderFactory, profile CapabilityProfile) {
	t.Helper()
	if newProvider == nil {
		t.Fatal("RunCapabilityConformance requires a provider factory")
	}

	probe := newProvider(t, CapabilityInvocation{})
	if nilProvider(probe) {
		t.Fatal("RunCapabilityConformance probe factory returned a nil provider")
	}
	providerName := probe.Name()
	probeClaims := reviewedClaims(probe.Capabilities())
	if err := validateCapabilityProfile(providerName, probeClaims, profile); err != nil {
		t.Fatal(err)
	}

	prepared := make([]preparedCapabilityInvocation, 0)
	for _, caseDef := range profile.Cases {
		for _, path := range caseDef.Paths {
			invocation := CapabilityInvocation{
				CaseName:   caseDef.Name,
				Capability: caseDef.Capability,
				Path:       path,
			}
			contextPrefix := capabilityContext(providerName, invocation)
			provider := newProvider(t, invocation)
			if nilProvider(provider) {
				t.Fatalf("%s: factory returned a nil provider", contextPrefix)
			}
			if got := reviewedClaims(provider.Capabilities()); !reflect.DeepEqual(got, probeClaims) {
				t.Fatalf("%s: factory changed reviewed capability claims: got %v, probe advertised %v", contextPrefix, sortedClaims(got), sortedClaims(probeClaims))
			}
			request := caseDef.Request()
			if request == nil {
				t.Fatalf("provider %q capability case %q (%s) path %q: Request returned nil", providerName, caseDef.Name, caseDef.Capability, path)
			}
			prepared = append(prepared, preparedCapabilityInvocation{caseDef: caseDef, invocation: invocation, provider: provider, request: request})
		}
	}

	for _, item := range prepared {
		item := item
		t.Run(item.caseDef.Name+"/"+string(item.invocation.Path), func(t *testing.T) {
			contextPrefix := capabilityContext(providerName, item.invocation)
			ctx, cancel := context.WithTimeout(context.Background(), capabilityConformanceStreamTimeout)
			defer cancel()
			var (
				response *llm.Response
				err      error
			)
			switch item.invocation.Path {
			case ConformanceChat:
				response, err = item.provider.Chat(ctx, item.request)
			case ConformanceStream:
				response, err = collectCapabilityStream(ctx, item.provider, item.request)
			}
			if err != nil {
				t.Fatalf("%s: provider call failed: %v", contextPrefix, err)
			}
			if response == nil {
				t.Fatalf("%s: provider call returned a nil response", contextPrefix)
			}
			if err := item.caseDef.Assert(response); err != nil {
				t.Fatalf("%s: normalized response assertion failed: %v", contextPrefix, err)
			}
		})
	}
}

func validateCapabilityProfile(providerName string, advertised map[llm.Capability]struct{}, profile CapabilityProfile) error {
	caseNames := make(map[string]struct{}, len(profile.Cases))
	caseCapabilities := make(map[llm.Capability]struct{})
	for index, caseDef := range profile.Cases {
		if strings.TrimSpace(caseDef.Name) == "" {
			return fmt.Errorf("provider %q capability profile case %d has an empty name", providerName, index)
		}
		if _, duplicate := caseNames[caseDef.Name]; duplicate {
			return fmt.Errorf("provider %q capability profile has duplicate case name %q", providerName, caseDef.Name)
		}
		caseNames[caseDef.Name] = struct{}{}
		if err := validateCoveredCapability(providerName, "case "+fmt.Sprintf("%q", caseDef.Name), caseDef.Capability, advertised); err != nil {
			return err
		}
		if caseDef.Request == nil {
			return fmt.Errorf("provider %q capability case %q (%s) has a nil Request callback", providerName, caseDef.Name, caseDef.Capability)
		}
		if caseDef.Assert == nil {
			return fmt.Errorf("provider %q capability case %q (%s) has a nil Assert callback", providerName, caseDef.Name, caseDef.Capability)
		}
		if len(caseDef.Paths) == 0 {
			return fmt.Errorf("provider %q capability case %q (%s) has no paths", providerName, caseDef.Name, caseDef.Capability)
		}
		paths := make(map[ConformancePath]struct{}, len(caseDef.Paths))
		for _, path := range caseDef.Paths {
			if path != ConformanceChat && path != ConformanceStream {
				if path == "" {
					return fmt.Errorf("provider %q capability case %q (%s) has an empty path", providerName, caseDef.Name, caseDef.Capability)
				}
				return fmt.Errorf("provider %q capability case %q (%s) has unknown path %q", providerName, caseDef.Name, caseDef.Capability, path)
			}
			if _, duplicate := paths[path]; duplicate {
				return fmt.Errorf("provider %q capability case %q (%s) has duplicate path %q", providerName, caseDef.Name, caseDef.Capability, path)
			}
			paths[path] = struct{}{}
		}
		caseCapabilities[caseDef.Capability] = struct{}{}
	}

	exemptions := make(map[llm.Capability]struct{}, len(profile.Exemptions))
	for index, exemption := range profile.Exemptions {
		label := fmt.Sprintf("exemption %d", index)
		if err := validateCoveredCapability(providerName, label, exemption.Capability, advertised); err != nil {
			return err
		}
		if _, duplicate := exemptions[exemption.Capability]; duplicate {
			return fmt.Errorf("provider %q capability profile has duplicate exemption for %s", providerName, exemption.Capability)
		}
		if strings.TrimSpace(exemption.Reason) == "" {
			return fmt.Errorf("provider %q capability exemption for %s has a blank reason", providerName, exemption.Capability)
		}
		if _, covered := caseCapabilities[exemption.Capability]; covered {
			return fmt.Errorf("provider %q capability %s has both activation cases and an exemption", providerName, exemption.Capability)
		}
		exemptions[exemption.Capability] = struct{}{}
	}

	for _, capability := range reviewedCapabilityActivations {
		if _, claimed := advertised[capability]; !claimed {
			continue
		}
		_, covered := caseCapabilities[capability]
		_, exempt := exemptions[capability]
		if !covered && !exempt {
			return fmt.Errorf("provider %q advertises reviewed capability %s but profile has no activation case or exemption", providerName, capability)
		}
	}
	return nil
}

func validateCoveredCapability(providerName, entry string, capability llm.Capability, advertised map[llm.Capability]struct{}) error {
	if capability == "" {
		return fmt.Errorf("provider %q capability profile %s has an empty capability", providerName, entry)
	}
	if !isReviewedCapability(capability) {
		return fmt.Errorf("provider %q capability profile %s uses unreviewed capability %s", providerName, entry, capability)
	}
	if _, ok := advertised[capability]; !ok {
		return fmt.Errorf("provider %q capability profile %s covers capability %s that the provider does not advertise", providerName, entry, capability)
	}
	return nil
}

func isReviewedCapability(capability llm.Capability) bool {
	for _, reviewed := range reviewedCapabilityActivations {
		if capability == reviewed {
			return true
		}
	}
	return false
}

func reviewedClaims(capabilities []llm.Capability) map[llm.Capability]struct{} {
	claims := make(map[llm.Capability]struct{})
	for _, capability := range capabilities {
		if isReviewedCapability(capability) {
			claims[capability] = struct{}{}
		}
	}
	return claims
}

func sortedClaims(claims map[llm.Capability]struct{}) []llm.Capability {
	out := make([]llm.Capability, 0, len(claims))
	for _, capability := range reviewedCapabilityActivations {
		if _, ok := claims[capability]; ok {
			out = append(out, capability)
		}
	}
	return out
}

func nilProvider(provider llm.Provider) bool {
	if provider == nil {
		return true
	}
	value := reflect.ValueOf(provider)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

func capabilityContext(providerName string, invocation CapabilityInvocation) string {
	return fmt.Sprintf("provider %q capability case %q (%s) path %q", providerName, invocation.CaseName, invocation.Capability, invocation.Path)
}

func collectCapabilityStream(ctx context.Context, provider llm.Provider, request *llm.Request) (*llm.Response, error) {
	type collected struct {
		response *llm.Response
		err      error
	}
	done := make(chan collected, 1)
	go func() {
		response, err := llm.Collect(provider.ChatStream(ctx, request))
		done <- collected{response: response, err: err}
	}()
	select {
	case result := <-done:
		return result.response, result.err
	case <-ctx.Done():
		return nil, fmt.Errorf("stream did not finish within %s: %w", time.Duration(capabilityConformanceStreamTimeout), ctx.Err())
	}
}
