package llmtest

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"os"
	"os/exec"
	"reflect"
	"strings"
	"sync"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestRunCapabilityConformancePreservesModelAndInvocation(t *testing.T) {
	const model = "llmtest-conformance-tools:gpt-5.6/real?branch=lite"
	var mu sync.Mutex
	var invocations []CapabilityInvocation
	var providers []*Provider
	var order []string
	requestNumber := 0

	RunCapabilityConformance(t, func(t *testing.T, invocation CapabilityInvocation) llm.Provider {
		mu.Lock()
		invocations = append(invocations, invocation)
		order = append(order, "factory/"+string(invocation.Path))
		mu.Unlock()
		provider := New(WithName("activation-test"), WithCapabilities(llm.CapabilityStreaming, llm.CapabilityTools))
		if invocation == (CapabilityInvocation{}) {
			return provider
		}
		providers = append(providers, provider)
		switch invocation.Path {
		case ConformanceChat:
			provider.EnqueueResponse(capabilityResponse(model))
		case ConformanceStream:
			provider.EnqueueStream(
				llm.MessageStart{ID: "response", Provider: "activation-test", Model: model},
				llm.TextDelta{Index: 0, Text: "activated"},
				llm.MessageEnd{StopReason: llm.StopReasonEndTurn, Usage: llm.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2}},
			)
		}
		return provider
	}, CapabilityProfile{Cases: []CapabilityCase{{
		Name:       "native-tools",
		Capability: llm.CapabilityTools,
		Paths:      []ConformancePath{ConformanceChat, ConformanceStream},
		Request: func() *llm.Request {
			mu.Lock()
			requestNumber++
			order = append(order, fmt.Sprintf("request/%d", requestNumber))
			mu.Unlock()
			return &llm.Request{Model: model, Messages: []llm.Message{llm.UserText("activate")}}
		},
		Assert: func(response *llm.Response) error {
			if response.Model != model || response.Text() != "activated" {
				return fmt.Errorf("response = model %q text %q", response.Model, response.Text())
			}
			return nil
		},
	}}})

	mu.Lock()
	defer mu.Unlock()
	wantInvocations := []CapabilityInvocation{
		{},
		{CaseName: "native-tools", Capability: llm.CapabilityTools, Path: ConformanceChat},
		{CaseName: "native-tools", Capability: llm.CapabilityTools, Path: ConformanceStream},
	}
	if !reflect.DeepEqual(invocations, wantInvocations) {
		t.Fatalf("factory invocations = %#v, want %#v", invocations, wantInvocations)
	}
	wantOrder := []string{"factory/", "factory/chat", "request/1", "factory/stream", "request/2"}
	if !reflect.DeepEqual(order, wantOrder) {
		t.Fatalf("factory/request order = %#v, want %#v", order, wantOrder)
	}
	var requests []*llm.Request
	for _, provider := range providers {
		requests = append(requests, provider.Requests()...)
	}
	if len(requests) != 2 {
		t.Fatalf("recorded requests = %d, want 2", len(requests))
	}
	for _, request := range requests {
		if request.Model != model {
			t.Fatalf("request model = %q, want byte-for-byte %q", request.Model, model)
		}
	}
}

func TestValidateCapabilityProfile(t *testing.T) {
	validCase := func(capability llm.Capability) CapabilityCase {
		return CapabilityCase{
			Name:       "case",
			Capability: capability,
			Paths:      []ConformancePath{ConformanceChat},
			Request:    func() *llm.Request { return &llm.Request{} },
			Assert:     func(*llm.Response) error { return nil },
		}
	}
	advertised := reviewedClaims(reviewedCapabilityActivations)
	tests := []struct {
		name    string
		profile CapabilityProfile
		claims  map[llm.Capability]struct{}
		want    string
	}{
		{name: "valid cases and exemption", profile: CapabilityProfile{Cases: []CapabilityCase{validCase(llm.CapabilityTools)}, Exemptions: exemptionsExcept(llm.CapabilityTools)}, claims: advertised},
		{name: "empty name", profile: CapabilityProfile{Cases: []CapabilityCase{{Capability: llm.CapabilityTools}}}, claims: advertised, want: "empty name"},
		{name: "duplicate name", profile: CapabilityProfile{Cases: []CapabilityCase{validCase(llm.CapabilityTools), validCase(llm.CapabilityJSONSchema)}}, claims: advertised, want: "duplicate case name"},
		{name: "empty capability", profile: CapabilityProfile{Cases: []CapabilityCase{validCase("")}}, claims: advertised, want: "empty capability"},
		{name: "custom capability", profile: CapabilityProfile{Cases: []CapabilityCase{validCase("vendor/custom")}}, claims: advertised, want: "unreviewed capability"},
		{name: "out of target capability", profile: CapabilityProfile{Cases: []CapabilityCase{validCase(llm.CapabilityStreaming)}}, claims: advertised, want: "unreviewed capability"},
		{name: "stale capability", profile: CapabilityProfile{Cases: []CapabilityCase{validCase(llm.CapabilityTools)}}, claims: map[llm.Capability]struct{}{}, want: "does not advertise"},
		{name: "nil request", profile: CapabilityProfile{Cases: []CapabilityCase{replaceCase(validCase(llm.CapabilityTools), func(c *CapabilityCase) { c.Request = nil })}}, claims: advertised, want: "nil Request"},
		{name: "nil assert", profile: CapabilityProfile{Cases: []CapabilityCase{replaceCase(validCase(llm.CapabilityTools), func(c *CapabilityCase) { c.Assert = nil })}}, claims: advertised, want: "nil Assert"},
		{name: "no paths", profile: CapabilityProfile{Cases: []CapabilityCase{replaceCase(validCase(llm.CapabilityTools), func(c *CapabilityCase) { c.Paths = nil })}}, claims: advertised, want: "no paths"},
		{name: "empty path", profile: CapabilityProfile{Cases: []CapabilityCase{replaceCase(validCase(llm.CapabilityTools), func(c *CapabilityCase) { c.Paths = []ConformancePath{""} })}}, claims: advertised, want: "empty path"},
		{name: "unknown path", profile: CapabilityProfile{Cases: []CapabilityCase{replaceCase(validCase(llm.CapabilityTools), func(c *CapabilityCase) { c.Paths = []ConformancePath{"batch"} })}}, claims: advertised, want: "unknown path"},
		{name: "duplicate path", profile: CapabilityProfile{Cases: []CapabilityCase{replaceCase(validCase(llm.CapabilityTools), func(c *CapabilityCase) { c.Paths = []ConformancePath{ConformanceChat, ConformanceChat} })}}, claims: advertised, want: "duplicate path"},
		{name: "empty exemption", profile: CapabilityProfile{Exemptions: []CapabilityExemption{{Reason: "gap"}}}, claims: advertised, want: "empty capability"},
		{name: "custom exemption", profile: CapabilityProfile{Exemptions: []CapabilityExemption{{Capability: "vendor/custom", Reason: "gap"}}}, claims: advertised, want: "unreviewed capability"},
		{name: "stale exemption", profile: CapabilityProfile{Exemptions: []CapabilityExemption{{Capability: llm.CapabilityTools, Reason: "gap"}}}, claims: map[llm.Capability]struct{}{}, want: "does not advertise"},
		{name: "duplicate exemption", profile: CapabilityProfile{Exemptions: []CapabilityExemption{{Capability: llm.CapabilityTools, Reason: "one"}, {Capability: llm.CapabilityTools, Reason: "two"}}}, claims: advertised, want: "duplicate exemption"},
		{name: "blank exemption", profile: CapabilityProfile{Exemptions: []CapabilityExemption{{Capability: llm.CapabilityTools, Reason: " \t"}}}, claims: advertised, want: "blank reason"},
		{name: "contradictory exemption", profile: CapabilityProfile{Cases: []CapabilityCase{validCase(llm.CapabilityTools)}, Exemptions: []CapabilityExemption{{Capability: llm.CapabilityTools, Reason: "gap"}}}, claims: advertised, want: "both activation cases"},
		{name: "missing advertised capability", profile: CapabilityProfile{}, claims: map[llm.Capability]struct{}{llm.CapabilityTools: {}}, want: "no activation case or exemption"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateCapabilityProfile("fixture", tt.claims, tt.profile)
			if tt.want == "" {
				if err != nil {
					t.Fatalf("validateCapabilityProfile returned error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.want) || !strings.Contains(err.Error(), "fixture") {
				t.Fatalf("validateCapabilityProfile error = %v, want provider context and %q", err, tt.want)
			}
		})
	}
}

func TestRunCapabilityConformanceFailureContract(t *testing.T) {
	if failure := os.Getenv("GO_LLM_CAPABILITY_FAILURE"); failure != "" {
		runCapabilityFailureCase(t, failure)
		return
	}
	for _, failure := range []string{"nil_factory", "nil_probe", "typed_nil_probe", "nil_request_result", "nil_case_provider", "claim_drift", "provider_error", "nil_response", "assertion_error"} {
		t.Run(failure, func(t *testing.T) {
			command := exec.Command(os.Args[0], "-test.run=^TestRunCapabilityConformanceFailureContract$")
			command.Env = append(os.Environ(), "GO_LLM_CAPABILITY_FAILURE="+failure)
			output, err := command.CombinedOutput()
			if err == nil {
				t.Fatalf("failure child unexpectedly passed:\n%s", output)
			}
			text := string(output)
			if strings.Contains(text, "panic:") {
				t.Fatalf("failure child panicked instead of failing through testing.T:\n%s", text)
			}
			if !strings.Contains(text, failureExpectedText(failure)) {
				t.Fatalf("failure output missing %q:\n%s", failureExpectedText(failure), text)
			}
		})
	}
}

func TestCapabilityCallbacksRetainOrdinaryPanicSemantics(t *testing.T) {
	// validateCapabilityProfile deliberately checks callback presence without
	// invoking or recovering them. Panics from factories, Request, Assert, or
	// providers therefore remain ordinary caller test panics, outside the
	// malformed-profile guarantee.
	called := false
	profile := completeProfile(CapabilityCase{
		Name:       "panic-not-invoked",
		Capability: llm.CapabilityTools,
		Paths:      []ConformancePath{ConformanceChat},
		Request: func() *llm.Request {
			called = true
			panic("caller panic")
		},
		Assert: func(*llm.Response) error { return nil },
	})
	if err := validateCapabilityProfile("fixture", reviewedClaims(reviewedCapabilityActivations), profile); err != nil {
		t.Fatalf("validation returned error: %v", err)
	}
	if called {
		t.Fatal("structural validation invoked Request callback")
	}
}

func runCapabilityFailureCase(t *testing.T, failure string) {
	baseFactory := func(t *testing.T, invocation CapabilityInvocation) llm.Provider {
		provider := New(WithName("failure-fixture"), WithCapabilities(llm.CapabilityTools))
		if invocation != (CapabilityInvocation{}) {
			provider.EnqueueResponse(capabilityResponse("real-model"))
		}
		return provider
	}
	profile := CapabilityProfile{Cases: []CapabilityCase{{
		Name:       "tools-case",
		Capability: llm.CapabilityTools,
		Paths:      []ConformancePath{ConformanceChat},
		Request:    func() *llm.Request { return &llm.Request{Model: "real-model"} },
		Assert:     func(*llm.Response) error { return nil },
	}}}

	switch failure {
	case "nil_factory":
		RunCapabilityConformance(t, nil, CapabilityProfile{})
	case "nil_probe":
		RunCapabilityConformance(t, func(*testing.T, CapabilityInvocation) llm.Provider { return nil }, CapabilityProfile{})
	case "typed_nil_probe":
		RunCapabilityConformance(t, func(*testing.T, CapabilityInvocation) llm.Provider { return (*Provider)(nil) }, CapabilityProfile{})
	case "nil_request_result":
		profile.Cases[0].Request = func() *llm.Request { return nil }
		RunCapabilityConformance(t, baseFactory, profile)
	case "nil_case_provider":
		RunCapabilityConformance(t, func(t *testing.T, invocation CapabilityInvocation) llm.Provider {
			if invocation != (CapabilityInvocation{}) {
				return nil
			}
			return baseFactory(t, invocation)
		}, profile)
	case "claim_drift":
		RunCapabilityConformance(t, func(t *testing.T, invocation CapabilityInvocation) llm.Provider {
			if invocation != (CapabilityInvocation{}) {
				return New(WithName("failure-fixture"), WithCapabilities(llm.CapabilityJSONSchema))
			}
			return baseFactory(t, invocation)
		}, profile)
	case "provider_error":
		RunCapabilityConformance(t, func(t *testing.T, invocation CapabilityInvocation) llm.Provider {
			provider := New(WithName("failure-fixture"), WithCapabilities(llm.CapabilityTools))
			if invocation != (CapabilityInvocation{}) {
				provider.EnqueueError(errors.New("wire rejected"))
			}
			return provider
		}, profile)
	case "nil_response":
		RunCapabilityConformance(t, func(t *testing.T, invocation CapabilityInvocation) llm.Provider {
			return nilResponseProvider{Provider: baseFactory(t, invocation)}
		}, profile)
	case "assertion_error":
		profile.Cases[0].Assert = func(*llm.Response) error { return errors.New("normalized mismatch") }
		RunCapabilityConformance(t, baseFactory, profile)
	}
}

func failureExpectedText(failure string) string {
	switch failure {
	case "nil_factory":
		return "requires a provider factory"
	case "nil_probe", "typed_nil_probe":
		return "probe factory returned a nil provider"
	case "nil_request_result":
		return "Request returned nil"
	case "nil_case_provider":
		return "factory returned a nil provider"
	case "claim_drift":
		return "changed reviewed capability claims"
	case "provider_error":
		return "provider call failed"
	case "nil_response":
		return "nil response"
	case "assertion_error":
		return "normalized response assertion failed"
	default:
		return failure
	}
}

type nilResponseProvider struct{ llm.Provider }

func (nilResponseProvider) Chat(context.Context, *llm.Request) (*llm.Response, error) {
	return nil, nil
}

func (nilResponseProvider) ChatStream(context.Context, *llm.Request) iter.Seq2[llm.Event, error] {
	return func(func(llm.Event, error) bool) {}
}

func capabilityResponse(model string) *llm.Response {
	return &llm.Response{
		ID:         "response",
		Provider:   "activation-test",
		Model:      model,
		Parts:      []llm.Part{llm.Text("activated")},
		StopReason: llm.StopReasonEndTurn,
		Usage:      llm.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2},
	}
}

func replaceCase(caseDef CapabilityCase, replace func(*CapabilityCase)) CapabilityCase {
	replace(&caseDef)
	return caseDef
}

func exemptionsExcept(except llm.Capability) []CapabilityExemption {
	var exemptions []CapabilityExemption
	for _, capability := range reviewedCapabilityActivations {
		if capability != except {
			exemptions = append(exemptions, CapabilityExemption{Capability: capability, Reason: "reviewed fixture gap"})
		}
	}
	return exemptions
}

func completeProfile(caseDef CapabilityCase) CapabilityProfile {
	return CapabilityProfile{Cases: []CapabilityCase{caseDef}, Exemptions: exemptionsExcept(caseDef.Capability)}
}
