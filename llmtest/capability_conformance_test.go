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
	"time"

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
		return dispatchLoggingProvider{Provider: provider, log: func(path ConformancePath) {
			mu.Lock()
			order = append(order, "dispatch/"+string(path))
			mu.Unlock()
		}}
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
			if response.Model != model || response.Text() != "activated" || response.StopReason != llm.StopReasonEndTurn || response.Usage.InputTokens != 1 || response.Usage.OutputTokens != 1 || response.Usage.TotalTokens != 2 {
				return fmt.Errorf("response = model %q text %q stop %q usage %+v", response.Model, response.Text(), response.StopReason, response.Usage)
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
	wantOrder := []string{"factory/", "factory/chat", "request/1", "factory/stream", "request/2", "dispatch/chat", "dispatch/stream"}
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
	if selected := os.Getenv("GO_LLM_MALFORMED_PROFILE"); selected != "" {
		var index int
		if _, err := fmt.Sscanf(selected, "%d", &index); err != nil || index < 0 || index >= len(tests) {
			t.Fatalf("invalid malformed profile selector %q", selected)
		}
		tt := tests[index]
		RunCapabilityConformance(t, func(*testing.T, CapabilityInvocation) llm.Provider {
			return New(WithName("fixture"), WithCapabilities(claimSlice(tt.claims)...))
		}, tt.profile)
		return
	}

	for index, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.want != "" {
				command := exec.Command(os.Args[0], "-test.run=^TestValidateCapabilityProfile$")
				command.Env = append(os.Environ(), fmt.Sprintf("GO_LLM_MALFORMED_PROFILE=%d", index))
				output, err := command.CombinedOutput()
				if err == nil {
					t.Fatalf("malformed profile child unexpectedly passed:\n%s", output)
				}
				text := string(output)
				if strings.Contains(text, "panic:") || !strings.Contains(text, tt.want) || !strings.Contains(text, "fixture") {
					t.Fatalf("malformed profile output does not pin non-panic provider context and %q:\n%s", tt.want, text)
				}
				return
			}
			err := validateCapabilityProfile("fixture", tt.claims, tt.profile)
			if err != nil {
				t.Fatalf("validateCapabilityProfile returned error: %v", err)
			}
		})
	}
}

func TestRunCapabilityConformanceFailureContract(t *testing.T) {
	if failure := os.Getenv("GO_LLM_CAPABILITY_FAILURE"); failure != "" {
		runCapabilityFailureCase(t, failure)
		return
	}
	for _, failure := range []string{"nil_factory", "nil_probe", "typed_nil_probe", "nil_request_result", "late_nil_request", "nil_case_provider", "claim_drift", "provider_error", "stream_provider_error", "nil_response", "assertion_error", "stream_assertion_error", "stream_watchdog"} {
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
			for _, fragment := range failureExpectedText(failure) {
				if !strings.Contains(text, fragment) {
					t.Fatalf("failure output missing %q:\n%s", fragment, text)
				}
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
			if invocation.Path == ConformanceStream {
				provider.EnqueueStream(
					llm.MessageStart{ID: "response", Provider: "failure-fixture", Model: "real-model"},
					llm.TextDelta{Index: 0, Text: "activated"},
					llm.MessageEnd{StopReason: llm.StopReasonEndTurn, Usage: llm.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2}},
				)
			} else {
				provider.EnqueueResponse(capabilityResponse("real-model"))
			}
		}
		return provider
	}
	profile := CapabilityProfile{Cases: []CapabilityCase{{
		Name:       "tools-case",
		Capability: llm.CapabilityTools,
		Paths:      []ConformancePath{ConformanceChat},
		Request: func() *llm.Request {
			return &llm.Request{Model: "real-model", Messages: []llm.Message{llm.UserText("activate")}}
		},
		Assert: func(*llm.Response) error { return nil },
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
	case "late_nil_request":
		calls := 0
		defer func() {
			if calls != 0 {
				panic(fmt.Sprintf("dispatch occurred before preparation completed: %d calls", calls))
			}
		}()
		profile.Cases = append(profile.Cases, CapabilityCase{
			Name: "late-invalid", Capability: llm.CapabilityTools, Paths: []ConformancePath{ConformanceChat},
			Request: func() *llm.Request { return nil }, Assert: func(*llm.Response) error { return nil },
		})
		RunCapabilityConformance(t, func(t *testing.T, invocation CapabilityInvocation) llm.Provider {
			return callCountingProvider{Provider: baseFactory(t, invocation), calls: &calls}
		}, profile)
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
	case "stream_provider_error":
		profile.Cases[0].Paths = []ConformancePath{ConformanceStream}
		RunCapabilityConformance(t, func(t *testing.T, invocation CapabilityInvocation) llm.Provider {
			provider := New(WithName("failure-fixture"), WithCapabilities(llm.CapabilityTools))
			if invocation != (CapabilityInvocation{}) {
				provider.EnqueueError(errors.New("stream wire rejected"))
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
	case "stream_assertion_error":
		profile.Cases[0].Paths = []ConformancePath{ConformanceStream}
		profile.Cases[0].Assert = func(*llm.Response) error { return errors.New("stream normalized mismatch") }
		RunCapabilityConformance(t, baseFactory, profile)
	case "stream_watchdog":
		profile.Cases[0].Paths = []ConformancePath{ConformanceStream}
		capabilityConformanceStreamTimeout = 20 * time.Millisecond
		observedCancellation := make(chan struct{})
		defer func() {
			select {
			case <-observedCancellation:
			case <-time.After(time.Second):
				panic("stream watchdog did not cancel the request context")
			}
		}()
		RunCapabilityConformance(t, func(t *testing.T, invocation CapabilityInvocation) llm.Provider {
			return watchdogProvider{Provider: baseFactory(t, invocation), observedCancellation: observedCancellation}
		}, profile)
	}
}

func failureExpectedText(failure string) []string {
	switch failure {
	case "nil_factory":
		return []string{"requires a provider factory"}
	case "nil_probe", "typed_nil_probe":
		return []string{"probe factory returned a nil provider"}
	case "nil_request_result", "late_nil_request":
		return []string{"Request returned nil"}
	case "nil_case_provider":
		return []string{"factory returned a nil provider"}
	case "claim_drift":
		return []string{"changed reviewed capability claims"}
	case "provider_error", "stream_provider_error", "stream_watchdog":
		return append(failureContext(failure), "provider call failed")
	case "nil_response":
		return append(failureContext(failure), "nil response")
	case "assertion_error", "stream_assertion_error":
		return append(failureContext(failure), "normalized response assertion failed")
	default:
		return []string{failure}
	}
}

func failureContext(failure string) []string {
	path := `path "chat"`
	if strings.HasPrefix(failure, "stream_") {
		path = `path "stream"`
	}
	return []string{`provider "failure-fixture"`, `case "tools-case"`, "(tools)", path}
}

type nilResponseProvider struct{ llm.Provider }

func (nilResponseProvider) Chat(context.Context, *llm.Request) (*llm.Response, error) {
	return nil, nil
}

type dispatchLoggingProvider struct {
	*Provider
	log func(ConformancePath)
}

func (p dispatchLoggingProvider) Chat(ctx context.Context, request *llm.Request) (*llm.Response, error) {
	p.log(ConformanceChat)
	return p.Provider.Chat(ctx, request)
}

func (p dispatchLoggingProvider) ChatStream(ctx context.Context, request *llm.Request) iter.Seq2[llm.Event, error] {
	p.log(ConformanceStream)
	return p.Provider.ChatStream(ctx, request)
}

type callCountingProvider struct {
	llm.Provider
	calls *int
}

func (p callCountingProvider) Chat(ctx context.Context, request *llm.Request) (*llm.Response, error) {
	(*p.calls)++
	return p.Provider.Chat(ctx, request)
}

func (p callCountingProvider) ChatStream(ctx context.Context, request *llm.Request) iter.Seq2[llm.Event, error] {
	(*p.calls)++
	return p.Provider.ChatStream(ctx, request)
}

type watchdogProvider struct {
	llm.Provider
	observedCancellation chan<- struct{}
}

func (p watchdogProvider) ChatStream(ctx context.Context, _ *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		yield(llm.MessageStart{Provider: "failure-fixture", Model: "real-model"}, nil)
		<-ctx.Done()
		close(p.observedCancellation)
		select {}
	}
}

func claimSlice(claims map[llm.Capability]struct{}) []llm.Capability {
	result := make([]llm.Capability, 0, len(claims))
	for _, capability := range reviewedCapabilityActivations {
		if _, ok := claims[capability]; ok {
			result = append(result, capability)
		}
	}
	return result
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
