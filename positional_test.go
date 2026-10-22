package llm

import (
	"bytes"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
)

func tool(name string) Tool { return Tool{Name: name, InputSchema: map[string]any{"type": "object"}} }

func system(parts ...Part) Message { return Message{Role: RoleSystem, Parts: parts} }

func toolNames(tools []Tool) []string {
	var names []string
	for _, t := range tools {
		names = append(names, t.Name)
	}
	return names
}

func positionalTranscript() *Request {
	return &Request{
		Model:  "m",
		System: "base",
		Tools:  []Tool{tool("a"), tool("b")},
		Messages: []Message{
			system(Text("lead"), SystemDeltaPart{Section: "s1", Text: "one"}),
			UserText("u1"),
			AssistantText("x1"),
			system(Text("later"), SystemDeltaPart{Section: "s2", Text: "two"}, ToolsDeltaPart{Added: []Tool{tool("c")}, Removed: []string{"a"}}),
			UserText("u2"),
			AssistantText("x2"),
			system(SystemDeltaPart{Section: "s1", Text: "uno"}, SystemDeltaPart{Section: "s2"}, ToolsDeltaPart{Added: []Tool{tool("b")}}),
			UserText("u3"),
		},
	}
}

func TestFoldPositionalWithoutSupportReplaysEverything(t *testing.T) {
	req := positionalTranscript()
	before, _ := MarshalMessages(req.Messages)
	out := FoldPositional(req, PositionalSupport{})
	// pi: later text is appended to the base prompt; sections render after
	// it, patched by name.
	if want := "base\n\nlead\n\nlater\n\nuno"; out.System != want {
		t.Fatalf("System = %q, want %q", out.System, want)
	}
	if got, want := toolNames(out.Tools), []string{"b", "c"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Tools = %v, want %v (a removed, b redefined in place)", got, want)
	}
	for _, msg := range out.Messages {
		if msg.Role == RoleSystem {
			t.Fatalf("system message survived folding: %+v", out.Messages)
		}
	}
	if len(out.Messages) != 5 {
		t.Fatalf("messages = %d, want 5", len(out.Messages))
	}
	if after, _ := MarshalMessages(req.Messages); !bytes.Equal(before, after) || req.System != "base" || len(req.Tools) != 2 {
		t.Fatal("FoldPositional modified its input")
	}
	again := FoldPositional(positionalTranscript(), PositionalSupport{})
	if again.System != out.System || !reflect.DeepEqual(toolNames(again.Tools), toolNames(out.Tools)) {
		t.Fatal("folding is not deterministic")
	}
}

// Like pi's Map.set, an update keeps its position and only a deleted or
// removed entry that comes back moves to the end.
func TestFoldPositionalOrdersUpdatesLikePi(t *testing.T) {
	req := &Request{Model: "m", Tools: []Tool{tool("x"), tool("y")}, Messages: []Message{
		UserText("u"),
		system(SystemDeltaPart{Section: "a", Text: "A"}, SystemDeltaPart{Section: "b", Text: "B"}),
		AssistantText("r"),
		system(SystemDeltaPart{Section: "a", Text: "A2"}, ToolsDeltaPart{Added: []Tool{{Name: "x", Description: "v2"}}}),
		UserText("u2"),
	}}
	out := FoldPositional(req, PositionalSupport{})
	if out.System != "A2\n\nB" || !reflect.DeepEqual(toolNames(out.Tools), []string{"x", "y"}) || out.Tools[0].Description != "v2" {
		t.Fatalf("in-place updates moved: System=%q Tools=%+v", out.System, out.Tools)
	}
	req.Messages = append(req.Messages, system(
		SystemDeltaPart{Section: "a"}, SystemDeltaPart{Section: "a", Text: "A3"},
		ToolsDeltaPart{Removed: []string{"x"}, Added: []Tool{tool("x")}},
	))
	out = FoldPositional(req, PositionalSupport{})
	if out.System != "B\n\nA3" || !reflect.DeepEqual(toolNames(out.Tools), []string{"y", "x"}) {
		t.Fatalf("deleted and re-added entries must move to the end: System=%q Tools=%v", out.System, toolNames(out.Tools))
	}
}

func TestFoldPositionalKeepsWhatTheModelTakesInPlace(t *testing.T) {
	both := FoldPositional(positionalTranscript(), PositionalSupport{Instructions: true, Tools: true})
	if both.System != "base\n\nlead\n\none" || !reflect.DeepEqual(toolNames(both.Tools), []string{"a", "b"}) {
		t.Fatalf("only the leading run folds: System=%q Tools=%v", both.System, toolNames(both.Tools))
	}
	if len(both.Messages) != 7 || both.Messages[2].Role != RoleSystem || len(both.Messages[2].Parts) != 3 {
		t.Fatalf("later system messages must stay in place: %+v", both.Messages)
	}

	textOnly := FoldPositional(positionalTranscript(), PositionalSupport{Instructions: true})
	if !reflect.DeepEqual(toolNames(textOnly.Tools), []string{"b", "c"}) {
		t.Fatalf("tool changes fold into the final set: %v", toolNames(textOnly.Tools))
	}
	if len(textOnly.Messages[2].Parts) != 2 || len(textOnly.Messages[5].Parts) != 2 || HasToolsDelta(textOnly.Messages) {
		t.Fatalf("tool deltas must be removed from kept system messages: %+v", textOnly.Messages)
	}

	toolsOnly := FoldPositional(positionalTranscript(), PositionalSupport{Tools: true})
	if HasToolsDelta(toolsOnly.Messages) || len(toolsOnly.Messages) != 5 {
		t.Fatal("tools in place require instructions in place")
	}
}

func TestValidateRequestChecksPositionalMessages(t *testing.T) {
	caps := []Capability{CapabilityTools, CapabilityToolChoiceRequired}
	if err := ValidateRequest(caps, positionalTranscript()); err != nil {
		t.Fatalf("valid transcript rejected: %v", err)
	}
	cases := map[string]func(*Request){
		"delta outside a system message": func(r *Request) {
			r.Messages[1].Parts = append(r.Messages[1].Parts, SystemDeltaPart{Section: "x"})
		},
		"image in a system message": func(r *Request) { r.Messages[0].Parts = append(r.Messages[0].Parts, ImageURL("u")) },
		"unnamed section":           func(r *Request) { r.Messages[0].Parts = append(r.Messages[0].Parts, SystemDeltaPart{}) },
		"removing an unoffered tool": func(r *Request) {
			r.Messages[3].Parts = append(r.Messages[3].Parts, ToolsDeltaPart{Removed: []string{"zzz"}})
		},
		"duplicate added tool": func(r *Request) {
			r.Messages[3].Parts = append(r.Messages[3].Parts, ToolsDeltaPart{Added: []Tool{tool("d"), tool("d")}})
		},
		"tool call to a removed tool": func(r *Request) {
			r.Messages[5] = AssistantParts(ToolCall("1", "a", json.RawMessage(`{}`)))
		},
		"tool call before the tool is added": func(r *Request) {
			r.Messages[2] = AssistantParts(ToolCall("1", "c", json.RawMessage(`{}`)))
		},
		"tool choice of a removed tool": func(r *Request) { r.ToolChoice = ToolChoice{Mode: ToolChoiceTool, Name: "a"} },
	}
	for name, mutate := range cases {
		req := positionalTranscript()
		mutate(req)
		if err := ValidateRequest(caps, req); !errors.Is(err, ErrBadRequest) {
			t.Errorf("%s: err = %v, want ErrBadRequest", name, err)
		}
	}
	req := positionalTranscript()
	req.ToolChoice = ToolChoice{Mode: ToolChoiceTool, Name: "c"}
	if err := ValidateRequest(caps, req); err != nil {
		t.Fatalf("tool choice of an added tool = %v", err)
	}
	noTools := positionalTranscript()
	noTools.Tools = nil
	noTools.Messages[3].Parts[2] = ToolsDeltaPart{Added: []Tool{tool("c")}}
	if err := ValidateRequest(nil, noTools); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("tools added in place need CapabilityTools: %v", err)
	}
}

// Without tool deltas, history may name tools no longer offered.
func TestValidateRequestAllowsHistoricalToolCallsWithoutToolDeltas(t *testing.T) {
	req := &Request{Model: "m", Tools: []Tool{tool("a")}, Messages: []Message{
		UserText("u"), AssistantParts(ToolCall("1", "gone", json.RawMessage(`{}`))),
		{Role: RoleTool, Parts: []Part{ToolResult("1", "ok")}}, UserText("next"),
	}}
	if err := ValidateRequest([]Capability{CapabilityTools}, req); err != nil {
		t.Fatal(err)
	}
}

func TestPositionalPartsRoundTripAndBumpTheEnvelope(t *testing.T) {
	plain, err := MarshalMessages([]Message{UserText("hi")})
	if err != nil || !bytes.HasPrefix(plain, []byte(`{"version":1,`)) {
		t.Fatalf("plain history must stay version 1: %s %v", plain, err)
	}
	msgs := []Message{system(
		SystemDeltaPart{Section: "s", Text: "t", Cache: &CacheHint{}},
		ToolsDeltaPart{
			Added:   []Tool{{Name: "c", Description: "d", InputSchema: map[string]any{"type": "object"}, Strict: true, Annotations: ToolAnnotations{ReadOnly: true}}},
			Removed: []string{"a"},
		},
	), UserText("hi")}
	data, err := MarshalMessages(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(data, []byte(`{"version":2,`)) {
		t.Fatalf("positional history must be version 2: %s", data)
	}
	got, err := UnmarshalMessages(data)
	if err != nil {
		t.Fatal(err)
	}
	delta := got[0].Parts[1].(ToolsDeltaPart)
	if schema, ok := delta.Added[0].InputSchema.(json.RawMessage); !ok || string(schema) != `{"type":"object"}` {
		t.Fatalf("InputSchema = %#v, want json.RawMessage", delta.Added[0].InputSchema)
	}
	again, err := MarshalMessages(got)
	if err != nil || !bytes.Equal(again, data) {
		t.Fatalf("round trip changed bytes:\n%s\n%s", data, again)
	}
	section := got[0].Parts[0].(SystemDeltaPart)
	if section.Section != "s" || section.Text != "t" || section.Cache == nil || !delta.Added[0].Strict || !delta.Added[0].Annotations.ReadOnly || delta.Removed[0] != "a" {
		t.Fatalf("decoded parts = %+v", got[0].Parts)
	}
}

func TestEstimateRequestTokensCountsPositionalParts(t *testing.T) {
	base := &Request{Model: "m", Messages: []Message{UserText("hi")}}
	with := &Request{Model: "m", Messages: []Message{system(SystemDeltaPart{Section: "s", Text: "some words here"}, ToolsDeltaPart{Added: []Tool{tool("c")}}), UserText("hi")}}
	if EstimateRequestTokens(with) <= EstimateRequestTokens(base) {
		t.Fatal("positional parts must count toward the estimate")
	}
}

func TestModelTableCarriesPositionalBehavior(t *testing.T) {
	for model, want := range map[string]PositionalSupport{
		"claude-haiku-5-5":          {Instructions: true, Tools: true},
		"claude-sonnet-5":           {Instructions: true},
		"claude-sonnet-4-6":         {},
		"claude-haiku-4-5-20251001": {},
	} {
		info, ok := LookupModelInfo("anthropic", model)
		if !ok || info.Behavior.PositionalSupport() != want {
			t.Errorf("anthropic/%s support = %+v (found %v), want %+v", model, info.Behavior.PositionalSupport(), ok, want)
		}
	}
	if info, _ := LookupModelInfo("openai", "gpt-5.6-luna"); info.Behavior.PositionalSupport() != (PositionalSupport{Instructions: true, Tools: true}) {
		t.Errorf("openai/gpt-5.6-luna behavior = %+v", info.Behavior)
	}
	if info, _ := LookupModelInfo("openrouter", "anthropic/claude-haiku-5-5"); info.Behavior != (ModelBehavior{}) {
		t.Errorf("aggregator rows must not inherit behavior: %+v", info.Behavior)
	}
}
