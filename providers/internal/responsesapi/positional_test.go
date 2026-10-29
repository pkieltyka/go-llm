package responsesapi_test

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
	"github.com/pkieltyka/go-llm/providers/internal/responsesapi"
)

func positionalAdapter() responsesapi.Adapter {
	return responsesapi.Adapter{ProviderName: "openai", Capabilities: []llm.Capability{llm.CapabilityTools, llm.CapabilityToolChoiceRequired}}
}

func positionalRequest(model string) *llm.Request {
	tool := func(name string) llm.Tool {
		return llm.Tool{Name: name, InputSchema: map[string]any{"type": "object"}}
	}
	return &llm.Request{
		Model:  model,
		System: "base",
		Tools:  []llm.Tool{tool("get_time")},
		Messages: []llm.Message{
			llm.UserText("hello"),
			llm.AssistantText("hi"),
			{Role: llm.RoleSystem, Parts: []llm.Part{
				llm.SystemDeltaPart{Section: "style", Text: "Use French."},
				llm.ToolsDeltaPart{Added: []llm.Tool{tool("get_weather")}, Removed: []string{"get_time"}},
			}},
			llm.UserText("weather?"),
			llm.AssistantText("checking"),
			{Role: llm.RoleSystem, Parts: []llm.Part{llm.ToolsDeltaPart{Added: []llm.Tool{tool("get_news")}}}},
			llm.UserText("news?"),
		},
	}
}

// inputSummary renders each input item as "kind:detail" for layout checks.
func inputSummary(t *testing.T, body string) (items []string, topTools []string) {
	t.Helper()
	var decoded struct {
		Input []map[string]any `json:"input"`
		Tools []struct {
			Name string `json:"name"`
		} `json:"tools"`
	}
	if err := json.Unmarshal([]byte(body), &decoded); err != nil {
		t.Fatal(err)
	}
	for _, item := range decoded.Input {
		switch {
		case item["type"] == "additional_tools":
			var names []string
			for _, tool := range item["tools"].([]any) {
				names = append(names, tool.(map[string]any)["name"].(string))
			}
			items = append(items, "additional_tools:"+strings.Join(names, ","))
		case item["role"] == "developer":
			items = append(items, "developer:"+item["content"].(string))
		default:
			items = append(items, item["role"].(string))
		}
	}
	for _, tool := range decoded.Tools {
		topTools = append(topTools, tool.Name)
	}
	return items, topTools
}

// Supported models get pi's additional_tools layout: every tool declared in
// place, removals announced as developer text and restated afterwards.
func TestResponsesSendsPositionalChangesInPlace(t *testing.T) {
	params, err := positionalAdapter().BuildParams(positionalRequest("gpt-5.6-luna"), false)
	if err != nil {
		t.Fatal(err)
	}
	items, top := inputSummary(t, testutil.MustCompactJSON(t, params))
	notice := "developer:This is an incremental tools update. Previously declared tools remain available unless explicitly marked unavailable. A tool declared again here is available again, and its latest definition replaces the earlier one."
	want := []string{
		"additional_tools:get_time", "user", "assistant",
		notice, "additional_tools:get_weather",
		"developer:Updated system prompt section \"style\":\n\nUse French.\n\nThe following tools are no longer available. Do not call them:\n- get_time",
		"user", "assistant",
		notice + "\nStill unavailable: get_time.", "additional_tools:get_news",
		"user",
	}
	if !slices.Equal(items, want) || len(top) != 0 {
		t.Fatalf("input = %q, top-level tools = %q\nwant input %q and no top-level tools", items, top, want)
	}
}

func TestResponsesFoldsPositionalChangesForOtherModels(t *testing.T) {
	params, err := positionalAdapter().BuildParams(positionalRequest("gpt-4.1"), false)
	if err != nil {
		t.Fatal(err)
	}
	items, top := inputSummary(t, testutil.MustCompactJSON(t, params))
	if want := []string{"user", "assistant", "user", "assistant", "user"}; !slices.Equal(items, want) {
		t.Fatalf("input = %q, want %q", items, want)
	}
	if want := []string{"get_weather", "get_news"}; !slices.Equal(top, want) {
		t.Fatalf("top-level tools = %q, want %q", top, want)
	}
	if params.Instructions.Value != "base\n\nUse French." {
		t.Fatalf("instructions = %q", params.Instructions.Value)
	}
}

func TestResponsesGPT54And55ReplayWithTopLevelTools(t *testing.T) {
	for _, model := range []string{"gpt-5.4", "gpt-5.4-mini", "gpt-5.4-pro", "gpt-5.5"} {
		for _, withDelta := range []bool{false, true} {
			name := model + "/initial"
			if withDelta {
				name = model + "/tool-deltas"
			}
			t.Run(name, func(t *testing.T) {
				tool := func(name string) llm.Tool {
					return llm.Tool{Name: name, InputSchema: map[string]any{"type": "object"}}
				}
				req := &llm.Request{
					Model: model, System: "base", Tools: []llm.Tool{tool("get_time")},
					Messages: []llm.Message{
						llm.UserText("time?"),
						{Role: llm.RoleAssistant, Provider: "openai", Model: "gpt-5.6-luna", Parts: []llm.Part{
							llm.ToolCallPart{ID: "call_time", Name: "get_time", Args: json.RawMessage(`{}`)},
						}},
						{Role: llm.RoleTool, Parts: []llm.Part{
							llm.ToolResultPart{ToolCallID: "call_time", Name: "get_time", Content: []llm.Part{llm.Text("noon")}},
						}},
						{Role: llm.RoleSystem, Parts: []llm.Part{llm.SystemDeltaPart{Section: "style", Text: "Use French."}}},
					},
				}
				wantTools := []string{"get_time"}
				if withDelta {
					req.Messages = append(req.Messages, llm.Message{Role: llm.RoleSystem, Parts: []llm.Part{
						llm.ToolsDeltaPart{Added: []llm.Tool{tool("get_weather")}, Removed: []string{"get_time"}},
					}})
					wantTools = []string{"get_weather"}
				}
				req.Messages = append(req.Messages, llm.UserText("continue"))
				params, err := positionalAdapter().BuildParams(req, false)
				if err != nil {
					t.Fatal(err)
				}
				var wire struct {
					Instructions string           `json:"instructions"`
					Input        []map[string]any `json:"input"`
					Tools        []struct {
						Name string `json:"name"`
					} `json:"tools"`
				}
				if err := json.Unmarshal([]byte(testutil.MustCompactJSON(t, params)), &wire); err != nil {
					t.Fatal(err)
				}
				var tools []string
				for _, tool := range wire.Tools {
					tools = append(tools, tool.Name)
				}
				if !slices.Equal(tools, wantTools) || wire.Instructions != "base" {
					t.Fatalf("top-level tools = %q, instructions = %q; want %q and base", tools, wire.Instructions, wantTools)
				}
				if len(wire.Input) != 5 {
					t.Fatalf("unexpected input: %+v", wire.Input)
				}
				call, result, developer := wire.Input[1], wire.Input[2], wire.Input[3]
				if call["type"] != "function_call" || call["call_id"] != "call_time" || call["name"] != "get_time" || call["arguments"] != "{}" {
					t.Fatalf("replayed call = %+v", call)
				}
				if result["type"] != "function_call_output" || result["call_id"] != "call_time" || result["output"] != "noon" {
					t.Fatalf("replayed result = %+v", result)
				}
				if developer["role"] != "developer" || developer["content"] != "Updated system prompt section \"style\":\n\nUse French." {
					t.Fatalf("positional instruction = %+v", developer)
				}
				for _, item := range wire.Input {
					if item["type"] == "additional_tools" {
						t.Fatalf("unexpected positional tool declaration: %+v", item)
					}
				}
			})
		}
	}
}

func TestResponsesPositionalEdgeCases(t *testing.T) {
	tool := func(name string) llm.Tool { return llm.Tool{Name: name, InputSchema: map[string]any{"type": "object"}} }
	req := positionalRequest("gpt-5.6-luna")
	// Pointer parts are normalized like value parts.
	delta := req.Messages[2].Parts[1].(llm.ToolsDeltaPart)
	req.Messages[2].Parts[1] = &delta
	// Declaring get_time again makes it available again.
	req.Messages = append(req.Messages, llm.AssistantText("ok"),
		llm.Message{Role: llm.RoleSystem, Parts: []llm.Part{llm.ToolsDeltaPart{Added: []llm.Tool{tool("get_time")}}}},
		llm.UserText("time?"), llm.AssistantText("noted"),
		llm.Message{Role: llm.RoleSystem, Parts: []llm.Part{llm.ToolsDeltaPart{Added: []llm.Tool{tool("get_date")}}}},
		llm.UserText("date?"))
	params, err := positionalAdapter().BuildParams(req, false)
	if err != nil {
		t.Fatal(err)
	}
	items, _ := inputSummary(t, testutil.MustCompactJSON(t, params))
	notice := "developer:This is an incremental tools update. Previously declared tools remain available unless explicitly marked unavailable. A tool declared again here is available again, and its latest definition replaces the earlier one."
	if got := items[len(items)-3:]; !slices.Equal(got, []string{notice, "additional_tools:get_date", "user"}) {
		t.Fatalf("after redeclaring get_time the notice must not list it: %q", got)
	}
	if _, err := positionalAdapter().BuildInput(positionalRequest("gpt-5.6-luna").Messages); err == nil || !strings.Contains(err.Error(), "not folded") {
		t.Fatalf("BuildInput with an unfolded tools delta = %v", err)
	}
}

// Deltas in one message apply in order: re-adding a tool a part after
// removing it cancels the removal notice, and later notices never list it.
func TestResponsesReAddInSameMessageCancelsRemoval(t *testing.T) {
	tool := func(name string) llm.Tool { return llm.Tool{Name: name, InputSchema: map[string]any{"type": "object"}} }
	req := &llm.Request{
		Model: "gpt-5.6-luna",
		Tools: []llm.Tool{tool("a")},
		Messages: []llm.Message{
			llm.UserText("hello"),
			llm.AssistantText("hi"),
			{Role: llm.RoleSystem, Parts: []llm.Part{
				llm.ToolsDeltaPart{Removed: []string{"a"}},
				llm.ToolsDeltaPart{Added: []llm.Tool{tool("a")}},
			}},
			llm.UserText("again"),
			llm.AssistantText("ok"),
			{Role: llm.RoleSystem, Parts: []llm.Part{llm.ToolsDeltaPart{Added: []llm.Tool{tool("b")}}}},
			llm.UserText("more"),
		},
		ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "a"},
	}
	params, err := positionalAdapter().BuildParams(req, false)
	if err != nil {
		t.Fatal(err)
	}
	items, _ := inputSummary(t, testutil.MustCompactJSON(t, params))
	for _, item := range items {
		if strings.Contains(item, "no longer available") || strings.Contains(item, "Still unavailable") {
			t.Fatalf("wire marks the re-added tool unavailable: %q", items)
		}
	}
}
