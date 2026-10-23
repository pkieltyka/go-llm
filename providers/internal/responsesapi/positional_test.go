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
