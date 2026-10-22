package providerutil

import (
	"encoding/json"
	"reflect"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestReplayableHistory(t *testing.T) {
	reasoning := llm.ReasoningPart{Provider: "p", Text: "why", Raw: json.RawMessage(`{"sig":"x"}`)}
	call := llm.ToolCall("call_1", "lookup", json.RawMessage(`{}`))
	history := []llm.Message{
		llm.UserText("hi"),
		{Role: llm.RoleAssistant, Model: "model-a-2026-01-02", Parts: []llm.Part{reasoning, llm.Text("same")}},
		{Role: llm.RoleAssistant, Model: "model-b", Parts: []llm.Part{&reasoning, call}},
		{Role: llm.RoleAssistant, Model: "model-b", Parts: []llm.Part{reasoning}},
		{Role: llm.RoleAssistant, Parts: []llm.Part{reasoning, llm.Text("legacy")}},
		{Role: llm.RoleUser, Model: "model-b", Parts: []llm.Part{reasoning}},
	}

	got := ReplayableHistory(history, "model-a")
	want := []llm.Message{
		history[0],
		history[1],
		{Role: llm.RoleAssistant, Model: "model-b", Parts: []llm.Part{call}},
		// history[3] is reasoning-only from model-b: dropped entirely.
		history[4],
		history[5],
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ReplayableHistory =\n%#v\nwant\n%#v", got, want)
	}
	if len(history[2].Parts) != 2 {
		t.Fatal("ReplayableHistory mutated the caller's message parts")
	}

	unchanged := history[:2]
	if out := ReplayableHistory(unchanged, "model-a"); &out[0] != &unchanged[0] || len(out) != len(unchanged) {
		t.Fatal("ReplayableHistory must return the input slice when nothing is removed")
	}
	foreignFirst := []llm.Message{
		{Role: llm.RoleAssistant, Model: "model-b", Parts: []llm.Part{reasoning, llm.Text("x")}},
		llm.UserText("next"),
	}
	gotFirst := ReplayableHistory(foreignFirst, "model-a")
	wantFirst := []llm.Message{{Role: llm.RoleAssistant, Model: "model-b", Parts: []llm.Part{llm.Text("x")}}, foreignFirst[1]}
	if !reflect.DeepEqual(gotFirst, wantFirst) {
		t.Fatalf("foreign first message = %#v, want %#v", gotFirst, wantFirst)
	}
	if out := ReplayableHistory(nil, "model-a"); out != nil {
		t.Fatalf("ReplayableHistory(nil) = %#v, want nil", out)
	}
}
