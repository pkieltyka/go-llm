package llmtest

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

// AppendOnlyPrefixConfig wires RunAppendOnlyPrefix to one provider's fixture
// server.
type AppendOnlyPrefixConfig struct {
	// NewProvider returns a provider pointed at a fixture server that
	// answers the session's first turn with exactly one call to the
	// "lookup" tool, answers the second turn with a final text message,
	// and records every request body it receives.
	NewProvider func(t *testing.T) llm.Provider
	// Requests returns the raw request bodies captured so far, in order.
	Requests func() [][]byte
	// Model is the request model.
	Model string
	// MessagesField is the top-level request array holding serialized
	// conversation items ("messages" for Chat Completions and Anthropic,
	// "input" for Responses).
	MessagesField string
	// StableFields are top-level request fields that must be deep-equal
	// across the two turns (tools, model, cache/session keys, system).
	// Fields absent from both turns compare equal.
	StableFields []string
}

// RunAppendOnlyPrefix machine-checks the append-only wire-prefix property
// provider prompt caches rely on: within a session, each turn's serialized
// request must extend the previous one — the second turn's conversation
// items must start with exactly the first turn's items, and prefix-relevant
// fields (tools, model, cache keys, system) must not change between turns.
func RunAppendOnlyPrefix(t *testing.T, cfg AppendOnlyPrefixConfig) {
	t.Helper()
	if cfg.NewProvider == nil || cfg.Requests == nil || cfg.MessagesField == "" {
		t.Fatal("RunAppendOnlyPrefix requires NewProvider, Requests, and MessagesField")
	}

	session := llm.NewSession(cfg.NewProvider(t), cfg.Model,
		llm.WithSessionSystem("You are a prefix-conformance probe."),
		llm.WithSessionTools(llm.Tool{
			Name:        "lookup",
			Description: "Look something up.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`),
		}),
		llm.WithSessionID("prefix-conformance-session"),
	)

	resp, err := session.ChatText(context.Background(), "call the lookup tool")
	if err != nil {
		t.Fatalf("turn 1 returned error: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("turn 1 tool calls = %d, want 1 (fixture must answer turn 1 with one lookup call)", len(calls))
	}
	session.AddToolResults(llm.ToolResult(calls[0].ID, "result-data"))
	if _, err := session.Continue(context.Background()); err != nil {
		t.Fatalf("turn 2 returned error: %v", err)
	}

	bodies := cfg.Requests()
	if len(bodies) != 2 {
		t.Fatalf("captured %d request bodies, want 2", len(bodies))
	}
	turn1 := decodeRequestBody(t, bodies[0])
	turn2 := decodeRequestBody(t, bodies[1])

	items1, ok := turn1[cfg.MessagesField].([]any)
	if !ok || len(items1) == 0 {
		t.Fatalf("turn 1 %q = %v, want a non-empty array", cfg.MessagesField, turn1[cfg.MessagesField])
	}
	items2, ok := turn2[cfg.MessagesField].([]any)
	if !ok || len(items2) <= len(items1) {
		t.Fatalf("turn 2 %q has %d items, want more than turn 1's %d", cfg.MessagesField, len(items2), len(items1))
	}
	if !reflect.DeepEqual(items1, items2[:len(items1)]) {
		t.Fatalf("turn 2 %s prefix diverged from turn 1 (cache-breaking rewrite)\nturn 1: %s\nturn 2 prefix: %s",
			cfg.MessagesField, mustJSON(t, items1), mustJSON(t, items2[:len(items1)]))
	}
	for _, field := range cfg.StableFields {
		if !reflect.DeepEqual(turn1[field], turn2[field]) {
			t.Fatalf("field %q changed between turns (cache-breaking drift)\nturn 1: %s\nturn 2: %s",
				field, mustJSON(t, turn1[field]), mustJSON(t, turn2[field]))
		}
	}
}

func decodeRequestBody(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var decoded map[string]any
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("request body is not a JSON object: %v\n%s", err, body)
	}
	return decoded
}

func mustJSON(t *testing.T, value any) []byte {
	t.Helper()
	out, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal diagnostic value: %v", err)
	}
	return out
}
