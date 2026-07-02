package llm

import (
	"bytes"
	"encoding/json"
	"errors"
	"testing"
)

func init() {
	if err := RegisterPartType("test/registered", func(raw json.RawMessage) (Part, error) {
		var wire struct {
			Type  string `json:"type"`
			Value string `json:"value"`
		}
		if err := json.Unmarshal(raw, &wire); err != nil {
			return nil, err
		}
		return registeredPart{Value: wire.Value}, nil
	}); err != nil {
		panic(err)
	}
}

type registeredPart struct {
	ExtensionPartBase
	Value string
}

func (registeredPart) ExtensionProvider() string { return "test" }

func (p registeredPart) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type  string `json:"type"`
		Value string `json:"value"`
	}{
		Type:  "test/registered",
		Value: p.Value,
	})
}

func TestMarshalMessagesRoundTripByteIdentity(t *testing.T) {
	msgs := []Message{
		{
			Role: RoleUser,
			Parts: []Part{
				TextPart{Text: "hello", Cache: &CacheHint{}},
				ImageData([]byte{1, 2, 3}, "image/png"),
			},
		},
		{
			Role:     RoleAssistant,
			Provider: "openai",
			Model:    "gpt-test",
			Parts: []Part{
				ReasoningPart{
					Text:     "because",
					Raw:      json.RawMessage(`{ "encrypted": "<payload>", "index": 1 }`),
					Provider: "openai",
				},
				ToolCall("call_1", "lookup", json.RawMessage(`{"q":"go"}`)),
			},
		},
		{
			Role: RoleTool,
			Parts: []Part{
				ToolResultPart{
					ToolCallID: "call_1",
					Name:       "lookup",
					Parts:      []Part{Text("result")},
					IsError:    true,
				},
			},
		},
	}

	first, err := MarshalMessages(msgs)
	if err != nil {
		t.Fatalf("MarshalMessages returned error: %v", err)
	}
	decoded, err := UnmarshalMessages(first)
	if err != nil {
		t.Fatalf("UnmarshalMessages returned error: %v", err)
	}
	second, err := MarshalMessages(decoded)
	if err != nil {
		t.Fatalf("MarshalMessages after decode returned error: %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("round trip bytes differ:\nfirst:  %s\nsecond: %s", first, second)
	}

	if decoded[1].Provider != "openai" || decoded[1].Model != "gpt-test" {
		t.Fatalf("assistant provenance = (%q, %q), want (openai, gpt-test)", decoded[1].Provider, decoded[1].Model)
	}
	reasoning, ok := decoded[1].Parts[0].(ReasoningPart)
	if !ok {
		t.Fatalf("decoded reasoning type = %T, want ReasoningPart", decoded[1].Parts[0])
	}
	if string(reasoning.Raw) != `{ "encrypted": "<payload>", "index": 1 }` || reasoning.Provider != "openai" {
		t.Fatalf("reasoning raw/provider = %s/%q", reasoning.Raw, reasoning.Provider)
	}
	if !bytes.Contains(first, []byte(`"raw":{ "encrypted": "<payload>", "index": 1 }`)) {
		t.Fatalf("serialized reasoning raw was not byte-preserved: %s", first)
	}
}

func TestMarshalMessagePreservesRawPayloadBytes(t *testing.T) {
	reasoningRaw := json.RawMessage(`{ "html": "<tag>", "x": 1 }`)
	argsRaw := json.RawMessage(`{ "query": "<go>", "limit": 2 }`)
	unknownRaw := json.RawMessage(`{"type":"future/part","payload":{ "html": "<tag>", "x": 1 }}`)
	msg := Message{
		Role:     RoleAssistant,
		Provider: "openai",
		Model:    "gpt-test",
		Parts: []Part{
			ReasoningPart{Raw: reasoningRaw, Provider: "openai"},
			ToolCall("call_1", "lookup", argsRaw),
			UnknownPart{Type: "future/part", Data: unknownRaw},
		},
	}

	first, err := MarshalMessage(msg)
	if err != nil {
		t.Fatalf("MarshalMessage returned error: %v", err)
	}
	for _, want := range [][]byte{
		[]byte(`"raw":{ "html": "<tag>", "x": 1 }`),
		[]byte(`"args":{ "query": "<go>", "limit": 2 }`),
		unknownRaw,
	} {
		if !bytes.Contains(first, want) {
			t.Fatalf("MarshalMessage output missing raw payload %s in %s", want, first)
		}
	}

	decoded, err := UnmarshalMessage(first)
	if err != nil {
		t.Fatalf("UnmarshalMessage returned error: %v", err)
	}
	second, err := MarshalMessage(decoded)
	if err != nil {
		t.Fatalf("MarshalMessage after decode returned error: %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("message round trip bytes differ:\nfirst:  %s\nsecond: %s", first, second)
	}
}

func TestUnmarshalMessagesPreservesUnknownPart(t *testing.T) {
	input := []byte(`{"version":1,"messages":[{"role":"user","parts":[{"type":"future/part","payload":{ "x": 1, "html": "<tag>" }}]}]}`)

	msgs, err := UnmarshalMessages(input)
	if err != nil {
		t.Fatalf("UnmarshalMessages returned error: %v", err)
	}
	part, ok := msgs[0].Parts[0].(UnknownPart)
	if !ok {
		t.Fatalf("part type = %T, want UnknownPart", msgs[0].Parts[0])
	}
	if part.Type != "future/part" || string(part.Data) != `{"type":"future/part","payload":{ "x": 1, "html": "<tag>" }}` {
		t.Fatalf("unknown part = %+v, data %s", part, part.Data)
	}

	out, err := MarshalMessages(msgs)
	if err != nil {
		t.Fatalf("MarshalMessages returned error: %v", err)
	}
	if !bytes.Equal(out, input) {
		t.Fatalf("unknown part did not re-marshal identically:\n got: %s\nwant: %s", out, input)
	}
}

func TestRegisterPartTypeReturnsErrors(t *testing.T) {
	tests := []struct {
		name   string
		typ    string
		decode func(json.RawMessage) (Part, error)
	}{
		{name: "empty", decode: func(json.RawMessage) (Part, error) { return Text("x"), nil }},
		{name: "nil decoder", typ: "test/nil"},
		{name: "built in", typ: "text", decode: func(json.RawMessage) (Part, error) { return Text("x"), nil }},
		{name: "duplicate", typ: "test/registered", decode: func(json.RawMessage) (Part, error) { return Text("x"), nil }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := RegisterPartType(tt.typ, tt.decode)
			if !errors.Is(err, ErrBadRequest) {
				t.Fatalf("error = %v, want ErrBadRequest", err)
			}
		})
	}
}

func TestRegisterPartTypeDecodesExtensionPart(t *testing.T) {
	input := []byte(`{"version":1,"messages":[{"role":"user","parts":[{"type":"test/registered","value":"ok"}]}]}`)

	msgs, err := UnmarshalMessages(input)
	if err != nil {
		t.Fatalf("UnmarshalMessages returned error: %v", err)
	}
	part, ok := msgs[0].Parts[0].(registeredPart)
	if !ok {
		t.Fatalf("part type = %T, want registeredPart", msgs[0].Parts[0])
	}
	if part.Value != "ok" {
		t.Fatalf("registered part value = %q, want ok", part.Value)
	}

	out, err := MarshalMessages(msgs)
	if err != nil {
		t.Fatalf("MarshalMessages returned error: %v", err)
	}
	if !bytes.Equal(out, input) {
		t.Fatalf("registered part did not re-marshal identically:\n got: %s\nwant: %s", out, input)
	}
}

func TestMarshalResponsePreservesRawPayloadBytes(t *testing.T) {
	reasoningRaw := json.RawMessage(`{ "html": "<tag>", "x": 1 }`)
	resp := &Response{
		ID:       "resp_1",
		Provider: "openai",
		Model:    "gpt-test",
		Parts: []Part{
			ReasoningPart{Raw: reasoningRaw, Provider: "openai"},
		},
		Usage: Usage{InputTokens: 1, OutputTokens: 2, TotalTokens: 3, Raw: map[string]any{"sdk_usage": "secret"}},
		Raw:   map[string]any{"sdk": "secret"},
	}

	first, err := MarshalResponse(resp)
	if err != nil {
		t.Fatalf("MarshalResponse returned error: %v", err)
	}
	if !bytes.Contains(first, []byte(`"raw":{ "html": "<tag>", "x": 1 }`)) {
		t.Fatalf("MarshalResponse output did not preserve raw payload: %s", first)
	}
	if bytes.Contains(first, []byte("secret")) || bytes.Contains(first, []byte("sdk")) {
		t.Fatalf("MarshalResponse leaked raw SDK data: %s", first)
	}

	decoded, err := UnmarshalResponse(first)
	if err != nil {
		t.Fatalf("UnmarshalResponse returned error: %v", err)
	}
	second, err := MarshalResponse(decoded)
	if err != nil {
		t.Fatalf("MarshalResponse after decode returned error: %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("response round trip bytes differ:\nfirst:  %s\nsecond: %s", first, second)
	}
}

func TestResponseJSONExcludesRaw(t *testing.T) {
	cost := 0.25
	resp := &Response{
		ID:            "resp_1",
		Provider:      "openrouter",
		Model:         "model-a",
		Parts:         []Part{Text("done")},
		StopReason:    StopReasonEndTurn,
		StopReasonRaw: "stop",
		Usage: Usage{
			InputTokens:      1,
			OutputTokens:     2,
			TotalTokens:      3,
			CacheReadTokens:  4,
			CacheWriteTokens: 5,
			ReasoningTokens:  6,
			CostUSD:          &cost,
			Raw:              map[string]any{"sdk_usage": "secret"},
		},
		Raw: map[string]any{"sdk": "secret"},
	}

	first, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("Marshal response returned error: %v", err)
	}
	if bytes.Contains(first, []byte("secret")) || bytes.Contains(first, []byte("sdk")) {
		t.Fatalf("serialized response leaked raw data: %s", first)
	}

	var decoded Response
	if err := json.Unmarshal(first, &decoded); err != nil {
		t.Fatalf("Unmarshal response returned error: %v", err)
	}
	if decoded.Raw != nil || decoded.Usage.Raw != nil {
		t.Fatalf("decoded raw fields = response:%v usage:%v, want nil", decoded.Raw, decoded.Usage.Raw)
	}
	second, err := json.Marshal(&decoded)
	if err != nil {
		t.Fatalf("Marshal decoded response returned error: %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("response round trip bytes differ:\nfirst:  %s\nsecond: %s", first, second)
	}
}
