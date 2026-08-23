package chatcompletions_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
)

func TestBuildParamsNormalizesObjectToolProperties(t *testing.T) {
	var nilProperties map[string]any
	populated := map[string]any{"query": map[string]any{"type": "string"}}
	tests := []struct {
		name       string
		schema     any
		wantType   any
		wantFields map[string]any
	}{
		{name: "omitted properties", schema: map[string]any{"type": "object"}, wantType: "object"},
		{name: "typed nil properties", schema: map[string]any{"type": "object", "properties": nilProperties}, wantType: "object"},
		{name: "raw null properties", schema: json.RawMessage(`{"type":"object","properties":null}`), wantType: "object"},
		{name: "absent root type", schema: map[string]any{"title": "No arguments"}, wantFields: map[string]any{"title": "No arguments"}},
		{name: "empty object", schema: map[string]any{"type": "object", "properties": map[string]any{}}, wantType: "object"},
		{
			name:     "populated object and other keywords",
			schema:   map[string]any{"type": "object", "properties": populated, "required": []string{"query"}, "additionalProperties": false},
			wantType: "object",
			wantFields: map[string]any{
				"required":             []any{"query"},
				"additionalProperties": false,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params, err := newReplayProvider(t).BuildParams(toolSchemaRequest(tt.schema), false)
			if err != nil {
				t.Fatalf("BuildParams returned error: %v", err)
			}
			schema := builtToolSchema(t, params)
			properties, ok := schema["properties"].(map[string]any)
			if !ok || properties == nil {
				t.Fatalf("properties = %#v (%T), want object", schema["properties"], schema["properties"])
			}
			if tt.name == "populated object and other keywords" && !reflect.DeepEqual(properties, populated) {
				t.Fatalf("properties = %#v, want %#v", properties, populated)
			}
			if tt.wantType != nil && schema["type"] != tt.wantType {
				t.Fatalf("type = %#v, want %#v", schema["type"], tt.wantType)
			}
			for key, want := range tt.wantFields {
				if !reflect.DeepEqual(schema[key], want) {
					t.Fatalf("%s = %#v, want %#v", key, schema[key], want)
				}
			}
		})
	}
}

func TestBuildParamsRejectsInvalidToolSchemaStructure(t *testing.T) {
	var nilRoot map[string]any
	tests := []struct {
		name   string
		schema any
	}{
		{name: "typed nil root", schema: nilRoot},
		{name: "raw null root", schema: json.RawMessage(`null`)},
		{name: "null root type", schema: map[string]any{"type": nil}},
		{name: "string root", schema: map[string]any{"type": "string"}},
		{name: "array root", schema: map[string]any{"type": "array"}},
		{name: "number root", schema: map[string]any{"type": "number"}},
		{name: "integer root", schema: map[string]any{"type": "integer"}},
		{name: "boolean root", schema: map[string]any{"type": "boolean"}},
		{name: "array properties", schema: map[string]any{"type": "object", "properties": []any{}}},
		{name: "string properties", schema: map[string]any{"type": "object", "properties": "none"}},
		{name: "number properties", schema: map[string]any{"type": "object", "properties": 1}},
		{name: "bool properties", schema: map[string]any{"type": "object", "properties": false}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := newReplayProvider(t).BuildParams(toolSchemaRequest(tt.schema), false)
			if !errors.Is(err, llm.ErrBadRequest) {
				t.Fatalf("BuildParams error = %v, want ErrBadRequest", err)
			}
			if err == nil || !containsAll(err.Error(), replayDialectName, `tool "lookup"`) {
				t.Fatalf("BuildParams error lacks provider/tool context: %v", err)
			}
		})
	}
}

func TestToolSchemaNormalizationDoesNotMutateCaller(t *testing.T) {
	nested := map[string]any{"query": map[string]any{"type": "string"}}
	original := map[string]any{
		"type":                 "object",
		"properties":           nested,
		"additionalProperties": false,
	}
	want := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{"type": "string"},
		},
		"additionalProperties": false,
	}
	params, err := newReplayProvider(t).BuildParams(toolSchemaRequest(original), false)
	if err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	built := builtToolSchema(t, params)
	built["properties"].(map[string]any)["added"] = map[string]any{"type": "boolean"}
	if !reflect.DeepEqual(original, want) {
		t.Fatalf("caller schema mutated: got %#v, want %#v", original, want)
	}

	omitted := map[string]any{"type": "object"}
	if _, err := newReplayProvider(t).BuildParams(toolSchemaRequest(omitted), false); err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	if _, exists := omitted["properties"]; exists {
		t.Fatalf("normalization added properties to caller map: %#v", omitted)
	}

	raw := json.RawMessage(`{"type":"object","properties":null,"title":"raw"}`)
	wantRaw := append(json.RawMessage(nil), raw...)
	if _, err := newReplayProvider(t).BuildParams(toolSchemaRequest(raw), false); err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	if !bytes.Equal(raw, wantRaw) {
		t.Fatalf("caller RawMessage mutated: got %s, want %s", raw, wantRaw)
	}
}

func TestToolSchemaWireShapeMatchesBlockingAndStreaming(t *testing.T) {
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		bodies = append(bodies, body)
		if stream, _ := body["stream"].(bool); stream {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]}`)
	}))
	t.Cleanup(server.Close)
	p, err := chatcompletions.New(server.URL,
		chatcompletions.WithHTTPClient(server.Client()),
		chatcompletions.WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	req := toolSchemaRequest(json.RawMessage(`{"type":"object","properties":null}`))
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if _, err := llm.Collect(p.ChatStream(context.Background(), req)); err != nil {
		t.Fatalf("ChatStream returned error: %v", err)
	}
	if len(bodies) != 2 {
		t.Fatalf("requests = %d, want 2", len(bodies))
	}
	for i, body := range bodies {
		properties := wireToolSchema(t, body)["properties"]
		object, ok := properties.(map[string]any)
		if !ok || object == nil || len(object) != 0 {
			t.Fatalf("request %d properties = %#v (%T), want {}", i, properties, properties)
		}
	}
}

func TestMalformedToolPropertiesFailBeforeNetwork(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		requests++
	}))
	t.Cleanup(server.Close)
	p, err := chatcompletions.New(server.URL,
		chatcompletions.WithHTTPClient(server.Client()),
		chatcompletions.WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	for _, properties := range []any{[]any{}, "none", 1, false} {
		req := toolSchemaRequest(map[string]any{"type": "object", "properties": properties})
		if _, err := p.Chat(context.Background(), req); !errors.Is(err, llm.ErrBadRequest) {
			t.Fatalf("Chat properties %T error = %v, want ErrBadRequest", properties, err)
		}
		if _, err := llm.Collect(p.ChatStream(context.Background(), req)); !errors.Is(err, llm.ErrBadRequest) {
			t.Fatalf("ChatStream properties %T error = %v, want ErrBadRequest", properties, err)
		}
	}
	if requests != 0 {
		t.Fatalf("malformed schemas made %d requests, want 0", requests)
	}
}

func toolSchemaRequest(schema any) *llm.Request {
	return &llm.Request{
		Model:    "m",
		Messages: []llm.Message{llm.UserText("hi")},
		Tools:    []llm.Tool{{Name: "lookup", InputSchema: schema}},
	}
}

func builtToolSchema(t *testing.T, params any) map[string]any {
	t.Helper()
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("decode params: %v", err)
	}
	return wireToolSchema(t, body)
}

func wireToolSchema(t *testing.T, body map[string]any) map[string]any {
	t.Helper()
	tools, ok := body["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %#v, want one tool", body["tools"])
	}
	function := tools[0].(map[string]any)["function"].(map[string]any)
	return function["parameters"].(map[string]any)
}

func containsAll(value string, fragments ...string) bool {
	for _, fragment := range fragments {
		if !strings.Contains(value, fragment) {
			return false
		}
	}
	return true
}
