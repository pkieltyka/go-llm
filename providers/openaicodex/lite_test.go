package openaicodex

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
)

func codexLiteFixtureRequest(model string) *llm.Request {
	return &llm.Request{
		Model:  model,
		System: "You are a probe.",
		Messages: []llm.Message{
			llm.UserText("call the lookup tool"),
		},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Description: "Look something up.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`),
		}},
	}
}

// TestOpenAICodexResponsesLiteBodyGolden pins the exact Lite request shape
// (item order included) for the gpt-5.6 family: tools as a leading
// additional_tools developer item, instructions as a developer message,
// parallel_tool_calls forced false, reasoning.context forced all_turns.
func TestOpenAICodexResponsesLiteBodyGolden(t *testing.T) {
	params, err := (&Provider{}).adapter().BuildParams(codexLiteFixtureRequest("gpt-5.6"), true)
	if err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	body, err := codexStreamingBody(params)
	if err != nil {
		t.Fatalf("codexStreamingBody returned error: %v", err)
	}
	testutil.AssertJSONEqual(t, string(body), `{
		"include":["reasoning.encrypted_content"],
		"input":[
			{"type":"additional_tools","role":"developer","tools":[{"type":"function","name":"lookup","description":"Look something up.","parameters":{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]},"strict":false}]},
			{"type":"message","role":"developer","content":[{"type":"input_text","text":"You are a probe."}]},
			{"content":[{"text":"call the lookup tool","type":"input_text"}],"role":"user"}
		],
		"model":"gpt-5.6",
		"parallel_tool_calls":false,
		"reasoning":{"context":"all_turns"},
		"store":false,
		"stream":true,
		"tool_choice":"auto"
	}`)
}

// Pre-5.6 requests keep the existing wire body byte-for-byte: top-level
// tools and instructions, no Lite fields.
func TestOpenAICodexPre56BodyUnchanged(t *testing.T) {
	params, err := (&Provider{}).adapter().BuildParams(codexLiteFixtureRequest("gpt-5.5"), true)
	if err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	body, err := codexStreamingBody(params)
	if err != nil {
		t.Fatalf("codexStreamingBody returned error: %v", err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		t.Fatalf("body is not JSON: %v", err)
	}
	if _, ok := fields["tools"]; !ok {
		t.Fatal("pre-5.6 body lost top-level tools")
	}
	if _, ok := fields["instructions"]; !ok {
		t.Fatal("pre-5.6 body lost top-level instructions")
	}
	if _, ok := fields["parallel_tool_calls"]; ok {
		t.Fatal("pre-5.6 body gained forced parallel_tool_calls")
	}
	if raw, ok := fields["reasoning"]; ok {
		var reasoning map[string]json.RawMessage
		if err := json.Unmarshal(raw, &reasoning); err == nil {
			if _, ok := reasoning["context"]; ok {
				t.Fatal("pre-5.6 body gained reasoning.context")
			}
		}
	}
}

func TestOpenAICodexLiteHeaderGate(t *testing.T) {
	for _, tc := range []struct {
		model string
		want  string
	}{
		{"gpt-5.6", "true"},
		{"gpt-5.6-luna", "true"},
		{"gpt-5.6-sol-2026-07-09", "true"},
		{"gpt-5.5", ""},
	} {
		var gotHeader, gotUA string
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotHeader = r.Header.Get(codexResponsesLiteHeader)
			gotUA = r.Header.Get("User-Agent")
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"type":"response.created","response":{"id":"resp_1","model":"`+tc.model+`","status":"in_progress","output":[]}}`+"\n\n")
			_, _ = io.WriteString(w, `data: {"type":"response.completed","response":{"id":"resp_1","model":"`+tc.model+`","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok","annotations":[]}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}}`+"\n\n")
		}))
		p, err := New(
			WithOAuth(llm.AuthCredential{Type: "oauth", Access: fakeCodexJWT(t, "acct-lite"), Refresh: "refresh"}, func(ctx context.Context, _ llm.AuthCredential) error {
				return ctx.Err()
			}),
			WithBaseURL(server.URL),
			withOAuthTokenURL(server.URL+"/oauth/token"),
			WithHTTPClient(server.Client()),
			WithMaxRetries(0),
		)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		if _, err := p.Chat(context.Background(), &llm.Request{
			Model:    tc.model,
			Messages: []llm.Message{llm.UserText("hi")},
		}); err != nil {
			t.Fatalf("model %s: Chat returned error: %v", tc.model, err)
		}
		if gotHeader != tc.want {
			t.Fatalf("model %s: lite header = %q, want %q", tc.model, gotHeader, tc.want)
		}
		if gotUA != "codex_cli_rs/0.144.0" {
			t.Fatalf("model %s: User-Agent = %q, want codex_cli_rs/0.144.0", tc.model, gotUA)
		}
		server.Close()
	}
}
