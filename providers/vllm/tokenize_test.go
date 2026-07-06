package vllm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
)

// TestTokenizeBuildsChatShapedBody pins the /tokenize wire shape: the body
// reuses the engine's message/tool conversion verbatim (system prompt, text
// blocks, tool defs), adds add_generation_prompt, and carries the provider's
// chat_template_kwargs handling so counts match a real chat request.
func TestTokenizeBuildsChatShapedBody(t *testing.T) {
	var gotPath, gotMethod string
	var gotBody []byte
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotMethod = r.URL.Path, r.Method
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"count":3,"max_model_len":131072,"tokens":[1,2,3]}`)
	})
	enableThinking := false
	result, err := p.Tokenize(context.Background(), &llm.Request{
		Model:  "Qwen/Qwen3.6-27B-FP8",
		System: "be terse",
		Messages: []llm.Message{
			llm.UserText("Output exactly this word and no other text: pong"),
		},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Description: "Look up a short value.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`),
		}},
		ProviderOptions: Options{
			EnableThinking:     &enableThinking,
			ChatTemplateKwargs: map[string]any{"custom_flag": "x"},
		},
	})
	if err != nil {
		t.Fatalf("Tokenize returned error: %v", err)
	}
	if gotMethod != http.MethodPost || gotPath != "/tokenize" {
		t.Fatalf("tokenize request = %s %s, want POST /tokenize", gotMethod, gotPath)
	}
	want := `{
	"model": "Qwen/Qwen3.6-27B-FP8",
	"add_generation_prompt": true,
	"chat_template_kwargs": {
		"custom_flag": "x",
		"enable_thinking": false
	},
	"messages": [
		{
			"content": [
				{
					"text": "be terse",
					"type": "text"
				}
			],
			"role": "system"
		},
		{
			"content": [
				{
					"text": "Output exactly this word and no other text: pong",
					"type": "text"
				}
			],
			"role": "user"
		}
	],
	"tools": [
		{
			"function": {
				"description": "Look up a short value.",
				"name": "lookup",
				"parameters": {
					"properties": {
						"q": {
							"type": "string"
						}
					},
					"required": [
						"q"
					],
					"type": "object"
				},
				"strict": false
			},
			"type": "function"
		}
	]
}`
	testutil.AssertJSONEqual(t, string(gotBody), want)
	if result.Count != 3 || result.MaxModelLen != 131072 || len(result.Tokens) != 3 {
		t.Fatalf("result = %+v", result)
	}
	usage := result.ContextUsage()
	if usage.UsedTokens != 3 || usage.Window != 131072 || usage.Remaining != 131069 || usage.UsedPercent <= 0 {
		t.Fatalf("ContextUsage bridge = %+v", usage)
	}
}

// TestTokenizeHitsServerRootNotV1 pins the path answer probed live: the
// tokenizer endpoints live at the server root while the OpenAI surface hangs
// off /v1 (POST /v1/tokenize is 404), so a conventional ".../v1" base URL
// must produce a root-relative /tokenize request.
func TestTokenizeHitsServerRootNotV1(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"count":1,"max_model_len":10,"tokens":[7]}`)
	}))
	t.Cleanup(server.Close)
	p, err := New(server.URL+"/v1", WithMaxRetries(0), WithHTTPClient(server.Client()))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if _, err := p.Tokenize(context.Background(), &llm.Request{
		Model:    "m",
		Messages: []llm.Message{llm.UserText("hi")},
	}); err != nil {
		t.Fatalf("Tokenize returned error: %v", err)
	}
	if gotPath != "/tokenize" {
		t.Fatalf("tokenize path = %q, want /tokenize (root-relative)", gotPath)
	}
}

// TestTokenizeEffortInjection mirrors vLLM's server-side reasoning_effort →
// enable_thinking injection so tokenize counts match real chat requests
// (live-verified parity: effort none adds the empty-think-block tokens).
// Explicit chat_template_kwargs win over the injection.
func TestTokenizeEffortInjection(t *testing.T) {
	enableThinking := true
	cases := map[string]struct {
		effort  llm.Effort
		options llm.ProviderOptions
		want    string // "" = no chat_template_kwargs in the body
	}{
		"no_effort_no_kwargs": {effort: "", want: ""},
		"effort_none":         {effort: llm.EffortNone, want: `{"enable_thinking":false}`},
		"effort_high":         {effort: llm.EffortHigh, want: `{"enable_thinking":true}`},
		"explicit_wins": {
			effort:  llm.EffortNone,
			options: Options{EnableThinking: &enableThinking},
			want:    `{"enable_thinking":true}`,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			var gotBody []byte
			p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
				gotBody, _ = io.ReadAll(r.Body)
				w.Header().Set("Content-Type", "application/json")
				mustWrite(t, w, `{"count":1,"max_model_len":10,"tokens":[7]}`)
			})
			_, err := p.Tokenize(context.Background(), &llm.Request{
				Model:           "m",
				Effort:          tc.effort,
				Messages:        []llm.Message{llm.UserText("hi")},
				ProviderOptions: tc.options,
			})
			if err != nil {
				t.Fatalf("Tokenize returned error: %v", err)
			}
			var body struct {
				ChatTemplateKwargs json.RawMessage `json:"chat_template_kwargs"`
			}
			if err := json.Unmarshal(gotBody, &body); err != nil {
				t.Fatalf("unmarshal body: %v", err)
			}
			switch {
			case tc.want == "" && body.ChatTemplateKwargs != nil:
				t.Fatalf("chat_template_kwargs = %s, want absent", body.ChatTemplateKwargs)
			case tc.want != "" && string(body.ChatTemplateKwargs) != tc.want:
				t.Fatalf("chat_template_kwargs = %s, want %s", body.ChatTemplateKwargs, tc.want)
			}
		})
	}
}

// TestTokenizeValidatesLikeChat pins that the extension inherits the engine's
// request validation (same build path as Chat) rather than sending garbage.
func TestTokenizeValidatesLikeChat(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("request must not reach the server")
	})
	if _, err := p.Tokenize(context.Background(), &llm.Request{Model: "m"}); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("empty-messages error = %v, want ErrBadRequest", err)
	}
	_, err := p.Tokenize(context.Background(), &llm.Request{
		Model:           "m",
		Messages:        []llm.Message{llm.UserText("hi")},
		ProviderOptions: fakeOptions{},
	})
	if !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("foreign options error = %v, want ErrBadRequest", err)
	}
}

func TestDetokenize(t *testing.T) {
	var gotPath string
	var gotBody []byte
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"prompt":"<|im_start|>user\nhello<|im_end|>"}`)
	})
	prompt, err := p.Detokenize(context.Background(), []int{248045, 846})
	if err != nil {
		t.Fatalf("Detokenize returned error: %v", err)
	}
	if gotPath != "/detokenize" {
		t.Fatalf("detokenize path = %q", gotPath)
	}
	testutil.AssertJSONEqual(t, string(gotBody), `{"tokens":[248045,846]}`)
	if !strings.Contains(prompt, "hello") {
		t.Fatalf("prompt = %q", prompt)
	}

	// nil tokens normalize to an empty list (the server rejects null).
	if _, err := p.Detokenize(context.Background(), nil); err != nil {
		t.Fatalf("Detokenize(nil) returned error: %v", err)
	}
	testutil.AssertJSONEqual(t, string(gotBody), `{"tokens":[]}`)
}

func TestTokenizerInfo(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tokenizer_info" || r.Method != http.MethodGet {
			t.Fatalf("tokenizer_info request = %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"tokenizer_class":"Qwen2Tokenizer","max_model_len":131072,"extra_field":true}`)
	})
	info, err := p.TokenizerInfo(context.Background())
	if err != nil {
		t.Fatalf("TokenizerInfo returned error: %v", err)
	}
	// Raw passthrough: version-varying fields survive verbatim.
	if !strings.Contains(string(info), `"extra_field":true`) {
		t.Fatalf("info = %s", info)
	}
}

// TestTokenizerInfoNotEnabled maps the gated-off endpoint (404 without
// --enable-tokenizer-info-endpoint, observed live) to ErrNotFound with
// provider metadata.
func TestTokenizerInfoNotEnabled(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		mustWrite(t, w, `{"detail":"Not Found"}`)
	})
	_, err := p.TokenizerInfo(context.Background())
	if !errors.Is(err, llm.ErrNotFound) {
		t.Fatalf("gated endpoint error = %v, want ErrNotFound", err)
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Provider != providerName {
		t.Fatalf("error missing vllm provider metadata: %v", err)
	}
}
