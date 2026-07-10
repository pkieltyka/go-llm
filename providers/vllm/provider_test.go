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
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
)

func newTestProvider(t *testing.T, handler http.HandlerFunc, opts ...Option) *Provider {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	all := append([]Option{WithMaxRetries(0), WithHTTPClient(server.Client())}, opts...)
	p, err := New(server.URL, all...)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	return p
}

func mustWrite(t *testing.T, w io.Writer, s string) {
	t.Helper()
	if _, err := io.WriteString(w, s); err != nil {
		t.Fatalf("WriteString returned error: %v", err)
	}
}

func TestNewRequiresBaseURL(t *testing.T) {
	if _, err := New(""); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("New(\"\") error = %v, want ErrBadRequest", err)
	}
}

func TestNewIsKeyOptional(t *testing.T) {
	if _, err := New("http://localhost:8000/v1"); err != nil {
		t.Fatalf("keyless New returned error: %v", err)
	}
}

// TestBuildRequestGolden pins the vLLM wire shape: reasoning_effort from the
// unified Effort, typed Options extras (sampling, chat_template_kwargs with
// the EnableThinking merge, vllm_xargs), same-provider plain-text reasoning
// replayed under `reasoning`, and foreign reasoning dropped.
func TestBuildRequestGolden(t *testing.T) {
	p, err := New("http://vllm.test/v1")
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	topK := 40
	minP := 0.05
	repetitionPenalty := 1.05
	enableThinking := false
	params, err := p.inner.BuildParams(&llm.Request{
		Model:     "Qwen/Qwen3.6-27B-FP8",
		MaxTokens: 64,
		Effort:    llm.EffortHigh,
		Messages: []llm.Message{
			llm.UserText("solve"),
			llm.AssistantParts(
				llm.ReasoningPart{Provider: providerName, Text: "prior thinking"},
				llm.ReasoningPart{Provider: "anthropic", Text: "foreign thinking"},
				llm.Text("prior answer"),
			),
			llm.UserText("continue"),
		},
		ProviderOptions: Options{
			TopK:               &topK,
			MinP:               &minP,
			RepetitionPenalty:  &repetitionPenalty,
			StopTokenIDs:       []int{151645},
			EnableThinking:     &enableThinking,
			ChatTemplateKwargs: map[string]any{"enable_thinking": true, "custom_flag": "x"},
			XArgs:              map[string]any{"speculative_len": 4},
		},
	}, false)
	if err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	got := testutil.MustCompactJSON(t, params)
	want := `{
	"chat_template_kwargs": {
		"custom_flag": "x",
		"enable_thinking": false
	},
	"max_tokens": 64,
	"messages": [
		{
			"content": [
				{
					"text": "solve",
					"type": "text"
				}
			],
			"role": "user"
		},
		{
			"content": "prior answer",
			"reasoning": "prior thinking",
			"role": "assistant"
		},
		{
			"content": [
				{
					"text": "continue",
					"type": "text"
				}
			],
			"role": "user"
		}
	],
	"model": "Qwen/Qwen3.6-27B-FP8",
	"n": 1,
	"reasoning_effort": "high",
	"repetition_penalty": 1.05,
	"stop_token_ids": [
		151645
	],
	"top_k": 40,
	"min_p": 0.05,
	"vllm_xargs": {
		"speculative_len": 4
	}
}`
	testutil.AssertJSONEqual(t, got, want)
	if strings.Contains(got, "foreign thinking") {
		t.Fatalf("foreign reasoning was replayed: %s", got)
	}
}

func TestLegacyEraReplaysReasoningContent(t *testing.T) {
	p, err := New("http://vllm.test/v1", WithLegacyEra())
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	params, err := p.inner.BuildParams(&llm.Request{
		Model: "m",
		Messages: []llm.Message{
			llm.UserText("q"),
			llm.AssistantParts(
				llm.ReasoningPart{Provider: providerName, Text: "prior thinking"},
				llm.Text("a"),
			),
			llm.UserText("next"),
		},
	}, false)
	if err != nil {
		t.Fatalf("BuildParams returned error: %v", err)
	}
	raw, _ := json.Marshal(params)
	if !strings.Contains(string(raw), `"reasoning_content":"prior thinking"`) {
		t.Fatalf("legacy era wire missing reasoning_content: %s", raw)
	}
	if strings.Contains(string(raw), `"reasoning":"`) {
		t.Fatalf("legacy era wire used modern reasoning field: %s", raw)
	}
}

func TestEffortMapping(t *testing.T) {
	p, err := New("http://vllm.test/v1")
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	cases := []struct {
		effort llm.Effort
		want   string // "" = reasoning_effort absent
	}{
		{"", ""},
		{llm.EffortNone, "none"},
		{llm.EffortMinimal, "minimal"},
		{llm.EffortLow, "low"},
		{llm.EffortMedium, "medium"},
		{llm.EffortHigh, "high"},
		{llm.EffortXHigh, "high"}, // nearest vLLM level: no xhigh upstream
		{llm.EffortMax, "max"},    // vLLM-specific level
	}
	for _, tc := range cases {
		params, err := p.inner.BuildParams(&llm.Request{
			Model:    "m",
			Effort:   tc.effort,
			Messages: []llm.Message{llm.UserText("hi")},
		}, false)
		if err != nil {
			t.Fatalf("BuildParams(%q) returned error: %v", tc.effort, err)
		}
		raw, _ := json.Marshal(params)
		var body struct {
			ReasoningEffort *string `json:"reasoning_effort"`
		}
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Fatalf("unmarshal params: %v", err)
		}
		switch {
		case tc.want == "" && body.ReasoningEffort != nil:
			t.Fatalf("effort %q sent reasoning_effort %q, want absent", tc.effort, *body.ReasoningEffort)
		case tc.want != "" && (body.ReasoningEffort == nil || *body.ReasoningEffort != tc.want):
			t.Fatalf("effort %q reasoning_effort = %v, want %q", tc.effort, body.ReasoningEffort, tc.want)
		}
	}
}

// vllmChatBody is a verbatim-shaped vLLM 0.24 blocking response: content is
// null while reasoning carries the parser's output, and vLLM-only fields
// (stop_reason, prompt_token_ids, kv_transfer_params) ride along.
const vllmChatBody = `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"Qwen/Qwen3.6-27B-FP8",` +
	`"choices":[{"index":0,"message":{"role":"assistant","content":"pong","refusal":null,"annotations":null,"audio":null,"function_call":null,"reasoning":"user wants pong"},` +
	`"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],` +
	`"service_tier":null,"system_fingerprint":"vllm-0.24.0","usage":{"prompt_tokens":12,"total_tokens":72,"completion_tokens":60,"prompt_tokens_details":null},` +
	`"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}`

func TestChatMapsReasoningField(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, vllmChatBody)
	})
	resp, err := p.Chat(context.Background(), &llm.Request{
		Model:    "Qwen/Qwen3.6-27B-FP8",
		Messages: []llm.Message{llm.UserText("Say pong")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Text() != "pong" || resp.Reasoning() != "user wants pong" {
		t.Fatalf("text/reasoning = %q/%q", resp.Text(), resp.Reasoning())
	}
	reasoning, ok := resp.Parts[0].(llm.ReasoningPart)
	if !ok || reasoning.Provider != providerName || len(reasoning.Raw) != 0 {
		t.Fatalf("reasoning part = %+v", resp.Parts[0])
	}
	if resp.StopReason != llm.StopReasonEndTurn {
		t.Fatalf("stop reason = %q", resp.StopReason)
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 60 || resp.Usage.TotalTokens != 72 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	if resp.Usage.CostUSD != nil {
		t.Fatalf("self-hosted cost should be nil: %+v", resp.Usage)
	}
	raw, ok := resp.Raw.(chatcompletions.JSONObject)
	if !ok || raw["system_fingerprint"] != "vllm-0.24.0" {
		t.Fatalf("raw extras = %#v", resp.Raw)
	}
}

func TestChatMapsLegacyReasoningContent(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"id":"chatcmpl-1","model":"m","choices":[{"index":0,"finish_reason":"stop",`+
			`"message":{"role":"assistant","content":"hi","reasoning_content":"legacy thinking"}}],`+
			`"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`)
	})
	resp, err := p.Chat(context.Background(), &llm.Request{
		Model:    "m",
		Messages: []llm.Message{llm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Reasoning() != "legacy thinking" {
		t.Fatalf("legacy reasoning_content not mapped: %+v", resp.Parts)
	}
}

func TestChatUsageCachedTokens(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"id":"chatcmpl-1","model":"m","choices":[{"index":0,"finish_reason":"stop",`+
			`"message":{"role":"assistant","content":"hi"}}],`+
			`"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":4}}}`)
	})
	resp, err := p.Chat(context.Background(), &llm.Request{
		Model:    "m",
		Messages: []llm.Message{llm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	usage := resp.Usage
	if usage.InputTokens != 6 || usage.CacheReadTokens != 4 || usage.OutputTokens != 5 || usage.TotalTokens != 15 {
		t.Fatalf("usage = %+v", usage)
	}
}

// TestStreamReasoningDeltas covers the vLLM streaming shape: delta.reasoning
// fragments then content, a finish chunk, and a trailing usage-only chunk.
// The collected ReasoningPart must carry plain text tagged with the vllm
// provider (Collect-equivalence with the blocking path) and no Raw payload.
func TestStreamReasoningDeltas(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		mustWrite(t, w, `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"reasoning":"think "},"finish_reason":null}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"reasoning":"hard"},"finish_reason":null}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"pong"},"finish_reason":"stop","stop_reason":null}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"c1","model":"m","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":50,"total_tokens":62}}`+"\n\n")
		mustWrite(t, w, "data: [DONE]\n\n")
	})
	resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{
		Model:    "m",
		Messages: []llm.Message{llm.UserText("Say pong")},
	}))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if resp.Text() != "pong" || resp.Reasoning() != "think hard" {
		t.Fatalf("text/reasoning = %q/%q", resp.Text(), resp.Reasoning())
	}
	reasoning, ok := resp.Parts[0].(llm.ReasoningPart)
	if !ok || reasoning.Provider != providerName || len(reasoning.Raw) != 0 {
		t.Fatalf("collected reasoning part = %+v", resp.Parts[0])
	}
	if resp.Usage.OutputTokens != 50 || resp.Usage.TotalTokens != 62 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	if resp.StopReason != llm.StopReasonEndTurn {
		t.Fatalf("stop reason = %q", resp.StopReason)
	}
}

// TestStreamMidStreamErrorSniff covers the goose-crash case: after HTTP 200,
// vLLM emits a choice-less SSE data event whose payload is the error JSON.
// Both the nested (current) and flat legacy shapes must map to a normalized
// in-stream error, with the partial response still collectable.
func TestStreamMidStreamErrorSniff(t *testing.T) {
	cases := map[string]struct {
		payload string
		want    error
	}{
		"nested": {
			payload: `{"error":{"message":"Internal server error","type":"InternalServerError","param":null,"code":500}}`,
			want:    llm.ErrServer,
		},
		"flat_legacy": {
			payload: `{"object":"error","message":"engine died","code":500}`,
			want:    llm.ErrServer,
		},
		// Status-less chunk errors carry a numeric code mirroring an HTTP
		// status; it classifies through the canonical status table.
		"numeric_code_status_table": {
			payload: `{"object":"error","message":"too many requests","code":429}`,
			want:    llm.ErrRateLimited,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				mustWrite(t, w, `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"par"},"finish_reason":null}]}`+"\n\n")
				mustWrite(t, w, "data: "+tc.payload+"\n\n")
			})
			resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{
				Model:    "m",
				Messages: []llm.Message{llm.UserText("hi")},
			}))
			if !errors.Is(err, tc.want) {
				t.Fatalf("mid-stream error = %v, want %v", err, tc.want)
			}
			var providerErr *llm.ProviderError
			if !errors.As(err, &providerErr) || providerErr.Provider != providerName {
				t.Fatalf("mid-stream error missing provider metadata: %v", err)
			}
			if resp == nil || resp.Text() != "par" {
				t.Fatalf("partial response = %+v, want partial text", resp)
			}
		})
	}
}

// TestStreamTrailingUsageChunkIsNotSniffed guards the sniff against false
// positives: the trailing usage chunk carries "choices":[] (present key) and
// must map to usage, not an error.
func TestStreamTrailingUsageChunkIsNotSniffed(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		mustWrite(t, w, `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\n")
		mustWrite(t, w, `data: {"id":"c1","model":"m","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\n")
		mustWrite(t, w, "data: [DONE]\n\n")
	})
	resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{
		Model:    "m",
		Messages: []llm.Message{llm.UserText("hi")},
	}))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if resp.Usage.TotalTokens != 2 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestModelsSurfacesMaxModelLen(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Fatalf("models path = %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"object":"list","data":[`+
			`{"id":"Qwen/Qwen3.6-27B-FP8","object":"model","owned_by":"vllm","root":"Qwen/Qwen3.6-27B-FP8","parent":null,"max_model_len":131072},`+
			`{"id":"my-lora","object":"model","owned_by":"vllm","root":"/adapters/my-lora","parent":"Qwen/Qwen3.6-27B-FP8","max_model_len":131072}]}`)
	})
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("models len = %d", len(models))
	}
	if models[0].ID != "Qwen/Qwen3.6-27B-FP8" || models[0].ContextWindow != 131072 {
		t.Fatalf("base model = %+v", models[0])
	}
	raw, ok := models[1].Raw.(json.RawMessage)
	if !ok || !strings.Contains(string(raw), `"parent":"Qwen/Qwen3.6-27B-FP8"`) {
		t.Fatalf("LoRA row raw = %+v", models[1].Raw)
	}
}

func TestResolveModel(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Fatalf("models path = %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"object":"list","data":[`+
			`{"id":"meta-llama/Llama-3.3-70B-Instruct","max_model_len":131072},`+
			`{"id":"nvidia/Qwen-3.6-27B-NVFP4","max_model_len":131072},`+
			`{"id":"Qwen/Qwen3.6-27B-FP8","max_model_len":131072}]}`)
	})

	cases := []struct {
		name       string
		preference string
		want       string
	}{
		{
			name:       "empty_prefers_qwen",
			preference: "",
			want:       "nvidia/Qwen-3.6-27B-NVFP4",
		},
		{
			name:       "short_qwen_prefers_first_qwen",
			preference: "qwen",
			want:       "nvidia/Qwen-3.6-27B-NVFP4",
		},
		{
			name:       "substring_ignores_case",
			preference: "qwen-3.6-27b",
			want:       "nvidia/Qwen-3.6-27B-NVFP4",
		},
		{
			name:       "old_precision_suffix_fuzzy_matches_new_quantization",
			preference: "Qwen/Qwen3.6-27B-FP8",
			want:       "Qwen/Qwen3.6-27B-FP8",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := p.ResolveModel(context.Background(), tc.preference)
			if err != nil {
				t.Fatalf("ResolveModel returned error: %v", err)
			}
			if got.ID != tc.want {
				t.Fatalf("ResolveModel(%q) = %q, want %q", tc.preference, got.ID, tc.want)
			}
		})
	}
}

func TestResolveModelFuzzyMatchesChangedServedID(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Fatalf("models path = %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"data":[`+
			`{"id":"meta-llama/Llama-3.3-70B-Instruct","max_model_len":131072},`+
			`{"id":"nvidia/Qwen-3.6-27B-NVFP4","max_model_len":131072}]}`)
	})
	got, err := p.ResolveModel(context.Background(), "Qwen/Qwen3.6-27B-FP8")
	if err != nil {
		t.Fatalf("ResolveModel returned error: %v", err)
	}
	if got.ID != "nvidia/Qwen-3.6-27B-NVFP4" {
		t.Fatalf("ResolveModel fuzzy match = %q", got.ID)
	}
}

func TestResolveModelFallsBackToFirstServedModel(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"data":[{"id":"model-a"},{"id":"model-b"}]}`)
	})
	got, err := p.ResolveModel(context.Background(), "missing")
	if err != nil {
		t.Fatalf("ResolveModel returned error: %v", err)
	}
	if got.ID != "model-a" {
		t.Fatalf("fallback model = %q", got.ID)
	}
}

func TestResolveModelErrorsOnEmptyModelsList(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"data":[]}`)
	})
	if _, err := p.ResolveModel(context.Background(), "qwen"); !errors.Is(err, llm.ErrNotFound) {
		t.Fatalf("ResolveModel error = %v, want ErrNotFound", err)
	}
}

// TestKeylessSendsNoAuthorization pins keyless construction end to end: no
// Authorization header on the SDK blocking path, the direct SSE path, or
// models listing — and WithAPIKey restores the bearer header.
func TestKeylessSendsNoAuthorization(t *testing.T) {
	t.Setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer ambient-secret\nX-Ambient-Safe: retained")
	var authHeaders []string
	var ambientHeaders []string
	handler := func(w http.ResponseWriter, r *http.Request) {
		value, present := r.Header["Authorization"]
		if present {
			authHeaders = append(authHeaders, strings.Join(value, ","))
		} else {
			authHeaders = append(authHeaders, "<absent>")
		}
		ambientHeaders = append(ambientHeaders, r.Header.Get("X-Ambient-Safe"))
		if r.Header.Get("Accept") == "text/event-stream" {
			w.Header().Set("Content-Type", "text/event-stream")
			mustWrite(t, w, `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\n")
			mustWrite(t, w, "data: [DONE]\n\n")
			return
		}
		if r.URL.Path == "/models" {
			w.Header().Set("Content-Type", "application/json")
			mustWrite(t, w, `{"data":[{"id":"m","max_model_len":100}]}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}

	p := newTestProvider(t, handler)
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if _, err := llm.Collect(p.ChatStream(context.Background(), req)); err != nil {
		t.Fatalf("stream returned error: %v", err)
	}
	if _, err := p.Models(context.Background()); err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	for i, header := range authHeaders {
		if header != "<absent>" {
			t.Fatalf("request %d sent Authorization %q, want none", i, header)
		}
		if ambientHeaders[i] != "retained" {
			t.Fatalf("request %d X-Ambient-Safe = %q", i, ambientHeaders[i])
		}
	}

	authHeaders = nil
	ambientHeaders = nil
	keyed := newTestProvider(t, handler, WithAPIKey("secret-key"))
	if _, err := keyed.Chat(context.Background(), req); err != nil {
		t.Fatalf("keyed Chat returned error: %v", err)
	}
	if _, err := llm.Collect(keyed.ChatStream(context.Background(), req)); err != nil {
		t.Fatalf("keyed stream returned error: %v", err)
	}
	if _, err := keyed.Models(context.Background()); err != nil {
		t.Fatalf("keyed Models returned error: %v", err)
	}
	for i, header := range authHeaders {
		if header != "Bearer secret-key" {
			t.Fatalf("keyed request %d Authorization = %q", i, header)
		}
		if ambientHeaders[i] != "retained" {
			t.Fatalf("keyed request %d X-Ambient-Safe = %q", i, ambientHeaders[i])
		}
	}
}

func TestErrorMapping(t *testing.T) {
	cases := map[string]struct {
		status int
		body   string
		want   error
	}{
		"nested_not_found": {
			status: http.StatusNotFound,
			body:   `{"error":{"message":"The model does not exist.","type":"NotFoundError","param":"model","code":404}}`,
			want:   llm.ErrNotFound,
		},
		"nested_bad_request": {
			status: http.StatusBadRequest,
			body:   `{"error":{"message":"max_tokens=131073 cannot be greater than max_model_len","type":"BadRequestError","param":"max_tokens","code":400}}`,
			want:   llm.ErrBadRequest,
		},
		"flat_legacy": {
			status: http.StatusBadRequest,
			body:   `{"object":"error","message":"bad request","type":"invalid_request_error","code":400}`,
			want:   llm.ErrBadRequest,
		},
		"auth": {
			status: http.StatusUnauthorized,
			body:   `{"error":{"message":"Unauthorized","type":"AuthenticationError","code":401}}`,
			want:   llm.ErrAuth,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.status)
				mustWrite(t, w, tc.body)
			})
			_, err := p.Chat(context.Background(), &llm.Request{
				Model:    "m",
				Messages: []llm.Message{llm.UserText("hi")},
			})
			if !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want %v", err, tc.want)
			}
			var providerErr *llm.ProviderError
			if !errors.As(err, &providerErr) || providerErr.HTTPStatus != tc.status {
				t.Fatalf("provider error metadata = %+v", err)
			}
		})
	}
}

// TestForcedToolCallStopNormalizes covers the vLLM forced-call semantic:
// named/required tool choice ends with finish_reason "stop" even though the
// message carries tool calls; the preset upgrades the mapped stop reason to
// tool_use (FS §5) while keeping the raw wire value, on both paths.
func TestForcedToolCallStopNormalizes(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Accept") == "text/event-stream" {
			w.Header().Set("Content-Type", "text/event-stream")
			mustWrite(t, w, `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"chatcmpl-tool-1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"go\"}"}}]},"finish_reason":null}]}`+"\n\n")
			mustWrite(t, w, `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop","stop_reason":null}]}`+"\n\n")
			mustWrite(t, w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"stop","stop_reason":null,`+
			`"message":{"role":"assistant","content":null,"tool_calls":[{"id":"chatcmpl-tool-1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"go\"}"}}]}}],`+
			`"usage":{"prompt_tokens":5,"completion_tokens":5,"total_tokens":10}}`)
	}
	p := newTestProvider(t, handler)
	req := &llm.Request{
		Model:    "m",
		Messages: []llm.Message{llm.UserText("look up go")},
		Tools: []llm.Tool{{
			Name:        "lookup",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`),
		}},
		ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "lookup"},
	}

	resp, err := p.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.StopReason != llm.StopReasonToolUse || resp.StopReasonRaw != "stop" {
		t.Fatalf("blocking stop = %q (raw %q), want tool_use/stop", resp.StopReason, resp.StopReasonRaw)
	}
	if calls := resp.ToolCalls(); len(calls) != 1 || calls[0].Name != "lookup" {
		t.Fatalf("blocking tool calls = %+v", calls)
	}

	streamed, err := llm.Collect(p.ChatStream(context.Background(), req))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if streamed.StopReason != llm.StopReasonToolUse || streamed.StopReasonRaw != "stop" {
		t.Fatalf("streamed stop = %q (raw %q), want tool_use/stop", streamed.StopReason, streamed.StopReasonRaw)
	}
	if calls := streamed.ToolCalls(); len(calls) != 1 || calls[0].Name != "lookup" {
		t.Fatalf("streamed tool calls = %+v", calls)
	}
}

func TestStopReasonAbort(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mustWrite(t, w, `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"abort",`+
			`"message":{"role":"assistant","content":"partial"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	})
	resp, err := p.Chat(context.Background(), &llm.Request{
		Model:    "m",
		Messages: []llm.Message{llm.UserText("hi")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.StopReason != llm.StopReasonError || resp.StopReasonRaw != "abort" {
		t.Fatalf("stop reason = %q (raw %q)", resp.StopReason, resp.StopReasonRaw)
	}
}

func TestOptionsRejectedForOtherProvider(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		t.Fatal("request must not reach the server")
	})
	_, err := p.Chat(context.Background(), &llm.Request{
		Model:           "m",
		Messages:        []llm.Message{llm.UserText("hi")},
		ProviderOptions: fakeOptions{},
	})
	if !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("foreign options error = %v, want ErrBadRequest", err)
	}
}

type fakeOptions struct{}

func (fakeOptions) ForProvider() string { return "openrouter" }

var _ llm.Provider = (*Provider)(nil)
