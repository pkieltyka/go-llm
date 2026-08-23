package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestParsePricing(t *testing.T) {
	tests := []struct {
		name                  string
		prompt, completion    string
		cacheRead, cacheWrite string
		want                  *llm.ModelPricing
	}{
		{
			name:       "ordinary positive rates",
			prompt:     "0.000001",
			completion: "0.0000025",
			want:       &llm.ModelPricing{InputPerMTok: 1, OutputPerMTok: 2.5},
		},
		{
			name:       "explicit free rates",
			prompt:     "0",
			completion: "0.0",
			want:       &llm.ModelPricing{},
		},
		{name: "negative prompt invalidates token pricing", prompt: "-1", completion: "0.000002"},
		{name: "negative completion invalidates token pricing", prompt: "0.000001", completion: " -1 "},
		{
			name:       "negative cache rate is omitted",
			prompt:     "0.000001",
			completion: "0.000002",
			cacheRead:  "-1",
			cacheWrite: "0.0000005",
			want:       &llm.ModelPricing{InputPerMTok: 1, OutputPerMTok: 2, CacheWritePerMTok: 0.5},
		},
		{
			name:       "cache rates",
			cacheRead:  "0.00000025",
			cacheWrite: "0.00000075",
			want:       &llm.ModelPricing{CacheReadPerMTok: 0.25, CacheWritePerMTok: 0.75},
		},
		{name: "all absent"},
		{
			name:   "surrounding whitespace",
			prompt: " 0.0000015\t",
			want:   &llm.ModelPricing{InputPerMTok: 1.5},
		},
		{
			name:       "malformed component preserves valid component",
			prompt:     "not-a-number",
			completion: "0.000002",
			want:       &llm.ModelPricing{OutputPerMTok: 2},
		},
		{name: "all malformed", prompt: "wat", completion: ""},
		{name: "parse overflow", prompt: "1e309"},
		{name: "nan", prompt: "NaN"},
		{name: "positive infinity", prompt: "+Inf"},
		{name: "negative infinity", prompt: "-Infinity"},
		{name: "per-million overflow", prompt: "1e303"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parsePricing(tt.prompt, tt.completion, tt.cacheRead, tt.cacheWrite)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("parsePricing = %+v, want %+v", got, tt.want)
			}
			if got != nil {
				for name, value := range map[string]float64{
					"input": got.InputPerMTok, "output": got.OutputPerMTok,
					"cache read": got.CacheReadPerMTok, "cache write": got.CacheWritePerMTok,
				} {
					if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
						t.Fatalf("%s price leaked invalid value %v", name, value)
					}
				}
			}
		})
	}
}

func TestOpenRouterModelPricingPreservesFreeCacheAndRawValues(t *testing.T) {
	p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Fatalf("path = %s", r.URL.Path)
		}
		mustWrite(t, w, `{"data":[
			{"id":"free","pricing":{"prompt":"0","completion":"0","input_cache_read":"0","input_cache_write":"0","request":"9"}},
			{"id":"dynamic","pricing":{"prompt":"-1","completion":"0.000002","input_cache_read":"0.0000002","input_cache_write":"0.0000003"}},
			{"id":"fixed-cache","pricing":{"prompt":"0.000001","completion":"0.000002","input_cache_read":"0.00000025","input_cache_write":"0.00000075","future_tier":{"kept":true}}},
			{"id":"unknown-cache","pricing":{"prompt":"0.000001","completion":"0.000002","input_cache_read":"-1","input_cache_write":"wat"}}
		]}`)
	})
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	if len(models) != 4 {
		t.Fatalf("models = %d, want 4", len(models))
	}
	if models[0].Pricing == nil || !reflect.DeepEqual(models[0].Pricing, &llm.ModelPricing{}) {
		t.Fatalf("free pricing = %+v, want non-nil zero rates", models[0].Pricing)
	}
	if models[1].Pricing != nil {
		t.Fatalf("dynamic pricing = %+v, want nil", models[1].Pricing)
	}
	if got := models[2].Pricing; got == nil || got.InputPerMTok != 1 || got.OutputPerMTok != 2 || got.CacheReadPerMTok != 0.25 || got.CacheWritePerMTok != 0.75 {
		t.Fatalf("fixed cache pricing = %+v", got)
	}
	if got := models[3].Pricing; got == nil || got.InputPerMTok != 1 || got.OutputPerMTok != 2 || got.CacheReadPerMTok != 0 || got.CacheWritePerMTok != 0 {
		t.Fatalf("unknown cache pricing = %+v", got)
	}
	for _, model := range models {
		if model.Pricing != nil && (model.Pricing.InputPerMTok < 0 || model.Pricing.OutputPerMTok < 0 || model.Pricing.CacheReadPerMTok < 0 || model.Pricing.CacheWritePerMTok < 0) {
			t.Fatalf("model %q leaked negative pricing: %+v", model.ID, model.Pricing)
		}
	}
	raw := models[2].Raw.(json.RawMessage)
	for _, key := range []string{`"prompt"`, `"completion"`, `"input_cache_read"`, `"input_cache_write"`, `"future_tier"`} {
		if !bytes.Contains(raw, []byte(key)) {
			t.Fatalf("raw row missing %s: %s", key, raw)
		}
	}
}

func TestOpenRouterReasoningMetadataNormalization(t *testing.T) {
	rows := `
		{"id":"descending","reasoning":{"supported_efforts":["max","high","low","none"],"default_effort":"high","mandatory":false}},
		{"id":"ascending","reasoning":{"supported_efforts":["none","low","medium","high"],"default_effort":"medium"}},
		{"id":"duplicates","reasoning":{"supported_efforts":["high","low","high","medium","low"]}},
		{"id":"omitted-reasoning"},
		{"id":"omitted-ladder","reasoning":{"default_effort":"high"}},
		{"id":"null-ladder","reasoning":{"supported_efforts":null,"default_effort":"none"}},
		{"id":"null-copy","reasoning":{"supported_efforts":null}},
		{"id":"mandatory-null","reasoning":{"supported_efforts":null,"default_effort":"high","mandatory":true}},
		{"id":"empty-ladder","reasoning":{"supported_efforts":[],"default_effort":"high"}},
		{"id":"unknown-only","reasoning":{"supported_efforts":["turbo"],"default_effort":"high","future_policy":{"kept":true}}},
		{"id":"mandatory","reasoning":{"supported_efforts":["none","high","low"],"default_effort":"none","mandatory":true}},
		{"id":"off-by-default","reasoning":{"supported_efforts":["low","none"],"default_effort":"none","mandatory":false}},
		{"id":"contradictory-default","reasoning":{"supported_efforts":["low","medium"],"default_effort":"high"}},
		{"id":"unknown-default","reasoning":{"supported_efforts":["low"],"default_effort":"turbo"}},
		{"id":"scalar-reasoning","reasoning":"automatic"},
		{"id":"wrong-shape","reasoning":{"supported_efforts":{"low":true},"default_effort":"high","mandatory":true}},
		{"id":"mixed-types","reasoning":{"supported_efforts":["low",3],"default_effort":"high"}},
		{"id":"non-string-default","reasoning":{"supported_efforts":["high"],"default_effort":3}},
		{"id":"non-bool-mandatory","reasoning":{"supported_efforts":["none","low"],"default_effort":"none","mandatory":"yes"}}
	`
	p := newTestProvider(t, func(w http.ResponseWriter, _ *http.Request) {
		mustWrite(t, w, `{"data":[`+rows+`]}`)
	})
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	wantOrder := []string{"descending", "ascending", "duplicates", "omitted-reasoning", "omitted-ladder", "null-ladder", "null-copy", "mandatory-null", "empty-ladder", "unknown-only", "mandatory", "off-by-default", "contradictory-default", "unknown-default", "scalar-reasoning", "wrong-shape", "mixed-types", "non-string-default", "non-bool-mandatory"}
	if len(models) != len(wantOrder) {
		t.Fatalf("models = %d, want %d", len(models), len(wantOrder))
	}
	byID := make(map[string]llm.ModelInfo, len(models))
	for i, model := range models {
		if model.ID != wantOrder[i] {
			t.Fatalf("model %d ID = %q, want %q", i, model.ID, wantOrder[i])
		}
		byID[model.ID] = model
	}

	all := []llm.Effort{llm.EffortNone, llm.EffortMinimal, llm.EffortLow, llm.EffortMedium, llm.EffortHigh, llm.EffortXHigh, llm.EffortMax}
	assertReasoningModel(t, byID["descending"], []llm.Effort{llm.EffortNone, llm.EffortLow, llm.EffortHigh, llm.EffortMax}, llm.EffortHigh, false)
	assertReasoningModel(t, byID["ascending"], []llm.Effort{llm.EffortNone, llm.EffortLow, llm.EffortMedium, llm.EffortHigh}, llm.EffortMedium, false)
	assertReasoningModel(t, byID["duplicates"], []llm.Effort{llm.EffortLow, llm.EffortMedium, llm.EffortHigh}, "", false)
	assertReasoningModel(t, byID["omitted-reasoning"], nil, "", false)
	assertReasoningModel(t, byID["omitted-ladder"], nil, llm.EffortHigh, false)
	assertReasoningModel(t, byID["null-ladder"], all, llm.EffortNone, false)
	assertReasoningModel(t, byID["mandatory-null"], all[1:], llm.EffortHigh, true)
	assertReasoningModel(t, byID["empty-ladder"], []llm.Effort{}, "", false)
	assertReasoningModel(t, byID["unknown-only"], []llm.Effort{}, "", false)
	assertReasoningModel(t, byID["mandatory"], []llm.Effort{llm.EffortLow, llm.EffortHigh}, "", true)
	assertReasoningModel(t, byID["off-by-default"], []llm.Effort{llm.EffortNone, llm.EffortLow}, llm.EffortNone, false)
	assertReasoningModel(t, byID["contradictory-default"], []llm.Effort{llm.EffortLow, llm.EffortMedium}, "", false)
	assertReasoningModel(t, byID["unknown-default"], []llm.Effort{llm.EffortLow}, "", false)
	assertReasoningModel(t, byID["scalar-reasoning"], nil, "", false)
	assertReasoningModel(t, byID["wrong-shape"], nil, llm.EffortHigh, true)
	assertReasoningModel(t, byID["mixed-types"], nil, llm.EffortHigh, false)
	assertReasoningModel(t, byID["non-string-default"], []llm.Effort{llm.EffortHigh}, "", false)
	assertReasoningModel(t, byID["non-bool-mandatory"], []llm.Effort{llm.EffortNone, llm.EffortLow}, llm.EffortNone, false)

	for id, fragments := range map[string][]string{
		"unknown-only":     {`"turbo"`, `"future_policy"`, `"kept"`},
		"scalar-reasoning": {`"reasoning":"automatic"`},
		"wrong-shape":      {`"supported_efforts":{"low":true}`},
		"mixed-types":      {`"supported_efforts":["low",3]`},
	} {
		raw := byID[id].Raw.(json.RawMessage)
		for _, fragment := range fragments {
			if !bytes.Contains(raw, []byte(fragment)) {
				t.Fatalf("%s Raw missing %s: %s", id, fragment, raw)
			}
		}
	}
	// Explicit-null ladders must be copied rather than sharing the package's
	// canonical effort slice or another row's mutable slice.
	nullLadder := byID["null-ladder"]
	nullLadder.SupportedEfforts[0] = llm.EffortMax
	if byID["null-copy"].SupportedEfforts[0] != llm.EffortNone || knownEfforts[0] != llm.EffortNone {
		t.Fatalf("effort slices share caller-mutable storage")
	}
}

func TestOpenRouterModelDiscoveryHasNoHiddenFetchOrRequestGating(t *testing.T) {
	counts := map[string]int{}
	var chatBodies [][]byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			counts["models"]++
			mustWrite(t, w, `{"data":[{"id":"mandatory-model","reasoning":{"supported_efforts":["low","high"],"default_effort":"high","mandatory":true}}]}`)
		case "/chat/completions":
			counts["chat"]++
			body, _ := io.ReadAll(r.Body)
			chatBodies = append(chatBodies, body)
			if r.Header.Get("Accept") == "text/event-stream" {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, `data: {"id":"c1","model":"mandatory-model","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\n")
				_, _ = io.WriteString(w, "data: [DONE]\n\n")
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"c1","model":"mandatory-model","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]}`)
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
		}
	}))
	t.Cleanup(server.Close)
	p, err := New(WithAPIKey("test"), WithBaseURL(server.URL), WithHTTPClient(server.Client()), WithMaxRetries(0))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if len(counts) != 0 {
		t.Fatalf("New made requests: %+v", counts)
	}
	req := &llm.Request{Model: "mandatory-model", Effort: llm.EffortNone, Messages: []llm.Message{llm.UserText("hi")}}
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if counts["models"] != 0 {
		t.Fatalf("Chat fetched models: %+v", counts)
	}
	if _, err := llm.Collect(p.ChatStream(context.Background(), req)); err != nil {
		t.Fatalf("ChatStream returned error: %v", err)
	}
	if counts["models"] != 0 || counts["chat"] != 2 {
		t.Fatalf("chat paths requests = %+v", counts)
	}
	models, err := p.Models(context.Background())
	if err != nil || len(models) != 1 || !models[0].ReasoningRequired {
		t.Fatalf("Models = %+v, %v", models, err)
	}
	if counts["models"] != 1 {
		t.Fatalf("explicit Models requests = %d, want 1", counts["models"])
	}
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatalf("post-discovery EffortNone Chat returned error: %v", err)
	}
	if counts["models"] != 1 || counts["chat"] != 3 {
		t.Fatalf("post-discovery requests = %+v", counts)
	}
	for i, body := range chatBodies {
		if !strings.Contains(string(body), `"reasoning":{"enabled":false}`) {
			t.Fatalf("chat body %d changed EffortNone mapping: %s", i, body)
		}
	}
}

func assertReasoningModel(t *testing.T, model llm.ModelInfo, efforts []llm.Effort, defaultEffort llm.Effort, required bool) {
	t.Helper()
	if !reflect.DeepEqual(model.SupportedEfforts, efforts) || model.DefaultEffort != defaultEffort || model.ReasoningRequired != required {
		t.Fatalf("model %q reasoning = efforts %v default %q required %v, want %v %q %v", model.ID, model.SupportedEfforts, model.DefaultEffort, model.ReasoningRequired, efforts, defaultEffort, required)
	}
}
