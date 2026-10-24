package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func threeQuestions() *llm.ClassifyRequest {
	return &llm.ClassifyRequest{
		Model: "typesafe/jev-1.14",
		State: json.RawMessage(`{"message":"The change works perfectly, thanks."}`),
		Questions: []llm.Question{
			llm.ChoiceQuestion("category", "Classify the message.",
				llm.Choice{Value: "approval", Description: "The user approves of the result"},
				llm.Choice{Value: "correction", Description: "The user requests a correction"}),
			llm.ScoreQuestion("satisfaction", "Score user satisfaction.", "dissatisfied", "neutral", "satisfied"),
			llm.BoolQuestion("approved", "Does the user approve?", "Approval", "No approval"),
		},
	}
}

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return data
}

type captured struct {
	path    string
	headers http.Header
	body    string
}

// serve answers every request with status and body and records the last
// request.
func serve(t *testing.T, status int, body []byte) (*httptest.Server, *captured) {
	t.Helper()
	got := &captured{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		got.path, got.headers, got.body = r.URL.Path, r.Header.Clone(), string(data)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write(body)
	}))
	t.Cleanup(server.Close)
	return server, got
}

func engine(t *testing.T, cfg Config) *Engine {
	t.Helper()
	if cfg.Provider == "" {
		cfg.Provider = "openrouter"
	}
	e, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestClassifyGoldenRequestAndLiveResponses(t *testing.T) {
	for _, tc := range []struct {
		fixture, model string
		choice         string
		score, bool_   float64
		levels         []float64
		cost           float64
	}{
		{"openrouter_jev.json", "typesafe/jev-1.14-20261014", "approval", 2, 0.97, []float64{0, 0, 1}, 0.000016842},
		{"openrouter_clef_flash.json", "cloudflare/clef-flash", "approval", 1.8352, 0.947, []float64{0.0318, 0.1012, 0.867}, 0.000006111},
		{"openrouter_luna.json", "openai/gpt-6-luna-decisions-20261006", "approval", 2, 1, []float64{0, 0, 1}, 0.0000393},
	} {
		t.Run(tc.fixture, func(t *testing.T) {
			server, got := serve(t, http.StatusOK, fixture(t, tc.fixture))
			e := engine(t, Config{BaseURL: server.URL + "/api/v1", APIKey: func(context.Context) (string, error) { return "test-key", nil }})
			result, err := e.Classify(context.Background(), threeQuestions())
			if err != nil {
				t.Fatal(err)
			}
			want := `{"model":"typesafe/jev-1.14","questions":{"category":{"type":"choice","instructions":"Classify the message.","criteria":{"approval":"The user approves of the result","correction":"The user requests a correction"}},"satisfaction":{"type":"score","instructions":"Score user satisfaction.","criteria":["dissatisfied","neutral","satisfied"]},"approved":{"type":"noul","instructions":"Does the user approve?","criteria":{"true":"Approval","false":"No approval"}}},"state":{"message":"The change works perfectly, thanks."}}`
			if got.body != want {
				t.Fatalf("request body\n got %s\nwant %s", got.body, want)
			}
			if got.path != "/api/v1/systemone" || got.headers.Get("Authorization") != "Bearer test-key" || got.headers.Get(imageOptInHeader) != "" {
				t.Fatalf("path %q headers %v", got.path, got.headers)
			}
			if result.Model != tc.model || !strings.HasPrefix(result.ID, "gen-dec-") || result.Provider != "openrouter" {
				t.Fatalf("result identity = %+v", result)
			}
			if a := result.Answers["category"]; a.Type != llm.QuestionChoice || a.Choice != tc.choice || len(a.Probabilities) != 2 {
				t.Fatalf("choice = %+v", a)
			}
			if a := result.Answers["satisfaction"]; a.Type != llm.QuestionScore || a.Score != tc.score || len(a.LevelProbabilities) != 3 || a.LevelProbabilities[2] != tc.levels[2] {
				t.Fatalf("score = %+v", a)
			}
			if a := result.Answers["approved"]; a.Type != llm.QuestionBool || a.Probability != tc.bool_ {
				t.Fatalf("bool = %+v", a)
			}
			if result.Usage.CostUSD == nil || *result.Usage.CostUSD != tc.cost || result.Usage.CostSource != llm.CostSourceNative || result.Usage.TotalTokens != result.Usage.InputTokens+result.Usage.OutputTokens {
				t.Fatalf("usage = %+v", result.Usage)
			}
		})
	}
}

func TestClassifyHTTPErrors(t *testing.T) {
	for _, tc := range []struct {
		fixture string
		status  int
		kind    error
		message string
	}{
		{"openrouter_bool_rejected.json", 400, llm.ErrBadRequest, "Invalid discriminator value"},
		{"openrouter_image_dimensions.json", 400, llm.ErrBadRequest, "Invalid image dimensions"},
		{"openrouter_upstream_502.json", 502, llm.ErrServer, "502"},
	} {
		t.Run(tc.fixture, func(t *testing.T) {
			server, _ := serve(t, tc.status, fixture(t, tc.fixture))
			_, err := engine(t, Config{BaseURL: server.URL}).Classify(context.Background(), threeQuestions())
			var providerErr *llm.ProviderError
			if !errors.Is(err, tc.kind) || !errors.As(err, &providerErr) || providerErr.HTTPStatus != tc.status || !strings.Contains(providerErr.Message, tc.message) {
				t.Fatalf("err = %v", err)
			}
		})
	}
	server, _ := serve(t, 401, []byte(`{"detail":"Invalid API key"}`))
	_, err := engine(t, Config{BaseURL: server.URL}).Classify(context.Background(), threeQuestions())
	if !errors.Is(err, llm.ErrAuth) || !strings.Contains(err.Error(), "Invalid API key") {
		t.Fatalf("FastAPI detail error = %v", err)
	}
}

// TestClassifyMalformedAnswersKeepUsage: a billed response with a missing
// or mistyped answer returns the result with its usage and ErrServer.
func TestClassifyMalformedAnswersKeepUsage(t *testing.T) {
	for name, body := range map[string]string{
		"missing":    `{"answers":{"category":{"type":"choice","choice":"approval","probabilities":{"approval":1,"correction":0},"confidence":1}},"usage":{"input_tokens":10,"output_tokens":2,"cost":0.001}}`,
		"wrong type": `{"answers":{"category":{"type":"noul","noul":1},"satisfaction":{"type":"score","score":1,"confidence":1},"approved":{"type":"noul","noul":1}},"usage":{"input_tokens":10,"output_tokens":2,"cost":0.001}}`,
		"no number":  `{"answers":{"category":{"type":"choice","choice":"approval","confidence":1},"satisfaction":{"type":"score","confidence":1},"approved":{"type":"noul","noul":1}},"usage":{"input_tokens":10,"output_tokens":2,"cost":0.001}}`,
	} {
		t.Run(name, func(t *testing.T) {
			server, _ := serve(t, http.StatusOK, []byte(body))
			result, err := engine(t, Config{BaseURL: server.URL}).Classify(context.Background(), threeQuestions())
			if !errors.Is(err, llm.ErrServer) || result == nil || result.Usage.InputTokens != 10 || result.Usage.CostUSD == nil {
				t.Fatalf("result %+v err %v", result, err)
			}
		})
	}
	server, _ := serve(t, http.StatusOK, []byte(`not json`))
	if _, err := engine(t, Config{BaseURL: server.URL}).Classify(context.Background(), threeQuestions()); !errors.Is(err, llm.ErrServer) {
		t.Fatalf("unreadable body = %v", err)
	}
}

func TestClassifyEstimatesCostWithoutNativeCost(t *testing.T) {
	server, _ := serve(t, http.StatusOK, []byte(`{"answers":{"approved":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1000000,"output_tokens":0}}`))
	table := llm.PriceTable{"typesafe/jev-latest": {InputPerMTok: 0.042}}
	req := &llm.ClassifyRequest{Model: "jev-latest", State: json.RawMessage(`{}`), Questions: []llm.Question{llm.BoolQuestion("approved", "?", "", "")}}
	result, err := engine(t, Config{Provider: "typesafe", BaseURL: server.URL, PriceTable: table}).Classify(context.Background(), req)
	if err != nil || result.Usage.CostUSD == nil || *result.Usage.CostUSD != 0.042 || result.Usage.CostSource != llm.CostSourceEstimated {
		t.Fatalf("usage = %+v, %v", result.Usage, err)
	}
	if result.Model != "jev-latest" {
		t.Fatalf("model falls back to the requested one: %q", result.Model)
	}
}

func TestClassifyImageForms(t *testing.T) {
	png := []byte{0x89, 'P', 'N', 'G'}
	image := llm.ImagePart{Data: png, MediaType: "image/png"}
	answer := []byte(`{"answers":{"approved":{"type":"noul","noul":0.5}}}`)
	req := func(images ...llm.ImagePart) *llm.ClassifyRequest {
		return &llm.ClassifyRequest{Model: "m", State: json.RawMessage(`{"a":1}`), Images: images,
			Questions: []llm.Question{llm.BoolQuestion("approved", "?", "", "")}}
	}
	for _, tc := range []struct {
		name   string
		cfg    Config
		want   []string
		optIn  bool
		reject []llm.ImagePart
	}{
		{"assets with opt-in", Config{ImageOptIn: true},
			[]string{`"assets":{"<image-1>":{"data_base64":"iVBORw==","format":"png","type":"image"}}`, `"state":["<image-1>",{"a":1}]`}, true,
			[]llm.ImagePart{{Data: png, MediaType: "image/webp"}}},
		{"data URLs", Config{ImageForm: func(string) ImageForm { return ImageDataURLs }},
			[]string{`"images":["data:image/png;base64,iVBORw=="]`, `"state":{"a":1}`}, false,
			[]llm.ImagePart{{Data: png, MediaType: "image/gif"}}},
		{"state URLs", Config{ImageForm: func(string) ImageForm { return ImageStateURLs }, ImageOptIn: true},
			[]string{`"state":[{"image_url":{"url":"data:image/png;base64,iVBORw=="},"type":"image_url"},{"a":1}]`}, false, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server, got := serve(t, http.StatusOK, answer)
			tc.cfg.BaseURL = server.URL
			e := engine(t, tc.cfg)
			if _, err := e.Classify(context.Background(), req(image)); err != nil {
				t.Fatal(err)
			}
			for _, want := range tc.want {
				if !strings.Contains(got.body, want) {
					t.Fatalf("body %s\nmissing %s", got.body, want)
				}
			}
			if (got.headers.Get(imageOptInHeader) == "true") != tc.optIn {
				t.Fatalf("opt-in header = %q", got.headers.Get(imageOptInHeader))
			}
			if tc.reject != nil {
				if _, err := e.Classify(context.Background(), req(tc.reject...)); !errors.Is(err, llm.ErrBadRequest) {
					t.Fatalf("rejected images = %v", err)
				}
			}
		})
	}
	server, _ := serve(t, http.StatusOK, answer)
	if _, err := engine(t, Config{BaseURL: server.URL, TextOnly: true}).Classify(context.Background(), req(image)); !errors.Is(err, llm.ErrUnsupported) {
		t.Fatalf("text-only service = %v", err)
	}
}

func TestClassifyValidatesBeforeIO(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls++ }))
	defer server.Close()
	req := threeQuestions()
	req.Questions = nil
	if _, err := engine(t, Config{BaseURL: server.URL}).Classify(context.Background(), req); !errors.Is(err, llm.ErrBadRequest) || calls != 0 {
		t.Fatalf("err %v calls %d", err, calls)
	}
}

func TestClassifyEnvelopeAndKeyless(t *testing.T) {
	server, got := serve(t, http.StatusOK, []byte(`{"success":true,"result":{"answers":{"approved":{"type":"noul","noul":0.25}}}}`))
	e := engine(t, Config{
		Provider: "wrapped",
		Endpoint: func(context.Context) (string, string, error) { return server.URL + "/ai/run", "", nil },
		Payload: func(model string, request map[string]any) any {
			return map[string]any{"model": model, "input": request}
		},
		Output: func(_ int, body []byte) (json.RawMessage, error) {
			var envelope struct {
				Result json.RawMessage `json:"result"`
			}
			if err := json.Unmarshal(body, &envelope); err != nil {
				return nil, err
			}
			return envelope.Result, nil
		},
		Headers: http.Header{"X-Extra": {"1"}},
	})
	req := &llm.ClassifyRequest{Model: "@cf/cloudflare/clef", State: json.RawMessage(`{}`), Questions: []llm.Question{llm.BoolQuestion("approved", "?", "", "")}}
	result, err := e.Classify(context.Background(), req)
	if err != nil || result.Answers["approved"].Probability != 0.25 {
		t.Fatalf("result %+v err %v", result, err)
	}
	if got.path != "/ai/run" || !strings.HasPrefix(got.body, `{"input":{"questions"`) || got.headers.Get("Authorization") != "" || got.headers.Get("X-Extra") != "1" {
		t.Fatalf("path %q body %s headers %v", got.path, got.body, got.headers)
	}
}

func TestNewRejectsIncompleteConfig(t *testing.T) {
	for _, cfg := range []Config{{BaseURL: "http://x"}, {Provider: "p"}, {Provider: "p", BaseURL: "http://x", MaxRetries: -1}} {
		if _, err := New(cfg); !errors.Is(err, llm.ErrBadRequest) {
			t.Fatalf("New(%+v) = %v", cfg, err)
		}
	}
}

func TestClassifyServiceErrorShapes(t *testing.T) {
	for _, tc := range []struct {
		body   string
		status int
		kind   error
		code   string
		msg    string
	}{
		{`{"detail":{"error_type":"authentication_error","message":"Must supply an API key! Check your request and try again."}}`, 403, llm.ErrAuth, "authentication_error", "Must supply an API key"},
		{`{"message":"questions: At least one question is required","error_type":"invalid_request"}`, 400, llm.ErrBadRequest, "invalid_request", "At least one question"},
		{`{"error":{"type":"server_error","message":"Upstream request failed: Model is unavailable."}}`, 400, llm.ErrBadRequest, "server_error", "Model is unavailable"},
	} {
		server, _ := serve(t, tc.status, []byte(tc.body))
		_, err := engine(t, Config{BaseURL: server.URL}).Classify(context.Background(), threeQuestions())
		var providerErr *llm.ProviderError
		if !errors.Is(err, tc.kind) || !errors.As(err, &providerErr) || providerErr.Code != tc.code || !strings.Contains(providerErr.Message, tc.msg) {
			t.Fatalf("%s: err = %v (code %q)", tc.body, err, providerErr.Code)
		}
	}
}

func TestCloudflareErrorShapeAndURLError(t *testing.T) {
	server, _ := serve(t, 400, []byte(`{"success":false,"errors":[{"code":5006,"message":"Invalid input"},{"code":5007,"message":"questions required"}]}`))
	_, err := engine(t, Config{Endpoint: func(context.Context) (string, string, error) { return server.URL + "/run", "k", nil }}).Classify(context.Background(), threeQuestions())
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Code != "5006" || providerErr.Message != "Invalid input; questions required" {
		t.Fatalf("err = %v", err)
	}
	_, err = engine(t, Config{Endpoint: func(context.Context) (string, string, error) { return "", "", llm.ErrAuth }}).Classify(context.Background(), threeQuestions())
	if !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("URL error = %v", err)
	}
}

// TestAssetNamesAvoidCallerText: a literal "<image-" in the state or the
// questions moves the asset names to another deterministic prefix, so
// TypeSafe's substitution never rewrites caller text.
func TestAssetNamesAvoidCallerText(t *testing.T) {
	image := llm.ImagePart{Data: []byte{1}, MediaType: "image/png"}
	for _, tc := range []struct {
		name, state, instructions, want string
	}{
		{"plain", `{"a":"text"}`, "?", `"<image-1>"`},
		{"state", `{"log":"see <image-1> above"}`, "?", `"<image1-1>"`},
		{"escaped state", `{"log":"\u003cimage-2>"}`, "?", `"<image1-1>"`},
		{"instructions", `{"a":1}`, "Compare <image-1> with <image1-1>", `"<image2-1>"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server, got := serve(t, http.StatusOK, []byte(`{"answers":{"approved":{"type":"noul","noul":0.5}}}`))
			req := &llm.ClassifyRequest{Model: "m", State: json.RawMessage(tc.state), Images: []llm.ImagePart{image},
				Questions: []llm.Question{llm.BoolQuestion("approved", tc.instructions, "", "")}}
			if _, err := engine(t, Config{BaseURL: server.URL}).Classify(context.Background(), req); err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(got.body, `"assets":{`+tc.want) || !strings.Contains(got.body, `"state":[`+tc.want) {
				t.Fatalf("body %s lacks %s", got.body, tc.want)
			}
		})
	}
}

func TestServiceErrorCodesAndPreflight(t *testing.T) {
	server, _ := serve(t, http.StatusBadRequest, []byte(`{"error":{"message":"state too long","code":"max_tokens_exceeded"}}`))
	if _, err := engine(t, Config{BaseURL: server.URL}).Classify(context.Background(), threeQuestions()); !errors.Is(err, llm.ErrContextTooLong) {
		t.Fatalf("max_tokens_exceeded = %v", err)
	}
	hooked, _ := serve(t, http.StatusBadRequest, []byte(`{"errors":[{"code":42,"message":"custom"}]}`))
	hook := func(_ int, code, _ string) error {
		if code == "42" {
			return llm.ErrNotFound
		}
		return nil
	}
	if _, err := engine(t, Config{BaseURL: hooked.URL, ErrorKind: hook}).Classify(context.Background(), threeQuestions()); !errors.Is(err, llm.ErrNotFound) {
		t.Fatalf("service hook = %v", err)
	}
	calls := 0
	counted := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls++ }))
	defer counted.Close()
	limit := func(*llm.ClassifyRequest) error { return fmt.Errorf("%w: too many", llm.ErrBadRequest) }
	if _, err := engine(t, Config{BaseURL: counted.URL, Preflight: limit}).Classify(context.Background(), threeQuestions()); !errors.Is(err, llm.ErrBadRequest) || calls != 0 {
		t.Fatalf("preflight = %v (calls %d)", err, calls)
	}
}
