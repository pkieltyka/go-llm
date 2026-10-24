package openrouter

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
	"github.com/pkieltyka/go-llm/llmtest"
)

func classifyRequest() *llm.ClassifyRequest {
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

// TestClassifyOverSystemOne: OpenRouter classifies beside chat, on the same
// key and base URL, with its native cost (live fixture from Phase 1).
func TestClassifyOverSystemOne(t *testing.T) {
	live, err := os.ReadFile("../internal/systemone/testdata/openrouter_jev.json")
	if err != nil {
		t.Fatal(err)
	}
	var path, auth, referer, body string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		path, auth, referer, body = r.URL.Path, r.Header.Get("Authorization"), r.Header.Get("HTTP-Referer"), string(data)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(live)
	}))
	defer server.Close()
	p, err := New(WithAPIKey("test-key"), WithBaseURL(server.URL+"/api/v1"), WithHTTPClient(server.Client()), WithAttribution("https://example.test", "Example"))
	if err != nil {
		t.Fatal(err)
	}
	jev := llm.NewClassifierModel(p, "typesafe/jev-1.14")
	req := classifyRequest()
	result, err := jev.ClassifyRequest(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if path != "/api/v1/systemone" || auth != "Bearer test-key" || referer != "https://example.test" || !strings.Contains(body, `"type":"noul"`) {
		t.Fatalf("path %q auth %q referer %q body %s", path, auth, referer, body)
	}
	if result.Provider != "openrouter" || result.Model != "typesafe/jev-1.14-20261014" || result.Answers["approved"].Probability != 0.97 {
		t.Fatalf("result = %+v", result)
	}
	if result.Usage.CostSource != llm.CostSourceNative || result.Usage.CostUSD == nil {
		t.Fatalf("usage = %+v", result.Usage)
	}
	if !containsCapability(p.Capabilities(), llm.CapabilityClassify) {
		t.Fatal("OpenRouter does not advertise CapabilityClassify")
	}

	req.Images = []llm.ImagePart{{Data: []byte{1, 2}, MediaType: "image/png"}}
	if _, err := p.Classify(context.Background(), req); err != nil || !strings.Contains(body, `"state":[{"image_url":{"url":"data:image/png;base64,AQI="},"type":"image_url"}`) {
		t.Fatalf("image body %s err %v", body, err)
	}
}

func TestClassifyCredentialsAndMissingKey(t *testing.T) {
	t.Setenv(apiKeyEnv, "")
	var auth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		_, _ = io.WriteString(w, `{"answers":{"approved":{"type":"noul","noul":0.5}}}`)
	}))
	defer server.Close()
	req := &llm.ClassifyRequest{Model: "m", State: json.RawMessage(`{}`), Questions: []llm.Question{llm.BoolQuestion("approved", "?", "", "")}}
	store := authstore.NewMemory(llm.AuthFile{"openrouter": {Type: "api_key", Key: "stored-key"}})
	p, err := New(WithCredentials(store), WithBaseURL(server.URL), WithHTTPClient(server.Client()))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Classify(context.Background(), req); err != nil || auth != "Bearer stored-key" {
		t.Fatalf("stored key: auth %q err %v", auth, err)
	}
	// No key at all fails at construction, for chat and classify alike.
	if _, err := New(WithAPIKey(""), WithBaseURL(server.URL)); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("empty key = %v", err)
	}
}

func TestModelsListsClassifiers(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Query().Get("output_modalities") == "decisions" {
			_, _ = io.WriteString(w, `{"data":[{"id":"typesafe/jev-1.14","name":"TypeSafe: Jev 1.14","context_length":262144,"architecture":{"input_modalities":["text","image"],"output_modalities":["decisions"]},"pricing":{"prompt":"0.000000042","completion":"0"}},{"id":"openai/gpt-chat","architecture":{"output_modalities":["text"]}}]}`)
			return
		}
		_, _ = io.WriteString(w, `{"data":[{"id":"openai/gpt-chat","architecture":{"input_modalities":["text"],"output_modalities":["text"]},"supported_parameters":["tools"]}]}`)
	}))
	defer server.Close()
	p, err := New(WithAPIKey("k"), WithBaseURL(server.URL), WithHTTPClient(server.Client()))
	if err != nil {
		t.Fatal(err)
	}
	models, err := p.Models(context.Background())
	if err != nil || len(models) != 2 {
		t.Fatalf("models = %+v, %v", models, err)
	}
	if containsCapability(models[0].Capabilities, llm.CapabilityClassify) {
		t.Fatalf("chat model marked classify: %+v", models[0])
	}
	jev := models[1]
	if jev.ID != "typesafe/jev-1.14" || !containsCapability(jev.Capabilities, llm.CapabilityClassify) || !containsCapability(jev.Capabilities, llm.CapabilityImageInput) {
		t.Fatalf("classifier row = %+v", jev)
	}
	if jev.Pricing == nil || jev.Pricing.InputPerMTok < 0.0419 || jev.Pricing.InputPerMTok > 0.0421 || jev.ContextWindow != 262144 {
		t.Fatalf("classifier pricing = %+v window %d", jev.Pricing, jev.ContextWindow)
	}
}

// TestModelsToleratesFailedDecisionsListing: a host that rejects the
// decisions filter (a proxy or mirror) still lists its chat catalog.
func TestModelsToleratesFailedDecisionsListing(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Query().Get("output_modalities") == "decisions" {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"error":{"message":"unknown parameter"}}`)
			return
		}
		_, _ = io.WriteString(w, `{"data":[{"id":"openai/gpt-chat","architecture":{"input_modalities":["text"],"output_modalities":["text"]}}]}`)
	}))
	defer server.Close()
	p, err := New(WithAPIKey("k"), WithBaseURL(server.URL), WithHTTPClient(server.Client()))
	if err != nil {
		t.Fatal(err)
	}
	if models, err := p.Models(context.Background()); err != nil || len(models) != 1 || models[0].ID != "openai/gpt-chat" {
		t.Fatalf("models = %+v, %v", models, err)
	}
}

func containsCapability(caps []llm.Capability, want llm.Capability) bool {
	for _, c := range caps {
		if c == want {
			return true
		}
	}
	return false
}

func TestClassifierConformance(t *testing.T) {
	llmtest.RunClassifierConformance(t, func(t *testing.T) llm.Provider {
		p, err := New(WithAPIKey("k"), WithBaseURL("http://127.0.0.1:1"), WithMaxRetries(0))
		if err != nil {
			t.Fatal(err)
		}
		return p
	})
}
