package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
)

func TestAPIModelsImageAuthority(t *testing.T) {
	// Exact aliases and snapshots from the nine official pages linked in the
	// dated contract amendment; arbitrary suffixes must never inherit support.
	positive := []string{
		"gpt-4.1", "gpt-4.1-2025-04-14", "gpt-4.1-mini", "gpt-4.1-mini-2025-04-14",
		"gpt-4.1-nano", "gpt-4.1-nano-2025-04-14", "gpt-4o", "gpt-4o-2024-05-13",
		"gpt-4o-2024-08-06", "gpt-4o-2024-11-20", "gpt-4o-mini", "gpt-4o-mini-2024-07-18",
		"gpt-5", "gpt-5-2025-08-07", "gpt-5-mini", "gpt-5-mini-2025-08-07",
		"gpt-5-nano", "gpt-5-nano-2025-08-07", "gpt-5.1", "gpt-5.1-2025-11-13",
	}
	unknown := []string{
		"gpt-unknown", "gpt-5-custom", "gpt-5-2099-01-01", "gpt-5:variant", "ft:gpt-4.1:custom",
		"openai/gpt-5", "gpt-3.5-turbo", "claude-sonnet-4-5", "GPT-5",
	}
	ids := append(slices.Clone(positive), unknown...)
	rows := make([]map[string]any, len(ids))
	for i, id := range ids {
		rows[i] = map[string]any{"id": id, "object": "model", "created": 0, "owned_by": "openai"}
	}
	var requests atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		if r.URL.Path != "/models" || r.Method != http.MethodGet {
			t.Errorf("unexpected discovery request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": rows})
	}))
	defer server.Close()
	p, err := New(WithAPIKey("test-key"), WithBaseURL(server.URL), WithHTTPClient(server.Client()), WithMaxRetries(0))
	if err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 0 {
		t.Fatal("construction fetched a catalog")
	}
	models, err := p.Models(context.Background())
	if err != nil || len(models) != len(ids) {
		t.Fatalf("Models = %+v, %v", models, err)
	}
	for i, model := range models {
		if model.ID != ids[i] || slices.Contains(model.Capabilities, llm.CapabilityImageInput) != (i < len(positive)) {
			t.Errorf("catalog row %d = %+v", i, model)
		}
	}
	models[0].Capabilities[0] = llm.CapabilityReasoning
	again, err := p.Models(context.Background())
	if err != nil || !slices.Equal(again[0].Capabilities, []llm.Capability{llm.CapabilityImageInput}) {
		t.Fatalf("catalog capability mutation leaked: %+v, %v", again, err)
	}
	if requests.Load() != 2 {
		t.Fatalf("discovery made %d requests, want only the two explicit calls", requests.Load())
	}
}

func TestChatGPTImageAuthorityStaysInLiveCatalog(t *testing.T) {
	for _, tt := range []struct {
		row   string
		image bool
	}{
		{`{"slug":"gpt-5","visibility":"list"}`, false},
		{`{"slug":"gpt-5","visibility":"list","input_modalities":["text"]}`, false},
		{`{"slug":"gpt-unknown","visibility":"list","input_modalities":["text","image"]}`, true},
	} {
		model, ok := chatGPTModelInfo(json.RawMessage(tt.row))
		if !ok || slices.Contains(model.Capabilities, llm.CapabilityImageInput) != tt.image {
			t.Errorf("ChatGPT row %s = %+v, %v", tt.row, model, ok)
		}
	}
}

func TestOAuthDataRowsDoNotInheritAPIImageEvidence(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": "gpt-5", "object": "model", "owned_by": "openai"}}})
	}))
	defer server.Close()
	store := authstore.NewMemory(llm.AuthFile{"openai": {
		Type: "oauth", Access: "synthetic-access", Refresh: "synthetic-refresh", ClientID: "synthetic-client",
		Expires: time.Now().Add(time.Hour).UnixMilli(),
	}})
	p, err := New(WithCredentials(store), WithBaseURL(server.URL), WithHTTPClient(server.Client()), WithMaxRetries(0))
	if err != nil {
		t.Fatal(err)
	}
	models, err := p.Models(context.Background())
	if err != nil || len(models) != 1 || len(models[0].Capabilities) != 0 {
		t.Fatalf("OAuth row inherited API image evidence: %+v, %v", models, err)
	}
}
