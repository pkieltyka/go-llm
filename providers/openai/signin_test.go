package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

const signInCompletedSSE = "event: response.output_text.delta\n" +
	`data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"pong","logprobs":[]}` + "\n\n" +
	"event: response.completed\n" +
	`data: {"type":"response.completed","sequence_number":2,"response":{"id":"resp_1","model":"gpt-test","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"pong","annotations":[]}]}],"usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2}}}` + "\n\n"

// signInServer mimics api.openai.com under a Sign in with ChatGPT token:
// streaming only, with FastAPI-style {"detail": ...} rejections.
type signInServer struct {
	mu     sync.Mutex
	bodies []map[string]any
	after  func(w http.ResponseWriter, r *http.Request) // runs after response.completed
	server *httptest.Server
}

func newSignInServer(t *testing.T) *signInServer {
	s := &signInServer{}
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			_, _ = io.WriteString(w, `{"models":[`+
				`{"slug":"gpt-sub","display_name":"GPT Sub","visibility":"list","context_window":272000,"input_modalities":["text","image"],"supports_parallel_tool_calls":true,"default_reasoning_level":"medium","supported_reasoning_levels":[{"effort":"low"},{"effort":"medium"},{"effort":"ultra"}]},`+
				`{"slug":"internal-review","visibility":"hide"}]}`)
			return
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		s.mu.Lock()
		s.bodies = append(s.bodies, body)
		s.mu.Unlock()
		if body["stream"] != true {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"detail":"Stream must be set to true"}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, signInCompletedSSE)
		w.(http.Flusher).Flush()
		if s.after != nil {
			s.after(w, r)
		}
	}))
	t.Cleanup(s.server.Close)
	return s
}

func (s *signInServer) provider(t *testing.T, opts ...Option) *Provider {
	t.Helper()
	store := authstore.NewMemory(llm.AuthFile{"openai": {Type: "oauth", Access: "chatgpt-token", Refresh: "r", ClientID: "c", Expires: time.Now().Add(time.Hour).UnixMilli()}})
	p, err := New(append([]Option{WithCredentials(store), WithBaseURL(s.server.URL + "/v1"), WithMaxRetries(0)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestSignInWithChatGPTChatStreamsAndOmitsRejectedFields(t *testing.T) {
	s := newSignInServer(t)
	resp, err := s.provider(t).Chat(context.Background(), &llm.Request{
		Model:       "gpt-sub",
		MaxTokens:   64,
		Temperature: llm.Ptr(0.2),
		TopP:        llm.Ptr(0.9),
		Messages:    []llm.Message{llm.UserText("ping")},
	})
	if err != nil || resp.Text() != "pong" {
		t.Fatalf("Chat = %+v, %v", resp, err)
	}
	body := s.bodies[0]
	for _, field := range []string{"temperature", "top_p", "max_output_tokens", "prompt_cache_retention", "prompt_cache_options"} {
		if _, ok := body[field]; ok {
			t.Errorf("request carried %s, which Sign in with ChatGPT rejects", field)
		}
	}
}

func TestAPIKeyChatKeepsBlockingRequestAndFields(t *testing.T) {
	s := newSignInServer(t)
	p, err := New(WithAPIKey("sk-test"), WithBaseURL(s.server.URL+"/v1"), WithMaxRetries(0))
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Chat(context.Background(), &llm.Request{Model: "gpt-sub", MaxTokens: 64, Temperature: llm.Ptr(0.2), Messages: []llm.Message{llm.UserText("ping")}})
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Message != "Stream must be set to true" || !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("detail error = %v, want the mapped detail message", err)
	}
	if s.bodies[0]["temperature"] != 0.2 || s.bodies[0]["max_output_tokens"] != float64(64) {
		t.Fatalf("API-key request dropped fields: %v", s.bodies[0])
	}
}

func TestSignInWithChatGPTModelsParsesCatalog(t *testing.T) {
	models, err := newSignInServer(t).provider(t).Models(context.Background())
	if err != nil || len(models) != 1 {
		t.Fatalf("Models = %+v, %v, want only the listed row", models, err)
	}
	m := models[0]
	if m.ID != "gpt-sub" || m.DisplayName != "GPT Sub" || m.ContextWindow != 272000 || m.DefaultEffort != llm.EffortMedium {
		t.Fatalf("model = %+v", m)
	}
	if len(m.SupportedEfforts) != 2 || m.SupportedEfforts[0] != llm.EffortLow || m.SupportedEfforts[1] != llm.EffortMedium {
		t.Fatalf("efforts = %v, want unknown levels dropped", m.SupportedEfforts)
	}
	want := []llm.Capability{llm.CapabilityReasoning, llm.CapabilityParallelTools, llm.CapabilityImageInput}
	if len(m.Capabilities) != len(want) {
		t.Fatalf("capabilities = %v, want %v", m.Capabilities, want)
	}
	for i := range want {
		if m.Capabilities[i] != want[i] {
			t.Fatalf("capabilities = %v, want %v", m.Capabilities, want)
		}
	}
}

func TestStreamReadsToEOFAfterCompletion(t *testing.T) {
	s := newSignInServer(t)
	s.after = func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(20 * time.Millisecond)
		_, _ = io.WriteString(w, ": trailing keep-alive\n\n")
	}
	var mu sync.Mutex
	var captures []llm.WireCapture
	p := s.provider(t, WithWireCapture(func(c llm.WireCapture) {
		mu.Lock()
		captures = append(captures, c)
		mu.Unlock()
	}))
	if _, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{Model: "gpt-sub", Messages: []llm.Message{llm.UserText("ping")}})); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(captures) != 1 || captures[0].ResponseIncomplete {
		t.Fatalf("captures = %d, incomplete = %v; the body must be read to EOF", len(captures), len(captures) == 1 && captures[0].ResponseIncomplete)
	}
}

func TestStreamDrainIsBoundedWhenServerHoldsOpen(t *testing.T) {
	old := providerutil.StreamDrainGrace
	providerutil.StreamDrainGrace = 30 * time.Millisecond
	t.Cleanup(func() { providerutil.StreamDrainGrace = old })
	s := newSignInServer(t)
	s.after = func(_ http.ResponseWriter, r *http.Request) { <-r.Context().Done() }
	start := time.Now()
	resp, err := llm.Collect(s.provider(t).ChatStream(context.Background(), &llm.Request{Model: "gpt-sub", Messages: []llm.Message{llm.UserText("ping")}}))
	if err != nil || !strings.Contains(resp.Text(), "pong") {
		t.Fatalf("held-open stream = %+v, %v; the completed response must still be returned", resp, err)
	}
	if elapsed := time.Since(start); elapsed > 500*time.Millisecond {
		t.Fatalf("drain took %v, want it bounded by the grace", elapsed)
	}
}

// countingStore counts reads so tests can pin one store read per request.
type countingStore struct {
	llm.CredentialStore
	mu    sync.Mutex
	reads int
}

func (c *countingStore) Read(ctx context.Context, providerID string) (llm.AuthCredential, bool, error) {
	c.mu.Lock()
	c.reads++
	c.mu.Unlock()
	return c.CredentialStore.Read(ctx, providerID)
}

func TestStoredAPIKeyChatIsBlockingAndReadsTheStoreOnce(t *testing.T) {
	s := newSignInServer(t)
	store := &countingStore{CredentialStore: authstore.NewMemory(llm.AuthFile{"openai": {Type: "api_key", Key: "sk-stored"}})}
	p, err := New(WithCredentials(store), WithBaseURL(s.server.URL+"/v1"), WithMaxRetries(0))
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Chat(context.Background(), &llm.Request{Model: "gpt-sub", MaxTokens: 64, Temperature: llm.Ptr(0.2), Messages: []llm.Message{llm.UserText("ping")}})
	if !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("stored API key Chat = %v, want the blocking request (rejected by this fake)", err)
	}
	if s.bodies[0]["stream"] == true || s.bodies[0]["temperature"] != 0.2 {
		t.Fatalf("stored API key request = %v, want blocking with fields kept", s.bodies[0])
	}
	if store.reads != 1 {
		t.Fatalf("store reads = %d, want 1 per request", store.reads)
	}
}

func TestSignInWithChatGPTChatReadsTheStoreOnce(t *testing.T) {
	s := newSignInServer(t)
	store := &countingStore{CredentialStore: authstore.NewMemory(llm.AuthFile{"openai": {Type: "oauth", Access: "chatgpt-token", Refresh: "r", ClientID: "c", Expires: time.Now().Add(time.Hour).UnixMilli()}})}
	p, err := New(WithCredentials(store), WithBaseURL(s.server.URL+"/v1"), WithMaxRetries(0))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Chat(context.Background(), &llm.Request{Model: "gpt-sub", Messages: []llm.Message{llm.UserText("ping")}}); err != nil {
		t.Fatal(err)
	}
	if store.reads != 1 {
		t.Fatalf("store reads = %d, want 1", store.reads)
	}
}
