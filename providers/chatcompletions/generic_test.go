package chatcompletions_test

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
)

func newGenericServer(t *testing.T) (*httptest.Server, *[]http.Header) {
	t.Helper()
	var headers []http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers = append(headers, r.Header.Clone())
		if r.Header.Get("Accept") == "text/event-stream" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"po"},"finish_reason":null}]}`+"\n\n")
			_, _ = io.WriteString(w, `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"ng"},"finish_reason":"stop"}]}`+"\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		if r.URL.Path == "/models" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"object":"list","data":[{"id":"m","owned_by":"me"}]}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"pong"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(server.Close)
	return server, &headers
}

// TestGenericNew covers the public key-optional constructor end to end
// against a plain OpenAI-compatible fixture server: default name, keyless
// requests without Authorization, chat/stream/models, and raw extras.
func TestGenericNew(t *testing.T) {
	server, headers := newGenericServer(t)
	p, err := chatcompletions.New(server.URL,
		chatcompletions.WithHTTPClient(server.Client()),
		chatcompletions.WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if p.Name() != chatcompletions.DefaultProviderName {
		t.Fatalf("Name = %q, want %q", p.Name(), chatcompletions.DefaultProviderName)
	}

	ctx := context.Background()
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}
	resp, err := p.Chat(ctx, req)
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Text() != "pong" || resp.Provider != chatcompletions.DefaultProviderName {
		t.Fatalf("chat response = %+v", resp)
	}
	if raw, ok := resp.Raw.(chatcompletions.JSONObject); !ok || raw["id"] != "c1" {
		t.Fatalf("raw extras = %#v", resp.Raw)
	}

	streamed, err := llm.Collect(p.ChatStream(ctx, req))
	if err != nil {
		t.Fatalf("stream returned error: %v", err)
	}
	if streamed.Text() != "pong" {
		t.Fatalf("streamed text = %q", streamed.Text())
	}

	models, err := p.Models(ctx)
	if err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	if len(models) != 1 || models[0].ID != "m" {
		t.Fatalf("models = %+v", models)
	}

	for i, header := range *headers {
		if _, ok := header["Authorization"]; ok {
			t.Fatalf("keyless request %d sent Authorization %q", i, header.Get("Authorization"))
		}
	}
}

func TestGenericNewKeepsHarmlessAmbientHeadersWithoutAmbientAuth(t *testing.T) {
	t.Setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer ambient-secret\nX-Ambient-Safe: retained")
	server, headers := newGenericServer(t)
	p, err := chatcompletions.New(server.URL,
		chatcompletions.WithHTTPClient(server.Client()),
		chatcompletions.WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}
	if _, err := p.Chat(context.Background(), req); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if _, err := llm.Collect(p.ChatStream(context.Background(), req)); err != nil {
		t.Fatalf("ChatStream returned error: %v", err)
	}
	if _, err := p.Models(context.Background()); err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	for i, header := range *headers {
		if got := header.Get("Authorization"); got != "" {
			t.Fatalf("request %d Authorization = %q, want absent", i, got)
		}
		if got := header.Get("X-Ambient-Safe"); got != "retained" {
			t.Fatalf("request %d X-Ambient-Safe = %q", i, got)
		}
	}
}

func TestGenericNewOptions(t *testing.T) {
	server, headers := newGenericServer(t)
	var captures []llm.WireCapture
	p, err := chatcompletions.New(server.URL,
		chatcompletions.WithName("myserver"),
		chatcompletions.WithAPIKey("server-key"),
		chatcompletions.WithHTTPClient(server.Client()),
		chatcompletions.WithMaxRetries(0),
		chatcompletions.WithCapabilities(llm.CapabilityStreaming),
		chatcompletions.WithTimeout(time.Minute),
		chatcompletions.WithPriceTable(llm.PriceTable{"myserver/m": llm.ModelPricing{InputPerMTok: 1, OutputPerMTok: 1}}),
		chatcompletions.WithLogger(slog.New(slog.DiscardHandler)),
		chatcompletions.WithWireCapture(func(c llm.WireCapture) { captures = append(captures, c) }),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if p.Name() != "myserver" {
		t.Fatalf("Name = %q", p.Name())
	}
	if p.Client() == nil {
		t.Fatalf("Client() = nil")
	}
	if caps := p.Capabilities(); len(caps) != 1 || caps[0] != llm.CapabilityStreaming {
		t.Fatalf("capabilities = %+v", caps)
	}

	ctx := context.Background()
	resp, err := p.Chat(ctx, &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Provider != "myserver" {
		t.Fatalf("response provider = %q", resp.Provider)
	}
	if resp.Usage.CostUSD == nil || *resp.Usage.CostUSD <= 0 {
		t.Fatalf("price table estimation missing: %+v", resp.Usage)
	}
	for i, header := range *headers {
		if got := header.Get("Authorization"); got != "Bearer server-key" {
			t.Fatalf("request %d Authorization = %q", i, got)
		}
	}
	if len(captures) == 0 {
		t.Fatalf("wire capture never fired")
	}

	// Narrowed capabilities reject unsupported features before any call.
	_, err = p.Chat(ctx, &llm.Request{
		Model:    "m",
		Messages: []llm.Message{llm.UserText("hi")},
		Tools:    []llm.Tool{{Name: "lookup"}},
	})
	if !errors.Is(err, llm.ErrUnsupported) {
		t.Fatalf("undeclared tools error = %v, want ErrUnsupported", err)
	}
}

func TestGenericStreamWireCaptureTreatsDoneAsComplete(t *testing.T) {
	streamBody := "data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"pong\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
	client := &http.Client{Transport: testutil.RoundTripFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode:    http.StatusOK,
			ContentLength: -1,
			Header:        http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:          io.NopCloser(strings.NewReader(streamBody)),
		}, nil
	})}
	var captures []llm.WireCapture
	p, err := chatcompletions.New("https://example.test",
		chatcompletions.WithHTTPClient(client),
		chatcompletions.WithMaxRetries(0),
		chatcompletions.WithWireCapture(func(c llm.WireCapture) { captures = append(captures, c) }),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{
		Model:    "m",
		Messages: []llm.Message{llm.UserText("hi")},
	}))
	if err != nil {
		t.Fatalf("ChatStream returned error: %v", err)
	}
	if resp.Text() != "pong" {
		t.Fatalf("stream response text = %q", resp.Text())
	}
	if len(captures) != 1 {
		t.Fatalf("captures = %d, want 1", len(captures))
	}
	if captures[0].ResponseIncomplete || captures[0].Err != nil {
		t.Fatalf("terminal [DONE] capture = %+v", captures[0])
	}
	if !strings.HasSuffix(string(captures[0].ResponseBody), "data: [DONE]\n\n") {
		t.Fatalf("captured stream body = %q", captures[0].ResponseBody)
	}
}

// TestGenericNewAPIKeyFunc covers the per-request key resolver on both the
// SDK blocking path and the direct SSE path.
func TestGenericNewAPIKeyFunc(t *testing.T) {
	server, headers := newGenericServer(t)
	p, err := chatcompletions.New(server.URL,
		chatcompletions.WithAPIKeyFunc(func(context.Context) (string, error) { return "rotating-key", nil }),
		chatcompletions.WithHTTPClient(server.Client()),
		chatcompletions.WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	ctx := context.Background()
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}
	if _, err := p.Chat(ctx, req); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if _, err := llm.Collect(p.ChatStream(ctx, req)); err != nil {
		t.Fatalf("stream returned error: %v", err)
	}
	for i, header := range *headers {
		if got := header.Get("Authorization"); got != "Bearer rotating-key" {
			t.Fatalf("request %d Authorization = %q", i, got)
		}
	}
}

// TestGenericNewCompatSniff exercises Compat.SniffMidStreamErrors through the
// public constructor: a choice-less error data event maps to a normalized
// in-stream error for both the flat legacy and nested shapes.
func TestGenericNewCompatSniff(t *testing.T) {
	for name, payload := range map[string]string{
		"flat":   `{"object":"error","message":"engine died","code":503}`,
		"nested": `{"error":{"message":"engine died","type":"ServiceUnavailable","code":503}}`,
	} {
		t.Run(name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"par"},"finish_reason":null}]}`+"\n\n")
				_, _ = io.WriteString(w, "data: "+payload+"\n\n")
			}))
			t.Cleanup(server.Close)
			p, err := chatcompletions.New(server.URL,
				chatcompletions.WithCompat(chatcompletions.Compat{SniffMidStreamErrors: true}),
				chatcompletions.WithHTTPClient(server.Client()),
				chatcompletions.WithMaxRetries(0),
			)
			if err != nil {
				t.Fatalf("New returned error: %v", err)
			}
			resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{
				Model:    "m",
				Messages: []llm.Message{llm.UserText("hi")},
			}))
			if !errors.Is(err, llm.ErrServer) {
				t.Fatalf("sniffed error = %v, want ErrServer", err)
			}
			if resp == nil || resp.Text() != "par" {
				t.Fatalf("partial response = %+v", resp)
			}
		})
	}
}

func TestGenericNewRequiresBaseURL(t *testing.T) {
	if _, err := chatcompletions.New(""); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("New(\"\") error = %v, want ErrBadRequest", err)
	}
}

// TestGenericNewDefaultEffortSpelling pins the engine's nil-MapEffort default
// to the plain chat-completions reasoning_effort spelling.
func TestGenericNewDefaultEffortSpelling(t *testing.T) {
	var body []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(server.Close)
	p, err := chatcompletions.New(server.URL,
		chatcompletions.WithHTTPClient(server.Client()),
		chatcompletions.WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if _, err := p.Chat(context.Background(), &llm.Request{
		Model:    "m",
		Effort:   llm.EffortLow,
		Messages: []llm.Message{llm.UserText("hi")},
	}); err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if !strings.Contains(string(body), `"reasoning_effort":"low"`) {
		t.Fatalf("wire body missing reasoning_effort: %s", body)
	}
}
