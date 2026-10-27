package systemone_test

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
	"github.com/pkieltyka/go-llm/authstore"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/providers/systemone"
)

type seen struct {
	path, auth, optIn, extra, body string
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func server(t *testing.T) (*httptest.Server, *seen) {
	t.Helper()
	got := &seen{}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		got.path, got.auth, got.optIn, got.extra, got.body = r.URL.Path, r.Header.Get("Authorization"),
			r.Header.Get("TypeSafe-Danger-Zone-Images"), r.Header.Get("X-Extra"), string(body)
		_, _ = io.WriteString(w, `{"model":"jev-latest","answers":{"approved":{"type":"noul","noul":0.8}},"usage":{"input_tokens":12,"output_tokens":3}}`)
	}))
	t.Cleanup(s.Close)
	return s, got
}

func request(images ...llm.ImagePart) *llm.ClassifyRequest {
	return &llm.ClassifyRequest{Model: "jev-latest", State: json.RawMessage(`{"m":"hi"}`), Images: images,
		Questions: []llm.Question{llm.BoolQuestion("approved", "Approve?", "yes", "no")}}
}

func clearEnv(t *testing.T) {
	for _, name := range []string{"TYPESAFE_API_KEY", "AI_GATEWAY_API_KEY", "OPENCODE_API_KEY"} {
		t.Setenv(name, "")
	}
}

func TestNewIsKeylessAndClassifies(t *testing.T) {
	s, got := server(t)
	p, err := systemone.New(s.URL+"/v1", systemone.WithHeaders(http.Header{"X-Extra": {"1"}}))
	if err != nil {
		t.Fatal(err)
	}
	result, err := llm.Classify(context.Background(), p, request())
	if err != nil || result.Answers["approved"].Probability != 0.8 || result.Provider != "systemone" {
		t.Fatalf("result %+v err %v", result, err)
	}
	if got.path != "/v1/systemone" || got.auth != "" || got.extra != "1" {
		t.Fatalf("request = %+v", got)
	}
	if _, err := systemone.New(""); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("empty base URL = %v", err)
	}
	if _, err := p.Chat(context.Background(), &llm.Request{}); !errors.Is(err, llm.ErrUnsupported) {
		t.Fatalf("Chat = %v", err)
	}
	for _, err := range p.ChatStream(context.Background(), &llm.Request{}) {
		if !errors.Is(err, llm.ErrUnsupported) {
			t.Fatalf("ChatStream = %v", err)
		}
	}
	if _, err := p.Models(context.Background()); !errors.Is(err, llm.ErrUnsupported) {
		t.Fatalf("Models = %v", err)
	}
	if caps := p.Capabilities(); len(caps) != 2 || caps[0] != llm.CapabilityClassify {
		t.Fatalf("capabilities = %v", caps)
	}
}

func TestPresets(t *testing.T) {
	clearEnv(t)
	for _, tc := range []struct {
		name string
		env  string
		make func(...systemone.Option) (*systemone.Provider, error)
		text bool
	}{
		{"typesafe", "TYPESAFE_API_KEY", systemone.NewTypeSafe, false},
		{"vercel-ai-gateway", "AI_GATEWAY_API_KEY", systemone.NewVercelAIGateway, true},
		{"opencode", "OPENCODE_API_KEY", systemone.NewOpenCode, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.make(); !errors.Is(err, llm.ErrAuth) || !strings.Contains(err.Error(), tc.env) {
				t.Fatalf("no key = %v", err)
			}
			t.Setenv(tc.env, "env-key")
			s, got := server(t)
			p, err := tc.make(systemone.WithHTTPClient(s.Client()), withBase(s.URL))
			if err != nil {
				t.Fatal(err)
			}
			if p.Name() != tc.name {
				t.Fatalf("name = %q", p.Name())
			}
			if _, err := p.Classify(context.Background(), request()); err != nil || got.auth != "Bearer env-key" {
				t.Fatalf("classify: auth %q err %v", got.auth, err)
			}
			png := llm.ImagePart{Data: []byte{1}, MediaType: "image/png"}
			_, err = p.Classify(context.Background(), request(png))
			if tc.text {
				if !errors.Is(err, llm.ErrUnsupported) {
					t.Fatalf("text-only images = %v", err)
				}
				return
			}
			if err != nil || got.optIn != "true" || !strings.Contains(got.body, `"assets"`) {
				t.Fatalf("typesafe images: optIn %q body %s err %v", got.optIn, got.body, err)
			}
		})
	}
}

func withBase(url string) systemone.Option { return systemone.WithBaseURL(url) }

func TestOpenCodeDefaultEndpoint(t *testing.T) {
	t.Setenv("OPENCODE_API_KEY", "opencode-key")
	var gotURL, gotAuth, gotMethod string
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		gotURL, gotAuth, gotMethod = r.URL.String(), r.Header.Get("Authorization"), r.Method
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(`{"model":"jev-1.13","answers":{"approved":{"type":"noul","noul":0.8}}}`)),
		}, nil
	})}
	p, err := systemone.NewOpenCode(systemone.WithHTTPClient(client))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Classify(context.Background(), request()); err != nil {
		t.Fatal(err)
	}
	if gotURL != "https://opencode.ai/inference/systemone/v1/systemone" || gotAuth != "Bearer opencode-key" || gotMethod != http.MethodPost {
		t.Fatalf("request: method %q URL %q auth %q", gotMethod, gotURL, gotAuth)
	}
}

func TestCredentials(t *testing.T) {
	clearEnv(t)
	s, got := server(t)
	store := authstore.NewMemory(llm.AuthFile{"typesafe": {Type: "api_key", Key: "stored-key"}})
	p, err := systemone.NewTypeSafe(systemone.WithCredentials(store), withBase(s.URL))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Classify(context.Background(), request()); err != nil || got.auth != "Bearer stored-key" {
		t.Fatalf("stored key: auth %q err %v", got.auth, err)
	}
	if _, err := systemone.NewTypeSafe(systemone.WithCredentials(store), systemone.WithAPIKey("k")); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("credentials + key = %v", err)
	}
	oauth := authstore.NewMemory(llm.AuthFile{"typesafe": {Type: "oauth", Access: "a"}})
	p, _ = systemone.NewTypeSafe(systemone.WithCredentials(oauth), withBase(s.URL))
	if _, err := p.Classify(context.Background(), request()); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("oauth entry = %v", err)
	}

	// A keyless server with an auth file: no entry sends no key; an entry's
	// key is used.
	empty := authstore.NewMemory(nil)
	keyless, _ := systemone.New(s.URL, systemone.WithName("local"), systemone.WithCredentials(empty))
	if _, err := keyless.Classify(context.Background(), request()); err != nil || got.auth != "" {
		t.Fatalf("keyless: auth %q err %v", got.auth, err)
	}
	stored := authstore.NewMemory(llm.AuthFile{"local": {Type: "api_key", Key: "local-key"}})
	keyed, _ := systemone.New(s.URL, systemone.WithName("local"), systemone.WithCredentials(stored))
	if _, err := keyed.Classify(context.Background(), request()); err != nil || got.auth != "Bearer local-key" {
		t.Fatalf("keyed local: auth %q err %v", got.auth, err)
	}
}

func TestImageFormOption(t *testing.T) {
	s, got := server(t)
	png := llm.ImagePart{Data: []byte{1}, MediaType: "image/png"}
	// New defaults to llama.cpp's images list of data URLs.
	local, _ := systemone.New(s.URL)
	if _, err := local.Classify(context.Background(), request(png)); err != nil || !strings.Contains(got.body, `"images":["data:image/png;base64,AQ=="]`) || got.optIn != "" {
		t.Fatalf("default form: body %s optIn %q err %v", got.body, got.optIn, err)
	}
	p, _ := systemone.New(s.URL, systemone.WithImageForm(systemone.ImageStateURLs))
	if _, err := p.Classify(context.Background(), request(png)); err != nil || !strings.Contains(got.body, `"image_url"`) || got.optIn != "" {
		t.Fatalf("state URLs: body %s optIn %q err %v", got.body, got.optIn, err)
	}
}

func TestClassifierConformance(t *testing.T) {
	llmtest.RunClassifierConformance(t, func(t *testing.T) llm.Provider {
		p, err := systemone.New("http://127.0.0.1:1", systemone.WithMaxRetries(0))
		if err != nil {
			t.Fatal(err)
		}
		return p
	})
}
