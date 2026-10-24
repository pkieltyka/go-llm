package cloudflare_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/providers/cloudflare"
)

type seen struct{ path, auth, body string }

func server(t *testing.T, respond string) (*httptest.Server, *seen) {
	t.Helper()
	got := &seen{}
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		got.path, got.auth, got.body = r.URL.Path, r.Header.Get("Authorization"), string(data)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, respond)
	}))
	t.Cleanup(s.Close)
	return s, got
}

func request(model string, images ...llm.ImagePart) *llm.ClassifyRequest {
	return &llm.ClassifyRequest{Model: model, State: json.RawMessage(`{"m":"hi"}`), Images: images,
		Questions: []llm.Question{llm.BoolQuestion("approved", "Approve?", "yes", "no")}}
}

func clearEnv(t *testing.T) {
	t.Setenv("CLOUDFLARE_API_KEY", "")
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
}

func TestEnvelopes(t *testing.T) {
	clearEnv(t)
	for _, tc := range []struct {
		name, respond string
		probability   float64
	}{
		{"cloudflare-hosted (Clef)", `{"success":true,"result":{"model":"@cf/cloudflare/clef","answers":{"approved":{"type":"noul","noul":0.7}},"usage":{"input_tokens":20,"output_tokens":0}}}`, 0.7},
		{"third-party run record (Jev)", `{"success":true,"result":{"state":"Completed","result":{"answers":{"approved":{"type":"noul","noul":0.4}},"usage":{"input_tokens":30,"output_tokens":5}}}}`, 0.4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, got := server(t, tc.respond)
			p, err := cloudflare.New(cloudflare.WithAPIKey("cf-token"), cloudflare.WithAccountID("acct-1"), cloudflare.WithBaseURL(s.URL+"/client/v4/accounts/{account}/ai"))
			if err != nil {
				t.Fatal(err)
			}
			result, err := llm.Classify(context.Background(), p, request("@cf/cloudflare/clef"))
			if err != nil || result.Answers["approved"].Probability != tc.probability || result.Provider != "cloudflare-workers-ai" {
				t.Fatalf("result %+v err %v", result, err)
			}
			if got.path != "/client/v4/accounts/acct-1/ai/run" || got.auth != "Bearer cf-token" || !strings.HasPrefix(got.body, `{"input":{"questions":`) || !strings.Contains(got.body, `"model":"@cf/cloudflare/clef"`) {
				t.Fatalf("request = %+v", got)
			}
		})
	}
}

func TestEnvelopeFailures(t *testing.T) {
	clearEnv(t)
	for name, respond := range map[string]string{
		"success false":  `{"success":false,"errors":[{"code":7003,"message":"No route for that URI"}]}`,
		"run incomplete": `{"success":true,"result":{"state":"Running"}}`,
		"unreadable":     `nope`,
	} {
		t.Run(name, func(t *testing.T) {
			s, _ := server(t, respond)
			p, _ := cloudflare.New(cloudflare.WithAPIKey("k"), cloudflare.WithAccountID("a"), cloudflare.WithBaseURL(s.URL))
			if _, err := p.Classify(context.Background(), request("typesafe/jev")); !errors.Is(err, llm.ErrServer) {
				t.Fatalf("err = %v", err)
			}
		})
	}
}

func TestImageFormsByModel(t *testing.T) {
	clearEnv(t)
	s, got := server(t, `{"success":true,"result":{"answers":{"approved":{"type":"noul","noul":0.5}}}}`)
	p, _ := cloudflare.New(cloudflare.WithAPIKey("k"), cloudflare.WithAccountID("a"), cloudflare.WithBaseURL(s.URL))
	png := llm.ImagePart{Data: []byte{1}, MediaType: "image/png"}
	if _, err := p.Classify(context.Background(), request("@cf/cloudflare/clef-flash", png)); err != nil || !strings.Contains(got.body, `"images":["data:image/png;base64,AQ=="]`) {
		t.Fatalf("clef images: body %s err %v", got.body, err)
	}
	if _, err := p.Classify(context.Background(), request("typesafe/jev-1.14.0", png)); err != nil || !strings.Contains(got.body, `"assets":{"<image-1>"`) {
		t.Fatalf("jev assets: body %s err %v", got.body, err)
	}
}

func TestCredentialResolution(t *testing.T) {
	clearEnv(t)
	if _, err := cloudflare.New(); !errors.Is(err, llm.ErrAuth) || !strings.Contains(err.Error(), "CLOUDFLARE_API_KEY") {
		t.Fatalf("no token = %v", err)
	}
	if _, err := cloudflare.New(cloudflare.WithAPIKey("k")); !errors.Is(err, llm.ErrAuth) || !strings.Contains(err.Error(), "CLOUDFLARE_ACCOUNT_ID") {
		t.Fatalf("no account = %v", err)
	}
	s, got := server(t, `{"success":true,"result":{"answers":{"approved":{"type":"noul","noul":0.5}}}}`)
	proxy, err := cloudflare.New(cloudflare.WithAPIKey("k"), cloudflare.WithBaseURL(s.URL+"/proxy"))
	if err != nil {
		t.Fatalf("proxy without {account} = %v", err)
	}
	if _, err := proxy.Classify(context.Background(), request("typesafe/jev")); err != nil || got.path != "/proxy/run" {
		t.Fatalf("proxy: %+v %v", got, err)
	}

	t.Setenv("CLOUDFLARE_API_KEY", "env-token")
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "env-acct")
	p, err := cloudflare.New(cloudflare.WithBaseURL(s.URL + "/accounts/{account}/ai"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Classify(context.Background(), request("typesafe/jev")); err != nil || got.auth != "Bearer env-token" || got.path != "/accounts/env-acct/ai/run" {
		t.Fatalf("env: %+v %v", got, err)
	}

	// pi's per-field merge: the stored key and the entry's env account win
	// over the environment; a key-only entry still takes the env account.
	store := authstore.NewMemory(llm.AuthFile{"cloudflare-workers-ai": {Type: "api_key", Key: "stored-token", Env: map[string]string{"CLOUDFLARE_ACCOUNT_ID": "stored-acct"}}})
	p, _ = cloudflare.New(cloudflare.WithCredentials(store), cloudflare.WithBaseURL(s.URL+"/accounts/{account}/ai"))
	if _, err := p.Classify(context.Background(), request("typesafe/jev")); err != nil || got.auth != "Bearer stored-token" || got.path != "/accounts/stored-acct/ai/run" {
		t.Fatalf("stored: %+v %v", got, err)
	}
	keyOnly := authstore.NewMemory(llm.AuthFile{"cloudflare-workers-ai": {Type: "api_key", Key: "stored-token"}})
	p, _ = cloudflare.New(cloudflare.WithCredentials(keyOnly), cloudflare.WithBaseURL(s.URL+"/accounts/{account}/ai"))
	if _, err := p.Classify(context.Background(), request("typesafe/jev")); err != nil || got.path != "/accounts/env-acct/ai/run" {
		t.Fatalf("key-only entry: %+v %v", got, err)
	}
	oauth := authstore.NewMemory(llm.AuthFile{"cloudflare-workers-ai": {Type: "oauth", Access: "a"}})
	p, _ = cloudflare.New(cloudflare.WithCredentials(oauth), cloudflare.WithBaseURL(s.URL))
	if _, err := p.Classify(context.Background(), request("typesafe/jev")); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("oauth entry = %v", err)
	}
}

func TestChatUnsupportedAndConformance(t *testing.T) {
	clearEnv(t)
	p, _ := cloudflare.New(cloudflare.WithAPIKey("k"), cloudflare.WithAccountID("a"))
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
	llmtest.RunClassifierConformance(t, func(t *testing.T) llm.Provider {
		p, err := cloudflare.New(cloudflare.WithAPIKey("k"), cloudflare.WithAccountID("a"), cloudflare.WithBaseURL("http://127.0.0.1:1"), cloudflare.WithMaxRetries(0))
		if err != nil {
			t.Fatal(err)
		}
		return p
	})
}

// countingStore counts reads and rotates the entry after each one.
type countingStore struct {
	llm.CredentialStore
	reads int
}

func (s *countingStore) Read(ctx context.Context, id string) (llm.AuthCredential, bool, error) {
	s.reads++
	n := strconv.Itoa(s.reads)
	return llm.AuthCredential{Type: "api_key", Key: "key-" + n, Env: map[string]string{"CLOUDFLARE_ACCOUNT_ID": "acct-" + n}}, true, nil
}

// TestOneCredentialReadPerRequest: the key and the account come from one
// read, so a rotating entry never mixes them.
func TestOneCredentialReadPerRequest(t *testing.T) {
	clearEnv(t)
	s, got := server(t, `{"success":true,"result":{"answers":{"approved":{"type":"noul","noul":0.5}}}}`)
	store := &countingStore{}
	p, _ := cloudflare.New(cloudflare.WithCredentials(store), cloudflare.WithBaseURL(s.URL+"/accounts/{account}/ai"))
	for i := 1; i <= 2; i++ {
		if _, err := p.Classify(context.Background(), request("typesafe/jev")); err != nil {
			t.Fatal(err)
		}
		n := strconv.Itoa(i)
		if store.reads != i || got.auth != "Bearer key-"+n || got.path != "/accounts/acct-"+n+"/ai/run" {
			t.Fatalf("request %d: reads %d, %+v", i, store.reads, got)
		}
	}
	if _, err := cloudflare.New(cloudflare.WithAPIKey("k"), cloudflare.WithAccountID("../other")); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("path-like account = %v", err)
	}
}

func TestClefPreflightAndErrorCodes(t *testing.T) {
	clearEnv(t)
	calls := 0
	counted := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls++ }))
	defer counted.Close()
	p, _ := cloudflare.New(cloudflare.WithAPIKey("k"), cloudflare.WithAccountID("a"), cloudflare.WithBaseURL(counted.URL))
	png := llm.ImagePart{Data: []byte{1}, MediaType: "image/png"}
	badName := request("@cf/cloudflare/clef")
	badName.Questions = []llm.Question{llm.BoolQuestion("customer approved?", "x", "", "")}
	for name, req := range map[string]*llm.ClassifyRequest{
		"five images":  request("@cf/cloudflare/clef", png, png, png, png, png),
		"invalid name": badName,
	} {
		if _, err := p.Classify(context.Background(), req); !errors.Is(err, llm.ErrBadRequest) || calls != 0 {
			t.Fatalf("%s = %v (calls %d)", name, err, calls)
		}
	}
	for respond, want := range map[string]error{
		`{"success":false,"errors":[{"code":5007,"message":"No such model"}]}`: llm.ErrNotFound,
		`{"success":false,"errors":[{"code":3007,"message":"Timed out"}]}`:     llm.ErrTimeout,
		`{"success":false,"errors":[{"code":9999,"message":"Other"}]}`:         llm.ErrServer,
	} {
		s, _ := server(t, respond)
		p, _ := cloudflare.New(cloudflare.WithAPIKey("k"), cloudflare.WithAccountID("a"), cloudflare.WithBaseURL(s.URL))
		_, err := p.Classify(context.Background(), request("typesafe/jev"))
		var providerErr *llm.ProviderError
		if !errors.Is(err, want) || !errors.As(err, &providerErr) || providerErr.Code == "" {
			t.Fatalf("%s = %v", respond, err)
		}
	}
	notFound := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"success":false,"errors":[{"code":5007,"message":"No such model"}]}`)
	}))
	defer notFound.Close()
	p, _ = cloudflare.New(cloudflare.WithAPIKey("k"), cloudflare.WithAccountID("a"), cloudflare.WithBaseURL(notFound.URL))
	if _, err := p.Classify(context.Background(), request("typesafe/jev")); !errors.Is(err, llm.ErrNotFound) {
		t.Fatalf("HTTP 400 code 5007 = %v", err)
	}
}

func TestStringErrorCode(t *testing.T) {
	clearEnv(t)
	s, _ := server(t, `{"success":false,"errors":[{"code":"not_found","message":"No such model"}]}`)
	p, _ := cloudflare.New(cloudflare.WithAPIKey("k"), cloudflare.WithAccountID("a"), cloudflare.WithBaseURL(s.URL))
	_, err := p.Classify(context.Background(), request("typesafe/jev"))
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Code != "not_found" || !strings.Contains(providerErr.Message, "No such model") {
		t.Fatalf("string code = %v", err)
	}
}
