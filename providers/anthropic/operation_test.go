package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
)

type operationTransport func(*http.Request) (*http.Response, error)

func (f operationTransport) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }
func operationReply(req *http.Request, status int, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(body)), Request: req}
}
func operationCredential() llm.AuthCredential {
	return llm.AuthCredential{Type: "oauth", Access: "synthetic-access", Refresh: "synthetic-refresh", Expires: time.Now().Add(time.Hour).UnixMilli()}
}
func operationRequest() *llm.Request {
	return &llm.Request{Model: "claude-test", Messages: []llm.Message{llm.UserParts(llm.Text("hi"))}}
}

const operationMessage = `{"id":"msg_synthetic","type":"message","role":"assistant","model":"claude-test","content":[{"type":"tool_use","id":"tool_synthetic","name":"Read","input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`

func operationProvider(t *testing.T, store llm.CredentialStore, transport operationTransport, opts ...Option) *Provider {
	t.Helper()
	options := []Option{WithCredentials(store), WithHTTPClient(&http.Client{Transport: transport}), WithMaxRetries(0)}
	p, err := New(append(options, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return p
}
func assertOAuthHeaders(t *testing.T, req *http.Request) {
	t.Helper()
	for name, want := range map[string]string{"Authorization": "Bearer synthetic-access", "X-Api-Key": "", "Accept": "application/json", "anthropic-version": "2023-06-01", "anthropic-dangerous-direct-browser-access": "true", "User-Agent": "claude-cli/2.1.280", "x-app": "cli"} {
		if req.Header.Get(name) != want {
			t.Errorf("incorrect OAuth header %s", name)
		}
	}
	for _, beta := range []string{"claude-code-20250219", "oauth-2025-04-20"} {
		if strings.Count(req.Header.Get("anthropic-beta"), beta) != 1 {
			t.Errorf("missing or duplicate beta %s", beta)
		}
	}
}

func TestAnthropicOAuthPreparedOperations(t *testing.T) {
	store := authstore.NewMemory(llm.AuthFile{providerName: operationCredential()})
	var calls atomic.Int32
	p := operationProvider(t, store, func(req *http.Request) (*http.Response, error) {
		calls.Add(1)
		assertOAuthHeaders(t, req)
		if req.URL.Path == "/v1/models" {
			return operationReply(req, 200, `{"data":[{"id":"claude-test","display_name":"Test","type":"model","created_at":"2026-01-01T00:00:00Z"}],"has_more":false}`), nil
		}
		var body struct {
			System []struct {
				Text  string         `json:"text"`
				Cache map[string]any `json:"cache_control"`
			} `json:"system"`
			Tools []struct {
				Name string `json:"name"`
			} `json:"tools"`
			Choice struct {
				Name string `json:"name"`
			} `json:"tool_choice"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if len(body.System) != 2 || body.System[0].Text != oauthSystemPreamble || body.System[1].Text != "original system" || body.System[0].Cache["ttl"] != "1h" {
			t.Error("system/cache transformation missing")
		}
		if len(body.Tools) != 1 || body.Tools[0].Name != "Read" || body.Choice.Name != "Read" {
			t.Error("tool transformation missing")
		}
		if req.URL.Path == "/v1/messages/count_tokens" {
			return operationReply(req, 200, `{"input_tokens":42}`), nil
		}
		return operationReply(req, 200, operationMessage), nil
	})
	req := operationRequest()
	req.System = "original system"
	req.SystemCache = &llm.CacheHint{TTL: time.Hour}
	req.Tools = []llm.Tool{{Name: "rEaD", InputSchema: map[string]any{"type": "object"}}}
	req.ToolChoice = llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "rEaD"}
	before, _ := json.Marshal(req)
	resp, err := p.Chat(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Parts[0].(llm.ToolCallPart).Name != "rEaD" {
		t.Error("response lost original tool name")
	}
	count, err := p.CountTokens(context.Background(), req)
	if err != nil || count.Tokens != 42 {
		t.Fatalf("count: %v", err)
	}
	models, err := p.Models(context.Background())
	if err != nil || len(models) != 1 {
		t.Fatalf("models: %v", err)
	}
	after, _ := json.Marshal(req)
	if string(before) != string(after) {
		t.Error("caller request mutated")
	}
	if calls.Load() != 3 {
		t.Fatal("unexpected HTTP count")
	}
}

func TestAnthropicOAuthRawSDKFailsClosed(t *testing.T) {
	for _, expired := range []bool{false, true} {
		cred := operationCredential()
		if expired {
			cred.Expires = 1
		}
		var calls atomic.Int32
		p := operationProvider(t, authstore.NewMemory(llm.AuthFile{providerName: cred}), func(req *http.Request) (*http.Response, error) {
			calls.Add(1)
			return operationReply(req, 200, operationMessage), nil
		})
		_, err := p.Client().Messages.New(context.Background(), sdk.MessageNewParams{})
		if !errors.Is(err, llm.ErrUnsupported) {
			t.Fatalf("raw message: %v", err)
		}
		stream := p.Client().Messages.NewStreaming(context.Background(), sdk.MessageNewParams{})
		for stream.Next() {
		}
		err = stream.Err()
		stream.Close()
		if !errors.Is(err, llm.ErrUnsupported) {
			t.Fatalf("raw stream: %v", err)
		}
		_, err = p.Client().Messages.CountTokens(context.Background(), sdk.MessageCountTokensParams{})
		if !errors.Is(err, llm.ErrUnsupported) {
			t.Fatalf("raw count: %v", err)
		}
		_, err = p.Client().Models.List(context.Background(), sdk.ModelListParams{})
		if !errors.Is(err, llm.ErrUnsupported) {
			t.Fatalf("raw models: %v", err)
		}
		if calls.Load() != 0 {
			t.Fatal("raw SDK used OAuth transport")
		}
	}
}

func TestAnthropicOAuthAPIKeyTransitionsAndRawSDK(t *testing.T) {
	store := authstore.NewMemory(llm.AuthFile{providerName: {Type: "api_key", Key: "synthetic-sk-ant-oat-looking-key"}})
	oauth := false
	p := operationProvider(t, store, func(req *http.Request) (*http.Response, error) {
		var body map[string]any
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if oauth {
			assertOAuthHeaders(t, req)
			if body["system"] == nil {
				t.Error("OAuth system missing")
			}
		} else {
			for _, name := range []string{"Authorization", "x-app", "anthropic-dangerous-direct-browser-access", "anthropic-beta"} {
				if req.Header.Get(name) != "" {
					t.Errorf("API-key OAuth header %s", name)
				}
			}
			if req.Header.Get("X-Api-Key") != "synthetic-sk-ant-oat-looking-key" || body["system"] != nil {
				t.Error("API-key behavior changed")
			}
		}
		return operationReply(req, 200, operationMessage), nil
	})
	if _, err := p.Client().Messages.New(context.Background(), sdk.MessageNewParams{}); err != nil {
		t.Fatal(err)
	}
	for _, mode := range []bool{true, false} {
		oauth = mode
		cred := operationCredential()
		if !mode {
			cred = llm.AuthCredential{Type: "api_key", Key: "synthetic-sk-ant-oat-looking-key"}
		}
		_, _, err := store.Modify(context.Background(), providerName, func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) { return cred, true, nil })
		if err != nil {
			t.Fatal(err)
		}
		if _, err := p.Chat(context.Background(), operationRequest()); err != nil {
			t.Fatal(err)
		}
	}
}

func TestAnthropicOAuthRejectsInvalidIdentityAndEndpoint(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*llm.AuthCredential)
		opts []Option
		kind error
	}{
		{"client", func(c *llm.AuthCredential) { c.ClientID = "unsupported-client" }, nil, llm.ErrAuth},
		{"configured-endpoint", func(*llm.AuthCredential) {}, []Option{WithBaseURL("https://example.invalid")}, llm.ErrUnsupported},
		{"stored-endpoint", func(c *llm.AuthCredential) { c.BaseURL = "https://example.invalid" }, nil, llm.ErrUnsupported},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cred := operationCredential()
			tc.edit(&cred)
			p := operationProvider(t, authstore.NewMemory(llm.AuthFile{providerName: cred}), func(*http.Request) (*http.Response, error) { t.Fatal("invalid OAuth reached network"); return nil, nil }, tc.opts...)
			if _, err := p.Chat(context.Background(), operationRequest()); !errors.Is(err, tc.kind) {
				t.Fatalf("wrong error: %v", err)
			}
			if _, err := p.Models(context.Background()); !errors.Is(err, tc.kind) {
				t.Fatalf("wrong discovery error: %v", err)
			}
		})
	}
}

type operationReadHook struct {
	llm.CredentialStore
	reads int
	hook  func(int)
}

func (s *operationReadHook) Read(ctx context.Context, id string) (llm.AuthCredential, bool, error) {
	s.reads++
	s.hook(s.reads)
	return s.CredentialStore.Read(ctx, id)
}
func TestAnthropicOAuthOwnershipRecheckedBeforeDispatch(t *testing.T) {
	for _, change := range []string{"delete", "mode", "client"} {
		t.Run(change, func(t *testing.T) {
			store := authstore.NewMemory(llm.AuthFile{providerName: operationCredential()})
			hooked := &operationReadHook{CredentialStore: store, hook: func(read int) {
				if read != 3 {
					return
				}
				if change == "delete" {
					if err := store.Delete(context.Background(), providerName); err != nil {
						t.Fatal(err)
					}
					return
				}
				cred := operationCredential()
				if change == "mode" {
					cred = llm.AuthCredential{Type: "api_key", Key: "synthetic-key"}
				} else {
					cred.ClientID = "foreign"
				}
				_, _, err := store.Modify(context.Background(), providerName, func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) { return cred, true, nil })
				if err != nil {
					t.Fatal(err)
				}
			}}
			p := operationProvider(t, hooked, func(*http.Request) (*http.Response, error) { t.Fatal("stale operation dispatched"); return nil, nil })
			if _, err := p.Chat(context.Background(), operationRequest()); !errors.Is(err, llm.ErrAuth) {
				t.Fatalf("wrong error: %v", err)
			}
		})
	}
}

func TestAnthropicOAuthRefreshOnceAndPersistsBeforeInference(t *testing.T) {
	cred := operationCredential()
	cred.Expires = 1
	store := authstore.NewMemory(llm.AuthFile{providerName: cred})
	var refreshes, inferences atomic.Int32
	p := operationProvider(t, store, func(req *http.Request) (*http.Response, error) {
		if req.URL.Host == "platform.claude.com" {
			refreshes.Add(1)
			return operationReply(req, 200, `{"access_token":"synthetic-renewed","refresh_token":"synthetic-rotated","expires_in":3600}`), nil
		}
		saved, _, err := store.Read(req.Context(), providerName)
		if err != nil || saved.Access != "synthetic-renewed" || req.Header.Get("Authorization") != "Bearer synthetic-renewed" {
			t.Error("inference preceded persistence")
		}
		inferences.Add(1)
		return operationReply(req, 200, operationMessage), nil
	})
	var wg sync.WaitGroup
	for range 12 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := p.Chat(context.Background(), operationRequest()); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if refreshes.Load() != 1 || inferences.Load() != 12 {
		t.Fatalf("refresh/inference counts: %d/%d", refreshes.Load(), inferences.Load())
	}
}

func TestAnthropicOAuthCanceledRefreshPersistsWithoutInference(t *testing.T) {
	cred := operationCredential()
	cred.Expires = 1
	store := authstore.NewMemory(llm.AuthFile{providerName: cred})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	p := operationProvider(t, store, func(req *http.Request) (*http.Response, error) {
		if req.URL.Host != "platform.claude.com" {
			t.Fatal("canceled operation dispatched inference")
		}
		cancel()
		return operationReply(req, 200, `{"access_token":"synthetic-renewed","refresh_token":"synthetic-rotated","expires_in":3600}`), nil
	})
	if _, err := p.Chat(ctx, operationRequest()); err != context.Canceled {
		t.Fatalf("cancellation: %v", err)
	}
	saved, _, err := store.Read(context.Background(), providerName)
	if err != nil || saved.Access != "synthetic-renewed" {
		t.Fatal("rotation not persisted")
	}
}

func TestAnthropicOAuthAuthRetryAndSafeErrors(t *testing.T) {
	for _, status := range []int{401, 403} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			store := authstore.NewMemory(llm.AuthFile{providerName: operationCredential()})
			var refreshes, attempts int
			p := operationProvider(t, store, func(req *http.Request) (*http.Response, error) {
				if req.URL.Host == "platform.claude.com" {
					refreshes++
					return operationReply(req, 200, `{"access_token":"synthetic-renewed","refresh_token":"synthetic-rotated","expires_in":3600}`), nil
				}
				attempts++
				return operationReply(req, status, `{"type":"error","error":{"type":"authentication_error","message":"synthetic-secret-marker"}}`), nil
			})
			_, err := p.Chat(context.Background(), operationRequest())
			if err == nil || strings.Contains(fmt.Sprintf("%+v %#v", err, err), "synthetic-secret-marker") {
				t.Fatal("remote credential detail escaped")
			}
			want := 1
			if status == 401 {
				want = 2
			}
			if attempts != want || refreshes != want-1 {
				t.Fatalf("attempts/refreshes %d/%d", attempts, refreshes)
			}
		})
	}
	for _, cause := range []error{context.Canceled, context.DeadlineExceeded, llm.ErrServer} {
		p := operationProvider(t, authstore.NewMemory(llm.AuthFile{providerName: operationCredential()}), func(*http.Request) (*http.Response, error) {
			return nil, fmt.Errorf("synthetic-secret-marker: %w", cause)
		})
		_, err := p.Chat(context.Background(), operationRequest())
		if !errors.Is(err, cause) || strings.Contains(fmt.Sprintf("%+v %#v", err, err), "synthetic-secret-marker") {
			t.Fatal("transport error escaped or lost classification")
		}
		if cause != llm.ErrServer && err != cause {
			t.Fatal("noncanonical cancellation")
		}
	}
}

func TestAnthropicOAuthToolMappingIsolationAndReplay(t *testing.T) {
	p := operationProvider(t, authstore.NewMemory(llm.AuthFile{providerName: operationCredential()}), func(req *http.Request) (*http.Response, error) {
		return operationReply(req, 200, operationMessage), nil
	})
	var wg sync.WaitGroup
	for _, name := range []string{"read", "rEaD", "READ"} {
		wg.Add(1)
		go func(name string) {
			defer wg.Done()
			req := operationRequest()
			req.Tools = []llm.Tool{{Name: name, InputSchema: map[string]any{"type": "object"}}}
			resp, err := p.Chat(context.Background(), req)
			if err != nil {
				t.Error(err)
				return
			}
			if resp.Parts[0].(llm.ToolCallPart).Name != name {
				t.Error("operation tool map leaked")
			}
		}(name)
	}
	wg.Wait()
	req := operationRequest()
	req.Tools = []llm.Tool{{Name: "read", InputSchema: map[string]any{"type": "object"}}, {Name: "Read", InputSchema: map[string]any{"type": "object"}}}
	if _, err := p.Chat(context.Background(), req); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("collision: %v", err)
	}
	for folded, canonical := range claudeCodeTools {
		original := strings.ToUpper(folded)
		req := operationRequest()
		req.Tools = []llm.Tool{{Name: original}}
		req.Messages = []llm.Message{{Role: llm.RoleAssistant, Parts: []llm.Part{llm.ToolCallPart{ID: "id", Name: original}}}, {Role: llm.RoleSystem, Parts: []llm.Part{llm.ToolsDeltaPart{Added: []llm.Tool{{Name: original}}, Removed: []string{original}}}}}
		mapped, reverse, err := oauthRequest(req)
		if err != nil {
			t.Fatal(err)
		}
		if mapped.Tools[0].Name != canonical || mapped.Messages[0].Parts[0].(llm.ToolCallPart).Name != canonical || mapped.Messages[1].Parts[0].(llm.ToolsDeltaPart).Removed[0] != canonical || reverse[folded] != original {
			t.Errorf("mapping %s", canonical)
		}
	}
}

func TestAnthropicOAuthStreamingRestoresToolName(t *testing.T) {
	const events = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-test\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool\",\"name\":\"Read\",\"input\":{}}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":1}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	p := operationProvider(t, authstore.NewMemory(llm.AuthFile{providerName: operationCredential()}), func(req *http.Request) (*http.Response, error) {
		assertOAuthHeaders(t, req)
		r := operationReply(req, 200, events)
		r.Header.Set("Content-Type", "text/event-stream")
		return r, nil
	})
	req := operationRequest()
	req.Tools = []llm.Tool{{Name: "read", InputSchema: map[string]any{"type": "object"}}}
	calls := 0
	for event, err := range p.ChatStream(context.Background(), req) {
		if err != nil {
			t.Fatal(err)
		}
		switch e := event.(type) {
		case llm.ToolCallStart:
			if e.Name != "read" {
				t.Error("stream start lost tool spelling")
			}
			calls++
		}
	}
	if calls != 1 {
		t.Fatalf("tool starts %d", calls)
	}
}

type operationFailingStore struct{ llm.CredentialStore }

func (s operationFailingStore) Modify(ctx context.Context, id string, fn func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error)) (llm.AuthCredential, bool, error) {
	current, ok, err := s.Read(ctx, id)
	if err != nil {
		return llm.AuthCredential{}, false, err
	}
	if _, _, err := fn(current, ok); err != nil {
		return llm.AuthCredential{}, false, err
	}
	return llm.AuthCredential{}, false, errors.New("synthetic-secret-marker persistence failure")
}
func TestAnthropicOAuthPersistenceFailureBlocksInference(t *testing.T) {
	cred := operationCredential()
	cred.Expires = 1
	var refreshes int
	p := operationProvider(t, operationFailingStore{authstore.NewMemory(llm.AuthFile{providerName: cred})}, func(req *http.Request) (*http.Response, error) {
		if req.URL.Host != "platform.claude.com" {
			t.Fatal("unpersisted token used")
		}
		refreshes++
		return operationReply(req, 200, `{"access_token":"synthetic-renewed","refresh_token":"synthetic-rotated","expires_in":3600}`), nil
	})
	_, err := p.Chat(context.Background(), operationRequest())
	if !errors.Is(err, llm.ErrAuth) || strings.Contains(fmt.Sprint(err), "synthetic-secret-marker") || refreshes != 1 {
		t.Fatalf("persistence failure: %v, refreshes %d", err, refreshes)
	}
}

func TestAnthropicOAuthTransportRetryRechecksOwnership(t *testing.T) {
	store := authstore.NewMemory(llm.AuthFile{providerName: operationCredential()})
	calls := 0
	p := operationProvider(t, store, func(req *http.Request) (*http.Response, error) {
		calls++
		if err := store.Delete(context.Background(), providerName); err != nil {
			t.Fatal(err)
		}
		return operationReply(req, 503, `{"type":"error","error":{"type":"overloaded_error","message":"try later"}}`), nil
	}, WithMaxRetries(1), WithResponseRetries(true))
	if _, err := p.Chat(context.Background(), operationRequest()); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("retry: %v", err)
	}
	if calls != 1 {
		t.Fatal("retry dispatched after logout")
	}
}

func TestAnthropicOAuthExpiryAlreadyAdjustedAndRedirect(t *testing.T) {
	cred := operationCredential()
	cred.Expires = time.Now().Add(time.Minute).UnixMilli()
	calls := 0
	p := operationProvider(t, authstore.NewMemory(llm.AuthFile{providerName: cred}), func(req *http.Request) (*http.Response, error) {
		calls++
		if req.URL.Host != "api.anthropic.com" {
			t.Fatal("unnecessary refresh or redirect")
		}
		r := operationReply(req, 302, "")
		r.Header.Set("Location", "https://example.invalid/steal")
		return r, nil
	})
	if _, err := p.Chat(context.Background(), operationRequest()); err == nil {
		t.Fatal("redirect accepted")
	}
	if calls != 1 {
		t.Fatal("redirect followed or double expiry margin applied")
	}
}

func TestAnthropicOAuthRawAPIKeySwitchFailsClosed(t *testing.T) {
	store := authstore.NewMemory(llm.AuthFile{providerName: {Type: "api_key", Key: "synthetic-key"}})
	hooked := &operationReadHook{CredentialStore: store, hook: func(n int) {
		if n == 2 {
			_, _, err := store.Modify(context.Background(), providerName, func(llm.AuthCredential, bool) (llm.AuthCredential, bool, error) {
				return operationCredential(), true, nil
			})
			if err != nil {
				t.Fatal(err)
			}
		}
	}}
	p := operationProvider(t, hooked, func(*http.Request) (*http.Response, error) { t.Fatal("raw call acquired OAuth"); return nil, nil })
	if _, err := p.Client().Messages.New(context.Background(), sdk.MessageNewParams{}); !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("raw mode switch: %v", err)
	}
}

func TestAnthropicOAuthForeignProviderMarkerRejected(t *testing.T) {
	store := authstore.NewMemory(llm.AuthFile{providerName: operationCredential()})
	transport := func(*http.Request) (*http.Response, error) { t.Fatal("foreign operation dispatched"); return nil, nil }
	first := operationProvider(t, store, transport)
	second := operationProvider(t, store, transport)
	ctx, _, err := first.prepareOperation(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := second.Client().Messages.New(ctx, sdk.MessageNewParams{}); !errors.Is(err, llm.ErrUnsupported) {
		t.Fatalf("foreign marker: %v", err)
	}
}

func TestAnthropicOAuthRefreshErrorCategories(t *testing.T) {
	for _, tc := range []struct {
		status int
		want   error
	}{{429, llm.ErrRateLimited}, {408, llm.ErrTimeout}, {500, llm.ErrServer}, {401, llm.ErrAuth}} {
		t.Run(fmt.Sprint(tc.status), func(t *testing.T) {
			cred := operationCredential()
			cred.Expires = 1
			p := operationProvider(t, authstore.NewMemory(llm.AuthFile{providerName: cred}), func(req *http.Request) (*http.Response, error) {
				if req.URL.Host != "platform.claude.com" {
					t.Fatal("inference after refresh failure")
				}
				return operationReply(req, tc.status, `{"error_description":"synthetic-secret-marker"}`), nil
			})
			_, err := p.Chat(context.Background(), operationRequest())
			if !errors.Is(err, tc.want) || strings.Contains(fmt.Sprint(err), "synthetic-secret-marker") {
				t.Fatalf("category: %v", err)
			}
		})
	}
}

func TestAnthropicOAuthWireCaptureWithholdsEchoedCredentials(t *testing.T) {
	for _, transportError := range []bool{false, true} {
		t.Run(fmt.Sprint(transportError), func(t *testing.T) {
			captures := 0
			p := operationProvider(t, authstore.NewMemory(llm.AuthFile{providerName: operationCredential()}), func(req *http.Request) (*http.Response, error) {
				if transportError {
					return nil, errors.New("synthetic-secret-marker")
				}
				r := operationReply(req, 403, `{"error":{"type":"permission_error","message":"synthetic-secret-marker"}}`)
				r.Header.Set("X-Secret-Echo", "synthetic-secret-marker")
				return r, nil
			}, WithWireCapture(func(c llm.WireCapture) {
				captures++
				if len(c.RequestBody) > 0 || len(c.ResponseBody) > 0 || len(c.RequestHeaders) > 0 || len(c.ResponseHeaders) > 0 || strings.Contains(fmt.Sprintf("%+v", c), "synthetic-secret-marker") {
					t.Error("OAuth capture leaked payload")
				}
			}))
			if _, err := p.Chat(context.Background(), operationRequest()); err == nil {
				t.Fatal("expected failure")
			}
			if captures != 1 {
				t.Fatalf("capture count %d", captures)
			}
		})
	}
}

func TestAnthropicOAuthRefreshesActuallyDispatchedBearer(t *testing.T) {
	store := authstore.NewMemory(llm.AuthFile{providerName: operationCredential()})
	hooked := &operationReadHook{CredentialStore: store, hook: func(n int) {
		if n == 3 { // Rotation between middleware resolution and dispatch reread.
			_, _, err := store.Modify(context.Background(), providerName, func(c llm.AuthCredential, _ bool) (llm.AuthCredential, bool, error) {
				c.Access = "synthetic-concurrently-renewed"
				return c, true, nil
			})
			if err != nil {
				t.Fatal(err)
			}
		}
	}}
	refreshes := 0
	p := operationProvider(t, hooked, func(req *http.Request) (*http.Response, error) {
		if req.URL.Host == "platform.claude.com" {
			refreshes++
			return operationReply(req, 200, `{"access_token":"synthetic-final","refresh_token":"synthetic-final-refresh","expires_in":3600}`), nil
		}
		if req.Header.Get("Authorization") == "Bearer synthetic-final" {
			return operationReply(req, 200, operationMessage), nil
		}
		return operationReply(req, 401, `{"error":{"type":"authentication_error","message":"reject"}}`), nil
	})
	_, err := p.Chat(context.Background(), operationRequest())
	if err != nil || refreshes != 1 {
		t.Fatalf("401 for dispatched credential was not refreshed: refreshes=%d err=%v", refreshes, err)
	}
}
