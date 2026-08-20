package openaicodex

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

func TestOpenAICodexModelsRequestParseCacheAndIsolation(t *testing.T) {
	t.Setenv(providerutil.CustomHeadersEnv, "Authorization: Bearer ambient-secret\nContent-Type: text/event-stream\nOpenAI-Beta: responses=experimental\nX-Catalog-Test: retained")
	access := fakeCodexJWT(t, "acct-models")
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.Method != http.MethodGet || r.URL.Path != "/models" {
			t.Errorf("request = %s %s, want GET /models", r.Method, r.URL.Path)
		}
		if got := r.URL.Query()["client_version"]; !reflect.DeepEqual(got, []string{defaultCodexClientVersion}) {
			t.Errorf("client_version = %#v", got)
		}
		for name, want := range map[string]string{
			"Accept":         "application/json",
			"Authorization":  "Bearer " + access,
			accountIDHeader:  "acct-models",
			originatorHeader: defaultOriginator,
			"User-Agent":     defaultCodexUserAgent,
			"X-Catalog-Test": "retained",
		} {
			if got := r.Header.Get(name); got != want {
				t.Errorf("%s = %q, want %q", name, got, want)
			}
		}
		if got := r.Header.Get("OpenAI-Beta"); got != "" {
			t.Errorf("OpenAI-Beta = %q, want empty", got)
		}
		if got := r.Header.Get("Content-Type"); got != "" {
			t.Errorf("Content-Type = %q, want empty", got)
		}
		body, _ := io.ReadAll(r.Body)
		if len(body) != 0 {
			t.Errorf("request body = %q, want empty", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"models":[
			{"id":"gpt-5.4","visibility":"list","supported_reasoning_levels":[{"effort":"low"},{"effort":"ultra"},{"effort":"high"}],"supported_parameters":["tools","tool_choice","structured_outputs","reasoning","stop","prompt_cache_key","future_parameter"],"supports_parallel_tool_calls":true,"architecture":{"input_modalities":["text","image"]},"service_tiers":["priority"]},
			{"id":"hidden","visibility":"hide"},
			{"id":"codex-auto-review","visibility":"list"},
			{"id":"gpt-5.4","name":"duplicate"},
			5,
			{"slug":"new-visible","description":"New Visible","max_context_window":64000,"max_output_tokens":8000,"reasoning_efforts":["minimal","unknown"]}
		]}`)
	}))
	defer server.Close()

	p, err := New(
		WithOAuth(llm.AuthCredential{Type: "oauth", Access: access}, nil),
		WithBaseURL(server.URL+"/responses"),
		WithHTTPClient(server.Client()),
		WithMaxRetries(0),
		WithPriceTable(llm.PriceTable{
			providerName + "/gpt-5.4": {
				InputPerMTok: 1,
				Tiers:        []llm.ModelPricingTier{{InputTokensAbove: 1000, InputPerMTok: 2}},
			},
		}),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	if len(models) != 2 || models[0].ID != "gpt-5.4" || models[1].ID != "new-visible" || models[1].ContextWindow != 64000 {
		t.Fatalf("models = %+v", models)
	}
	if models[0].DisplayName != "GPT-5.4" || models[0].ContextWindow != 272000 || models[0].MaxOutputTokens != 128000 {
		t.Fatalf("enriched model = %+v", models[0])
	}
	if want := []llm.Effort{llm.EffortLow, llm.EffortHigh}; !reflect.DeepEqual(models[0].SupportedEfforts, want) {
		t.Fatalf("efforts = %+v, want %+v", models[0].SupportedEfforts, want)
	}
	wantCapabilities := []llm.Capability{
		llm.CapabilityTools,
		llm.CapabilityToolChoiceRequired,
		llm.CapabilityParallelTools,
		llm.CapabilityJSONSchema,
		llm.CapabilityReasoning,
		llm.CapabilityImageInput,
		llm.CapabilityStopSequences,
		llm.CapabilityPromptCaching,
	}
	if !reflect.DeepEqual(models[0].Capabilities, wantCapabilities) {
		t.Fatalf("capabilities = %+v, want %+v", models[0].Capabilities, wantCapabilities)
	}
	raw, ok := models[0].Raw.(json.RawMessage)
	if !ok || !bytes.Contains(raw, []byte(`"ultra"`)) || !bytes.Contains(raw, []byte(`"service_tiers"`)) {
		t.Fatalf("raw = %T %s", models[0].Raw, raw)
	}
	if models[0].Pricing == nil || len(models[0].Pricing.Tiers) != 1 {
		t.Fatalf("pricing = %+v", models[0].Pricing)
	}

	models[0].SupportedEfforts[0] = llm.EffortMax
	models[0].Capabilities[0] = llm.CapabilityPDFInput
	models[0].Pricing.Tiers[0].InputPerMTok = 99
	raw[0] = '['
	second, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("second Models returned error: %v", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("catalog calls = %d, want 1", calls.Load())
	}
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Models(canceled); !errors.Is(err, context.Canceled) {
		t.Fatalf("cached Models with canceled context = %v", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("canceled cached Models made a request; calls = %d", calls.Load())
	}
	if second[0].SupportedEfforts[0] != llm.EffortLow || second[0].Capabilities[0] != llm.CapabilityTools || second[0].Pricing.Tiers[0].InputPerMTok != 2 {
		t.Fatalf("cached model was mutated: %+v", second[0])
	}
	if cachedRaw := second[0].Raw.(json.RawMessage); len(cachedRaw) == 0 || cachedRaw[0] != '{' {
		t.Fatalf("cached raw was mutated: %s", cachedRaw)
	}
}

func TestParseOpenAICodexModelsDataShapeAndEmpty(t *testing.T) {
	models, err := parseCodexCatalog([]byte(`{"data":[{"id":"data-model","name":"Data Model","visibility":""}]}`), nil)
	if err != nil || len(models) != 1 || models[0].ID != "data-model" {
		t.Fatalf("parse data shape = %+v, %v", models, err)
	}
	for _, body := range []string{
		`{"models":[]}`,
		`{"models":[{"id":"codex-auto-review"},{"id":"hidden","visibility":"internal"}]}`,
		`{"data":[5,{"id":""}]}`,
	} {
		if _, err := parseCodexCatalog([]byte(body), nil); err == nil {
			t.Fatalf("parseCodexCatalog(%s) returned nil error", body)
		}
	}
	var ultraOnly codexCatalogRow
	if err := json.Unmarshal([]byte(`{"supported_reasoning_levels":[{"effort":"ultra"}]}`), &ultraOnly); err != nil {
		t.Fatalf("decode ultra-only row: %v", err)
	}
	if efforts := codexReasoningEfforts(ultraOnly); len(efforts) != 0 {
		t.Fatalf("ultra-only typed efforts = %+v, want empty", efforts)
	}
	if capabilities := codexModelCapabilities(ultraOnly); !reflect.DeepEqual(capabilities, []llm.Capability{llm.CapabilityReasoning}) {
		t.Fatalf("ultra-only capabilities = %+v, want reasoning", capabilities)
	}
	for _, raw := range []string{
		`{"supported_reasoning_levels":[]}`,
		`{"supported_reasoning_levels":[""]}`,
		`{"supported_reasoning_levels":[null]}`,
		`{"supported_reasoning_levels":[{}]}`,
		`{"reasoning_efforts":["  "]}`,
	} {
		var malformed codexCatalogRow
		if err := json.Unmarshal([]byte(raw), &malformed); err != nil {
			t.Fatalf("decode malformed reasoning row %s: %v", raw, err)
		}
		if capabilities := codexModelCapabilities(malformed); len(capabilities) != 0 {
			t.Fatalf("malformed reasoning row %s capabilities = %+v, want empty", raw, capabilities)
		}
	}
}

func TestOpenAICodexModelsTTLsAndFallbacks(t *testing.T) {
	access := fakeCodexJWT(t, "acct-cache")
	var calls atomic.Int32
	var fail atomic.Bool
	var generation atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		if fail.Load() {
			http.Error(w, `{"error":{"message":"temporary"}}`, http.StatusInternalServerError)
			return
		}
		id := generation.Load()
		_, _ = io.WriteString(w, `{"models":[{"id":"live-`+string(rune('a'+id))+`"}]}`)
	}))
	defer server.Close()
	p, err := New(
		WithOAuth(llm.AuthCredential{Type: "oauth", Access: access}, nil),
		WithBaseURL(server.URL),
		WithHTTPClient(server.Client()),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	now := time.Date(2026, 8, 20, 12, 0, 0, 0, time.UTC)
	p.modelsNow = func() time.Time { return now }

	first, err := p.Models(context.Background())
	if err != nil || len(first) != 1 || first[0].ID != "live-a" {
		t.Fatalf("first Models = %+v, %v", first, err)
	}
	if _, err := p.Models(context.Background()); err != nil || calls.Load() != 1 {
		t.Fatalf("cached Models calls/error = %d/%v", calls.Load(), err)
	}

	now = now.Add(modelsSuccessTTL + time.Second)
	fail.Store(true)
	stale, err := p.Models(context.Background())
	if err != nil || len(stale) != 1 || stale[0].ID != "live-a" || calls.Load() != 2 {
		t.Fatalf("stale Models = %+v, calls=%d, err=%v", stale, calls.Load(), err)
	}
	if _, err := p.Models(context.Background()); err != nil || calls.Load() != 2 {
		t.Fatalf("failure-suppressed Models calls/error = %d/%v", calls.Load(), err)
	}

	now = now.Add(modelsFailureTTL + time.Second)
	fail.Store(false)
	generation.Store(1)
	refreshed, err := p.Models(context.Background())
	if err != nil || len(refreshed) != 1 || refreshed[0].ID != "live-b" || calls.Load() != 3 {
		t.Fatalf("refreshed Models = %+v, calls=%d, err=%v", refreshed, calls.Load(), err)
	}
}

func TestOpenAICodexModelsInitialFailureUsesStaticAndSuppressesRetry(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		http.Error(w, "temporarily unavailable", http.StatusServiceUnavailable)
	}))
	defer server.Close()
	p, err := New(
		WithOAuth(llm.AuthCredential{Type: "oauth", Access: fakeCodexJWT(t, "acct-static")}, nil),
		WithBaseURL(server.URL),
		WithHTTPClient(server.Client()),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	models, err := p.Models(context.Background())
	if err != nil || len(models) != len(staticModels) || calls.Load() != 1 {
		t.Fatalf("Models = %d rows, calls=%d, err=%v", len(models), calls.Load(), err)
	}
	if _, err := p.Models(context.Background()); err != nil || calls.Load() != 1 {
		t.Fatalf("suppressed Models calls/error = %d/%v", calls.Load(), err)
	}
}

func TestOpenAICodexModelsCoalescesAndWaiterCanCancel(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		once.Do(func() { close(started) })
		<-release
		_, _ = io.WriteString(w, `{"models":[{"id":"shared","supported_parameters":["tools"]}]}`)
	}))
	defer server.Close()
	p, err := New(
		WithOAuth(llm.AuthCredential{Type: "oauth", Access: fakeCodexJWT(t, "acct-shared")}, nil),
		WithBaseURL(server.URL),
		WithHTTPClient(server.Client()),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	type result struct {
		models []llm.ModelInfo
		err    error
	}
	leader := make(chan result, 1)
	go func() {
		models, err := p.Models(context.Background())
		leader <- result{models, err}
	}()
	<-started

	waitCtx, cancel := context.WithCancel(context.Background())
	canceled := make(chan error, 1)
	go func() {
		_, err := p.Models(waitCtx)
		canceled <- err
	}()
	follower := make(chan result, 1)
	go func() {
		models, err := p.Models(context.Background())
		follower <- result{models, err}
	}()
	cancel()
	if err := <-canceled; !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled waiter error = %v", err)
	}
	close(release)
	first, second := <-leader, <-follower
	if first.err != nil || second.err != nil || len(first.models) != 1 || len(second.models) != 1 || calls.Load() != 1 {
		t.Fatalf("results = %+v / %+v, calls=%d", first, second, calls.Load())
	}
	first.models[0].Capabilities[0] = llm.CapabilityPDFInput
	if second.models[0].Capabilities[0] != llm.CapabilityTools {
		t.Fatal("coalesced callers shared a mutable capability slice")
	}
}

func TestOpenAICodexModelsAuthErrorVisibleAndNotCached(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		http.Error(w, `{"error":{"code":"invalid_token","message":"expired","type":"authentication_error"}}`, http.StatusUnauthorized)
	}))
	defer server.Close()
	p, err := New(
		WithOAuth(llm.AuthCredential{Type: "oauth", Access: fakeCodexJWT(t, "acct-auth")}, nil),
		WithBaseURL(server.URL),
		WithHTTPClient(server.Client()),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	for i := int32(1); i <= 2; i++ {
		if _, err := p.Models(context.Background()); !errors.Is(err, llm.ErrAuth) {
			t.Fatalf("Models error = %v, want ErrAuth", err)
		}
		if calls.Load() != i {
			t.Fatalf("calls = %d after attempt %d", calls.Load(), i)
		}
	}
}

func TestOpenAICodexModelsContextErrorVisibleAndNotCached(t *testing.T) {
	var calls atomic.Int32
	started := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := calls.Add(1)
		if attempt == 1 {
			close(started)
			<-r.Context().Done()
			return
		}
		_, _ = io.WriteString(w, `{"models":[{"id":"after-cancel"}]}`)
	}))
	defer server.Close()
	p, err := New(
		WithOAuth(llm.AuthCredential{Type: "oauth", Access: fakeCodexJWT(t, "acct-context")}, nil),
		WithBaseURL(server.URL),
		WithHTTPClient(server.Client()),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	first := make(chan error, 1)
	go func() {
		_, err := p.Models(ctx)
		first <- err
	}()
	<-started
	cancel()
	if err := <-first; !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled Models error = %v", err)
	}
	models, err := p.Models(context.Background())
	if err != nil || len(models) != 1 || models[0].ID != "after-cancel" || calls.Load() != 2 {
		t.Fatalf("second Models = %+v, calls=%d, err=%v", models, calls.Load(), err)
	}
}

func TestOpenAICodexModelsRefreshesOAuth(t *testing.T) {
	oldAccess := fakeCodexJWT(t, "acct-old-models")
	newAccess := fakeCodexJWT(t, "acct-new-models")
	var modelCalls atomic.Int32
	var persisted atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			call := modelCalls.Add(1)
			if call == 1 {
				http.Error(w, `{"error":{"code":"invalid_token","message":"expired","type":"authentication_error"}}`, http.StatusUnauthorized)
				return
			}
			if got := r.Header.Get("Authorization"); got != "Bearer "+newAccess {
				t.Errorf("retried Authorization = %q", got)
			}
			_, _ = io.WriteString(w, `{"models":[{"id":"refreshed"}]}`)
		case "/oauth/token":
			_, _ = io.WriteString(w, `{"access_token":`+strconvQuote(newAccess)+`,"refresh_token":"new-refresh","expires_in":3600}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	p, err := New(
		WithOAuth(llm.AuthCredential{Type: "oauth", Access: oldAccess, Refresh: "old-refresh"}, func(context.Context, llm.AuthCredential) error {
			persisted.Add(1)
			return nil
		}),
		WithBaseURL(server.URL),
		withOAuthTokenURL(server.URL+"/oauth/token"),
		WithHTTPClient(server.Client()),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	models, err := p.Models(context.Background())
	if err != nil || len(models) != 1 || models[0].ID != "refreshed" || modelCalls.Load() != 2 || persisted.Load() != 1 {
		t.Fatalf("Models = %+v, model calls=%d, persisted=%d, err=%v", models, modelCalls.Load(), persisted.Load(), err)
	}
}

func TestOpenAICodexModelsPersistenceErrorVisibleAndNotCached(t *testing.T) {
	oldAccess := fakeCodexJWT(t, "acct-old-persistence")
	newAccess := fakeCodexJWT(t, "acct-new-persistence")
	persistErr := errors.New("persist models credential")
	var modelCalls atomic.Int32
	var tokenCalls atomic.Int32
	var persistenceCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			modelCalls.Add(1)
			http.Error(w, `{"error":{"code":"invalid_token","message":"expired","type":"authentication_error"}}`, http.StatusUnauthorized)
		case "/oauth/token":
			tokenCalls.Add(1)
			_, _ = io.WriteString(w, `{"access_token":`+strconvQuote(newAccess)+`,"refresh_token":"new-refresh","expires_in":3600}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	p, err := New(
		WithOAuth(llm.AuthCredential{Type: "oauth", Access: oldAccess, Refresh: "old-refresh"}, func(context.Context, llm.AuthCredential) error {
			persistenceCalls.Add(1)
			return persistErr
		}),
		WithBaseURL(server.URL),
		withOAuthTokenURL(server.URL+"/oauth/token"),
		WithHTTPClient(server.Client()),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	for attempt := int32(1); attempt <= 2; attempt++ {
		models, err := p.Models(context.Background())
		if !errors.Is(err, persistErr) {
			t.Fatalf("Models attempt %d error = %v, want persistence error", attempt, err)
		}
		if models != nil {
			t.Fatalf("Models attempt %d = %+v, want nil", attempt, models)
		}
		if modelCalls.Load() != attempt || tokenCalls.Load() != attempt || persistenceCalls.Load() != attempt {
			t.Fatalf("attempt %d calls = models:%d tokens:%d persistence:%d", attempt, modelCalls.Load(), tokenCalls.Load(), persistenceCalls.Load())
		}
	}
}
