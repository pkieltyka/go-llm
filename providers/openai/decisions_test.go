package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/authstore"
	"github.com/pkieltyka/go-llm/llmtest"
)

func decisionsRequest() *llm.ClassifyRequest {
	return &llm.ClassifyRequest{
		Model: "gpt-6-luna",
		State: json.RawMessage(`{"message":"The change works perfectly, thanks."}`),
		Questions: []llm.Question{
			llm.ChoiceQuestion("category", "Classify the message.",
				llm.Choice{Value: "approval", Description: "The user approves"}, llm.Choice{Value: "correction"}),
			llm.ScoreQuestion("satisfaction", "Score user satisfaction.", "dissatisfied", "neutral", "satisfied"),
			llm.BoolQuestion("approved", "Does the user approve?", "Approval", "No approval"),
		},
	}
}

const decisionsAnswers = `{"id":"dec_1","model":"gpt-6-luna-2026-10-06","answers":[
	{"name":"category","type":"choice","choice":"approval","probabilities":[{"value":"approval","probability":0.98},{"value":"correction","probability":0.02}],"confidence":0.96},
	{"name":"satisfaction","type":"score","score":2,"confidence":0.9},
	{"name":"approved","type":"predicate","probability":0.97}],
	"usage":{"input_tokens":1000000}}`

func decisionsServer(t *testing.T, status int, respond string) (*httptest.Server, *string, *string) {
	t.Helper()
	var body, auth string
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		if r.URL.Path != "/v1/decisions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		body, auth = string(data), r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, respond)
	}))
	t.Cleanup(s.Close)
	return s, &body, &auth
}

func TestDecisionsClassify(t *testing.T) {
	s, body, auth := decisionsServer(t, http.StatusOK, decisionsAnswers)
	table := llm.PriceTable{"openai/gpt-6-luna": {InputPerMTok: 0.1}}
	p, err := New(WithAPIKey("sk-test"), WithBaseURL(s.URL+"/v1"), WithHTTPClient(s.Client()), WithPriceTable(table))
	if err != nil {
		t.Fatal(err)
	}
	result, err := llm.Classify(context.Background(), p, decisionsRequest())
	if err != nil {
		t.Fatal(err)
	}
	want := `{"input":"{\"message\":\"The change works perfectly, thanks.\"}","model":"gpt-6-luna","questions":[` +
		`{"type":"choice","name":"category","instructions":"Classify the message.","choices":[{"value":"approval","description":"The user approves"},{"value":"correction"}]},` +
		`{"type":"score","name":"satisfaction","instructions":"Score user satisfaction.","levels":[{"label":"dissatisfied"},{"label":"neutral"},{"label":"satisfied"}]},` +
		`{"type":"predicate","name":"approved","instructions":"Does the user approve?\n\nTrue means: Approval\nFalse means: No approval"}]}`
	if strings.TrimSpace(*body) != want {
		t.Fatalf("body\n got %s\nwant %s", *body, want)
	}
	if *auth != "Bearer sk-test" {
		t.Fatalf("auth = %q", *auth)
	}
	if result.Model != "gpt-6-luna-2026-10-06" || result.ID != "dec_1" || result.Provider != "openai" {
		t.Fatalf("identity = %+v", result)
	}
	if a := result.Answers["category"]; a.Choice != "approval" || a.Probabilities["correction"] != 0.02 || a.Confidence != 0.96 {
		t.Fatalf("choice = %+v", a)
	}
	if a := result.Answers["approved"]; a.Type != llm.QuestionBool || a.Probability != 0.97 {
		t.Fatalf("predicate = %+v", a)
	}
	if result.Usage.InputTokens != 1000000 || result.Usage.CostUSD == nil || *result.Usage.CostUSD != 0.1 || result.Usage.CostSource != llm.CostSourceEstimated {
		t.Fatalf("usage = %+v", result.Usage)
	}
}

func TestDecisionsImagesRefusalAndErrors(t *testing.T) {
	s, body, _ := decisionsServer(t, http.StatusOK, decisionsAnswers)
	p, _ := New(WithAPIKey("sk-test"), WithBaseURL(s.URL+"/v1"), WithHTTPClient(s.Client()))
	req := decisionsRequest()
	req.Images = []llm.ImagePart{{Data: []byte{1, 2}, MediaType: "image/png"}}
	if _, err := p.Classify(context.Background(), req); err != nil || !strings.Contains(*body, `"input":[{"content":[{"text":"{\"message\":\"The change works perfectly, thanks.\"}","type":"input_text"},{"image_url":"data:image/png;base64,AQI=","type":"input_image"}],"role":"user"}]`) {
		t.Fatalf("image body %s err %v", *body, err)
	}
	req.Images = make([]llm.ImagePart, 129)
	for i := range req.Images {
		req.Images[i] = llm.ImagePart{Data: []byte{1}, MediaType: "image/png"}
	}
	if _, err := p.Classify(context.Background(), req); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("too many images = %v", err)
	}

	refused, _, _ := decisionsServer(t, http.StatusOK, `{"answers":[{"name":"category","type":"refusal"},{"name":"approved","type":"predicate","probability":0.6}],"usage":{"input_tokens":5}}`)
	p, _ = New(WithAPIKey("sk-test"), WithBaseURL(refused.URL+"/v1"), WithHTTPClient(refused.Client()))
	result, err := p.Classify(context.Background(), decisionsRequest())
	if !errors.Is(err, llm.ErrContentFiltered) || result == nil || result.Usage.InputTokens != 5 || result.Answers["approved"].Probability != 0.6 {
		t.Fatalf("refusal = %+v, %v", result, err)
	}

	missing, _, _ := decisionsServer(t, http.StatusOK, `{"answers":[],"usage":{"input_tokens":5}}`)
	p, _ = New(WithAPIKey("sk-test"), WithBaseURL(missing.URL+"/v1"), WithHTTPClient(missing.Client()))
	if result, err := p.Classify(context.Background(), decisionsRequest()); !errors.Is(err, llm.ErrServer) || result.Usage.InputTokens != 5 {
		t.Fatalf("missing answers = %+v, %v", result, err)
	}

	gateway, _, _ := decisionsServer(t, http.StatusGatewayTimeout, `<html>504</html>`)
	p, _ = New(WithAPIKey("sk-test"), WithBaseURL(gateway.URL+"/v1"), WithHTTPClient(gateway.Client()))
	if _, err := p.Classify(context.Background(), decisionsRequest()); !errors.Is(err, llm.ErrServer) || !strings.Contains(err.Error(), "gateway timed out") {
		t.Fatalf("504 = %v", err)
	}
}

// TestDecisionsUsageAndCost: cache reads and writes are split out of
// input_tokens, and without a price table only a classifier catalog row
// prices the call; gpt-6-luna's chat row does not.
func TestDecisionsUsageAndCost(t *testing.T) {
	s, _, _ := decisionsServer(t, http.StatusOK, `{"answers":[
		{"name":"category","type":"choice","choice":"approval","probabilities":[{"value":"approval","probability":0.9},{"value":"correction","probability":0.1}],"confidence":0.8},
		{"name":"satisfaction","type":"score","score":2,"confidence":0.9},
		{"name":"approved","type":"predicate","probability":0.97}],
		"usage":{"input_tokens":1000,"input_tokens_details":{"cached_tokens":600,"cache_write_tokens":100},"output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":1000}}`)
	p, _ := New(WithAPIKey("sk-test"), WithBaseURL(s.URL+"/v1"), WithHTTPClient(s.Client()))
	result, err := p.Classify(context.Background(), decisionsRequest())
	if err != nil {
		t.Fatal(err)
	}
	u := result.Usage
	if u.InputTokens != 300 || u.CacheReadTokens != 600 || u.CacheWriteTokens != 100 || u.TotalTokens != 1000 {
		t.Fatalf("usage = %+v", u)
	}
	if _, ok := llm.LookupModelInfo("openai", "gpt-6-luna"); !ok {
		t.Skip("gpt-6-luna is no longer in the embedded table")
	}
	if u.CostUSD != nil {
		t.Fatalf("chat-row price used for a classify call: %v", *u.CostUSD)
	}
}

func TestDecisionsLimitsAndPrecedence(t *testing.T) {
	s, _, _ := decisionsServer(t, http.StatusOK, `{"answers":[
		{"name":"category","type":"choice","choice":"elsewhere","confidence":0.5},
		{"name":"satisfaction","type":"refusal"},
		{"name":"approved","type":"predicate","probability":0.4},
		{"name":"extra","type":"predicate","probability":0.5}],"usage":{"input_tokens":5}}`)
	p, _ := New(WithAPIKey("sk-test"), WithBaseURL(s.URL+"/v1"), WithHTTPClient(s.Client()))
	result, err := p.Classify(context.Background(), decisionsRequest())
	if !errors.Is(err, llm.ErrContentFiltered) {
		t.Fatalf("refusal precedence = %v", err)
	}
	if _, ok := result.Answers["category"]; ok {
		t.Fatalf("an undeclared choice was kept: %+v", result.Answers)
	}
	if _, ok := result.Answers["extra"]; ok || result.Answers["approved"].Probability != 0.4 {
		t.Fatalf("answers = %+v", result.Answers)
	}

	req := decisionsRequest()
	levels := make([]string, 11)
	for i := range levels {
		levels[i] = "l" + strconv.Itoa(i)
	}
	req.Questions[1] = llm.ScoreQuestion("satisfaction", "Score it.", levels...)
	if _, err := p.Classify(context.Background(), req); !errors.Is(err, llm.ErrBadRequest) || !strings.Contains(err.Error(), "score levels") {
		t.Fatalf("11 levels = %v", err)
	}
	req = decisionsRequest()
	for i := range 400 {
		req.Questions = append(req.Questions, llm.BoolQuestion("q"+strconv.Itoa(i), "x", "", ""))
	}
	if _, err := p.Classify(context.Background(), req); !errors.Is(err, llm.ErrBadRequest) || !strings.Contains(err.Error(), "questions") {
		t.Fatalf("403 questions = %v", err)
	}
}

// TestDecisionsRejectsSignInWithChatGPT: the Decisions API takes API keys
// only, so a stored OAuth credential fails before any I/O.
func TestDecisionsRejectsSignInWithChatGPT(t *testing.T) {
	calls := 0
	s := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls++ }))
	defer s.Close()
	store := authstore.NewMemory(llm.AuthFile{"openai": {Type: "oauth", Access: "a", Refresh: "r", ClientID: "c", Expires: time.Now().Add(time.Hour).UnixMilli()}})
	p, err := New(WithCredentials(store), WithBaseURL(s.URL+"/v1"), WithHTTPClient(s.Client()))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Classify(context.Background(), decisionsRequest()); !errors.Is(err, llm.ErrUnsupported) || calls != 0 {
		t.Fatalf("oauth classify = %v (calls %d)", err, calls)
	}
}

func TestDecisionsConformance(t *testing.T) {
	llmtest.RunClassifierConformance(t, func(t *testing.T) llm.Provider {
		p, err := New(WithAPIKey("sk-test"), WithBaseURL("http://127.0.0.1:1/v1"), WithMaxRetries(0))
		if err != nil {
			t.Fatal(err)
		}
		return p
	})
}

// TestDecisionsMalformedAnswerKeepsTheRest: each requested answer decodes on
// its own, so a malformed one keeps the usage and its valid siblings, a
// requested refusal still takes precedence, and a malformed unrequested
// answer is ignored. Response text stays out of the operational log.
func TestDecisionsMalformedAnswerKeepsTheRest(t *testing.T) {
	const marker = "PRIVATE_RESPONSE_MARKER"
	var logs strings.Builder
	logger := slog.New(slog.NewJSONHandler(&logs, nil))
	s, _, _ := decisionsServer(t, http.StatusOK, `{"answers":[
		{"name":"category","type":"refusal"},
		{"name":"satisfaction","type":"score","score":"bad","confidence":0.9},
		{"name":"approved","type":"predicate","probability":0.7}],"usage":{"input_tokens":123}}`)
	p, _ := New(WithAPIKey("sk-test"), WithBaseURL(s.URL+"/v1"), WithHTTPClient(s.Client()), WithLogger(logger))
	result, err := p.Classify(context.Background(), decisionsRequest())
	if !errors.Is(err, llm.ErrContentFiltered) || result == nil || result.Usage.InputTokens != 123 || result.Answers["approved"].Probability != 0.7 {
		t.Fatalf("result %+v err %v", result, err)
	}

	s, _, _ = decisionsServer(t, http.StatusOK, `{"answers":[
		{"name":"category","type":"choice","choice":"approval","confidence":0.9},
		{"name":"satisfaction","type":"score","score":"`+marker+`","confidence":0.9},
		{"name":"approved","type":"predicate","probability":0.7},
		{"name":"extra","type":"predicate","probability":"bad"}],"usage":{"input_tokens":123}}`)
	p, _ = New(WithAPIKey("sk-test"), WithBaseURL(s.URL+"/v1"), WithHTTPClient(s.Client()), WithLogger(logger))
	result, err = p.Classify(context.Background(), decisionsRequest())
	if !errors.Is(err, llm.ErrServer) || !strings.Contains(err.Error(), "satisfaction") || result == nil || result.Usage.InputTokens != 123 || len(result.Answers) != 2 {
		t.Fatalf("result %+v err %v", result, err)
	}

	s, _, _ = decisionsServer(t, http.StatusOK, `{"answers":[
		{"name":"category","type":"choice","choice":"`+marker+`","confidence":0.9},
		{"name":"satisfaction","type":"score","score":1,"confidence":0.9},
		{"name":"approved","type":"predicate","probability":0.7}]}`)
	p, _ = New(WithAPIKey("sk-test"), WithBaseURL(s.URL+"/v1"), WithHTTPClient(s.Client()), WithLogger(logger))
	if _, err = p.Classify(context.Background(), decisionsRequest()); !errors.Is(err, llm.ErrServer) || !strings.Contains(err.Error(), marker) {
		t.Fatalf("undeclared choice = %v", err)
	}
	if strings.Count(logs.String(), "classify failed") != 3 || strings.Contains(logs.String(), marker) {
		t.Fatalf("log = %s", logs.String())
	}

	s, _, _ = decisionsServer(t, http.StatusOK, `{"answers":[
		{"name":"category","type":"choice","choice":"approval","confidence":0.9},
		{"name":"satisfaction","type":"score","score":1,"confidence":0.9},
		{"name":"approved","type":"predicate","probability":0.7},
		{"name":"extra","type":"predicate","probability":"bad"},
		"unreadable"]}`)
	p, _ = New(WithAPIKey("sk-test"), WithBaseURL(s.URL+"/v1"), WithHTTPClient(s.Client()))
	if result, err = p.Classify(context.Background(), decisionsRequest()); err != nil || len(result.Answers) != 3 {
		t.Fatalf("unrequested malformed answers: result %+v err %v", result, err)
	}
}

// blockingStore blocks every read until its context ends.
type blockingStore struct{ llm.CredentialStore }

func (blockingStore) Read(ctx context.Context, _ string) (llm.AuthCredential, bool, error) {
	<-ctx.Done()
	return llm.AuthCredential{}, false, ctx.Err()
}

// TestDecisionsTimeoutCoversCredentialRead: WithTimeout bounds the
// credential read, as it does for Chat.
func TestDecisionsTimeoutCoversCredentialRead(t *testing.T) {
	p, err := New(WithCredentials(blockingStore{}), WithBaseURL("http://127.0.0.1:1/v1"), WithTimeout(10*time.Millisecond))
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	if _, err := p.Classify(context.Background(), decisionsRequest()); !errors.Is(err, context.DeadlineExceeded) || time.Since(start) > 5*time.Second {
		t.Fatalf("classify = %v after %v", err, time.Since(start))
	}
}

// TestDecisionsDuplicateAnswerIsMalformed: a question answered twice is
// malformed in either order, so a later answer cannot hide a refusal.
func TestDecisionsDuplicateAnswerIsMalformed(t *testing.T) {
	valid := `{"name":"approved","type":"predicate","probability":0.7}`
	refusal := `{"name":"approved","type":"refusal"}`
	for name, pair := range map[string]string{"valid first": valid + "," + refusal, "refusal first": refusal + "," + valid} {
		s, _, _ := decisionsServer(t, http.StatusOK, `{"answers":[
			{"name":"category","type":"choice","choice":"approval","confidence":0.9},
			{"name":"satisfaction","type":"score","score":1,"confidence":0.9},`+pair+`],"usage":{"input_tokens":7}}`)
		p, _ := New(WithAPIKey("sk-test"), WithBaseURL(s.URL+"/v1"), WithHTTPClient(s.Client()))
		result, err := p.Classify(context.Background(), decisionsRequest())
		if !errors.Is(err, llm.ErrServer) || !strings.Contains(err.Error(), "more than once") || result.Usage.InputTokens != 7 || len(result.Answers) != 2 {
			t.Fatalf("%s: result %+v err %v", name, result, err)
		}
	}
}
