package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

const piQuestions = `{
	"urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
	"category": {"type": "choice", "instructions": "Classify it.", "criteria": {"question": "A question", "bug": "A defect"}},
	"approved": {"type": "bool", "instructions": "Approve?", "criteria": {"true": "yes", "false": "no"}}
}`

func TestParseQuestionsPiFormKeepsOrder(t *testing.T) {
	got, err := parseQuestions([]byte(piQuestions))
	if err != nil {
		t.Fatal(err)
	}
	want := []llm.Question{
		llm.ScoreQuestion("urgency", "How urgent?", "low", "medium", "high"),
		llm.ChoiceQuestion("category", "Classify it.", llm.Choice{Value: "question", Description: "A question"}, llm.Choice{Value: "bug", Description: "A defect"}),
		llm.BoolQuestion("approved", "Approve?", "yes", "no"),
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("questions\n got %+v\nwant %+v", got, want)
	}
	list, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := parseQuestions(list); err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("list form = %+v, %v", got, err)
	}
	bare, err := parseQuestions([]byte(`{"ok": {"type": "bool", "instructions": "Fine?"}}`))
	if err != nil || len(bare) != 1 || bare[0].True != "" {
		t.Fatalf("bool without criteria = %+v, %v", bare, err)
	}
}

func TestParseQuestionsErrors(t *testing.T) {
	for name, input := range map[string]string{
		"not json":            `nope`,
		"not an object":       `"x"`,
		"trailing data":       `{} {}`,
		"bad list":            `[1]`,
		"unknown type":        `{"q": {"type": "box", "instructions": "x"}}`,
		"question not object": `{"q": 1}`,
		"choice criteria":     `{"q": {"type": "choice", "instructions": "x", "criteria": ["a"]}}`,
		"choice description":  `{"q": {"type": "choice", "instructions": "x", "criteria": {"a": 1}}}`,
		"score criteria":      `{"q": {"type": "score", "instructions": "x", "criteria": {"a": "b"}}}`,
		"bool criteria":       `{"q": {"type": "bool", "instructions": "x", "criteria": ["a"]}}`,
	} {
		if _, err := parseQuestions([]byte(input)); !errors.Is(err, llm.ErrBadRequest) {
			t.Errorf("%s: err = %v", name, err)
		}
	}
}

func classifyFake() *llmtest.Provider {
	fake := llmtest.New(llmtest.WithName("llmtest"))
	fake.EnqueueClassifyResult(&llm.ClassifyResult{
		Provider: "llmtest", Model: "jev", ID: "c1",
		Answers: map[string]llm.Answer{
			"urgency":  {Type: llm.QuestionScore, Score: 1.8, LevelProbabilities: []float64{0.05, 0.1, 0.85}, Confidence: 0.8},
			"category": {Type: llm.QuestionChoice, Choice: "bug", Probabilities: map[string]float64{"bug": 0.9, "question": 0.1}, Confidence: 0.85},
			"approved": {Type: llm.QuestionBool, Probability: 0},
		},
		Usage: llm.Usage{InputTokens: 40, TotalTokens: 40},
	})
	return fake
}

func writeFile(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestRunClassify(t *testing.T) {
	questions := writeFile(t, "q.json", piQuestions)
	image := writeFile(t, "shot.png", "\x89PNG\r\n\x1a\nrest")

	fake := classifyFake()
	var stdout, stderr bytes.Buffer
	a := testApp(fake, &stdout, &stderr)
	err := a.run(context.Background(), []string{"classify", "-p", "llmtest", "-m", "jev", "--questions", questions,
		"--state", `{"message":"It crashes"}`, "--image", image, "--usage"})
	if err != nil {
		t.Fatal(err)
	}
	want := "urgency: 1.800 ≈ high (confidence 0.800)\ncategory: bug (p=0.900, confidence 0.850)\napproved: p(true)=0.000\n"
	if stdout.String() != want {
		t.Fatalf("stdout\n got %q\nwant %q", stdout.String(), want)
	}
	if !strings.Contains(stderr.String(), "usage input=40") {
		t.Fatalf("stderr = %q", stderr.String())
	}
	reqs := fake.ClassifyRequests()
	if len(reqs) != 1 || string(reqs[0].State) != `{"message":"It crashes"}` || len(reqs[0].Questions) != 3 ||
		len(reqs[0].Images) != 1 || reqs[0].Images[0].MediaType != "image/png" {
		t.Fatalf("request = %+v", reqs)
	}

	// The state from a file, and from stdin; --json prints the result.
	state := writeFile(t, "state.json", ` {"message":"from file"} `)
	for _, tc := range []struct {
		stdin, state, want string
	}{
		{"", state, `{"message":"from file"}`},
		{`{"message":"piped"}`, "", `{"message":"piped"}`},
		{`{"message":"dash"}`, "-", `{"message":"dash"}`},
	} {
		fake := classifyFake()
		stdout.Reset()
		a := testApp(fake, &stdout, &stderr)
		a.stdin = strings.NewReader(tc.stdin)
		args := []string{"classify", "-p", "llmtest", "-m", "jev", "--questions", questions, "--json"}
		if tc.state != "" {
			args = append(args, "--state", tc.state)
		}
		if err := a.run(context.Background(), args); err != nil {
			t.Fatal(err)
		}
		if got := string(fake.ClassifyRequests()[0].State); got != tc.want {
			t.Fatalf("state = %s, want %s", got, tc.want)
		}
		var result llm.ClassifyResult
		if err := json.Unmarshal(stdout.Bytes(), &result); err != nil || result.ID != "c1" || result.Answers["category"].Choice != "bug" {
			t.Fatalf("json = %s (%v)", stdout.String(), err)
		}
		if !strings.Contains(stdout.String(), `"probability": 0`) {
			t.Fatalf("a zero bool probability was dropped: %s", stdout.String())
		}
	}
}

func TestRunClassifyErrors(t *testing.T) {
	questions := writeFile(t, "q.json", piQuestions)
	for _, tc := range []struct {
		name  string
		args  []string
		stdin string
		want  error
	}{
		{"no model", []string{"--questions", questions, "--state", "{}"}, "", llm.ErrBadRequest},
		{"no questions", []string{"-m", "jev", "--state", "{}"}, "", llm.ErrBadRequest},
		{"no state", []string{"-m", "jev", "--questions", questions}, "", llm.ErrBadRequest},
		{"state not an object", []string{"-m", "jev", "--questions", questions, "--state", "-"}, "[1]", llm.ErrBadRequest},
		{"bad state json", []string{"-m", "jev", "--questions", questions, "--state", "{nope"}, "", llm.ErrBadRequest},
		{"missing state file", []string{"-m", "jev", "--questions", questions, "--state", "missing.json"}, "", os.ErrNotExist},
		{"missing questions file", []string{"-m", "jev", "--questions", "missing.json", "--state", "{}"}, "", os.ErrNotExist},
		{"missing image", []string{"-m", "jev", "--questions", questions, "--state", "{}", "--image", "missing.png"}, "", os.ErrNotExist},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := testApp(classifyFake(), io.Discard, io.Discard)
			a.stdin = strings.NewReader(tc.stdin)
			if tc.name == "no state" {
				a.stdin = nil
			}
			err := a.run(context.Background(), append([]string{"classify", "-p", "llmtest"}, tc.args...))
			if !errors.Is(err, tc.want) {
				t.Fatalf("err = %v, want %v", err, tc.want)
			}
		})
	}
	// A chat-only provider cannot classify.
	a := testApp(llmtest.New(), io.Discard, io.Discard)
	a.providerFactory = func(context.Context, providerConfig) (llm.Provider, error) {
		return chatOnly{llmtest.New()}, nil
	}
	if err := a.run(context.Background(), []string{"classify", "-p", "x", "-m", "m", "--questions", questions, "--state", "{}"}); !errors.Is(err, llm.ErrUnsupported) {
		t.Fatalf("chat-only provider = %v", err)
	}
	var stderr bytes.Buffer
	if err := (app{stderr: &stderr}).run(context.Background(), []string{"classify", "extra"}); !errors.Is(err, errUsage) || !strings.Contains(stderr.String(), `unexpected argument "extra"`) {
		t.Fatalf("extra argument = %v, %q", err, stderr.String())
	}
	if err := (app{stderr: io.Discard}).run(context.Background(), []string{"classify", "-h"}); err != nil {
		t.Fatalf("-h = %v", err)
	}
}

// chatOnly hides llmtest's Classify method.
type chatOnly struct{ llm.Provider }

// TestClassifyProviders: the classifier-only providers are selectable, take
// their keys from the flag, the environment or the auth file, and classify
// end to end over a System One server.
func TestClassifyProviders(t *testing.T) {
	clearProviderEnv(t)
	for _, tc := range []struct {
		name string
		cfg  providerConfig
		env  map[string]string
		want error
	}{
		{name: "typesafe", cfg: providerConfig{name: "typesafe", apiKey: "k"}},
		{name: "typesafe", cfg: providerConfig{name: "typesafe"}, env: map[string]string{"TYPESAFE_API_KEY": "k"}},
		{name: "typesafe", cfg: providerConfig{name: "typesafe"}, want: llm.ErrAuth},
		{name: "vercel-ai-gateway", cfg: providerConfig{name: "vercel-ai-gateway", apiKey: "k"}},
		{name: "opencode", cfg: providerConfig{name: "opencode"}, env: map[string]string{"OPENCODE_API_KEY": "k"}},
		{name: "cloudflare-workers-ai", cfg: providerConfig{name: "cloudflare-workers-ai", apiKey: "k"}, env: map[string]string{"CLOUDFLARE_ACCOUNT_ID": "a"}},
		{name: "cloudflare-workers-ai", cfg: providerConfig{name: "cloudflare-workers-ai", apiKey: "k"}, want: llm.ErrAuth},
		{name: "systemone", cfg: providerConfig{name: "systemone", baseURL: "http://localhost:8080/v1"}},
		{name: "systemone", cfg: providerConfig{name: "systemone"}, want: llm.ErrBadRequest},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for key, value := range tc.env {
				t.Setenv(key, value)
			}
			tc.cfg.timeout, tc.cfg.debug, tc.cfg.stderr = 5*time.Second, true, io.Discard
			p, err := newProvider(context.Background(), tc.cfg)
			if tc.want != nil {
				if !errors.Is(err, tc.want) {
					t.Fatalf("err = %v, want %v", err, tc.want)
				}
				return
			}
			if err != nil || p.Name() != tc.name {
				t.Fatalf("provider = %v, %v", p, err)
			}
			if _, ok := llm.FindClassifier(p); !ok {
				t.Fatalf("%s does not classify", tc.name)
			}
		})
	}

	var gotAuth, gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth, gotPath = r.Header.Get("Authorization"), r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"jev","answers":{"approved":{"type":"noul","noul":0.75}},"usage":{"input_tokens":9,"output_tokens":0}}`)
	}))
	defer server.Close()
	authPath := writeFile(t, "auth.json", `{"typesafe":{"type":"api_key","key":"stored-key"},"systemone":{"type":"api_key","key":"local-key"}}`)
	questions := writeFile(t, "q.json", `{"approved":{"type":"bool","instructions":"Approve?"}}`)
	for _, tc := range []struct{ provider, base, auth string }{
		{"typesafe", server.URL + "/v1", "Bearer stored-key"},
		{"systemone", server.URL + "/v1", "Bearer local-key"},
	} {
		var stdout bytes.Buffer
		a := app{stdout: &stdout, stderr: io.Discard, providerFactory: newProvider}
		err := a.run(context.Background(), []string{"classify", "-p", tc.provider, "--base-url", tc.base, "--auth-file", authPath,
			"-m", "jev", "--questions", questions, "--state", `{"m":"ok"}`})
		if err != nil || stdout.String() != "approved: p(true)=0.750\n" || gotAuth != tc.auth || gotPath != "/v1/systemone" {
			t.Fatalf("%s: out %q err %v auth %q path %q", tc.provider, stdout.String(), err, gotAuth, gotPath)
		}
	}

	// An authentication failure points at 'llm-cli login'.
	empty := writeFile(t, "empty.json", `{}`)
	a := app{stdout: io.Discard, stderr: io.Discard, providerFactory: newProvider}
	err := a.run(context.Background(), []string{"classify", "-p", "typesafe", "--auth-file", empty, "-m", "jev", "--questions", questions, "--state", "{}"})
	if !errors.Is(err, llm.ErrAuth) || !strings.Contains(err.Error(), "llm-cli login typesafe") || !strings.Contains(err.Error(), "TYPESAFE_API_KEY") {
		t.Fatalf("missing key = %v", err)
	}
}

// TestLoginCloudflareStoresAccountID: as in pi, the Workers AI login asks
// for the API key and then the account ID, stored in the entry's env.
func TestLoginCloudflareStoresAccountID(t *testing.T) {
	clearProviderEnv(t)
	authPath := filepath.Join(t.TempDir(), "auth.json")
	run := func(stdin string, args ...string) (string, string, error) {
		var stdout, stderr bytes.Buffer
		a := app{stdin: strings.NewReader(stdin), stdout: &stdout, stderr: &stderr}
		err := a.run(context.Background(), append(args, "--auth-file", authPath))
		return stdout.String(), stderr.String(), err
	}
	if _, _, err := run("cf-token\n", "login", "cloudflare-workers-ai"); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("missing account = %v", err)
	}
	if _, out, err := run("cf-token\nacct-1\n", "login", "cloudflare-workers-ai"); err != nil || strings.Contains(out, "cf-token") {
		t.Fatalf("login = %q, %v", out, err)
	}
	if _, _, err := run("ts-key", "login", "typesafe"); err != nil {
		t.Fatal(err)
	}
	auth, err := llm.LoadAuthFile(authPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := auth["cloudflare-workers-ai"]; got.Type != "api_key" || got.Key != "cf-token" || got.Env["CLOUDFLARE_ACCOUNT_ID"] != "acct-1" {
		t.Fatalf("cloudflare entry has type %q and env keys %d", got.Type, len(got.Env))
	}
	t.Setenv("AI_GATEWAY_API_KEY", "env-key")
	status, _, err := run("", "auth")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"cloudflare-workers-ai", "typesafe", "✓ env: AI_GATEWAY_API_KEY", "opencode", "systemone"} {
		if !strings.Contains(status, want) {
			t.Fatalf("auth status missing %q:\n%s", want, status)
		}
	}
	keyOnly := writeFile(t, "auth.json", `{"cloudflare-workers-ai":{"type":"api_key","key":"k"}}`)
	status = func() string {
		var stdout bytes.Buffer
		if err := (app{stdout: &stdout, stderr: io.Discard}).run(context.Background(), []string{"auth", "--auth-file", keyOnly}); err != nil {
			t.Fatal(err)
		}
		return stdout.String()
	}()
	if !strings.Contains(status, "✗ API key stored without Cloudflare account ID") {
		t.Fatalf("key-only status without the env account:\n%s", status)
	}
	commandKey := writeFile(t, "auth.json", `{"cloudflare-workers-ai":{"type":"api_key","key":"!op read x"}}`)
	var commandOut bytes.Buffer
	if err := (app{stdout: &commandOut, stderr: io.Discard}).run(context.Background(), []string{"auth", "--auth-file", commandKey}); err != nil ||
		!strings.Contains(commandOut.String(), "✗ API key stored without Cloudflare account ID") {
		t.Fatalf("command key without account = %q, %v", commandOut.String(), err)
	}
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
	var stdout bytes.Buffer
	if err := (app{stdout: &stdout, stderr: io.Discard}).run(context.Background(), []string{"auth", "--auth-file", keyOnly}); err != nil ||
		!strings.Contains(stdout.String(), "CLOUDFLARE_ACCOUNT_ID from the environment") {
		t.Fatalf("key-only status = %q, %v", stdout.String(), err)
	}
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "")
	t.Setenv("CLOUDFLARE_API_KEY", "env-token")
	empty := writeFile(t, "auth.json", `{}`)
	stdout.Reset()
	if err := (app{stdout: &stdout, stderr: io.Discard}).run(context.Background(), []string{"auth", "--auth-file", empty}); err != nil ||
		!strings.Contains(stdout.String(), "✗ env: CLOUDFLARE_API_KEY without CLOUDFLARE_ACCOUNT_ID") {
		t.Fatalf("env token without account = %q, %v", stdout.String(), err)
	}
}

// TestRunClassifyPartialResult: a billed call that fails still prints its
// valid answers and usage, then exits with the error.
func TestRunClassifyPartialResult(t *testing.T) {
	questions := writeFile(t, "q.json", piQuestions)
	fake := llmtest.New(llmtest.WithName("llmtest"))
	fake.EnqueueClassifyError(llm.ErrContentFiltered, &llm.ClassifyResult{Provider: "llmtest", Model: "jev",
		Answers: map[string]llm.Answer{"approved": {Type: llm.QuestionBool, Probability: 0.25}},
		Usage:   llm.Usage{InputTokens: 12, TotalTokens: 12}})
	var stdout, stderr bytes.Buffer
	err := testApp(fake, &stdout, &stderr).run(context.Background(), []string{"classify", "-p", "llmtest", "-m", "jev",
		"--questions", questions, "--state", "{}", "--usage"})
	if !errors.Is(err, llm.ErrContentFiltered) {
		t.Fatalf("err = %v", err)
	}
	want := "urgency: no valid answer\ncategory: no valid answer\napproved: p(true)=0.250\n"
	if stdout.String() != want || !strings.Contains(stderr.String(), "usage input=12") {
		t.Fatalf("stdout %q stderr %q", stdout.String(), stderr.String())
	}
}
