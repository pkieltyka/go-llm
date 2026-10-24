package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

func validClassifyRequest() *llm.ClassifyRequest {
	return &llm.ClassifyRequest{
		Model: "typesafe/jev-1.14",
		State: json.RawMessage(`{"message":"The change works perfectly, thanks."}`),
		Questions: []llm.Question{
			llm.ChoiceQuestion("category", "Classify the message.",
				llm.Choice{Value: "approval", Description: "The user approves"},
				llm.Choice{Value: "correction", Description: "The user wants a change"}),
			llm.ScoreQuestion("satisfaction", "Score user satisfaction.", "dissatisfied", "neutral", "satisfied"),
			llm.BoolQuestion("approved", "Does the user approve?", "Approval", "No approval"),
		},
	}
}

func TestValidateClassifyRequest(t *testing.T) {
	if err := llm.ValidateClassifyRequest(validClassifyRequest()); err != nil {
		t.Fatalf("valid request = %v", err)
	}
	for name, mutate := range map[string]func(*llm.ClassifyRequest){
		"no model":        func(r *llm.ClassifyRequest) { r.Model = "" },
		"array state":     func(r *llm.ClassifyRequest) { r.State = json.RawMessage(`[1,2]`) },
		"empty state":     func(r *llm.ClassifyRequest) { r.State = nil },
		"invalid state":   func(r *llm.ClassifyRequest) { r.State = json.RawMessage(`{"a":`) },
		"no questions":    func(r *llm.ClassifyRequest) { r.Questions = nil },
		"unnamed":         func(r *llm.ClassifyRequest) { r.Questions[0].Name = "" },
		"duplicate name":  func(r *llm.ClassifyRequest) { r.Questions[1].Name = "category" },
		"one choice":      func(r *llm.ClassifyRequest) { r.Questions[0].Choices = r.Questions[0].Choices[:1] },
		"repeated choice": func(r *llm.ClassifyRequest) { r.Questions[0].Choices[1].Value = "approval" },
		"empty choice":    func(r *llm.ClassifyRequest) { r.Questions[0].Choices[0].Value = "" },
		"one level":       func(r *llm.ClassifyRequest) { r.Questions[1].Levels = r.Questions[1].Levels[:1] },
		"empty level":     func(r *llm.ClassifyRequest) { r.Questions[1].Levels[0] = "" },
		"unknown type":    func(r *llm.ClassifyRequest) { r.Questions[2].Type = "box" },
		"url image":       func(r *llm.ClassifyRequest) { r.Images = []llm.ImagePart{{URL: "https://x/y.png"}} },
		"untyped image":   func(r *llm.ClassifyRequest) { r.Images = []llm.ImagePart{{Data: []byte{1}}} },
	} {
		t.Run(name, func(t *testing.T) {
			req := validClassifyRequest()
			mutate(req)
			if err := llm.ValidateClassifyRequest(req); !errors.Is(err, llm.ErrBadRequest) {
				t.Fatalf("err = %v, want ErrBadRequest", err)
			}
		})
	}
	if err := llm.ValidateClassifyRequest(nil); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("nil request = %v", err)
	}
}

func TestCheckClassifyAnswers(t *testing.T) {
	req := validClassifyRequest()
	good := map[string]llm.Answer{
		"category":     {Type: llm.QuestionChoice, Choice: "approval", Probabilities: map[string]float64{"approval": 1}, Confidence: 1},
		"satisfaction": {Type: llm.QuestionScore, Score: 2, Confidence: 1},
		"approved":     {Type: llm.QuestionBool, Probability: 0.97},
	}
	if err := llm.CheckClassifyAnswers(req, good); err != nil {
		t.Fatalf("good answers = %v", err)
	}
	for name, mutate := range map[string]func(map[string]llm.Answer){
		"missing":    func(a map[string]llm.Answer) { delete(a, "approved") },
		"wrong type": func(a map[string]llm.Answer) { a["approved"] = llm.Answer{Type: llm.QuestionScore} },
		"no choice":  func(a map[string]llm.Answer) { a["category"] = llm.Answer{Type: llm.QuestionChoice} },
		"NaN": func(a map[string]llm.Answer) {
			a["approved"] = llm.Answer{Type: llm.QuestionBool, Probability: math.NaN()}
		},
		"Inf probability": func(a map[string]llm.Answer) {
			a["category"] = llm.Answer{Type: llm.QuestionChoice, Choice: "approval", Probabilities: map[string]float64{"approval": math.Inf(1)}}
		},
		"undeclared choice": func(a map[string]llm.Answer) {
			a["category"] = llm.Answer{Type: llm.QuestionChoice, Choice: "elsewhere", Confidence: 0.5}
		},
		"undeclared probability key": func(a map[string]llm.Answer) {
			a["category"] = llm.Answer{Type: llm.QuestionChoice, Choice: "approval", Probabilities: map[string]float64{"elsewhere": 0.1}, Confidence: 0.5}
		},
		"probability above one": func(a map[string]llm.Answer) {
			a["approved"] = llm.Answer{Type: llm.QuestionBool, Probability: 1.2}
		},
		"negative confidence": func(a map[string]llm.Answer) {
			a["category"] = llm.Answer{Type: llm.QuestionChoice, Choice: "approval", Confidence: -0.1}
		},
		"score above the top level": func(a map[string]llm.Answer) {
			a["satisfaction"] = llm.Answer{Type: llm.QuestionScore, Score: 2.5, Confidence: 1}
		},
		"level probabilities count": func(a map[string]llm.Answer) {
			a["satisfaction"] = llm.Answer{Type: llm.QuestionScore, Score: 1, LevelProbabilities: []float64{0.5, 0.5}, Confidence: 1}
		},
	} {
		t.Run(name, func(t *testing.T) {
			answers := map[string]llm.Answer{}
			for k, v := range good {
				answers[k] = v
			}
			mutate(answers)
			if err := llm.CheckClassifyAnswers(req, answers); !errors.Is(err, llm.ErrServer) {
				t.Fatalf("err = %v, want ErrServer", err)
			}
			for name, answer := range answers {
				if err := llm.CheckClassifyAnswers(&llm.ClassifyRequest{Questions: []llm.Question{questionNamed(req, name)}}, map[string]llm.Answer{name: answer}); err != nil {
					t.Fatalf("invalid answer %q was kept: %v", name, err)
				}
			}
		})
	}
	// Unrequested answers are dropped; a fractional score and float
	// rounding just past 1 are fine.
	answers := map[string]llm.Answer{"extra": {Type: llm.QuestionBool, Probability: 0.5}}
	for k, v := range good {
		answers[k] = v
	}
	answers["satisfaction"] = llm.Answer{Type: llm.QuestionScore, Score: 1.37, LevelProbabilities: []float64{0.1, 0.4, 0.5}, Confidence: 0.9}
	answers["approved"] = llm.Answer{Type: llm.QuestionBool, Probability: 1 + 1e-9}
	if err := llm.CheckClassifyAnswers(req, answers); err != nil || len(answers) != 3 {
		t.Fatalf("answers = %+v, %v", answers, err)
	}
}

func questionNamed(req *llm.ClassifyRequest, name string) llm.Question {
	for _, q := range req.Questions {
		if q.Name == name {
			return q
		}
	}
	return llm.Question{}
}

// TestClassifyFindsClassifierThroughWrap: llm.Classify follows Unwrap like
// CountTokens; a provider without a classifier is ErrUnsupported.
func TestClassifyFindsClassifierThroughWrap(t *testing.T) {
	fake := llmtest.New()
	fake.EnqueueClassifyResult(&llm.ClassifyResult{Provider: "llmtest", Model: "m", Answers: map[string]llm.Answer{
		"approved": {Type: llm.QuestionBool, Probability: 0.9},
	}})
	wrapped := llm.Wrap(fake, llm.NewUsageTracker().Middleware())
	req := &llm.ClassifyRequest{Model: "m", State: json.RawMessage(`{}`), Questions: []llm.Question{llm.BoolQuestion("approved", "?", "", "")}}
	result, err := llm.Classify(context.Background(), wrapped, req)
	if err != nil || result.Answers["approved"].Probability != 0.9 {
		t.Fatalf("Classify = %+v, %v", result, err)
	}
	if got := fake.ClassifyRequests(); len(got) != 1 || got[0].Model != "m" {
		t.Fatalf("recorded = %+v", got)
	}
	if _, err := llm.Classify(context.Background(), chatOnly{}, req); !errors.Is(err, llm.ErrUnsupported) {
		t.Fatalf("chat-only provider = %v", err)
	}
	if _, err := llm.Classify(context.Background(), nil, req); !errors.Is(err, llm.ErrUnsupported) {
		t.Fatalf("nil provider = %v", err)
	}
}

type chatOnly struct{ llm.Provider }

func (chatOnly) Name() string { return "chat-only" }

func TestClassifierModel(t *testing.T) {
	fake := llmtest.New()
	answer := &llm.ClassifyResult{Answers: map[string]llm.Answer{"approved": {Type: llm.QuestionBool, Probability: 0.5}}}
	for range 4 {
		fake.EnqueueClassifyResult(answer)
	}
	bound := llm.BoolQuestion("approved", "Does the user approve?", "", "")
	jev := llm.NewClassifierModel(fake, "typesafe/jev-1.14", llm.WithClassifierQuestions(bound))
	if jev.Model() != "typesafe/jev-1.14" {
		t.Fatalf("Model = %q", jev.Model())
	}

	type state struct {
		Message string `json:"message"`
	}
	if _, err := jev.Classify(context.Background(), state{Message: "thanks"}); err != nil {
		t.Fatal(err)
	}
	other := llm.BoolQuestion("polite", "Is it polite?", "", "")
	if _, err := jev.Classify(context.Background(), map[string]any{"message": "hi"}, other); err != nil {
		t.Fatal(err)
	}
	if _, err := jev.Classify(context.Background(), json.RawMessage(`{"message":"raw"}`)); err != nil {
		t.Fatal(err)
	}
	if _, err := jev.ClassifyRequest(context.Background(), &llm.ClassifyRequest{Model: "ignored", State: json.RawMessage(`{}`), Questions: []llm.Question{bound}}); err != nil {
		t.Fatal(err)
	}
	got := fake.ClassifyRequests()
	if len(got) != 4 {
		t.Fatalf("requests = %d", len(got))
	}
	for _, req := range got {
		if req.Model != "typesafe/jev-1.14" {
			t.Fatalf("model = %q", req.Model)
		}
	}
	if string(got[0].State) != `{"message":"thanks"}` || got[0].Questions[0].Name != "approved" {
		t.Fatalf("bound questions = %+v", got[0])
	}
	if got[1].Questions[0].Name != "polite" || len(got[1].Questions) != 1 {
		t.Fatalf("call questions replace the bound ones: %+v", got[1].Questions)
	}
	for _, bad := range []any{[]int{1}, "text", json.RawMessage(`[1]`), make(chan int)} {
		if _, err := jev.Classify(context.Background(), bad); !errors.Is(err, llm.ErrBadRequest) {
			t.Fatalf("state %T = %v", bad, err)
		}
	}

	// Concurrent use is safe: classification keeps no history.
	shared := llmtest.New()
	for range 16 {
		shared.EnqueueClassifyResult(answer)
	}
	model := llm.NewClassifierModel(shared, "m", llm.WithClassifierQuestions(bound))
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := model.Classify(context.Background(), map[string]int{"n": 1}); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
}

func TestClassifyJSONRoundTrip(t *testing.T) {
	req := validClassifyRequest()
	req.Images = []llm.ImagePart{{Data: []byte{0x89, 'P', 'N', 'G'}, MediaType: "image/png"}}
	data, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"questions":[{"name":"category","type":"choice"`) {
		t.Fatalf("request JSON = %s", data)
	}
	var back llm.ClassifyRequest
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(&back, req) {
		t.Fatalf("round trip:\n got %+v\nwant %+v", back, *req)
	}

	cost := 0.0000168
	result := &llm.ClassifyResult{Provider: "openrouter", Model: "typesafe/jev-1.14-20261014", ID: "gen-dec-1",
		Answers: map[string]llm.Answer{
			"satisfaction": {Type: llm.QuestionScore, Score: 1.83, LevelProbabilities: []float64{0.03, 0.1, 0.87}, Confidence: 0.75},
		},
		Usage: llm.Usage{InputTokens: 401, OutputTokens: 66, CostUSD: &cost, CostSource: llm.CostSourceNative},
	}
	data, err = json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	var backResult llm.ClassifyResult
	if err := json.Unmarshal(data, &backResult); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(&backResult, result) {
		t.Fatalf("result round trip:\n got %+v\nwant %+v", backResult, *result)
	}
}

func TestAnswerJSONKeepsZeros(t *testing.T) {
	for _, tc := range []struct {
		answer llm.Answer
		want   string
	}{
		{llm.Answer{Type: llm.QuestionBool}, `{"type":"bool","probability":0}`},
		{llm.Answer{Type: llm.QuestionScore}, `{"type":"score","score":0,"confidence":0}`},
		{llm.Answer{Type: llm.QuestionChoice, Choice: "a"}, `{"type":"choice","choice":"a","confidence":0}`},
	} {
		data, err := json.Marshal(tc.answer)
		if err != nil || string(data) != tc.want {
			t.Fatalf("Marshal(%+v) = %s, %v; want %s", tc.answer, data, err, tc.want)
		}
		var back llm.Answer
		if err := json.Unmarshal(data, &back); err != nil || !reflect.DeepEqual(back, tc.answer) {
			t.Fatalf("round trip = %+v, %v", back, err)
		}
	}
}

// TestWrapDecoratesClassify: Wrap of a classifier is a Classifier whose
// calls pass through Middleware.Classify, outermost first; Wrap of a
// chat-only provider is not a Classifier.
func TestWrapDecoratesClassify(t *testing.T) {
	fake := llmtest.New(llmtest.WithName("fake"))
	fake.EnqueueClassifyResult(&llm.ClassifyResult{Provider: "fake", Model: "m", Answers: map[string]llm.Answer{
		"approved": {Type: llm.QuestionBool, Probability: 0.9}}, Usage: llm.Usage{InputTokens: 10, TotalTokens: 10}})
	var order []string
	tag := func(name string) llm.Middleware {
		return llm.Middleware{Classify: func(next llm.ClassifyFunc) llm.ClassifyFunc {
			return func(ctx context.Context, req *llm.ClassifyRequest) (*llm.ClassifyResult, error) {
				order = append(order, name)
				return next(ctx, req)
			}
		}}
	}
	tracker := llm.NewUsageTracker()
	wrapped := llm.Wrap(llm.Wrap(fake, tag("inner")), tag("outer"), tracker.Middleware())
	if _, ok := wrapped.(llm.Classifier); !ok {
		t.Fatal("Wrap of a classifier is not a Classifier")
	}
	req := &llm.ClassifyRequest{Model: "m", State: json.RawMessage(`{}`), Questions: []llm.Question{llm.BoolQuestion("approved", "?", "", "")}}
	if _, err := llm.Classify(context.Background(), wrapped, req); err != nil {
		t.Fatal(err)
	}
	if strings.Join(order, ",") != "outer,inner" {
		t.Fatalf("order = %v", order)
	}
	if stats := tracker.Stats(); stats.Calls != 1 || stats.Usage.InputTokens != 10 {
		t.Fatalf("tracker = %+v", stats)
	}
	if _, ok := llm.Wrap(chatOnly{}, tag("x")).(llm.Classifier); ok {
		t.Fatal("Wrap of a chat-only provider is a Classifier")
	}

	// A billed failure is counted with its usage and as an error.
	fake.EnqueueClassifyError(llm.ErrServer, &llm.ClassifyResult{Provider: "fake", Model: "m", Usage: llm.Usage{InputTokens: 5, TotalTokens: 5}})
	if _, err := llm.Classify(context.Background(), wrapped, req); !errors.Is(err, llm.ErrServer) {
		t.Fatalf("err = %v", err)
	}
	if stats := tracker.Stats(); stats.Calls != 2 || stats.Errors != 1 || stats.Usage.InputTokens != 15 {
		t.Fatalf("tracker after error = %+v", stats)
	}
}
