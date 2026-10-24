package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
)

// Classifier is an optional Provider extension for classifier (decision)
// models: models that answer typed questions about structured state instead
// of generating text (FS §13C). It is discovered by type assertion; see
// Classify.
type Classifier interface {
	Classify(ctx context.Context, req *ClassifyRequest) (*ClassifyResult, error)
}

// QuestionType is the kind of a classifier Question.
type QuestionType string

const (
	// QuestionChoice picks one of several labelled options.
	QuestionChoice QuestionType = "choice"
	// QuestionScore places the state on an ordered scale of levels.
	QuestionScore QuestionType = "score"
	// QuestionBool answers true or false.
	QuestionBool QuestionType = "bool"
)

// Question is one named question of a ClassifyRequest. Build it with
// ChoiceQuestion, ScoreQuestion or BoolQuestion.
type Question struct {
	// Name identifies the question; its answer is keyed by it.
	Name string       `json:"name"`
	Type QuestionType `json:"type"`
	// Instructions say what to decide.
	Instructions string `json:"instructions"`
	// Choices are the options of a choice question.
	Choices []Choice `json:"choices,omitempty"`
	// Levels are the ordered levels of a score question, lowest first.
	Levels []string `json:"levels,omitempty"`
	// True and False say what each answer to a bool question means.
	True  string `json:"true,omitempty"`
	False string `json:"false,omitempty"`
}

// Choice is one option of a choice question.
type Choice struct {
	Value       string `json:"value"`
	Description string `json:"description,omitempty"`
}

// ChoiceQuestion returns a question answered by one of choices.
func ChoiceQuestion(name, instructions string, choices ...Choice) Question {
	return Question{Name: name, Type: QuestionChoice, Instructions: instructions, Choices: append([]Choice(nil), choices...)}
}

// ScoreQuestion returns a question answered on the ordered scale levels,
// lowest first.
func ScoreQuestion(name, instructions string, levels ...string) Question {
	return Question{Name: name, Type: QuestionScore, Instructions: instructions, Levels: append([]string(nil), levels...)}
}

// BoolQuestion returns a true/false question; ifTrue and ifFalse say what
// each answer means and may be empty.
func BoolQuestion(name, instructions, ifTrue, ifFalse string) Question {
	return Question{Name: name, Type: QuestionBool, Instructions: instructions, True: ifTrue, False: ifFalse}
}

// ClassifyRequest asks a classifier model the Questions about State.
type ClassifyRequest struct {
	Model string `json:"model"`
	// State is the JSON object the questions are about.
	State json.RawMessage `json:"state"`
	// Images are judged together with State. They need inline Data and a
	// MediaType; the model must accept image input.
	Images    []ImagePart `json:"images,omitempty"`
	Questions []Question  `json:"questions"`
	// ProviderOptions carries provider-specific extensions; it is not
	// serialized.
	ProviderOptions ProviderOptions `json:"-"`
}

// ClassifyResult holds one Answer per requested question.
type ClassifyResult struct {
	Provider string `json:"provider"`
	// Model is the model that answered: the resolved model (for example a
	// dated snapshot) when the service reports one, else the requested one.
	Model string `json:"model"`
	// ID is the service's request ID, when it returns one.
	ID string `json:"id,omitempty"`
	// Answers are keyed by question name.
	Answers map[string]Answer `json:"answers"`
	Usage   Usage             `json:"usage"`
}

// Answer is the answer to one Question.
type Answer struct {
	Type QuestionType `json:"type"`
	// Choice is the chosen value of a choice question, and Probabilities
	// the probability of every value.
	Choice        string             `json:"choice,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	// Score is a score question's expected level index (0 is the first
	// level), fractional when the model is uncertain. LevelProbabilities
	// holds each level's probability, by index, when the service reports
	// them.
	Score              float64   `json:"score,omitempty"`
	LevelProbabilities []float64 `json:"level_probabilities,omitempty"`
	// Confidence is reported for choice and score questions.
	Confidence float64 `json:"confidence,omitempty"`
	// Probability is a bool question's probability of true.
	Probability float64 `json:"probability,omitempty"`
}

// MarshalJSON writes the fields of the answer's type, zeros included: a
// bool probability of 0 is an answer, not an absent one.
func (a Answer) MarshalJSON() ([]byte, error) {
	switch a.Type {
	case QuestionChoice:
		return json.Marshal(struct {
			Type          QuestionType       `json:"type"`
			Choice        string             `json:"choice"`
			Probabilities map[string]float64 `json:"probabilities,omitempty"`
			Confidence    float64            `json:"confidence"`
		}{a.Type, a.Choice, a.Probabilities, a.Confidence})
	case QuestionScore:
		return json.Marshal(struct {
			Type               QuestionType `json:"type"`
			Score              float64      `json:"score"`
			LevelProbabilities []float64    `json:"level_probabilities,omitempty"`
			Confidence         float64      `json:"confidence"`
		}{a.Type, a.Score, a.LevelProbabilities, a.Confidence})
	case QuestionBool:
		return json.Marshal(struct {
			Type        QuestionType `json:"type"`
			Probability float64      `json:"probability"`
		}{a.Type, a.Probability})
	}
	type plain Answer
	return json.Marshal(plain(a))
}

// Classify asks the classifier on p, or on the providers it wraps
// (following Unwrap() Provider, as CountTokens does), to answer req. A
// provider without one fails with ErrUnsupported. Chat middleware does not
// observe the call.
func Classify(ctx context.Context, p Provider, req *ClassifyRequest) (*ClassifyResult, error) {
	classifier, ok := FindClassifier(p)
	if !ok {
		name := "<nil>"
		if p != nil {
			name = p.Name()
		}
		return nil, fmt.Errorf("%w: provider %s does not classify", ErrUnsupported, name)
	}
	return classifier.Classify(ctx, req)
}

// FindClassifier returns the Classifier on p or on the providers it wraps.
func FindClassifier(p Provider) (Classifier, bool) {
	for depth := 0; p != nil && depth < 32; depth++ {
		if classifier, ok := p.(Classifier); ok {
			return classifier, true
		}
		unwrapper, ok := p.(interface{ Unwrap() Provider })
		if !ok {
			return nil, false
		}
		p = unwrapper.Unwrap()
	}
	return nil, false
}

// ValidateClassifyRequest checks req before any network I/O: a model, State
// a JSON object, at least one question with a unique non-empty name, each
// question well formed for its type (two or more unique choices; two or
// more levels), and images with inline data and a media type. Failures
// wrap ErrBadRequest.
func ValidateClassifyRequest(req *ClassifyRequest) error {
	if req == nil {
		return fmt.Errorf("%w: nil classify request", ErrBadRequest)
	}
	if req.Model == "" {
		return fmt.Errorf("%w: classify request needs a model", ErrBadRequest)
	}
	if !isJSONObject(req.State) {
		return fmt.Errorf("%w: classify state must be a JSON object", ErrBadRequest)
	}
	if len(req.Questions) == 0 {
		return fmt.Errorf("%w: classify request needs at least one question", ErrBadRequest)
	}
	names := make(map[string]struct{}, len(req.Questions))
	for i, q := range req.Questions {
		if q.Name == "" {
			return fmt.Errorf("%w: classify question %d has no name", ErrBadRequest, i)
		}
		if _, dup := names[q.Name]; dup {
			return fmt.Errorf("%w: duplicate classify question name %q", ErrBadRequest, q.Name)
		}
		names[q.Name] = struct{}{}
		if err := validateQuestion(q); err != nil {
			return err
		}
	}
	for i, image := range req.Images {
		if len(image.Data) == 0 || image.MediaType == "" {
			return fmt.Errorf("%w: classify image %d needs inline data and a media type", ErrBadRequest, i)
		}
	}
	return nil
}

func validateQuestion(q Question) error {
	switch q.Type {
	case QuestionChoice:
		if len(q.Choices) < 2 {
			return fmt.Errorf("%w: choice question %q needs at least two choices", ErrBadRequest, q.Name)
		}
		values := make(map[string]struct{}, len(q.Choices))
		for _, choice := range q.Choices {
			if choice.Value == "" {
				return fmt.Errorf("%w: choice question %q has an empty value", ErrBadRequest, q.Name)
			}
			if _, dup := values[choice.Value]; dup {
				return fmt.Errorf("%w: choice question %q repeats value %q", ErrBadRequest, q.Name, choice.Value)
			}
			values[choice.Value] = struct{}{}
		}
	case QuestionScore:
		if len(q.Levels) < 2 {
			return fmt.Errorf("%w: score question %q needs at least two levels", ErrBadRequest, q.Name)
		}
		for _, level := range q.Levels {
			if level == "" {
				return fmt.Errorf("%w: score question %q has an empty level", ErrBadRequest, q.Name)
			}
		}
	case QuestionBool:
	default:
		return fmt.Errorf("%w: question %q has unknown type %q", ErrBadRequest, q.Name, q.Type)
	}
	return nil
}

// CheckClassifyAnswers verifies that answers holds a valid answer for every
// question of req: of the requested type, with finite numbers, every
// probability and confidence within [0, 1], a choice among the declared
// values (probabilities only for declared values), a score within the
// level indices [0, len(Levels)-1], and level probabilities, when present,
// one per level. Answers to questions req does not ask are removed, and so
// is every invalid answer, so the answers left can be trusted. A
// Classifier implementation calls it on a billed response and, when it
// fails, returns the result (with its usage and the remaining answers)
// together with the error, which wraps ErrServer and names the first
// missing or invalid answer.
func CheckClassifyAnswers(req *ClassifyRequest, answers map[string]Answer) error {
	asked := make(map[string]bool, len(req.Questions))
	var first error
	for _, q := range req.Questions {
		asked[q.Name] = true
		answer, ok := answers[q.Name]
		if !ok {
			if first == nil {
				first = fmt.Errorf("%w: no answer for classify question %q", ErrServer, q.Name)
			}
			continue
		}
		if err := checkAnswer(q, answer); err != nil {
			delete(answers, q.Name)
			if first == nil {
				first = fmt.Errorf("%w: classify question %q: %v", ErrServer, q.Name, err)
			}
		}
	}
	for name := range answers {
		if !asked[name] {
			delete(answers, name)
		}
	}
	return first
}

// probabilitySlack absorbs float rounding in reported probabilities.
const probabilitySlack = 1e-6

func checkAnswer(q Question, answer Answer) error {
	if answer.Type != q.Type {
		return fmt.Errorf("got a %s answer, want %s", answer.Type, q.Type)
	}
	unit := func(name string, n float64) error {
		if math.IsNaN(n) || math.IsInf(n, 0) {
			return fmt.Errorf("%s is not a finite number", name)
		}
		if n < -probabilitySlack || n > 1+probabilitySlack {
			return fmt.Errorf("%s %v is outside [0, 1]", name, n)
		}
		return nil
	}
	switch q.Type {
	case QuestionChoice:
		declared := false
		for _, c := range q.Choices {
			declared = declared || c.Value == answer.Choice
		}
		if !declared {
			return fmt.Errorf("choice %q is not one of the declared values", answer.Choice)
		}
		for value, p := range answer.Probabilities {
			known := false
			for _, c := range q.Choices {
				known = known || c.Value == value
			}
			if !known {
				return fmt.Errorf("probability for undeclared value %q", value)
			}
			if err := unit("probability of "+strconv.Quote(value), p); err != nil {
				return err
			}
		}
		return unit("confidence", answer.Confidence)
	case QuestionScore:
		if math.IsNaN(answer.Score) || math.IsInf(answer.Score, 0) {
			return errors.New("score is not a finite number")
		}
		if top := float64(len(q.Levels) - 1); len(q.Levels) > 0 && (answer.Score < -probabilitySlack || answer.Score > top+probabilitySlack) {
			return fmt.Errorf("score %v is outside the level indices [0, %v]", answer.Score, top)
		}
		if answer.LevelProbabilities != nil && len(answer.LevelProbabilities) != len(q.Levels) {
			return fmt.Errorf("%d level probabilities for %d levels", len(answer.LevelProbabilities), len(q.Levels))
		}
		for i, p := range answer.LevelProbabilities {
			if err := unit("probability of level "+strconv.Itoa(i), p); err != nil {
				return err
			}
		}
		return unit("confidence", answer.Confidence)
	default:
		return unit("probability", answer.Probability)
	}
}

func isJSONObject(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return false
	}
	var object map[string]json.RawMessage
	return json.Unmarshal(trimmed, &object) == nil
}

// ClassifierModel binds a provider and a classifier model, optionally with
// default questions (FS §13C). It is immutable and safe for concurrent use.
type ClassifierModel struct {
	provider        Provider
	model           string
	questions       []Question
	providerOptions ProviderOptions
}

// ClassifierOption configures a ClassifierModel.
type ClassifierOption func(*ClassifierModel)

// WithClassifierQuestions sets the questions Classify asks when it is given
// none.
func WithClassifierQuestions(questions ...Question) ClassifierOption {
	owned := cloneQuestions(questions)
	return func(m *ClassifierModel) { m.questions = owned }
}

// WithClassifierProviderOptions sets the ProviderOptions of every request.
func WithClassifierProviderOptions(opts ProviderOptions) ClassifierOption {
	return func(m *ClassifierModel) { m.providerOptions = opts }
}

// NewClassifierModel binds p and model.
func NewClassifierModel(p Provider, model string, opts ...ClassifierOption) *ClassifierModel {
	m := &ClassifierModel{provider: p, model: model}
	for _, opt := range opts {
		if opt != nil {
			opt(m)
		}
	}
	return m
}

// Model returns the bound model ID.
func (m *ClassifierModel) Model() string { return m.model }

// Classify asks questions about state, or the bound questions when none are
// given. state is any value that marshals to a JSON object (a struct, a
// map, json.RawMessage or []byte of JSON).
func (m *ClassifierModel) Classify(ctx context.Context, state any, questions ...Question) (*ClassifyResult, error) {
	raw, err := stateJSON(state)
	if err != nil {
		return nil, err
	}
	if len(questions) == 0 {
		questions = m.questions
	}
	return m.ClassifyRequest(ctx, &ClassifyRequest{State: raw, Questions: cloneQuestions(questions)})
}

// ClassifyRequest sends req with the bound model, and the bound provider
// options when req has none. req is not modified.
func (m *ClassifierModel) ClassifyRequest(ctx context.Context, req *ClassifyRequest) (*ClassifyResult, error) {
	if req == nil {
		return nil, fmt.Errorf("%w: nil classify request", ErrBadRequest)
	}
	bound := *req
	bound.Model = m.model
	if bound.ProviderOptions == nil {
		bound.ProviderOptions = m.providerOptions
	}
	return Classify(ctx, m.provider, &bound)
}

func stateJSON(state any) (json.RawMessage, error) {
	var raw []byte
	switch v := state.(type) {
	case json.RawMessage:
		raw = v
	case []byte:
		raw = v
	default:
		encoded, err := json.Marshal(state)
		if err != nil {
			return nil, fmt.Errorf("%w: encode classify state: %v", ErrBadRequest, err)
		}
		raw = encoded
	}
	if !isJSONObject(raw) {
		return nil, fmt.Errorf("%w: classify state must encode to a JSON object", ErrBadRequest)
	}
	return append(json.RawMessage(nil), raw...), nil
}

func cloneQuestions(questions []Question) []Question {
	if questions == nil {
		return nil
	}
	out := make([]Question, len(questions))
	for i, q := range questions {
		q.Choices = append([]Choice(nil), q.Choices...)
		q.Levels = append([]string(nil), q.Levels...)
		out[i] = q
	}
	return out
}
