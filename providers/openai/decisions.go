package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"

	sdk "github.com/openai/openai-go/v3"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// The Decisions API's documented request limits, checked before I/O. The
// image limit is pi's.
const (
	maxDecisionImages    = 128
	maxDecisionQuestions = 400
	maxDecisionChoices   = 255
	maxDecisionLevels    = 10
)

var _ llm.Classifier = (*Provider)(nil)

// Classify answers req with OpenAI's Decisions API (POST /v1/decisions;
// GPT-6 Luna, FS §13C). Only API keys work: the endpoint rejects Sign in
// with ChatGPT tokens, so a stored OAuth credential fails with
// ErrUnsupported before any I/O. A bool question is sent as a predicate
// whose true/false meanings are appended to its instructions. A refusal
// fails with ErrContentFiltered beside the result, which keeps its usage
// and the answers to the questions that were not refused. Usage splits the
// cache reads and writes out of input_tokens, as chat does (FS §11). A 504
// gateway timeout is not retried.
func (p *Provider) Classify(ctx context.Context, req *llm.ClassifyRequest) (*llm.ClassifyResult, error) {
	start := time.Now()
	result, err := p.classify(ctx, req)
	if err != nil {
		if p.logger != nil {
			model := ""
			if req != nil {
				model = req.Model
			}
			p.logger.ErrorContext(ctx, "llm provider classify failed",
				slog.String("provider", providerName), slog.String("model", model),
				slog.Duration("duration", time.Since(start)), slog.String("error", llm.SafeError(err)))
		}
		return result, err
	}
	if p.logger != nil {
		p.logger.DebugContext(ctx, "llm provider classify",
			slog.String("provider", providerName), slog.String("model", result.Model),
			slog.Duration("duration", time.Since(start)), slog.Int("questions", len(req.Questions)),
			slog.Int64("input_tokens", result.Usage.InputTokens))
	}
	return result, nil
}

func (p *Provider) classify(ctx context.Context, req *llm.ClassifyRequest) (*llm.ClassifyResult, error) {
	if err := llm.ValidateClassifyRequest(req); err != nil {
		return nil, err
	}
	// The timeout covers the credential read too, as in Chat.
	ctx, cancel := p.contextWithTimeout(ctx)
	defer cancel()
	ctx, signIn, err := p.signInWithChatGPT(ctx)
	if err != nil {
		return nil, err
	}
	if signIn {
		return nil, fmt.Errorf("%w: OpenAI's Decisions API rejects Sign in with ChatGPT credentials; classify with an OpenAI API key", llm.ErrUnsupported)
	}
	if err := checkDecisionLimits(req); err != nil {
		return nil, err
	}
	input, err := decisionInput(req)
	if err != nil {
		return nil, err
	}
	questions := make([]decisionQuestion, len(req.Questions))
	for i, q := range req.Questions {
		questions[i] = wireDecisionQuestion(q)
	}
	var body []byte
	params := map[string]any{"model": req.Model, "input": input, "questions": questions}
	if err := p.client.Post(ctx, "decisions", params, &body); err != nil {
		return nil, decisionError(err)
	}
	return parseDecisions(req, body, p.priceTable)
}

type decisionQuestion struct {
	Type         string           `json:"type"`
	Name         string           `json:"name"`
	Instructions string           `json:"instructions"`
	Choices      []decisionChoice `json:"choices,omitempty"`
	Levels       []decisionLevel  `json:"levels,omitempty"`
}

type decisionChoice struct {
	Value       string `json:"value"`
	Description string `json:"description,omitempty"`
}

type decisionLevel struct {
	Label string `json:"label"`
}

func wireDecisionQuestion(q llm.Question) decisionQuestion {
	switch q.Type {
	case llm.QuestionChoice:
		choices := make([]decisionChoice, len(q.Choices))
		for i, choice := range q.Choices {
			choices[i] = decisionChoice{Value: choice.Value, Description: choice.Description}
		}
		return decisionQuestion{Type: "choice", Name: q.Name, Instructions: q.Instructions, Choices: choices}
	case llm.QuestionScore:
		levels := make([]decisionLevel, len(q.Levels))
		for i, level := range q.Levels {
			levels[i] = decisionLevel{Label: level}
		}
		return decisionQuestion{Type: "score", Name: q.Name, Instructions: q.Instructions, Levels: levels}
	default:
		return decisionQuestion{Type: "predicate", Name: q.Name, Instructions: predicateInstructions(q)}
	}
}

// predicateInstructions appends the meanings of true and false, which a
// predicate has no field for, as pi does.
func predicateInstructions(q llm.Question) string {
	var meanings []string
	if q.True != "" {
		meanings = append(meanings, "True means: "+q.True)
	}
	if q.False != "" {
		meanings = append(meanings, "False means: "+q.False)
	}
	if len(meanings) == 0 {
		return q.Instructions
	}
	return q.Instructions + "\n\n" + strings.Join(meanings, "\n")
}

// decisionInput is the state as JSON text, or with images one user message
// with the state followed by the images as data URLs.
func decisionInput(req *llm.ClassifyRequest) (any, error) {
	state := string(req.State)
	if len(req.Images) == 0 {
		return state, nil
	}
	if len(req.Images) > maxDecisionImages {
		return nil, fmt.Errorf("%w: OpenAI Decisions accepts at most %d images, got %d", llm.ErrBadRequest, maxDecisionImages, len(req.Images))
	}
	content := []map[string]string{{"type": "input_text", "text": state}}
	for _, image := range req.Images {
		content = append(content, map[string]string{
			"type":      "input_image",
			"image_url": "data:" + image.MediaType + ";base64," + base64.StdEncoding.EncodeToString(image.Data),
		})
	}
	return []map[string]any{{"role": "user", "content": content}}, nil
}

func checkDecisionLimits(req *llm.ClassifyRequest) error {
	if len(req.Questions) > maxDecisionQuestions {
		return fmt.Errorf("%w: OpenAI Decisions accepts at most %d questions, got %d", llm.ErrBadRequest, maxDecisionQuestions, len(req.Questions))
	}
	for _, q := range req.Questions {
		if len(q.Choices) > maxDecisionChoices {
			return fmt.Errorf("%w: OpenAI Decisions accepts at most %d choices, question %q has %d", llm.ErrBadRequest, maxDecisionChoices, q.Name, len(q.Choices))
		}
		if len(q.Levels) > maxDecisionLevels {
			return fmt.Errorf("%w: OpenAI Decisions accepts at most %d score levels, question %q has %d", llm.ErrBadRequest, maxDecisionLevels, q.Name, len(q.Levels))
		}
	}
	return nil
}

func decisionError(err error) error {
	var apiErr *sdk.Error
	if errors.As(err, &apiErr) && apiErr.StatusCode == 504 {
		mapped := mapError(err)
		return fmt.Errorf("%w (the Decisions gateway timed out; very large inputs can exceed its time limit)", mapped)
	}
	return mapError(err)
}

// decisionsResponse keeps the answers raw, so one malformed answer does
// not lose the usage or the other answers.
type decisionsResponse struct {
	ID      string          `json:"id"`
	Model   string          `json:"model"`
	Answers json.RawMessage `json:"answers"`
	Usage   *struct {
		InputTokens        int64 `json:"input_tokens"`
		InputTokensDetails struct {
			CachedTokens     int64 `json:"cached_tokens"`
			CacheWriteTokens int64 `json:"cache_write_tokens"`
		} `json:"input_tokens_details"`
		OutputTokens        int64 `json:"output_tokens"`
		OutputTokensDetails struct {
			ReasoningTokens int64 `json:"reasoning_tokens"`
		} `json:"output_tokens_details"`
		TotalTokens int64 `json:"total_tokens"`
	} `json:"usage"`
}

type decisionAnswer struct {
	Name          string  `json:"name"`
	Type          string  `json:"type"`
	Choice        *string `json:"choice"`
	Probabilities []struct {
		Value       string  `json:"value"`
		Probability float64 `json:"probability"`
	} `json:"probabilities"`
	Confidence  *float64 `json:"confidence"`
	Score       *float64 `json:"score"`
	Probability *float64 `json:"probability"`
}

// parseDecisions maps the Decisions response. Usage is recorded before the
// answers are read, and each requested answer is decoded on its own;
// unrequested answers are ignored. Precedence: a refusal of a requested
// question fails with ErrContentFiltered, else a malformed, missing or
// invalid answer with ErrServer; either way the valid answers stay in the
// result. Errors about the response are ProviderErrors, keeping response
// text out of logs.
func parseDecisions(req *llm.ClassifyRequest, body []byte, table llm.PriceTable) (*llm.ClassifyResult, error) {
	var wire decisionsResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, providerutil.ResponseError(providerName, fmt.Errorf("unreadable Decisions response: %w", err))
	}
	result := &llm.ClassifyResult{Provider: providerName, Model: req.Model, ID: wire.ID, Answers: map[string]llm.Answer{}}
	if wire.Model != "" {
		result.Model = wire.Model
	}
	if wire.Usage != nil {
		result.Usage = providerutil.EstimateClassifierCost(table, providerName, req.Model, decisionUsage(wire))
	}
	var malformed error
	var raws []json.RawMessage
	if err := json.Unmarshal(wire.Answers, &raws); err != nil && len(wire.Answers) > 0 {
		malformed = providerutil.ResponseError(providerName, fmt.Errorf("unreadable Decisions answers: %w", err))
	}
	// An answer whose name or type cannot be read is unidentifiable; a
	// requested one is then reported missing.
	byName := make(map[string]json.RawMessage, len(raws))
	types := make(map[string]string, len(raws))
	for _, raw := range raws {
		var head struct {
			Name string `json:"name"`
			Type string `json:"type"`
		}
		if json.Unmarshal(raw, &head) == nil {
			byName[head.Name], types[head.Name] = raw, head.Type
		}
	}
	var refused []string
	for _, q := range req.Questions {
		raw, ok := byName[q.Name]
		if !ok {
			continue // CheckClassifyAnswers reports it
		}
		if types[q.Name] == "refusal" {
			refused = append(refused, strconv.Quote(q.Name))
			continue
		}
		mapped, err := mapDecisionAnswer(raw)
		if err != nil {
			if malformed == nil {
				malformed = providerutil.ResponseError(providerName, fmt.Errorf("Decisions answer for %q: %w", q.Name, err))
			}
			continue
		}
		result.Answers[q.Name] = mapped
	}
	checkErr := llm.CheckClassifyAnswers(req, result.Answers)
	switch {
	case len(refused) > 0:
		return result, &llm.ProviderError{Provider: providerName, Message: "OpenAI Decisions refused to answer " + strings.Join(refused, ", "), Kind: llm.ErrContentFiltered}
	case malformed != nil:
		return result, malformed
	case checkErr != nil:
		return result, providerutil.ResponseError(providerName, checkErr)
	}
	return result, nil
}

// decisionUsage normalizes the wire usage: OpenAI reports cache reads and
// writes as subsets of input_tokens, and the additive contract (FS §11)
// wants InputTokens to exclude both, as the Responses adapter does.
func decisionUsage(wire decisionsResponse) llm.Usage {
	u := wire.Usage
	cacheRead, cacheWrite := max(u.InputTokensDetails.CachedTokens, 0), max(u.InputTokensDetails.CacheWriteTokens, 0)
	input := max(u.InputTokens, 0)
	if cacheRead > 0 && input >= cacheRead {
		input -= cacheRead
	}
	if cacheWrite > 0 && input >= cacheWrite {
		input -= cacheWrite
	}
	usage := llm.Usage{
		InputTokens:      input,
		CacheReadTokens:  cacheRead,
		CacheWriteTokens: cacheWrite,
		OutputTokens:     max(u.OutputTokens, 0),
		ReasoningTokens:  max(u.OutputTokensDetails.ReasoningTokens, 0),
		TotalTokens:      max(u.TotalTokens, 0),
	}
	if usage.TotalTokens == 0 {
		usage.TotalTokens = usage.InputTokens + usage.CacheReadTokens + usage.CacheWriteTokens + usage.OutputTokens
	}
	return usage
}

func mapDecisionAnswer(raw json.RawMessage) (llm.Answer, error) {
	var answer decisionAnswer
	if err := json.Unmarshal(raw, &answer); err != nil {
		return llm.Answer{}, err
	}
	switch answer.Type {
	case "choice":
		if answer.Choice == nil || answer.Confidence == nil {
			return llm.Answer{}, errors.New("choice answer lacks choice or confidence")
		}
		probabilities := make(map[string]float64, len(answer.Probabilities))
		for _, p := range answer.Probabilities {
			probabilities[p.Value] = p.Probability
		}
		return llm.Answer{Type: llm.QuestionChoice, Choice: *answer.Choice, Probabilities: probabilities, Confidence: *answer.Confidence}, nil
	case "score":
		if answer.Score == nil || answer.Confidence == nil {
			return llm.Answer{}, errors.New("score answer lacks score or confidence")
		}
		return llm.Answer{Type: llm.QuestionScore, Score: *answer.Score, Confidence: *answer.Confidence}, nil
	case "predicate":
		if answer.Probability == nil {
			return llm.Answer{}, errors.New("predicate answer lacks probability")
		}
		return llm.Answer{Type: llm.QuestionBool, Probability: *answer.Probability}, nil
	}
	return llm.Answer{}, fmt.Errorf("unexpected answer type %q", answer.Type)
}
