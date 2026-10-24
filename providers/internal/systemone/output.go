package systemone

import (
	"encoding/json"
	"fmt"
	"strconv"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

type wireOutput struct {
	Model   string                     `json:"model"`
	ID      string                     `json:"id"`
	Answers map[string]json.RawMessage `json:"answers"`
	Usage   *wireUsage                 `json:"usage"`
}

type wireUsage struct {
	InputTokens  *int64   `json:"input_tokens"`
	OutputTokens *int64   `json:"output_tokens"`
	Cost         *float64 `json:"cost"`
}

type wireAnswer struct {
	Type          string             `json:"type"`
	Choice        *string            `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    *float64           `json:"confidence"`
	Score         *float64           `json:"score"`
	Noul          *float64           `json:"noul"`
}

// ParseOutput reads a System One output ({model?, id?, answers, usage}) for
// req. Usage is recorded before the answers are checked, so a billed
// response with malformed answers returns the result with its usage and an
// error wrapping ErrServer. Errors about the response are ProviderErrors
// (providerutil.ResponseError), keeping response text out of logs.
func ParseOutput(provider string, req *llm.ClassifyRequest, output json.RawMessage, table llm.PriceTable) (*llm.ClassifyResult, error) {
	var wire wireOutput
	if err := json.Unmarshal(output, &wire); err != nil {
		return nil, providerutil.ResponseError(provider, fmt.Errorf("unreadable classify response: %w", err))
	}
	result := &llm.ClassifyResult{Provider: provider, Model: req.Model, ID: wire.ID, Answers: map[string]llm.Answer{}}
	if wire.Model != "" {
		result.Model = wire.Model
	}
	result.Usage = usageFor(provider, req.Model, wire.Usage, table)
	var parseErr error
	for _, q := range req.Questions {
		raw, ok := wire.Answers[q.Name]
		if !ok {
			continue // CheckClassifyAnswers reports it
		}
		answer, err := parseAnswer(q, raw)
		if err != nil {
			if parseErr == nil {
				parseErr = providerutil.ResponseError(provider, fmt.Errorf("answer for %q: %w", q.Name, err))
			}
			continue
		}
		result.Answers[q.Name] = answer
	}
	checkErr := llm.CheckClassifyAnswers(req, result.Answers)
	if parseErr != nil {
		return result, parseErr
	}
	if checkErr != nil {
		return result, providerutil.ResponseError(provider, checkErr)
	}
	return result, nil
}

func parseAnswer(q llm.Question, raw json.RawMessage) (llm.Answer, error) {
	var wire wireAnswer
	if err := json.Unmarshal(raw, &wire); err != nil {
		return llm.Answer{}, err
	}
	switch wire.Type {
	case "choice":
		if wire.Choice == nil || wire.Confidence == nil {
			return llm.Answer{}, fmt.Errorf("choice answer lacks choice or confidence")
		}
		return llm.Answer{Type: llm.QuestionChoice, Choice: *wire.Choice, Probabilities: wire.Probabilities, Confidence: *wire.Confidence}, nil
	case "score":
		if wire.Score == nil || wire.Confidence == nil {
			return llm.Answer{}, fmt.Errorf("score answer lacks score or confidence")
		}
		levels, err := levelProbabilities(wire.Probabilities, len(q.Levels))
		if err != nil {
			return llm.Answer{}, err
		}
		return llm.Answer{Type: llm.QuestionScore, Score: *wire.Score, LevelProbabilities: levels, Confidence: *wire.Confidence}, nil
	case "noul":
		if wire.Noul == nil {
			return llm.Answer{}, fmt.Errorf("bool answer lacks noul")
		}
		return llm.Answer{Type: llm.QuestionBool, Probability: *wire.Noul}, nil
	default:
		return llm.Answer{}, fmt.Errorf("unexpected answer type %q", wire.Type)
	}
}

// levelProbabilities turns {"0": p, "1": p, ...} into a slice by level
// index. An absent (or empty) distribution is unknown (nil); a present one
// needs exactly one canonical key ("0", not "00") per level.
func levelProbabilities(byIndex map[string]float64, levels int) ([]float64, error) {
	if len(byIndex) == 0 {
		return nil, nil
	}
	if len(byIndex) != levels {
		return nil, fmt.Errorf("%d level probabilities for %d levels", len(byIndex), levels)
	}
	out := make([]float64, levels)
	for key, p := range byIndex {
		index, err := strconv.Atoi(key)
		if err != nil || index < 0 || index >= levels || strconv.Itoa(index) != key {
			return nil, fmt.Errorf("level probability key %q is not a level index in [0, %d]", key, levels-1)
		}
		out[index] = p
	}
	return out, nil
}

// usageFor maps the wire usage: tokens as reported, a native cost when the
// service reports one (OpenRouter), else an estimate from classifier
// pricing evidence (providerutil.EstimateClassifierCost).
func usageFor(provider, model string, wire *wireUsage, table llm.PriceTable) llm.Usage {
	var usage llm.Usage
	if wire == nil {
		return usage
	}
	if wire.InputTokens != nil && *wire.InputTokens > 0 {
		usage.InputTokens = *wire.InputTokens
	}
	if wire.OutputTokens != nil && *wire.OutputTokens > 0 {
		usage.OutputTokens = *wire.OutputTokens
	}
	usage.TotalTokens = usage.InputTokens + usage.OutputTokens
	if wire.Cost != nil && *wire.Cost >= 0 {
		cost := *wire.Cost
		usage.CostUSD = &cost
		usage.CostSource = llm.CostSourceNative
		return usage
	}
	return providerutil.EstimateClassifierCost(table, provider, model, usage)
}
