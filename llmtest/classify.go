package llmtest

import (
	"context"
	"encoding/json"
	"fmt"

	llm "github.com/pkieltyka/go-llm"
)

// Classify implements llm.Classifier from a separate script: each call
// validates the request (llm.ValidateClassifyRequest), records it, and
// returns the next enqueued result or error. A scripted result is returned
// as given, so tests can script malformed answers; an exhausted script
// fails with ErrBadRequest.
func (p *Provider) Classify(ctx context.Context, req *llm.ClassifyRequest) (*llm.ClassifyResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := llm.ValidateClassifyRequest(req); err != nil {
		return nil, err
	}
	p.mu.Lock()
	p.classifyRequests = append(p.classifyRequests, cloneClassifyRequest(req))
	if len(p.classifySteps) == 0 {
		p.mu.Unlock()
		return nil, fmt.Errorf("%w: llmtest exhausted classify script", llm.ErrBadRequest)
	}
	st := p.classifySteps[0]
	p.classifySteps = p.classifySteps[1:]
	p.mu.Unlock()
	if st.err != nil {
		return cloneClassifyResult(st.result), st.err
	}
	return cloneClassifyResult(st.result), nil
}

type classifyStep struct {
	result *llm.ClassifyResult
	err    error
}

// EnqueueClassifyResult appends a canned Classify result.
func (p *Provider) EnqueueClassifyResult(result *llm.ClassifyResult) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.classifySteps = append(p.classifySteps, classifyStep{result: cloneClassifyResult(result)})
}

// EnqueueClassifyError appends a Classify failure. A non-nil result is
// returned beside the error, as a billed response with malformed answers is.
func (p *Provider) EnqueueClassifyError(err error, result *llm.ClassifyResult) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.classifySteps = append(p.classifySteps, classifyStep{result: cloneClassifyResult(result), err: err})
}

// ClassifyRequests returns defensive copies of the recorded Classify
// requests. ProviderOptions is shallow-copied, as in Requests.
func (p *Provider) ClassifyRequests() []*llm.ClassifyRequest {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]*llm.ClassifyRequest, len(p.classifyRequests))
	for i, req := range p.classifyRequests {
		out[i] = cloneClassifyRequest(req)
	}
	return out
}

func cloneClassifyRequest(req *llm.ClassifyRequest) *llm.ClassifyRequest {
	if req == nil {
		return nil
	}
	copied := *req
	copied.State = append(json.RawMessage(nil), req.State...)
	copied.Images = make([]llm.ImagePart, len(req.Images))
	for i, image := range req.Images {
		image.Data = append([]byte(nil), image.Data...)
		copied.Images[i] = image
	}
	if req.Images == nil {
		copied.Images = nil
	}
	copied.Questions = make([]llm.Question, len(req.Questions))
	for i, q := range req.Questions {
		q.Choices = append([]llm.Choice(nil), q.Choices...)
		q.Levels = append([]string(nil), q.Levels...)
		copied.Questions[i] = q
	}
	return &copied
}

func cloneClassifyResult(result *llm.ClassifyResult) *llm.ClassifyResult {
	if result == nil {
		return nil
	}
	copied := *result
	copied.Usage = cloneUsage(result.Usage)
	if result.Answers != nil {
		copied.Answers = make(map[string]llm.Answer, len(result.Answers))
		for name, answer := range result.Answers {
			if answer.Probabilities != nil {
				probabilities := make(map[string]float64, len(answer.Probabilities))
				for value, p := range answer.Probabilities {
					probabilities[value] = p
				}
				answer.Probabilities = probabilities
			}
			answer.LevelProbabilities = append([]float64(nil), answer.LevelProbabilities...)
			copied.Answers[name] = answer
		}
	}
	return &copied
}
