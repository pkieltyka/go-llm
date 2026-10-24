package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	llm "github.com/pkieltyka/go-llm"
)

// runClassify asks a classifier model the questions about the state and
// prints the answers (FS §13C, §19).
func (a app) runClassify(ctx context.Context, cfg classifyConfig) error {
	if cfg.model == "" {
		return fmt.Errorf("%w: classify needs -m/--model", llm.ErrBadRequest)
	}
	if cfg.questionsPath == "" {
		return fmt.Errorf("%w: classify needs --questions (a JSON file)", llm.ErrBadRequest)
	}
	state, err := a.classifyState(ctx, cfg.state)
	if err != nil {
		return err
	}
	data, err := os.ReadFile(cfg.questionsPath)
	if err != nil {
		return fmt.Errorf("read questions: %w", err)
	}
	questions, err := parseQuestions(data)
	if err != nil {
		return err
	}
	images := make([]llm.ImagePart, 0, len(cfg.imagePaths))
	for _, path := range cfg.imagePaths {
		image, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read image %q: %w", path, err)
		}
		images = append(images, llm.ImagePart{Data: image, MediaType: mediaType(path, image)})
	}
	pcfg, err := a.withAuthFile(providerConfigFromClassify(cfg, a.stderr))
	if err != nil {
		return err
	}
	provider, err := a.providerFactory(ctx, pcfg)
	if err != nil {
		return err
	}
	req := &llm.ClassifyRequest{Model: cfg.model, State: state, Questions: questions}
	if len(images) > 0 {
		req.Images = images
	}
	result, classifyErr := llm.Classify(ctx, provider, req)
	if result == nil {
		return classifyErr
	}
	// A billed call that failed (a refusal, a malformed answer) still
	// prints its valid answers and usage before the error exits nonzero.
	if err := a.printClassifyResult(cfg, questions, result); err != nil {
		return err
	}
	return classifyErr
}

func (a app) printClassifyResult(cfg classifyConfig, questions []llm.Question, result *llm.ClassifyResult) error {
	if cfg.usage {
		if err := printUsage(a.stderr, result.Usage); err != nil {
			return err
		}
	}
	if cfg.jsonOutput {
		encoded, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			return err
		}
		_, err = fmt.Fprintln(a.stdout, string(encoded))
		return err
	}
	for _, q := range questions {
		line := q.Name + ": no valid answer"
		if answer, ok := result.Answers[q.Name]; ok {
			line = formatAnswer(q, answer)
		}
		if _, err := fmt.Fprintln(a.stdout, line); err != nil {
			return fmt.Errorf("write answers: %w", err)
		}
	}
	return nil
}

// classifyState reads the state: inline JSON (starting with "{"), a file,
// "-" for stdin, or piped stdin when --state is absent.
func (a app) classifyState(ctx context.Context, value string) (json.RawMessage, error) {
	var raw []byte
	switch {
	case strings.HasPrefix(strings.TrimSpace(value), "{"):
		raw = []byte(value)
	case value == "-" || (value == "" && a.stdin != nil && shouldReadStdin(a.stdin)):
		text, err := readAllContext(ctx, a.stdin)
		if err != nil {
			return nil, err
		}
		raw = []byte(text)
	case value != "":
		data, err := os.ReadFile(value)
		if err != nil {
			return nil, fmt.Errorf("read state: %w", err)
		}
		raw = data
	default:
		return nil, fmt.Errorf("%w: classify needs --state (inline JSON, a file, or - for stdin)", llm.ErrBadRequest)
	}
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 && value == "" {
		return nil, fmt.Errorf("%w: classify needs --state (inline JSON, a file, or - for stdin)", llm.ErrBadRequest)
	}
	var object map[string]json.RawMessage
	if len(raw) == 0 || raw[0] != '{' || json.Unmarshal(raw, &object) != nil {
		return nil, fmt.Errorf("%w: classify state must be a JSON object", llm.ErrBadRequest)
	}
	return json.RawMessage(raw), nil
}

// parseQuestions reads questions in either form, keeping their order:
//   - pi's object form, {"<name>": {"type": "choice"|"score"|"bool",
//     "instructions": "...", "criteria": ...}}, where choice criteria map
//     values to descriptions, score criteria list the levels and bool
//     criteria are {"true": "...", "false": "..."};
//   - go-llm's canonical list, [{"name", "type", "instructions",
//     "choices", "levels", "true", "false"}].
func parseQuestions(data []byte) ([]llm.Question, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) > 0 && trimmed[0] == '[' {
		var questions []llm.Question
		if err := json.Unmarshal(trimmed, &questions); err != nil {
			return nil, fmt.Errorf("%w: parse questions: %v", llm.ErrBadRequest, err)
		}
		return questions, nil
	}
	members, err := orderedMembers(trimmed)
	if err != nil {
		return nil, fmt.Errorf("%w: parse questions: %v", llm.ErrBadRequest, err)
	}
	questions := make([]llm.Question, 0, len(members))
	for _, m := range members {
		var wire struct {
			Type         string          `json:"type"`
			Instructions string          `json:"instructions"`
			Criteria     json.RawMessage `json:"criteria"`
		}
		if err := json.Unmarshal(m.value, &wire); err != nil {
			return nil, fmt.Errorf("%w: question %q: %v", llm.ErrBadRequest, m.key, err)
		}
		q := llm.Question{Name: m.key, Type: llm.QuestionType(wire.Type), Instructions: wire.Instructions}
		switch q.Type {
		case llm.QuestionChoice:
			choices, err := orderedMembers(wire.Criteria)
			if err != nil {
				return nil, fmt.Errorf("%w: question %q criteria: %v", llm.ErrBadRequest, m.key, err)
			}
			for _, c := range choices {
				var description string
				if err := json.Unmarshal(c.value, &description); err != nil {
					return nil, fmt.Errorf("%w: question %q choice %q: %v", llm.ErrBadRequest, m.key, c.key, err)
				}
				q.Choices = append(q.Choices, llm.Choice{Value: c.key, Description: description})
			}
		case llm.QuestionScore:
			if err := json.Unmarshal(wire.Criteria, &q.Levels); err != nil {
				return nil, fmt.Errorf("%w: question %q levels: %v", llm.ErrBadRequest, m.key, err)
			}
		case llm.QuestionBool:
			var meanings struct {
				True  string `json:"true"`
				False string `json:"false"`
			}
			if len(wire.Criteria) > 0 {
				if err := json.Unmarshal(wire.Criteria, &meanings); err != nil {
					return nil, fmt.Errorf("%w: question %q criteria: %v", llm.ErrBadRequest, m.key, err)
				}
			}
			q.True, q.False = meanings.True, meanings.False
		default:
			return nil, fmt.Errorf("%w: question %q has unknown type %q (want choice, score or bool)", llm.ErrBadRequest, m.key, wire.Type)
		}
		questions = append(questions, q)
	}
	return questions, nil
}

type jsonMember struct {
	key   string
	value json.RawMessage
}

// orderedMembers decodes a JSON object's members in document order.
func orderedMembers(data []byte) ([]jsonMember, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	if delim, ok := token.(json.Delim); !ok || delim != '{' {
		return nil, errors.New("want a JSON object")
	}
	var members []jsonMember
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, err
		}
		key, _ := token.(string)
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil, err
		}
		members = append(members, jsonMember{key: key, value: value})
	}
	if _, err := decoder.Token(); err != nil {
		return nil, err
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return nil, errors.New("trailing data after the JSON object")
	}
	return members, nil
}

func formatAnswer(q llm.Question, a llm.Answer) string {
	switch q.Type {
	case llm.QuestionChoice:
		return fmt.Sprintf("%s: %s (p=%.3f, confidence %.3f)", q.Name, a.Choice, a.Probabilities[a.Choice], a.Confidence)
	case llm.QuestionScore:
		level := ""
		if index := int(a.Score + 0.5); index >= 0 && index < len(q.Levels) {
			level = " ≈ " + q.Levels[index]
		}
		return fmt.Sprintf("%s: %.3f%s (confidence %.3f)", q.Name, a.Score, level, a.Confidence)
	default:
		return fmt.Sprintf("%s: p(true)=%.3f", q.Name, a.Probability)
	}
}
