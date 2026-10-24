// Classify asks a classifier (decision) model typed questions about some
// state: TypeSafe's Jev here, through OpenRouter, which also serves
// Cloudflare's Clef, OpenAI's GPT-6 Luna Decisions and other decision
// models over the same System One protocol.
//
// A classifier answers with probabilities instead of prose: a choice among
// labelled options, a score on an ordered scale, or a true/false
// probability. llm.NewClassifierModel binds the provider, the model and the
// questions once; each call then passes only the state.
//
// With OPENROUTER_API_KEY set it calls OpenRouter; without it, it runs
// offline on the scripted llmtest provider.
package main

import (
	"context"
	"fmt"
	"os"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
	"github.com/pkieltyka/go-llm/providers/openrouter"
)

// review is the state the questions are about; any value that encodes to a
// JSON object works.
type review struct {
	Product string `json:"product"`
	Text    string `json:"text"`
}

func main() {
	ctx := context.Background()
	p := newProvider()

	triage := llm.NewClassifierModel(p, "typesafe/jev-1.14", llm.WithClassifierQuestions(
		llm.ChoiceQuestion("topic", "What is the review mainly about?",
			llm.Choice{Value: "quality", Description: "How well the product works or is made"},
			llm.Choice{Value: "shipping", Description: "Delivery, packaging or delays"},
			llm.Choice{Value: "price", Description: "Cost or value for money"}),
		llm.ScoreQuestion("sentiment", "How does the reviewer feel?", "negative", "mixed", "positive"),
		llm.BoolQuestion("needs_reply", "Should support reply to this review?",
			"The reviewer reports a problem support can fix", "No action is needed"),
	))

	for _, r := range []review{
		{Product: "kettle", Text: "Boils fast and looks great on the counter."},
		{Product: "headphones", Text: "Arrived two weeks late and the box was crushed."},
	} {
		result, err := triage.Classify(ctx, r)
		if err != nil {
			panic(err)
		}
		topic := result.Answers["topic"]
		sentiment := result.Answers["sentiment"]
		reply := result.Answers["needs_reply"]
		fmt.Printf("%s: topic=%s (%.0f%%), sentiment=%.1f of 0..2, needs reply=%.0f%%\n",
			r.Product, topic.Choice, 100*topic.Probabilities[topic.Choice], sentiment.Score, 100*reply.Probability)
	}
}

func newProvider() llm.Provider {
	if os.Getenv("OPENROUTER_API_KEY") != "" {
		p, err := openrouter.New() // reads OPENROUTER_API_KEY
		if err != nil {
			panic(err)
		}
		return p
	}
	fmt.Println("(offline demo — set OPENROUTER_API_KEY to call OpenRouter)")
	p := llmtest.New()
	p.EnqueueClassifyResult(&llm.ClassifyResult{Provider: "llmtest", Model: "typesafe/jev-1.14", Answers: map[string]llm.Answer{
		"topic":       {Type: llm.QuestionChoice, Choice: "quality", Probabilities: map[string]float64{"quality": 0.94, "shipping": 0.03, "price": 0.03}, Confidence: 0.9},
		"sentiment":   {Type: llm.QuestionScore, Score: 1.9, Confidence: 0.85},
		"needs_reply": {Type: llm.QuestionBool, Probability: 0.06},
	}})
	p.EnqueueClassifyResult(&llm.ClassifyResult{Provider: "llmtest", Model: "typesafe/jev-1.14", Answers: map[string]llm.Answer{
		"topic":       {Type: llm.QuestionChoice, Choice: "shipping", Probabilities: map[string]float64{"quality": 0.04, "shipping": 0.93, "price": 0.03}, Confidence: 0.88},
		"sentiment":   {Type: llm.QuestionScore, Score: 0.2, Confidence: 0.8},
		"needs_reply": {Type: llm.QuestionBool, Probability: 0.91},
	}})
	return p
}
