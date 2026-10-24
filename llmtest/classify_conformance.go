package llmtest

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

// RunClassifierConformance checks the provider-independent Classifier
// contract (FS §13C) for a provider built by newProvider:
//   - it implements llm.Classifier and advertises llm.CapabilityClassify;
//   - every invalid request fails with ErrBadRequest before any I/O;
//   - a canceled context fails with the context's error.
//
// Point the provider at an address that refuses connections (for example
// "http://127.0.0.1:1"): a request that reaches the network then fails with
// a transport error instead of the expected one, so I/O before validation
// is caught. Wire shapes and answer parsing are provider-specific and stay
// in the provider's own tests.
func RunClassifierConformance(t *testing.T, newProvider func(t *testing.T) llm.Provider) {
	t.Helper()
	p := newProvider(t)
	classifier, ok := p.(llm.Classifier)
	if !ok {
		t.Fatalf("%T does not implement llm.Classifier", p)
	}
	advertised := false
	for _, cap := range p.Capabilities() {
		advertised = advertised || cap == llm.CapabilityClassify
	}
	if !advertised {
		t.Fatalf("%s does not advertise CapabilityClassify", p.Name())
	}
	valid := func() *llm.ClassifyRequest {
		return &llm.ClassifyRequest{
			Model:     "conformance-model",
			State:     json.RawMessage(`{"conformance":true}`),
			Questions: []llm.Question{llm.BoolQuestion("ok", "Is this a conformance check?", "yes", "no")},
		}
	}
	t.Run("validation_before_io", func(t *testing.T) {
		for name, mutate := range map[string]func(*llm.ClassifyRequest){
			"no model":       func(r *llm.ClassifyRequest) { r.Model = "" },
			"array state":    func(r *llm.ClassifyRequest) { r.State = json.RawMessage(`[]`) },
			"no questions":   func(r *llm.ClassifyRequest) { r.Questions = nil },
			"duplicate name": func(r *llm.ClassifyRequest) { r.Questions = append(r.Questions, r.Questions[0]) },
			"one choice": func(r *llm.ClassifyRequest) {
				r.Questions = []llm.Question{llm.ChoiceQuestion("c", "?", llm.Choice{Value: "only"})}
			},
			"url image": func(r *llm.ClassifyRequest) { r.Images = []llm.ImagePart{{URL: "https://example.test/a.png"}} },
		} {
			req := valid()
			mutate(req)
			if _, err := classifier.Classify(context.Background(), req); !errors.Is(err, llm.ErrBadRequest) {
				t.Errorf("%s: err = %v, want ErrBadRequest before any I/O", name, err)
			}
		}
		if _, err := classifier.Classify(context.Background(), nil); !errors.Is(err, llm.ErrBadRequest) {
			t.Errorf("nil request: err = %v, want ErrBadRequest", err)
		}
	})
	t.Run("canceled_context", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := classifier.Classify(ctx, valid()); !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
	})
}
