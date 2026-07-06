package llm_test

import (
	"errors"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

func TestPromptTemplate(t *testing.T) {
	tmpl := llm.MustPromptTemplate("greeting", "Hello {{.Name}} from {{.Place}}")
	partial := tmpl.Partial(map[string]any{"Place": "Toronto"})
	got, err := partial.Format(struct {
		Name  string
		Place string
	}{Name: "Ada", Place: "Montreal"})
	if err != nil {
		t.Fatalf("Format returned error: %v", err)
	}
	if got != "Hello Ada from Montreal" {
		t.Fatalf("Format = %q", got)
	}
	if _, err := tmpl.Format(map[string]any{"Name": "Ada"}); err == nil {
		t.Fatalf("Format with missing var returned nil error")
	}
	got, err = partial.Format(map[string]any{"Name": "Grace"})
	if err != nil || got != "Hello Grace from Toronto" {
		t.Fatalf("partial Format = %q, %v", got, err)
	}
	got, err = tmpl.Format(map[string]any{"Name": "Linus", "Place": "Helsinki"})
	if err != nil || got != "Hello Linus from Helsinki" {
		t.Fatalf("original template mutated: %q, %v", got, err)
	}

	invalidPartial := tmpl.Partial(map[int]string{1: "bad"})
	if _, err := invalidPartial.Format(map[string]any{"Name": "Ada"}); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("invalid partial Format error = %v, want ErrBadRequest", err)
	}
}
