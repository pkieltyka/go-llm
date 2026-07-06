package llm

import (
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"text/template"
)

// PromptTemplate is an immutable strict text/template wrapper.
type PromptTemplate struct {
	name  string
	tmpl  *template.Template
	bound map[string]any
	err   error
}

// NewPromptTemplate parses text as a strict prompt template.
func NewPromptTemplate(name, text string) (*PromptTemplate, error) {
	tmpl, err := template.New(name).Option("missingkey=error").Parse(text)
	if err != nil {
		return nil, err
	}
	return &PromptTemplate{name: name, tmpl: tmpl}, nil
}

// MustPromptTemplate is NewPromptTemplate but panics on parse errors.
func MustPromptTemplate(name, text string) *PromptTemplate {
	t, err := NewPromptTemplate(name, text)
	if err != nil {
		panic(err)
	}
	return t
}

// Name returns the template name.
func (t *PromptTemplate) Name() string {
	if t == nil {
		return ""
	}
	return t.name
}

// Format renders the template. Missing variables return an error.
func (t *PromptTemplate) Format(vars any) (string, error) {
	if t == nil {
		return "", fmt.Errorf("%w: nil prompt template", ErrBadRequest)
	}
	if t.err != nil {
		return "", t.err
	}
	merged := cloneVars(t.bound)
	callVars, err := varsToMap(vars)
	if err != nil {
		return "", err
	}
	for key, value := range callVars {
		merged[key] = value
	}

	var b bytes.Buffer
	if err := t.tmpl.Execute(&b, merged); err != nil {
		return "", err
	}
	return b.String(), nil
}

// Partial returns a new template with vars pre-bound. Format-time vars win.
func (t *PromptTemplate) Partial(vars any) *PromptTemplate {
	if t == nil {
		return nil
	}
	merged := cloneVars(t.bound)
	if t.err != nil {
		return &PromptTemplate{name: t.name, tmpl: t.tmpl, bound: merged, err: t.err}
	}
	more, err := varsToMap(vars)
	if err != nil {
		return &PromptTemplate{name: t.name, tmpl: t.tmpl, bound: merged, err: err}
	}
	for key, value := range more {
		merged[key] = value
	}
	return &PromptTemplate{name: t.name, tmpl: t.tmpl, bound: merged}
}

func varsToMap(vars any) (map[string]any, error) {
	if vars == nil {
		return nil, nil
	}
	value := reflect.ValueOf(vars)
	for value.Kind() == reflect.Pointer {
		if value.IsNil() {
			return nil, nil
		}
		value = value.Elem()
	}

	switch value.Kind() {
	case reflect.Map:
		if value.Type().Key().Kind() != reflect.String {
			return nil, fmt.Errorf("%w: prompt vars map key must be string", ErrBadRequest)
		}
		out := make(map[string]any, value.Len())
		for iter := value.MapRange(); iter.Next(); {
			out[iter.Key().String()] = iter.Value().Interface()
		}
		return out, nil
	case reflect.Struct:
		out := make(map[string]any)
		typ := value.Type()
		for i := range typ.NumField() {
			field := typ.Field(i)
			if field.PkgPath != "" {
				continue
			}
			name := field.Name
			if tag := field.Tag.Get("json"); tag != "" {
				tagName, _, _ := strings.Cut(tag, ",")
				if tagName == "-" {
					continue
				}
				if tagName != "" {
					name = tagName
				}
			}
			out[name] = value.Field(i).Interface()
		}
		return out, nil
	default:
		return nil, fmt.Errorf("%w: prompt vars must be a map or struct", ErrBadRequest)
	}
}

func cloneVars(vars map[string]any) map[string]any {
	if len(vars) == 0 {
		return make(map[string]any)
	}
	out := make(map[string]any, len(vars))
	for key, value := range vars {
		out[key] = value
	}
	return out
}
