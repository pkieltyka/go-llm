package schema

import (
	"encoding/json"
	"errors"
	"fmt"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/schemajson"
)

// ValidateArgs checks model-emitted tool arguments against the supported
// strict-mode JSON Schema subset: type, required, properties,
// additionalProperties, items, and enum. Annotation keywords such as
// description and format are accepted in schemas but not enforced here.
func ValidateArgs(t llm.Tool, args json.RawMessage) error {
	err := schemajson.ValidateArgs(t.Name, t.InputSchema, args)
	if errors.Is(err, schemajson.ErrBadRequest) {
		return fmt.Errorf("%w: %s", llm.ErrBadRequest, schemajson.BadRequestDetail(err))
	}
	return err
}

// ValidateSchema checks that schema is inside the supported strict-mode JSON
// Schema subset (its shape only), without validating any arguments against it.
// It fails closed on a root missing "type", a union/nullable type, or an array
// without "items". Schemas produced by For are conformant by construction.
func ValidateSchema(schema any) error {
	err := schemajson.ValidateSchema(schema)
	if errors.Is(err, schemajson.ErrBadRequest) {
		return fmt.Errorf("%w: %s", llm.ErrBadRequest, schemajson.BadRequestDetail(err))
	}
	return err
}
