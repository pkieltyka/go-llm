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
