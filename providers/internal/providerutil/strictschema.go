package providerutil

// StrictSchemaSupported reports whether a JSON schema uses only constructs
// accepted by OpenAI-style strict mode (structured outputs / strict tools).
// Adapters use it to fail open: when a schema needs keywords strict mode
// rejects, the request degrades to strict:false instead of erroring
// (ARCH §3.3 fail-open schema adaptation). Shared by the Responses adapter
// (providers/internal/responsesapi) and the chat-completions adapter
// (providers/chatcompletions).
func StrictSchemaSupported(value any) bool {
	switch v := value.(type) {
	case map[string]any:
		for key, nested := range v {
			if unsupportedStrictSchemaKeyword(key) {
				return false
			}
			// Keys one level below "properties", "$defs", and "definitions"
			// are property/definition NAMES, not schema keywords — a property
			// or shared definition literally named "format" must not disable
			// strict mode. Recurse into each named subschema only.
			if key == "properties" || key == "$defs" || key == "definitions" {
				if named, ok := nested.(map[string]any); ok {
					for _, subschema := range named {
						if !StrictSchemaSupported(subschema) {
							return false
						}
					}
					continue
				}
			}
			if !StrictSchemaSupported(nested) {
				return false
			}
		}
	case []any:
		for _, nested := range v {
			if !StrictSchemaSupported(nested) {
				return false
			}
		}
	}
	return true
}

func unsupportedStrictSchemaKeyword(key string) bool {
	switch key {
	case "format", "pattern", "minimum", "maximum", "exclusiveMinimum",
		"exclusiveMaximum", "multipleOf", "minLength", "maxLength",
		"minItems", "maxItems", "uniqueItems", "oneOf", "anyOf", "allOf",
		"not", "if", "then", "else", "dependentRequired",
		"dependentSchemas", "patternProperties":
		return true
	default:
		return false
	}
}
