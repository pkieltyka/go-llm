package openaicodex

import (
	"encoding/json"
	"strings"
)

// The gpt-5.6 family is served by the codex backend only through the
// "Responses Lite" contract: an internal header plus a request-shape rewrite,
// mirroring Codex CLI 0.144.0. This is an undocumented upstream contract
// (plan 2 phase 3b); keep every Lite rule in this file so upstream changes
// have one place to land.
const codexResponsesLiteHeader = "X-OpenAI-Internal-Codex-Responses-Lite"

// isGPT56Model reports whether a model id is in the gpt-5.6 family
// (gpt-5.6 or gpt-5.6-*, including dated snapshots).
func isGPT56Model(model string) bool {
	return model == "gpt-5.6" || strings.HasPrefix(model, "gpt-5.6-")
}

// applyResponsesLite rewrites a codex request's top-level fields into the
// Lite shape:
//   - tools move from top-level `tools` into the input array as a leading
//     {"type":"additional_tools","role":"developer","tools":[...]} item;
//   - the system prompt moves from top-level `instructions` into a
//     `developer` role input message placed after the additional_tools item;
//   - `parallel_tool_calls` is forced false;
//   - `reasoning.context` is forced "all_turns".
//
// Pre-5.6 requests never reach this function.
func applyResponsesLite(fields map[string]json.RawMessage) error {
	var prefix []json.RawMessage

	if tools, ok := fields["tools"]; ok {
		item, err := json.Marshal(struct {
			Type  string          `json:"type"`
			Role  string          `json:"role"`
			Tools json.RawMessage `json:"tools"`
		}{Type: "additional_tools", Role: "developer", Tools: tools})
		if err != nil {
			return err
		}
		prefix = append(prefix, item)
		delete(fields, "tools")
	}

	if rawInstructions, ok := fields["instructions"]; ok {
		var instructions string
		if err := json.Unmarshal(rawInstructions, &instructions); err != nil {
			return err
		}
		delete(fields, "instructions")
		if instructions != "" {
			item, err := json.Marshal(struct {
				Type    string `json:"type"`
				Role    string `json:"role"`
				Content []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"content"`
			}{Type: "message", Role: "developer", Content: []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}{{Type: "input_text", Text: instructions}}})
			if err != nil {
				return err
			}
			prefix = append(prefix, item)
		}
	}

	if len(prefix) > 0 {
		var input []json.RawMessage
		if raw, ok := fields["input"]; ok {
			if err := json.Unmarshal(raw, &input); err != nil {
				return err
			}
		}
		combined, err := json.Marshal(append(prefix, input...))
		if err != nil {
			return err
		}
		fields["input"] = combined
	}

	fields["parallel_tool_calls"] = json.RawMessage("false")

	reasoning := map[string]json.RawMessage{}
	if raw, ok := fields["reasoning"]; ok {
		if err := json.Unmarshal(raw, &reasoning); err != nil {
			return err
		}
	}
	reasoning["context"] = json.RawMessage(`"all_turns"`)
	combined, err := json.Marshal(reasoning)
	if err != nil {
		return err
	}
	fields["reasoning"] = combined
	return nil
}
