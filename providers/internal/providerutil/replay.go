package providerutil

import llm "github.com/pkieltyka/go-llm"

// ReplayableHistory applies the model-aware reasoning replay rule (FS §18):
// ReasoningParts are removed from assistant messages that a different model
// produced, meaning Message.Model is set and !llm.SameModel(Message.Model,
// model). Cross-model reasoning is dropped, never converted to text. An
// assistant message left without parts is dropped, as adapters already do
// for messages emptied by provider filtering. Messages with an empty Model
// (legacy history) keep today's provider-gated replay. When nothing is
// removed the input slice is returned unchanged.
func ReplayableHistory(messages []llm.Message, model string) []llm.Message {
	var out []llm.Message
	for i, msg := range messages {
		if !foreignModelReasoning(msg, model) {
			if out != nil {
				out = append(out, msg)
			}
			continue
		}
		if out == nil {
			out = make([]llm.Message, i, len(messages))
			copy(out, messages[:i])
		}
		parts := make([]llm.Part, 0, len(msg.Parts))
		for _, part := range msg.Parts {
			if _, ok := DerefPart(part).(llm.ReasoningPart); !ok {
				parts = append(parts, part)
			}
		}
		if len(parts) == 0 {
			continue
		}
		msg.Parts = parts
		out = append(out, msg)
	}
	if out == nil {
		return messages
	}
	return out
}

func foreignModelReasoning(msg llm.Message, model string) bool {
	if msg.Role != llm.RoleAssistant || msg.Model == "" || llm.SameModel(msg.Model, model) {
		return false
	}
	for _, part := range msg.Parts {
		if _, ok := DerefPart(part).(llm.ReasoningPart); ok {
			return true
		}
	}
	return false
}
