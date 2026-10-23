package providerutil

import (
	"fmt"

	llm "github.com/pkieltyka/go-llm"
)

// PositionalSupport returns what providerName's model accepts in place,
// from the embedded models.json behavior (FS §13B). Unknown models fold.
func PositionalSupport(providerName, model string) llm.PositionalSupport {
	info, ok := llm.LookupModelInfo(providerName, model)
	if !ok {
		return llm.PositionalSupport{}
	}
	return info.Behavior.PositionalSupport()
}

// RenderSystemDelta renders a section update sent in place, framed by name
// so the model can relate it to the original prompt (pi's
// renderSystemMessageUpdate). The framing is request-time only.
func RenderSystemDelta(part llm.SystemDeltaPart) string {
	if part.Text == "" {
		return fmt.Sprintf("Removed system prompt section %q.", part.Section)
	}
	return fmt.Sprintf("Updated system prompt section %q:\n\n%s", part.Section, part.Text)
}
