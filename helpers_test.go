package llm_test

import (
	"strings"

	llm "github.com/pkieltyka/go-llm"
)

func msgText(msg llm.Message) string {
	var b strings.Builder
	for _, part := range msg.Parts {
		switch p := part.(type) {
		case llm.TextPart:
			b.WriteString(p.Text)
		case *llm.TextPart:
			if p != nil {
				b.WriteString(p.Text)
			}
		}
	}
	return b.String()
}
