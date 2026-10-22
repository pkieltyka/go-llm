package providerutil

import (
	"reflect"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

// DerefPart duplicates the root package's pointer normalization; it must
// agree with llm.CloneParts for every core part type.
func TestDerefPartMatchesRootNormalization(t *testing.T) {
	for _, part := range []llm.Part{
		&llm.TextPart{Text: "t"}, &llm.ImagePart{URL: "u"}, &llm.FilePart{URL: "u"},
		&llm.ToolCallPart{ID: "1"}, &llm.ToolResultPart{ToolCallID: "1"}, &llm.ReasoningPart{Text: "r"},
		&llm.UnknownPart{Type: "x/y"}, &llm.SystemDeltaPart{Section: "s"}, &llm.ToolsDeltaPart{Removed: []string{"a"}},
	} {
		want := reflect.TypeOf(llm.CloneParts([]llm.Part{part})[0])
		if got := reflect.TypeOf(DerefPart(part)); got != want {
			t.Errorf("DerefPart(%T) = %v, want %v", part, got, want)
		}
	}
}
