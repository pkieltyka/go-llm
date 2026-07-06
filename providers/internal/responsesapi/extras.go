package responsesapi

import (
	"github.com/openai/openai-go/v3/responses"
	llm "github.com/pkieltyka/go-llm"
)

// ResponseExtras carries Responses API payload details that have no home on
// the core llm.Response (ARCH §3.2 "annotations preserved in extras").
type ResponseExtras struct {
	// Annotations are the output_text annotations (URL citations, file
	// citations, file paths, ...) across all message output items, in
	// output order.
	Annotations []responses.ResponseOutputTextAnnotationUnion
}

// Extras extracts extras from a response produced by a Responses-backed
// provider. It works on both paths: the blocking path stores the SDK
// response in Response.Raw, and the stream path attaches the terminal SDK
// response to MessageEnd.Raw, which Collect installs as Response.Raw.
func Extras(resp *llm.Response) (*ResponseExtras, bool) {
	if resp == nil {
		return nil, false
	}
	raw, ok := resp.Raw.(*responses.Response)
	if !ok || raw == nil {
		return nil, false
	}
	extras := &ResponseExtras{}
	for _, item := range raw.Output {
		if item.Type != "message" {
			continue
		}
		for _, content := range item.Content {
			if content.Type != "output_text" {
				continue
			}
			extras.Annotations = append(extras.Annotations, content.Annotations...)
		}
	}
	return extras, true
}
