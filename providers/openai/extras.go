package openai

import (
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/responsesapi"
)

// ResponseExtras carries OpenAI Responses payload details that have no home
// on the core llm.Response, e.g. output_text annotations (URL citations,
// file citations, file paths).
type ResponseExtras = responsesapi.ResponseExtras

// Extras extracts OpenAI-specific extras from a response produced by this
// provider. It works on both paths: Chat stores the SDK response in
// Response.Raw, and ChatStream attaches the terminal SDK response to
// MessageEnd.Raw, which llm.Collect installs as Response.Raw.
func Extras(resp *llm.Response) (*ResponseExtras, bool) {
	if resp == nil || resp.Provider != providerName {
		return nil, false
	}
	return responsesapi.Extras(resp)
}
