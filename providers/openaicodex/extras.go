package openaicodex

import (
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/responsesapi"
)

// ResponseExtras carries Responses payload details that have no home on the
// core llm.Response, e.g. output_text annotations.
type ResponseExtras = responsesapi.ResponseExtras

// Extras extracts codex-specific extras from a response produced by this
// provider. It works on both paths: Chat and Collect(ChatStream) both end up
// with the terminal SDK response in Response.Raw.
func Extras(resp *llm.Response) (*ResponseExtras, bool) {
	if resp == nil || resp.Provider != providerName {
		return nil, false
	}
	return responsesapi.Extras(resp)
}
