package llm

import "context"

const droppedToolCallRetryPrompt = "The previous response contained malformed tool calls that could not be used. Retry with valid tool calls, including complete id/name fields and valid JSON arguments."

// RetryDroppedToolCalls retries Chat responses that contain malformed tool
// calls reported by a provider adapter. Streaming is passed through unchanged.
//
// When a retry attempt itself fails, the prior successful response is
// returned ALONGSIDE the retry error — both non-nil, mirroring Collect's
// partial-result contract — so callers can persist the salvageable turn
// while still observing the failure (including ctx cancellation).
func RetryDroppedToolCalls(n int) Middleware {
	if n <= 0 {
		return Middleware{}
	}
	return Middleware{
		Chat: func(next ChatFunc) ChatFunc {
			return func(ctx context.Context, req *Request) (*Response, error) {
				current := cloneRequest(req)
				var last *Response
				for attempt := 0; attempt <= n; attempt++ {
					resp, err := next(ctx, current)
					if err != nil {
						if last != nil {
							return last, err
						}
						return resp, err
					}
					last = resp
					if resp == nil || len(resp.DroppedToolCalls) == 0 || attempt == n {
						return resp, nil
					}
					current = retryRequestAfterDroppedToolCalls(current, resp)
				}
				return last, nil
			}
		},
	}
}

func retryRequestAfterDroppedToolCalls(req *Request, resp *Response) *Request {
	next := cloneRequest(req)
	next.Messages = append(next.Messages, Message{
		Role:     RoleAssistant,
		Parts:    cloneParts(resp.Parts),
		Provider: resp.Provider,
		Model:    resp.Model,
	})
	if results := retryCorrectionToolResults(resp); len(results) > 0 {
		next.Messages = append(next.Messages, Message{Role: RoleTool, Parts: results})
		return next
	}
	next.Messages = append(next.Messages, UserText(droppedToolCallRetryPrompt))
	return next
}

func retryCorrectionToolResults(resp *Response) []Part {
	calls := resp.ToolCalls()
	results := make([]Part, 0, len(calls))
	for _, call := range calls {
		if call.ID == "" || call.Name == "" {
			continue
		}
		result := ToolResult(call.ID, droppedToolCallRetryPrompt)
		result.Name = call.Name
		result.IsError = true
		results = append(results, result)
	}
	return results
}
