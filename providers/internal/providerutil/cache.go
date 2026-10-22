package providerutil

import (
	"time"

	llm "github.com/pkieltyka/go-llm"
)

// longCacheTTL is the TTL adapters map to their long cache tier: anything
// above five minutes selects Anthropic's one-hour cache_control.
const longCacheTTL = time.Hour

// ApplyCacheRetention returns req with Request.CacheRetention applied to
// every CacheHint (FS §15): none removes them, short resets their TTL to the
// provider default, and long sets the long tier. The caller's request and
// messages are never mutated: an empty retention returns req itself, and any
// other value returns a copy with new message and part slices.
func ApplyCacheRetention(req *llm.Request) *llm.Request {
	if req == nil || req.CacheRetention == llm.CacheRetentionDefault {
		return req
	}
	rewrite := func(hint *llm.CacheHint) *llm.CacheHint {
		if hint == nil {
			return nil
		}
		switch req.CacheRetention {
		case llm.CacheRetentionNone:
			return nil
		case llm.CacheRetentionShort:
			return &llm.CacheHint{}
		case llm.CacheRetentionLong:
			return &llm.CacheHint{TTL: longCacheTTL}
		default:
			return hint
		}
	}
	out := *req
	out.SystemCache = rewrite(req.SystemCache)
	out.Messages = make([]llm.Message, len(req.Messages))
	for i, msg := range req.Messages {
		msg.Parts = rewriteCacheParts(msg.Parts, rewrite)
		out.Messages[i] = msg
	}
	return &out
}

func rewriteCacheParts(parts []llm.Part, rewrite func(*llm.CacheHint) *llm.CacheHint) []llm.Part {
	if len(parts) == 0 {
		return parts
	}
	out := make([]llm.Part, len(parts))
	for i, part := range parts {
		switch p := DerefPart(part).(type) {
		case llm.TextPart:
			p.Cache = rewrite(p.Cache)
			out[i] = p
		case llm.ImagePart:
			p.Cache = rewrite(p.Cache)
			out[i] = p
		case llm.FilePart:
			p.Cache = rewrite(p.Cache)
			out[i] = p
		case llm.ToolResultPart:
			p.Content = rewriteCacheParts(p.Content, rewrite)
			out[i] = p
		case llm.SystemDeltaPart:
			p.Cache = rewrite(p.Cache)
			out[i] = p
		case llm.ToolsDeltaPart:
			p.Cache = rewrite(p.Cache)
			out[i] = p
		default:
			out[i] = part
		}
	}
	return out
}
