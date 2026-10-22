package providerutil

import (
	"testing"
	"time"

	llm "github.com/pkieltyka/go-llm"
)

func TestApplyCacheRetention(t *testing.T) {
	hint := &llm.CacheHint{TTL: 2 * time.Hour}
	req := &llm.Request{
		Model:       "m",
		SystemCache: hint,
		Messages: []llm.Message{
			llm.UserParts(llm.TextPart{Text: "a", Cache: hint}, &llm.ImagePart{URL: "u", Cache: hint}),
			{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResultParts("c", llm.TextPart{Text: "r", Cache: hint})}},
		},
	}
	hints := func(r *llm.Request) []*llm.CacheHint {
		out := []*llm.CacheHint{r.SystemCache}
		out = append(out, r.Messages[0].Parts[0].(llm.TextPart).Cache, r.Messages[0].Parts[1].(llm.ImagePart).Cache)
		out = append(out, r.Messages[1].Parts[0].(llm.ToolResultPart).Content[0].(llm.TextPart).Cache)
		return out
	}
	before := hints(req)
	if got := ApplyCacheRetention(req); got != req {
		t.Fatal("empty retention must return the request unchanged")
	}
	for _, tt := range []struct {
		retention llm.CacheRetention
		check     func(*llm.CacheHint) bool
	}{
		{llm.CacheRetentionNone, func(h *llm.CacheHint) bool { return h == nil }},
		{llm.CacheRetentionShort, func(h *llm.CacheHint) bool { return h != nil && h.TTL == 0 }},
		{llm.CacheRetentionLong, func(h *llm.CacheHint) bool { return h != nil && h.TTL == time.Hour }},
	} {
		in := *req
		in.CacheRetention = tt.retention
		for i, h := range hints(ApplyCacheRetention(&in)) {
			if !tt.check(h) {
				t.Errorf("%s: hint %d = %+v", tt.retention, i, h)
			}
		}
	}
	for i, h := range hints(req) {
		if h != before[i] || h.TTL != 2*time.Hour {
			t.Fatalf("ApplyCacheRetention mutated the caller's hint %d", i)
		}
	}
}
