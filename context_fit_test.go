package llm_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync/atomic"
	"testing"

	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/llmtest"
)

func TestEstimateRequestTokens(t *testing.T) {
	if got := llm.EstimateRequestTokens(nil); got != 0 {
		t.Fatalf("nil request = %d, want 0", got)
	}
	// 35 ASCII word chars → ceil(8.75) = 9, plus 4 per message and 8 per request.
	text := llm.EstimateRequestTokens(&llm.Request{Messages: []llm.Message{llm.UserText("abcdefghijklmnopqrstuvwxyzabcdefghi")}})
	if text != 21 {
		t.Fatalf("one 35-char message = %d, want 21", text)
	}
	withSystem := llm.EstimateRequestTokens(&llm.Request{System: "abcdefg", Messages: []llm.Message{llm.UserText("abcdefghijklmnopqrstuvwxyzabcdefghi")}})
	if withSystem != text+2 {
		t.Fatalf("system adds %d, want 2", withSystem-text)
	}
	image := llm.EstimateRequestTokens(&llm.Request{Messages: []llm.Message{llm.UserParts(llm.ImageURL("https://example.test/a.png"))}})
	if image != 8+4+1600 {
		t.Fatalf("image = %d, want 1612", image)
	}
	pdf := llm.EstimateRequestTokens(&llm.Request{Messages: []llm.Message{llm.UserParts(llm.FileData([]byte("%PDF-1.7 ..."), "application/pdf", "a.pdf"))}})
	if pdf != 8+4+2000 {
		t.Fatalf("pdf without page objects = %d, want the 2012 minimum", pdf)
	}
	var doc strings.Builder
	doc.WriteString("%PDF-1.7\n1 0 obj << /Type /Pages /Count 100 >>\n")
	for i := 0; i < 100; i++ {
		fmt.Fprintf(&doc, "%d 0 obj << /Type/Page /Parent 1 0 R >>\n", i+2)
	}
	hundred := llm.EstimateRequestTokens(&llm.Request{Messages: []llm.Message{llm.UserParts(llm.FileData([]byte(doc.String()), "application/pdf", "big.pdf"))}})
	if hundred != 8+4+100*2500 {
		t.Fatalf("100-page pdf = %d, want %d", hundred, 8+4+100*2500)
	}
	jsonMode := llm.EstimateRequestTokens(&llm.Request{ResponseFormat: &llm.ResponseFormat{Type: llm.FormatJSONMode}})
	if jsonMode != 8 {
		t.Fatalf("json mode without schema = %d, want only the request overhead", jsonMode)
	}
	withID := llm.EstimateRequestTokens(&llm.Request{Messages: []llm.Message{llm.AssistantParts(llm.ToolCall("toolu_01ABCDEFGHIJKLMNOPQRSTUV", "f", nil))}})
	withoutID := llm.EstimateRequestTokens(&llm.Request{Messages: []llm.Message{llm.AssistantParts(llm.ToolCall("", "f", nil))}})
	if withID <= withoutID {
		t.Fatal("tool call IDs must be estimated")
	}
	textFile := llm.EstimateRequestTokens(&llm.Request{Messages: []llm.Message{llm.UserParts(llm.FileData([]byte("abcdefg"), "text/plain; charset=utf-8", "a.txt"))}})
	if textFile != 8+4+2 {
		t.Fatalf("text file = %d, want 14", textFile)
	}
	tool := llm.Tool{Name: "lookup", Description: "Look up a value.", InputSchema: json.RawMessage(`{"type":"object"}`)}
	noTools := llm.EstimateRequestTokens(&llm.Request{Messages: []llm.Message{llm.UserText("hi")}})
	withTools := llm.EstimateRequestTokens(&llm.Request{Messages: []llm.Message{llm.UserText("hi")}, Tools: []llm.Tool{tool}})
	if withTools <= noTools+300+8 {
		t.Fatalf("tools added %d tokens, want the preamble, overhead, and definition", withTools-noTools)
	}
	loop := llm.EstimateRequestTokens(&llm.Request{Messages: []llm.Message{
		llm.AssistantParts(llm.ToolCall("call_1", "lookup", json.RawMessage(`{"q":"go"}`)), llm.ReasoningPart{Text: "thinking", Raw: json.RawMessage(`{"opaque":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}`)}),
		{Role: llm.RoleTool, Parts: []llm.Part{llm.ToolResult("call_1", "result text")}},
	}})
	if loop <= 0 {
		t.Fatalf("tool loop estimate = %d", loop)
	}
	// Raw reasoning payloads are not estimated: only the visible text counts.
	rawOnly := llm.EstimateRequestTokens(&llm.Request{Messages: []llm.Message{llm.AssistantParts(llm.ReasoningPart{Raw: json.RawMessage(`{"opaque":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}`)})}})
	if rawOnly != 8+4 {
		t.Fatalf("raw-only reasoning = %d, want request and message overhead only", rawOnly)
	}
}

func TestEstimateTextCharacterClasses(t *testing.T) {
	estimate := func(text string) int64 {
		return llm.EstimateRequestTokens(&llm.Request{Messages: []llm.Message{llm.UserText(text)}}) - 12
	}
	tests := []struct {
		text string
		want int64
	}{
		{"abcd", 1},       // 4 word chars
		{"a b c d", 1},    // single spaces between words are free
		{"{}[],:", 4},     // 6 punctuation → 3.6
		{"a\n\n    b", 4}, // each newline and each space inside the run: 6 × 0.5, plus 2 word chars → 3.5
		{"并发编程", 4},
		{"a\tb", 1},
		{"a\r\nb", 1},
		{"🙂🙂", 2}, // non-ASCII runes count a token each
	}
	for _, tt := range tests {
		if got := estimate(tt.text); got != tt.want {
			t.Errorf("estimate(%q) = %d, want %d", tt.text, got, tt.want)
		}
	}
}

func TestEstimateContinuationTokens(t *testing.T) {
	last := llm.Usage{InputTokens: 100, CacheReadTokens: 50, CacheWriteTokens: 25, OutputTokens: 10}
	got := llm.EstimateContinuationTokens(last, []llm.Message{llm.UserText("abcdefg")})
	if got != 185+4+2 {
		t.Fatalf("continuation = %d, want 191", got)
	}
}

type countingProvider struct {
	*llmtest.Provider
	tokens int64
	err    error
	calls  atomic.Int64
}

func (p *countingProvider) CountTokens(_ context.Context, _ *llm.Request) (llm.TokenCount, error) {
	p.calls.Add(1)
	if p.err != nil {
		return llm.TokenCount{}, p.err
	}
	return llm.TokenCount{Tokens: p.tokens, Exact: true}, nil
}

func TestCountTokensFindsCounterThroughWrapAndFallsBack(t *testing.T) {
	ctx := context.Background()
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("abcdefg")}}
	counter := &countingProvider{Provider: llmtest.New(), tokens: 42}
	wrapped := llm.Wrap(llm.Wrap(counter, llm.Middleware{}), llm.NewUsageTracker().Middleware())
	got, err := llm.CountTokens(ctx, wrapped, req)
	if err != nil || got != (llm.TokenCount{Tokens: 42, Exact: true}) {
		t.Fatalf("CountTokens through Wrap = %+v, %v", got, err)
	}
	estimate, err := llm.CountTokens(ctx, llmtest.New(), req)
	if err != nil || estimate.Exact || estimate.Tokens != llm.EstimateRequestTokens(req) {
		t.Fatalf("CountTokens fallback = %+v, %v", estimate, err)
	}
	counter.err = llm.ErrServer
	if _, err := llm.CountTokens(ctx, counter, req); !errors.Is(err, llm.ErrServer) {
		t.Fatalf("counter error = %v, want it returned unchanged", err)
	}
}

func TestResolveModelInfoMergesLiveAndTable(t *testing.T) {
	ctx := context.Background()
	p := llmtest.New(llmtest.WithName("anthropic"), llmtest.WithModels(llm.ModelInfo{ID: "claude-haiku-4-5-20251001", DisplayName: "Live Haiku"}))
	info, err := llm.ResolveModelInfo(ctx, p, "claude-haiku-4-5")
	if err != nil {
		t.Fatal(err)
	}
	table, ok := llm.LookupModelInfo("anthropic", "claude-haiku-4-5")
	if !ok {
		t.Fatal("embedded table lacks claude-haiku-4-5")
	}
	if info.DisplayName != "Live Haiku" || info.ContextWindow != table.ContextWindow || info.MaxOutputTokens != table.MaxOutputTokens || info.Pricing == nil {
		t.Fatalf("merged info = %+v, want live row with table fills", info)
	}
	if info.Pricing == table.Pricing {
		t.Fatal("pricing must be cloned from the table")
	}

	tableOnly, err := llm.ResolveModelInfo(ctx, llmtest.New(llmtest.WithName("anthropic")), "claude-haiku-4-5")
	if err != nil || tableOnly.ContextWindow != table.ContextWindow {
		t.Fatalf("table-only = %+v, %v", tableOnly, err)
	}
	if _, err := llm.ResolveModelInfo(ctx, llmtest.New(llmtest.WithName("nobody")), "nothing-here"); !errors.Is(err, llm.ErrNotFound) {
		t.Fatalf("unknown model err = %v, want ErrNotFound", err)
	}
	snapshots := llmtest.New(llmtest.WithName("x"), llmtest.WithModels(
		llm.ModelInfo{ID: "gpt-4o-2024-05-13", DisplayName: "old"},
		llm.ModelInfo{ID: "gpt-4o-2024-08-06", DisplayName: "new"},
	))
	if got, err := llm.ResolveModelInfo(ctx, snapshots, "gpt-4o"); err != nil || got.DisplayName != "new" {
		t.Fatalf("inexact match = %+v, %v; want the latest snapshot", got, err)
	}
	if _, err := llm.ResolveModelInfo(ctx, nil, "m"); !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("nil provider err = %v", err)
	}
}

func TestFitMaxTokens(t *testing.T) {
	tests := []struct {
		prompt, window    int64
		requested, margin int
		want              int
		wantErr           error
	}{
		{prompt: 900, window: 1000, requested: 50, margin: 0, want: 50},
		{prompt: 900, window: 1000, requested: 500, margin: 0, want: 100},
		{prompt: 900, window: 1000, requested: 0, margin: 10, want: 90},
		{prompt: 990, window: 1000, requested: 50, margin: 10, wantErr: llm.ErrContextTooLong},
		{prompt: 1, window: 0, requested: 1, wantErr: llm.ErrBadRequest},
		{prompt: 0, window: math.MaxInt64, requested: 0, want: math.MaxInt32},
		{prompt: math.MaxInt64, window: 1, requested: 0, margin: 3, wantErr: llm.ErrContextTooLong},
	}
	for _, tt := range tests {
		got, err := llm.FitMaxTokens(tt.prompt, tt.window, tt.requested, tt.margin)
		if tt.wantErr != nil {
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("FitMaxTokens(%+v) err = %v, want %v", tt, err, tt.wantErr)
			}
			continue
		}
		if err != nil || got != tt.want {
			t.Errorf("FitMaxTokens(%+v) = %d, %v, want %d", tt, got, err, tt.want)
		}
	}
}

func TestEstimateCostBound(t *testing.T) {
	pricing := llm.ModelPricing{InputPerMTok: 1, OutputPerMTok: 4, CacheReadPerMTok: 0.1, CacheWritePerMTok: 1.25}
	bound, ok := llm.EstimateCostBound(1_000_000, 1_000_000, pricing)
	if !ok || math.Abs(bound-5.25) > 1e-9 {
		t.Fatalf("bound = %v, %v, want 5.25 (cache-write interpretation)", bound, ok)
	}
	unknown := pricing
	unknown.Availability = &llm.ModelPricingAvailability{InputPerMTok: true, OutputPerMTok: true}
	if _, ok := llm.EstimateCostBound(10, 10, unknown); ok {
		t.Fatal("unknown cache rates must make the bound unknown")
	}
	if _, ok := llm.EstimateCostBound(-1, 10, pricing); ok {
		t.Fatal("negative tokens must fail")
	}
}
