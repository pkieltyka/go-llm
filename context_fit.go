package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// Request-size estimation constants (FS §13A). They are deliberate,
// documented heuristics, not a tokenizer: they lean high so that budget and
// compaction decisions trigger early rather than after the server rejects a
// request. Calibrated 2026-10-22 against vLLM's exact /tokenize counts
// (prose, Go code, indented JSON, tool loops, long conversations: 0.95-1.17x;
// CJK and tiny tool prompts run higher); evidence lives in FS §13A and go-llm
// plan 10.
const (
	estimateASCIIWordCharsPerToken = 4.0  // ASCII letters and digits
	estimatePunctuationTokens      = 0.6  // each ASCII punctuation or symbol
	estimateExtraWhitespaceTokens  = 0.5  // each newline and each whitespace char after the first in a run
	estimateNonASCIITokens         = 1.0  // each non-ASCII rune (CJK is roughly one token per character)
	estimateRequestOverhead        = 8    // chat-template framing per request
	estimateMessageOverhead        = 4    // role and separator tokens per message
	estimateToolsPreamble          = 300  // tool-use instructions providers inject when tools are offered
	estimateToolOverhead           = 8    // per tool definition, plus its JSON
	estimateImageTokens            = 1600 // per image: Anthropic's ~1600 cap; a floor for providers that tile large images
	estimateFileTokens             = 2000 // per non-text file, and the minimum for a PDF
	estimatePDFPageTokens          = 2500 // per PDF page (text plus page image)
)

// EstimateRequestTokens returns a provider-neutral estimate of the prompt
// tokens req would occupy: system text, every message part, tool
// definitions, and the response format schema. Text is estimated from its
// character classes (ASCII word characters at a quarter token each,
// punctuation and extra whitespace at fractional rates, non-ASCII runes at a
// full token), plus fixed request, message, and tool overheads, including a
// one-time tool-use preamble when tools are offered. Images count a fixed
// amount (Anthropic's per-image cap, so a floor where providers tile large
// images); an inline PDF counts per page; other non-text files count a fixed
// amount; and text-typed file
// data (text/*, JSON, XML) counts as text. Reasoning counts its visible text
// only (opaque raw payloads are not estimated), and provider extension parts
// count nothing. The estimate is intentionally conservative and never exact;
// use CountTokens for a provider's exact count where one exists. A nil
// request returns zero.
func EstimateRequestTokens(req *Request) int64 {
	if req == nil {
		return 0
	}
	total := addTokens(estimateRequestOverhead, estimateTextTokens(req.System))
	total = addTokens(total, estimateMessagesTokens(req.Messages))
	if len(req.Tools) > 0 {
		total = addTokens(total, estimateToolsPreamble)
	}
	for _, tool := range req.Tools {
		definition, _ := json.Marshal(struct {
			Name        string `json:"name"`
			Description string `json:"description,omitempty"`
			Schema      any    `json:"schema,omitempty"`
		}{tool.Name, tool.Description, tool.InputSchema})
		total = addTokens(total, estimateToolOverhead+estimateTextTokens(string(definition)))
	}
	if req.ResponseFormat != nil && req.ResponseFormat.Schema != nil {
		schema, _ := json.Marshal(req.ResponseFormat.Schema)
		total = addTokens(total, estimateToolOverhead+estimateTextTokens(string(schema)))
	}
	return total
}

// EstimateContinuationTokens estimates the prompt of the next request in a
// conversation from the usage of the last response plus the messages added
// after it (tool results, the next user turn). The last response's full
// occupancy (input, cache reads and writes, and output, which becomes part
// of the next prompt) is ground truth, so only trailing is estimated, with
// the same constants as EstimateRequestTokens. trailing must not include the
// assistant message built from that response, whose tokens are already in
// last.OutputTokens. Hidden reasoning that a provider does not replay is
// counted in OutputTokens, so the result can run high, which is the safe
// direction.
func EstimateContinuationTokens(last Usage, trailing []Message) int64 {
	return addTokens(last.ContextUsage(0).UsedTokens, estimateMessagesTokens(trailing))
}

func estimateMessagesTokens(messages []Message) int64 {
	var total int64
	for _, msg := range messages {
		total = addTokens(total, estimateMessageOverhead)
		for _, part := range msg.Parts {
			total = addTokens(total, estimatePartTokens(part))
		}
	}
	return total
}

func estimatePartTokens(part Part) int64 {
	switch p := derefPart(part).(type) {
	case TextPart:
		return estimateTextTokens(p.Text)
	case ImagePart:
		return estimateImageTokens
	case FilePart:
		if len(p.Data) > 0 && textMediaType(p.MediaType) {
			return estimateTextTokens(string(p.Data))
		}
		if pages := pdfPageCount(p.Data); pages > 0 {
			return max(estimateFileTokens, int64(pages)*estimatePDFPageTokens)
		}
		return estimateFileTokens
	case ToolCallPart:
		return addTokens(addTokens(estimateTextTokens(p.ID), estimateTextTokens(p.Name)), estimateTextTokens(string(p.Args)))
	case ToolResultPart:
		total := addTokens(estimateMessageOverhead, estimateTextTokens(p.ToolCallID))
		for _, content := range p.Content {
			total = addTokens(total, estimatePartTokens(content))
		}
		return total
	case ReasoningPart:
		return estimateTextTokens(p.Text)
	case UnknownPart:
		return estimateTextTokens(string(p.Data))
	default:
		return 0
	}
}

func estimateTextTokens(s string) int64 {
	if s == "" {
		return 0
	}
	var word, punct, extraSpace, nonASCII float64
	inSpace := false
	for _, r := range s {
		switch {
		case r == ' ' || r == '\t' || r == '\r' || r == '\n':
			if r == '\n' || inSpace {
				extraSpace++
			}
			inSpace = true
			continue
		case r >= 0x80:
			nonASCII++
		case (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9'):
			word++
		default:
			punct++
		}
		inSpace = false
	}
	estimate := word/estimateASCIIWordCharsPerToken + punct*estimatePunctuationTokens +
		extraSpace*estimateExtraWhitespaceTokens + nonASCII*estimateNonASCIITokens
	return int64(math.Ceil(estimate))
}

// pdfPageCount counts page objects ("/Type /Page", not "/Pages") in inline
// PDF data. It is a heuristic for estimation, not a PDF parser; data that is
// not a PDF, or whose pages hide in compressed object streams, counts zero.
func pdfPageCount(data []byte) int {
	if !bytes.HasPrefix(data, []byte("%PDF-")) {
		return 0
	}
	typeKey, pageName := []byte("/Type"), []byte("/Page")
	pages := 0
	for rest := data; ; {
		j := bytes.Index(rest, typeKey)
		if j < 0 {
			return pages
		}
		rest = bytes.TrimLeft(rest[j+len(typeKey):], " \t\r\n")
		if bytes.HasPrefix(rest, pageName) && (len(rest) == len(pageName) || rest[len(pageName)] != 's') {
			pages++
		}
	}
}

func textMediaType(mediaType string) bool {
	mediaType = strings.ToLower(strings.TrimSpace(mediaType))
	if i := strings.IndexByte(mediaType, ';'); i >= 0 {
		mediaType = strings.TrimSpace(mediaType[:i])
	}
	return strings.HasPrefix(mediaType, "text/") ||
		mediaType == "application/json" || mediaType == "application/xml" ||
		strings.HasSuffix(mediaType, "+json") || strings.HasSuffix(mediaType, "+xml")
}

func addTokens(a, b int64) int64 {
	if b > 0 && a > math.MaxInt64-b {
		return math.MaxInt64
	}
	return a + b
}

// TokenCount is a prompt-size measurement. Exact reports a provider-computed
// count; false marks an EstimateRequestTokens estimate.
type TokenCount struct {
	Tokens int64
	Exact  bool
}

// TokenCounter is an optional Provider extension for exact prompt token
// counting, discovered by type assertion. It converts req exactly as Chat
// would and asks the provider how many prompt tokens it occupies, without
// generating. It is never invoked implicitly by Chat or ChatStream.
type TokenCounter interface {
	CountTokens(ctx context.Context, req *Request) (TokenCount, error)
}

// CountTokens measures req for p. It uses the first TokenCounter found on p
// or on the providers it wraps (following Unwrap() Provider methods, which
// llm.Wrap providers expose), and returns its result and error unchanged. A
// provider without a counter yields EstimateRequestTokens with Exact false
// and a nil error.
func CountTokens(ctx context.Context, p Provider, req *Request) (TokenCount, error) {
	if counter, ok := findTokenCounter(p); ok {
		return counter.CountTokens(ctx, req)
	}
	return TokenCount{Tokens: EstimateRequestTokens(req)}, nil
}

func findTokenCounter(p Provider) (TokenCounter, bool) {
	for depth := 0; p != nil && depth < 32; depth++ {
		if counter, ok := p.(TokenCounter); ok {
			return counter, true
		}
		unwrapper, ok := p.(interface{ Unwrap() Provider })
		if !ok {
			return nil, false
		}
		p = unwrapper.Unwrap()
	}
	return nil, false
}

// ResolveModelInfo returns model metadata for model on p, merging the live
// catalog with the embedded table: the provider's Models() row (matched by
// exact ID, then by SameModel, preferring the lexically greatest ID, which is
// the latest dated snapshot, when several rows match inexactly) wins, and a zero context window, zero max output,
// nil pricing, nil effort ladder, or empty default effort is filled from
// LookupModelInfo. Nothing is invented: a field neither source knows stays
// zero. A Models() failure degrades to the table; the error is returned only
// when neither source knows the model. Each call performs provider model
// discovery, so callers should cache the result.
func ResolveModelInfo(ctx context.Context, p Provider, model string) (ModelInfo, error) {
	if p == nil {
		return ModelInfo{}, fmt.Errorf("%w: nil provider", ErrBadRequest)
	}
	if strings.TrimSpace(model) == "" {
		return ModelInfo{}, fmt.Errorf("%w: empty model", ErrBadRequest)
	}
	models, listErr := p.Models(ctx)
	live, liveOK := findModelRow(models, model)
	table, tableOK := LookupModelInfo(p.Name(), model)
	switch {
	case liveOK && tableOK:
		return mergeModelInfo(live, table), nil
	case liveOK:
		return live, nil
	case tableOK:
		return table, nil
	case listErr != nil:
		return ModelInfo{}, listErr
	default:
		return ModelInfo{}, fmt.Errorf("%w: model %q is not in the %s catalog or the embedded table", ErrNotFound, model, p.Name())
	}
}

func findModelRow(models []ModelInfo, model string) (ModelInfo, bool) {
	for _, row := range models {
		if row.ID == model {
			return row, true
		}
	}
	var best ModelInfo
	found := false
	for _, row := range models {
		if SameModel(row.ID, model) && (!found || row.ID > best.ID) {
			best, found = row, true
		}
	}
	return best, found
}

func mergeModelInfo(live, table ModelInfo) ModelInfo {
	if live.ContextWindow <= 0 {
		live.ContextWindow = table.ContextWindow
	}
	if live.MaxOutputTokens <= 0 {
		live.MaxOutputTokens = table.MaxOutputTokens
	}
	if live.Pricing == nil && table.Pricing != nil {
		live.Pricing = table.Pricing.Clone()
	}
	if live.SupportedEfforts == nil && table.SupportedEfforts != nil {
		live.SupportedEfforts = append([]Effort(nil), table.SupportedEfforts...)
	}
	if live.DefaultEffort == "" {
		live.DefaultEffort = table.DefaultEffort
	}
	return live
}

// FitMaxTokens returns the output budget that fits a prompt of prompt tokens
// into a context window of window tokens while reserving margin tokens:
// min(requested, window-prompt-margin). A requested value of zero or less
// means "as much as fits". It returns an error wrapping ErrContextTooLong
// when fewer than one output token fits, and ErrBadRequest for a
// non-positive window or negative prompt. It is a pure function: nothing is
// clamped unless the caller applies the result.
func FitMaxTokens(prompt, window int64, requested, margin int) (int, error) {
	if window <= 0 || prompt < 0 {
		return 0, fmt.Errorf("%w: fit max tokens needs a positive window and a non-negative prompt (window=%d prompt=%d)", ErrBadRequest, window, prompt)
	}
	if margin < 0 {
		margin = 0
	}
	available := window - addTokens(prompt, int64(margin))
	if available < 1 {
		return 0, fmt.Errorf("%w: prompt of %d tokens leaves no output room in a %d-token window (margin %d)", ErrContextTooLong, prompt, window, margin)
	}
	if requested > 0 && int64(requested) < available {
		return requested, nil
	}
	if available > math.MaxInt32 {
		available = math.MaxInt32
	}
	return int(available), nil
}

// EstimateCostBound returns an upper bound on the cost of a request whose
// prompt occupies promptTokens and which may generate up to maxOutput
// tokens: the prompt is priced as uncached input, as cache reads, and as
// cache writes, and the largest is returned, honoring pricing tiers. ok is
// false when a needed rate is unknown or the inputs are negative.
func EstimateCostBound(promptTokens, maxOutput int64, pricing ModelPricing) (bound float64, ok bool) {
	if promptTokens < 0 || maxOutput < 0 {
		return 0, false
	}
	for _, usage := range []Usage{
		{InputTokens: promptTokens, OutputTokens: maxOutput},
		{CacheReadTokens: promptTokens, OutputTokens: maxOutput},
		{CacheWriteTokens: promptTokens, OutputTokens: maxOutput},
	} {
		estimated := EstimateCost(usage, pricing)
		if estimated.CostUSD == nil || *estimated.CostUSD < 0 || math.IsNaN(*estimated.CostUSD) || math.IsInf(*estimated.CostUSD, 0) {
			return 0, false
		}
		bound = math.Max(bound, *estimated.CostUSD)
	}
	return bound, true
}
