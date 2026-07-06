package chatcompletions

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"maps"
	"net/http"
	"slices"
	"strings"
	"time"

	sdk "github.com/openai/openai-go/v3"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

func (p *Provider) streamEvents(ctx context.Context, req *llm.Request, params sdk.ChatCompletionNewParams) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		body, err := json.Marshal(params)
		if err != nil {
			yield(nil, err)
			return
		}
		body, err = withStreamEnabled(body)
		if err != nil {
			yield(nil, err)
			return
		}
		resp, err := p.openStream(ctx, req, body)
		if err != nil {
			yield(nil, err)
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			yield(nil, p.mapHTTPResponseError(resp))
			return
		}
		state := newStreamState(p)
		for payload, err := range ssePayloads(resp.Body) {
			if err != nil {
				yield(nil, err)
				return
			}
			if strings.TrimSpace(string(payload)) == "[DONE]" {
				for _, event := range state.finish() {
					if !yield(event, nil) {
						return
					}
				}
				if !state.everStarted {
					yield(nil, p.emptyStreamError())
				}
				return
			}
			if p.compat.SniffMidStreamErrors {
				if err := p.sniffStreamError(payload); err != nil {
					yield(nil, err)
					return
				}
			}
			var chunk rawChatCompletion
			if err := json.Unmarshal(payload, &chunk); err != nil {
				yield(nil, err)
				return
			}
			events, err := state.mapChunk(chunk, payload)
			if err != nil {
				yield(nil, err)
				return
			}
			for _, event := range events {
				if !yield(event, nil) {
					return
				}
			}
		}
		for _, event := range state.finish() {
			if !yield(event, nil) {
				return
			}
		}
		if !state.everStarted {
			yield(nil, p.emptyStreamError())
		}
	}
}

// sniffStreamError detects choice-less SSE data events that carry an error
// payload — vLLM emits these after HTTP 200 when generation fails mid-stream
// (Compat.SniffMidStreamErrors). Both the nested {"error":{...}} and the
// legacy flat {"object":"error","message":...,"code":N} shapes are
// recognized; events that carry a "choices" key (even an empty array, like
// trailing usage chunks) are never treated as errors here — in-choice errors
// stay with mapChunk.
func (p *Provider) sniffStreamError(payload []byte) error {
	var probe struct {
		Object  string          `json:"object"`
		Choices json.RawMessage `json:"choices"`
		Error   *rawError       `json:"error"`
		Message string          `json:"message"`
		Code    any             `json:"code"`
	}
	if err := json.Unmarshal(payload, &probe); err != nil {
		return nil // let the chunk decoder report malformed payloads
	}
	if probe.Choices != nil {
		return nil
	}
	if probe.Error != nil && (probe.Error.Message != "" || probe.Error.Code != nil) {
		return p.mapChunkError(probe.Error, payload)
	}
	if probe.Object == "error" {
		return p.mapChunkError(&rawError{Code: probe.Code, Message: probe.Message}, payload)
	}
	return nil
}

// emptyStreamError normalizes the empty-but-2xx SSE stream case (B4): zero
// events before EOF must surface as an in-stream ErrServer, never as a
// silent empty success that Collect would return with a nil error.
func (p *Provider) emptyStreamError() error {
	return &llm.ProviderError{
		Provider: p.Name(),
		Message:  "provider returned an empty stream",
		Kind:     llm.ErrServer,
	}
}

func (p *Provider) openStream(ctx context.Context, req *llm.Request, body []byte) (*http.Response, error) {
	var lastErr error
	for attempt := 0; attempt <= p.maxRetries; attempt++ {
		if attempt > 0 {
			if err := waitBeforeRetry(ctx, lastErr, attempt); err != nil {
				return nil, err
			}
		}
		resp, err := p.doStreamRequest(ctx, req, body)
		if err != nil {
			lastErr = p.mapError(err)
			if !streamRetryableError(ctx, err) || attempt == p.maxRetries {
				return nil, lastErr
			}
			continue
		}
		if retryableStreamStatus(resp.StatusCode) {
			lastErr = p.mapHTTPResponseError(resp)
			resp.Body.Close()
			if attempt == p.maxRetries {
				return nil, lastErr
			}
			continue
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			defer resp.Body.Close()
			return nil, p.mapHTTPResponseError(resp)
		}
		return resp, nil
	}
	return nil, lastErr
}

func (p *Provider) doStreamRequest(ctx context.Context, req *llm.Request, body []byte) (*http.Response, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(p.baseURL, "/")+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	key := p.apiKey
	if p.apiKeyFunc != nil {
		key, err = p.apiKeyFunc(ctx)
		if err != nil {
			return nil, err
		}
	}
	if key != "" {
		// Keyless providers (self-hosted servers) send no Authorization header.
		httpReq.Header.Set("Authorization", "Bearer "+key)
	}
	applyHeaders(httpReq.Header, p.requestHeaders(req))
	return p.httpClient.Do(httpReq)
}

func waitBeforeRetry(ctx context.Context, err error, attempt int) error {
	delay := streamRetryBackoff(attempt)
	var providerErr *llm.ProviderError
	if errors.As(err, &providerErr) && providerErr.RetryAfter > 0 {
		delay = providerErr.RetryAfter
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
	}
	return nil
}

func streamRetryBackoff(attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	delay := 100 * time.Millisecond
	for range min(attempt-1, 5) {
		delay *= 2
	}
	jitter := time.Duration((attempt*37)%100) * time.Millisecond
	delay += jitter
	if delay > 2*time.Second {
		return 2 * time.Second
	}
	return delay
}

func streamRetryableError(ctx context.Context, err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil {
		return false
	}
	return true
}

func retryableStreamStatus(status int) bool {
	return status == http.StatusRequestTimeout || status == http.StatusTooManyRequests || status >= 500
}

// withStreamEnabled splices "stream": true into the marshaled params.
// Numbers are decoded as json.Number so giant integers survive the
// round-trip verbatim instead of being mangled through float64.
func withStreamEnabled(body []byte) ([]byte, error) {
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var obj map[string]any
	if err := dec.Decode(&obj); err != nil {
		return nil, err
	}
	obj["stream"] = true
	return json.Marshal(obj)
}

func ssePayloads(r io.Reader) iter.Seq2[json.RawMessage, error] {
	return func(yield func(json.RawMessage, error) bool) {
		scanner := bufio.NewScanner(r)
		scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)
		var data bytes.Buffer
		flush := func() bool {
			if data.Len() == 0 {
				return true
			}
			payload := bytes.TrimSpace(data.Bytes())
			data.Reset()
			if len(payload) == 0 {
				return true
			}
			return yield(append(json.RawMessage(nil), payload...), nil)
		}
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				if !flush() {
					return
				}
				continue
			}
			if strings.HasPrefix(line, ":") {
				continue
			}
			if strings.HasPrefix(line, "data:") {
				if data.Len() > 0 {
					data.WriteByte('\n')
				}
				data.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			}
		}
		if err := scanner.Err(); err != nil {
			yield(nil, err)
			return
		}
		flush()
	}
}

type streamState struct {
	provider *Provider
	id       string
	model    string
	started  bool
	// everStarted stays true once any chunk arrived (started resets when
	// finish emits MessageEnd); it distinguishes a finished stream from one
	// that produced zero events (B4).
	everStarted  bool
	usage        *rawUsage
	finishReason string
	extraRoot    jsonObject
	extraChoice  rawChoice
	tools        map[int]*streamToolCall
	seenIDs      map[string]struct{}
	// toolCallsEmitted flips once any tool call completes (ToolCallEnd), so
	// MessageEnd can normalize an end-turn finish to tool_use (FS §5) —
	// matching the blocking path for servers whose forced tool calls end
	// with finish_reason "stop" (vLLM).
	toolCallsEmitted bool
	// reasoningDetails accumulates the elements of every chunk's
	// reasoning_details array per reasoning block index. ReasoningDelta.Raw
	// has REPLACE semantics (ARCH §2.5), so per-chunk emission would truncate
	// the block to its last chunk; the merged array is emitted as Raw once,
	// when the block is complete (stream end).
	reasoningDetails map[int][]json.RawMessage
	// reasoningTextBlocks records reasoning block indexes that streamed plain
	// text; blocks that never accumulate reasoning_details get a terminal
	// Provider-tagging ReasoningDelta so Collect matches the blocking path's
	// ReasoningPart.Provider (vLLM streams text-only reasoning).
	reasoningTextBlocks map[int]struct{}
	// annotations accumulates annotation elements across chunks for the
	// dialect's terminal extras.
	annotations []json.RawMessage
}

type streamToolCall struct {
	id         string
	name       string
	args       strings.Builder
	started    bool
	emittedLen int
}

func newStreamState(p *Provider) *streamState {
	return &streamState{
		provider:            p,
		tools:               map[int]*streamToolCall{},
		seenIDs:             map[string]struct{}{},
		reasoningDetails:    map[int][]json.RawMessage{},
		reasoningTextBlocks: map[int]struct{}{},
	}
}

func (s *streamState) mapChunk(chunk rawChatCompletion, raw []byte) ([]llm.Event, error) {
	var events []llm.Event
	if !s.started {
		s.started = true
		s.everStarted = true
		s.id = chunk.ID
		s.model = chunk.Model
		events = append(events, llm.MessageStart{ID: chunk.ID, Provider: s.provider.Name(), Model: chunk.Model})
	}
	if chunk.Usage.Raw != nil || chunk.Usage.TotalTokens != 0 || chunk.Usage.PromptTokens != 0 || chunk.Usage.CompletionTokens != 0 || chunk.Usage.Cost != nil {
		copied := chunk.Usage
		s.usage = &copied
	}
	s.captureExtras(chunk)
	for _, choice := range chunk.Choices {
		if choice.Error != nil || choice.FinishReason == "error" {
			if choice.Error == nil {
				choice.Error = &rawError{Code: "error", Message: "stream chunk finished with error"}
			}
			return events, s.provider.mapChunkError(choice.Error, raw)
		}
		if choice.FinishReason != "" {
			s.finishReason = choice.FinishReason
		}
		events = append(events, s.mapDelta(choice)...)
	}
	// MessageEnd is emitted only at stream end ([DONE] or EOF) via finish:
	// providers such as OpenRouter deliver usage/cost on a trailing
	// empty-choices chunk after finish_reason, and ending the message on the
	// first usage-bearing chunk would drop anything that follows.
	return events, nil
}

func (s *streamState) captureExtras(chunk rawChatCompletion) {
	if len(chunk.Raw) > 0 {
		if s.extraRoot == nil {
			s.extraRoot = jsonObject{}
		}
		for key, value := range chunk.Raw {
			if key == "choices" {
				continue
			}
			s.extraRoot[key] = value
		}
	}
	if len(chunk.Choices) == 0 {
		return
	}
	choice := chunk.Choices[0]
	if len(choice.Raw) > 0 {
		if s.extraChoice.Raw == nil {
			s.extraChoice.Raw = jsonObject{}
		}
		for key, value := range choice.Raw {
			if key == "delta" || key == "message" {
				continue
			}
			s.extraChoice.Raw[key] = value
		}
	}
	if choice.FinishReason != "" {
		s.extraChoice.FinishReason = choice.FinishReason
	}
	// Annotations and reasoning_details arrive as per-chunk ARRAYS that must
	// be accumulated — overwriting would keep only the last chunk's slice.
	s.annotations = appendRawArrayElements(s.annotations, choice.Message.Annotations)
	s.annotations = appendRawArrayElements(s.annotations, choice.Delta.Annotations)
	if len(choice.Message.ReasoningDetails) > 0 {
		s.appendReasoningDetails(choice.Index*streamBlockStride, choice.Message.ReasoningDetails)
	}
}

// appendReasoningDetails accumulates one chunk's reasoning_details array for
// the reasoning block at index.
func (s *streamState) appendReasoningDetails(index int, details json.RawMessage) {
	s.reasoningDetails[index] = appendRawArrayElements(s.reasoningDetails[index], details)
}

// appendRawArrayElements appends the elements of a raw JSON array to dst. A
// non-array payload (defensive: providers should always send arrays here) is
// kept whole as a single element.
func appendRawArrayElements(dst []json.RawMessage, raw json.RawMessage) []json.RawMessage {
	if len(raw) == 0 {
		return dst
	}
	var elements []json.RawMessage
	if err := json.Unmarshal(raw, &elements); err != nil {
		return append(dst, append(json.RawMessage(nil), raw...))
	}
	return append(dst, elements...)
}

// joinRawArray reassembles accumulated array elements into one JSON array,
// byte-preserving each element as it appeared on the wire.
func joinRawArray(elements []json.RawMessage) json.RawMessage {
	var buf bytes.Buffer
	buf.WriteByte('[')
	for i, element := range elements {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.Write(bytes.TrimSpace(element))
	}
	buf.WriteByte(']')
	return buf.Bytes()
}

// streamBlockStride spaces the per-choice block indexes: reasoning at
// choice*stride, text at +1, tool calls from +2.
const streamBlockStride = 10

func (s *streamState) mapDelta(choice rawChoice) []llm.Event {
	reasoningIndex := choice.Index * streamBlockStride
	textIndex := reasoningIndex + 1
	toolOffset := reasoningIndex + 2
	var events []llm.Event
	if text := choice.Delta.reasoningText(); text != "" {
		s.reasoningTextBlocks[reasoningIndex] = struct{}{}
		events = append(events, llm.ReasoningDelta{Index: reasoningIndex, Text: text})
	}
	if len(choice.Delta.ReasoningDetails) > 0 {
		// Accumulate only; the merged reasoning_details array is emitted as
		// ReasoningDelta.Raw (REPLACE semantics) once the block is complete.
		s.appendReasoningDetails(reasoningIndex, choice.Delta.ReasoningDetails)
	}
	if choice.Delta.Content != "" {
		events = append(events, llm.TextDelta{Index: textIndex, Text: choice.Delta.Content})
	}
	for _, call := range choice.Delta.ToolCalls {
		key := toolOffset
		if call.Index != nil {
			key = toolOffset + *call.Index
		}
		events = append(events, s.mapToolDelta(key, call)...)
	}
	if choice.FinishReason == "tool_calls" {
		events = append(events, s.finishPendingTools()...)
	}
	return events
}

// finishPendingTools flushes open tool-call blocks in deterministic
// ascending-index order.
func (s *streamState) finishPendingTools() []llm.Event {
	var events []llm.Event
	for _, key := range slices.Sorted(maps.Keys(s.tools)) {
		events = append(events, s.finishTool(key, s.tools[key])...)
	}
	return events
}

func (s *streamState) mapToolDelta(index int, delta rawToolCall) []llm.Event {
	call := s.tools[index]
	if call == nil {
		call = &streamToolCall{}
		s.tools[index] = call
	}
	if call.id == "" && delta.ID != "" {
		if _, exists := s.seenIDs[delta.ID]; !exists {
			call.id = delta.ID
			s.seenIDs[delta.ID] = struct{}{}
		}
	}
	if call.name == "" && delta.Function.Name != "" {
		call.name = delta.Function.Name
	}
	if delta.Function.Arguments != "" {
		call.args.WriteString(delta.Function.Arguments)
	}
	return startTool(index, call)
}

func startTool(index int, call *streamToolCall) []llm.Event {
	if call.id == "" || call.name == "" {
		return nil
	}
	var events []llm.Event
	if !call.started {
		call.started = true
		events = append(events, llm.ToolCallStart{Index: index, ID: call.id, Name: call.name})
	}
	if call.args.Len() > call.emittedLen {
		args := call.args.String()
		events = append(events, llm.ToolCallDelta{Index: index, ArgsFragment: args[call.emittedLen:]})
		call.emittedLen = len(args)
	}
	return events
}

func (s *streamState) finishTool(index int, call *streamToolCall) []llm.Event {
	if call.id == "" {
		call.id = providerutil.UniqueSyntheticToolCallID(index, s.seenIDs)
		s.seenIDs[call.id] = struct{}{}
	}
	if call.name == "" {
		delete(s.tools, index)
		return []llm.Event{llm.ToolCallDropped{Index: index, Reason: "missing tool name"}}
	}
	args := strings.TrimSpace(call.args.String())
	if args == "" {
		args = "{}"
	}
	if !json.Valid([]byte(args)) {
		delete(s.tools, index)
		return []llm.Event{llm.ToolCallDropped{Index: index, Reason: "invalid tool arguments JSON"}}
	}
	events := startTool(index, call)
	if call.emittedLen == 0 {
		events = append(events, llm.ToolCallDelta{Index: index, ArgsFragment: args})
		call.emittedLen = len(args)
	}
	events = append(events, llm.ToolCallEnd{Index: index})
	s.toolCallsEmitted = true
	delete(s.tools, index)
	return events
}

func (s *streamState) finish() []llm.Event {
	if !s.started {
		return nil
	}
	return s.finishEvents()
}

func (s *streamState) finishEvents() []llm.Event {
	events := s.completedReasoningRawEvents()
	events = append(events, s.finishPendingTools()...)
	usage := llm.Usage{}
	if s.usage != nil {
		usage = s.provider.dialect.MapUsage(s.model, *s.usage, s.provider.priceTable)
	}
	events = append(events, llm.MessageEnd{
		StopReason:    s.provider.normalizeToolUseStop(s.provider.dialect.MapStopReason(s.finishReason), s.toolCallsEmitted),
		StopReasonRaw: s.finishReason,
		Usage:         usage,
		Raw:           s.messageEndRaw(),
	})
	s.started = false
	return events
}

// completedReasoningRawEvents emits each reasoning block's merged
// reasoning_details array exactly once, now that the blocks are complete —
// matching the non-streaming mapping so Collect reconstructs a byte-identical
// ReasoningPart.Raw for replay. Text-only reasoning blocks (no
// reasoning_details anywhere in the stream — vLLM) get a terminal
// Provider-tagging delta instead so the collected ReasoningPart.Provider
// matches the blocking path.
func (s *streamState) completedReasoningRawEvents() []llm.Event {
	var events []llm.Event
	var all []json.RawMessage
	for _, index := range slices.Sorted(maps.Keys(s.reasoningTextBlocks)) {
		if _, hasRaw := s.reasoningDetails[index]; !hasRaw {
			events = append(events, llm.ReasoningDelta{Index: index, Provider: s.provider.Name()})
		}
	}
	for _, index := range slices.Sorted(maps.Keys(s.reasoningDetails)) {
		elements := s.reasoningDetails[index]
		if len(elements) == 0 {
			continue
		}
		events = append(events, llm.ReasoningDelta{Index: index, Raw: joinRawArray(elements), Provider: s.provider.Name()})
		all = append(all, elements...)
	}
	if len(all) > 0 {
		s.extraChoice.Message.ReasoningDetails = joinRawArray(all)
	}
	if len(s.annotations) > 0 {
		s.extraChoice.Message.Annotations = joinRawArray(s.annotations)
	}
	return events
}

func (s *streamState) messageEndRaw() any {
	if len(s.extraRoot) > 0 || len(s.extraChoice.Raw) > 0 || len(s.extraChoice.Message.Annotations) > 0 || len(s.extraChoice.Message.ReasoningDetails) > 0 {
		return s.provider.dialect.ExtractExtras(s.extraRoot, s.extraChoice)
	}
	return nil
}
