package chatcompletions_test

import (
	"context"
	"errors"
	"io"
	"iter"
	"net/http"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	sdk "github.com/openai/openai-go/v3"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/e2e"
	"github.com/pkieltyka/go-llm/internal/testutil"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
)

// replayDialect is a plain OpenAI-compatible dialect with no quirks: it
// defers part extraction to the adapter's default chat-completions mapping
// and uses the shared error-kind table, so replaying the recorded OpenRouter
// corpus (a standard chat-completions wire shape) exercises the adapter's own
// convert/response/stream/error paths directly.
type replayDialect struct{}

type choiceExtrasDialect struct{ replayDialect }

func (choiceExtrasDialect) ExtractExtras(_ chatcompletions.JSONObject, choice chatcompletions.RawChoice) any {
	return choice.Raw
}

const replayDialectName = "chatcompletions-replay"

func (replayDialect) Name() string           { return replayDialectName }
func (replayDialect) DefaultBaseURL() string { return "https://replay.invalid/v1" }
func (replayDialect) APIKeyEnv() string      { return "CHATCOMPLETIONS_REPLAY_API_KEY" }

func (replayDialect) Capabilities() []llm.Capability {
	return []llm.Capability{
		llm.CapabilityStreaming,
		llm.CapabilityTools,
		llm.CapabilityToolChoiceRequired,
		llm.CapabilityToolStreaming,
		llm.CapabilityParallelTools,
		llm.CapabilityStrictTools,
		llm.CapabilityJSONSchema,
		llm.CapabilityJSONMode,
		llm.CapabilityReasoning,
		llm.CapabilityImageInput,
		llm.CapabilityStopSequences,
		llm.CapabilityModelsListing,
	}
}

func (replayDialect) Compat() chatcompletions.Compat {
	return chatcompletions.Compat{StreamIncludeUsage: true}
}

func (replayDialect) ApplyRequest(*llm.Request, *sdk.ChatCompletionNewParams, chatcompletions.JSONObject) error {
	return nil
}

func (replayDialect) MapStopReason(raw string) llm.StopReason {
	switch raw {
	case "stop":
		return llm.StopReasonEndTurn
	case "length":
		return llm.StopReasonMaxTokens
	case "tool_calls", "function_call":
		return llm.StopReasonToolUse
	case "content_filter":
		return llm.StopReasonContentFilter
	case "error":
		return llm.StopReasonError
	case "":
		return ""
	default:
		return llm.StopReasonOther
	}
}

func (replayDialect) MapErrorStatus(status int, code, message string) error {
	return chatcompletions.DefaultErrorKind(status, code, message)
}

func (replayDialect) ExtractParts(chatcompletions.JSONObject, chatcompletions.RawMessage) ([]llm.Part, []llm.DroppedToolCall, error) {
	return nil, nil, nil
}

func (replayDialect) ExtractExtras(chatcompletions.JSONObject, chatcompletions.RawChoice) any {
	return nil
}

func (replayDialect) MapUsage(model string, raw chatcompletions.RawUsage, table llm.PriceTable) llm.Usage {
	cacheRead := raw.PromptTokensDetails.CachedTokens
	inputTokens := raw.PromptTokens
	if cacheRead > 0 && inputTokens >= cacheRead {
		inputTokens -= cacheRead
	}
	out := llm.Usage{
		InputTokens:     inputTokens,
		CacheReadTokens: cacheRead,
		OutputTokens:    raw.CompletionTokens,
		ReasoningTokens: raw.CompletionTokensDetails.ReasoningTokens,
		TotalTokens:     raw.TotalTokens,
		Raw:             raw.Raw,
	}
	if out.TotalTokens == 0 {
		out.TotalTokens = out.InputTokens + out.CacheReadTokens + out.OutputTokens
	}
	return out
}

func (replayDialect) Models(ctx context.Context, p *chatcompletions.Provider) ([]llm.ModelInfo, error) {
	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := p.DoJSON(ctx, http.MethodGet, "/models", nil, &payload); err != nil {
		return nil, err
	}
	models := make([]llm.ModelInfo, 0, len(payload.Data))
	for _, row := range payload.Data {
		models = append(models, llm.ModelInfo{ID: row.ID})
	}
	return models, nil
}

// TestReplayRecordedFixtures drives the shared chat-completions adapter
// (blocking, SSE streaming, models listing, error mapping) offline against
// the recorded OpenRouter live corpus using a quirk-free test dialect.
func TestReplayRecordedFixtures(t *testing.T) {
	root, err := e2e.RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	fixture := filepath.Join(root, "internal", "e2e", "fixtures", "openrouter", "live.json")
	e2e.ReplayExchanges(t, fixture, e2e.ReplayProfile{
		Provider: replayDialectName,
		New: func(t *testing.T, client *http.Client) llm.Provider {
			p, err := chatcompletions.NewWithDialect(chatcompletions.Config{
				Dialect:    replayDialect{},
				APIKey:     "replay-key",
				HTTPClient: client,
				MaxRetries: new(int),
			})
			if err != nil {
				t.Fatalf("chatcompletions.NewWithDialect returned error: %v", err)
			}
			return p
		},
		ToolCallMarkers:  []string{`"tool_calls"`},
		ReasoningMarkers: []string{`"reasoning_details"`},
	})
}

// TestEmptyStreamYieldsErrServer covers B4: a 2xx SSE response that ends
// (EOF or immediate [DONE]) without producing a single event must surface a
// normalized in-stream ErrServer instead of collecting into a silent empty
// success.
func TestEmptyStreamYieldsErrServer(t *testing.T) {
	for name, body := range map[string]string{
		"eof_without_data": "",
		"done_only":        "data: [DONE]\n\n",
	} {
		t.Run(name, func(t *testing.T) {
			exchange := e2e.RecordedExchange{
				Status:          http.StatusOK,
				ResponseHeaders: http.Header{"Content-Type": []string{"text/event-stream"}},
				ResponseBody:    body,
			}
			p, err := chatcompletions.NewWithDialect(chatcompletions.Config{
				Dialect:    replayDialect{},
				APIKey:     "replay-key",
				HTTPClient: e2e.NewReplayClient(exchange),
				MaxRetries: new(int),
			})
			if err != nil {
				t.Fatalf("chatcompletions.NewWithDialect returned error: %v", err)
			}
			resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{
				Model:    "replay-model",
				Messages: []llm.Message{llm.UserText("hi")},
			}))
			if !errors.Is(err, llm.ErrServer) {
				t.Fatalf("empty stream error = %v, want ErrServer", err)
			}
			if err == nil || !strings.Contains(err.Error(), "empty stream") {
				t.Fatalf("empty stream error text = %v, want mention of empty stream", err)
			}
			if resp != nil {
				t.Fatalf("empty stream response = %+v, want nil (no MessageStart seen)", resp)
			}
		})
	}
}

func newExchangeProvider(t *testing.T, dialect chatcompletions.Dialect, exchange e2e.RecordedExchange) *chatcompletions.Provider {
	t.Helper()
	p, err := chatcompletions.NewWithDialect(chatcompletions.Config{
		Dialect:    dialect,
		APIKey:     "replay-key",
		HTTPClient: e2e.NewReplayClient(exchange),
		MaxRetries: new(int),
	})
	if err != nil {
		t.Fatalf("chatcompletions.NewWithDialect returned error: %v", err)
	}
	return p
}

func streamExchange(body string) e2e.RecordedExchange {
	return e2e.RecordedExchange{
		Status:          http.StatusOK,
		ResponseHeaders: http.Header{"Content-Type": []string{"text/event-stream"}},
		ResponseBody:    body,
	}
}

func TestStreamUsesChoiceIndexZeroAndTrailingUsage(t *testing.T) {
	body := "" +
		`data: {"id":"c1","model":"m","choices":[{"index":7,"native_finish_reason":"wrong","delta":{"content":"wrong"}},{"index":0,"native_finish_reason":"right","delta":{"role":"assistant","content":"right"},"finish_reason":"stop"}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}` + "\n\n" +
		"data: [DONE]\n\n"
	p := newExchangeProvider(t, choiceExtrasDialect{}, streamExchange(body))
	resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if resp.Text() != "right" || resp.StopReason != llm.StopReasonEndTurn || resp.Usage.TotalTokens != 3 {
		t.Fatalf("response = %+v", resp)
	}
	extras, ok := resp.Raw.(chatcompletions.JSONObject)
	if !ok || extras["native_finish_reason"] != "right" {
		t.Fatalf("choice extras = %#v, want index-zero extras", resp.Raw)
	}
	if strings.Contains(resp.Text(), "wrong") {
		t.Fatalf("response included nonzero choice: %+v", resp)
	}
}

func TestBlockingUsesChoiceIndexZero(t *testing.T) {
	p := newExchangeProvider(t, replayDialect{}, e2e.RecordedExchange{
		Status:          http.StatusOK,
		ResponseHeaders: http.Header{"Content-Type": []string{"application/json"}},
		ResponseBody:    `{"id":"c1","model":"m","choices":[{"index":2,"finish_reason":"stop","message":{"content":"wrong"}},{"index":0,"finish_reason":"stop","message":{"content":"right"}}]}`,
	})
	resp, err := p.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Text() != "right" {
		t.Fatalf("response text = %q, want right", resp.Text())
	}
}

func TestStreamRejectsChoiceLessCompletion(t *testing.T) {
	for name, body := range map[string]string{
		"nonzero_only": `data: {"id":"c1","model":"m","choices":[{"index":1,"delta":{"content":"wrong"},"finish_reason":"stop"}]}` + "\n\ndata: [DONE]\n\n",
		"usage_only":   `data: {"id":"c1","model":"m","choices":[],"usage":{"total_tokens":1}}` + "\n\ndata: [DONE]\n\n",
	} {
		t.Run(name, func(t *testing.T) {
			p := newExchangeProvider(t, replayDialect{}, streamExchange(body))
			resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}))
			if !errors.Is(err, llm.ErrServer) {
				t.Fatalf("Collect error = %v, want ErrServer", err)
			}
			if resp != nil {
				t.Fatalf("response = %+v, want nil", resp)
			}
		})
	}
}

func TestStreamTruncatedEOFPreservesPartialResponse(t *testing.T) {
	body := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"partial"}}]}` + "\n\n"
	p := newExchangeProvider(t, replayDialect{}, streamExchange(body))
	resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}))
	if !errors.Is(err, llm.ErrServer) {
		t.Fatalf("Collect error = %v, want ErrServer", err)
	}
	if resp == nil || resp.Text() != "partial" {
		t.Fatalf("partial response = %+v", resp)
	}
}

func TestStreamDroppedToolIndexesMatchBlocking(t *testing.T) {
	blockingBody := `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{"reasoning":"why","content":"answer","tool_calls":[{"id":"bad","function":{"name":"lookup","arguments":"{"}},{"id":"good","function":{"name":"lookup","arguments":"{}"}}]}}]}`
	streamBody := "" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"reasoning":"why","content":"answer","tool_calls":[{"index":0,"id":"bad","function":{"name":"lookup","arguments":"{"}},{"index":1,"id":"good","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}
	blockingProvider := newExchangeProvider(t, replayDialect{}, e2e.RecordedExchange{
		Status:          http.StatusOK,
		ResponseHeaders: http.Header{"Content-Type": []string{"application/json"}},
		ResponseBody:    blockingBody,
	})
	streamProvider := newExchangeProvider(t, replayDialect{}, streamExchange(streamBody))
	blocking, err := blockingProvider.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	streamed, err := llm.Collect(streamProvider.ChatStream(context.Background(), req))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if !reflect.DeepEqual(streamed.Parts, blocking.Parts) {
		t.Fatalf("parts differ: stream=%#v blocking=%#v", streamed.Parts, blocking.Parts)
	}
	if !reflect.DeepEqual(streamed.DroppedToolCalls, blocking.DroppedToolCalls) {
		t.Fatalf("drops differ: stream=%#v blocking=%#v", streamed.DroppedToolCalls, blocking.DroppedToolCalls)
	}
	if len(streamed.DroppedToolCalls) != 1 || streamed.DroppedToolCalls[0].Index != 3 {
		t.Fatalf("drops = %#v, want stable tool position 3", streamed.DroppedToolCalls)
	}
}

func TestStreamToolCallIsIncrementallyObservable(t *testing.T) {
	body := "" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_live","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"go\"}"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	p := newExchangeProvider(t, replayDialect{}, streamExchange(body))
	var events []llm.Event
	for event, err := range p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}) {
		if err != nil {
			t.Fatalf("ChatStream returned error: %v", err)
		}
		events = append(events, event)
	}
	if len(events) != 6 {
		t.Fatalf("events = %#v, want start/tool-start/two deltas/tool-end/end", events)
	}
	start, ok := events[1].(llm.ToolCallStart)
	if !ok || start.Index != 3 || start.ID != "call_live" || start.Name != "lookup" {
		t.Fatalf("tool start = %#v", events[1])
	}
	first, ok := events[2].(llm.ToolCallDelta)
	if !ok || first.ArgsFragment != `{"q":` {
		t.Fatalf("first delta = %#v", events[2])
	}
	second, ok := events[3].(llm.ToolCallDelta)
	if !ok || second.ArgsFragment != `"go"}` {
		t.Fatalf("second delta = %#v", events[3])
	}
	if end, ok := events[4].(llm.ToolCallEnd); !ok || end.Index != 3 {
		t.Fatalf("tool end = %#v", events[4])
	}
}

func TestStreamToolCallUsesLateProviderID(t *testing.T) {
	streamBody := "" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"real_call_id","function":{"arguments":"\"go\"}"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	blockingBody := `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{"tool_calls":[{"id":"real_call_id","function":{"name":"lookup","arguments":"{\"q\":\"go\"}"}}]}}]}`
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}
	streamProvider := newExchangeProvider(t, replayDialect{}, streamExchange(streamBody))
	events := collectStreamEvents(t, streamProvider.ChatStream(context.Background(), req))
	var starts []llm.ToolCallStart
	startPosition, endPosition := -1, -1
	for position, event := range events {
		switch event := event.(type) {
		case llm.ToolCallStart:
			starts = append(starts, event)
			startPosition = position
		case llm.ToolCallEnd:
			endPosition = position
		}
	}
	if len(starts) != 1 || starts[0].ID != "real_call_id" || starts[0].Index != 3 || startPosition < 0 || endPosition <= startPosition {
		t.Fatalf("late-ID events = %#v, want one incremental real-ID call", events)
	}
	streamed, err := llm.Collect(testutil.EventSeq(events...))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	blockingProvider := newExchangeProvider(t, replayDialect{}, e2e.RecordedExchange{
		Status:          http.StatusOK,
		ResponseHeaders: http.Header{"Content-Type": []string{"application/json"}},
		ResponseBody:    blockingBody,
	})
	blocking, err := blockingProvider.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if !reflect.DeepEqual(streamed.Parts, blocking.Parts) {
		t.Fatalf("stream/blocking parts differ: %#v / %#v", streamed.Parts, blocking.Parts)
	}
}

func TestStreamToolCallSynthesizesIDOnlyOnError(t *testing.T) {
	body := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}` + "\n\n" +
		"data: {not-json}\n\n"
	p := newExchangeProvider(t, replayDialect{}, streamExchange(body))
	var events []llm.Event
	seq := p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}})
	resp, err := llm.Collect(func(yield func(llm.Event, error) bool) {
		for event, streamErr := range seq {
			if streamErr == nil {
				events = append(events, event)
			}
			if !yield(event, streamErr) {
				return
			}
		}
	})
	if !errors.Is(err, llm.ErrServer) {
		t.Fatalf("Collect error = %v, want ErrServer", err)
	}
	var starts []llm.ToolCallStart
	for _, event := range events {
		if start, ok := event.(llm.ToolCallStart); ok {
			starts = append(starts, start)
		}
	}
	if len(starts) != 1 || starts[0].ID != "call_3" || starts[0].Index != 3 {
		t.Fatalf("error-rescued starts = %#v, want deterministic synthetic ID", starts)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "call_3" || string(calls[0].Args) != `{"q":` {
		t.Fatalf("partial calls = %#v", calls)
	}
}

func TestToolAndLaterTextUseStableIndexes(t *testing.T) {
	body := "" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_0","function":{"name":"lookup","arguments":"{}"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"after"},"finish_reason":"tool_calls"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	p := newExchangeProvider(t, replayDialect{}, streamExchange(body))
	events := collectStreamEvents(t, p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}))
	var toolIndex, textIndex = -1, -1
	for _, event := range events {
		switch event := event.(type) {
		case llm.ToolCallStart:
			toolIndex = event.Index
		case llm.TextDelta:
			textIndex = event.Index
		}
	}
	if toolIndex != 3 || textIndex != 1 {
		t.Fatalf("tool/text indexes = %d/%d, want stable positions 3/1; events=%#v", toolIndex, textIndex, events)
	}
	resp, err := llm.Collect(testutil.EventSeq(events...))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if len(resp.Parts) != 2 {
		t.Fatalf("parts = %#v, want tool then text", resp.Parts)
	}
	if text, ok := resp.Parts[0].(llm.TextPart); !ok || text.Text != "after" {
		t.Fatalf("part 0 = %#v, want text at stable content position", resp.Parts[0])
	}
	if _, ok := resp.Parts[1].(llm.ToolCallPart); !ok {
		t.Fatalf("part 1 = %T, want ToolCallPart at stable tool position", resp.Parts[1])
	}
}

func TestMalformedCallReservesIndexBeforeRetainedTool(t *testing.T) {
	streamBody := "" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"answer"}}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"bad","function":{"name":"lookup","arguments":"{"}},{"index":1,"id":"good","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	blockingBody := `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{"content":"answer","tool_calls":[{"id":"bad","function":{"name":"lookup","arguments":"{"}},{"id":"good","function":{"name":"lookup","arguments":"{}"}}]}}]}`
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}
	streamProvider := newExchangeProvider(t, replayDialect{}, streamExchange(streamBody))
	events := collectStreamEvents(t, streamProvider.ChatStream(context.Background(), req))
	var textIndex, retainedToolIndex, droppedIndex = -1, -1, -1
	for _, event := range events {
		switch event := event.(type) {
		case llm.TextDelta:
			textIndex = event.Index
		case llm.ToolCallStart:
			if event.ID == "good" {
				retainedToolIndex = event.Index
			}
		case llm.ToolCallDropped:
			droppedIndex = event.Index
		}
	}
	if textIndex != 1 || droppedIndex != 3 || retainedToolIndex != 4 {
		t.Fatalf("text/drop/retained indexes = %d/%d/%d, want stable positions 1/3/4; events=%#v", textIndex, droppedIndex, retainedToolIndex, events)
	}
	streamed, err := llm.Collect(testutil.EventSeq(events...))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	blockingProvider := newExchangeProvider(t, replayDialect{}, e2e.RecordedExchange{
		Status:          http.StatusOK,
		ResponseHeaders: http.Header{"Content-Type": []string{"application/json"}},
		ResponseBody:    blockingBody,
	})
	blocking, err := blockingProvider.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if !reflect.DeepEqual(streamed.Parts, blocking.Parts) || !reflect.DeepEqual(streamed.DroppedToolCalls, blocking.DroppedToolCalls) {
		t.Fatalf("stream/blocking differ: %#v %#v / %#v %#v", streamed.Parts, streamed.DroppedToolCalls, blocking.Parts, blocking.DroppedToolCalls)
	}
}

func TestParallelToolCallsStreamIncrementallyAndCollectEquivalent(t *testing.T) {
	streamBody := "" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"lookup","arguments":"{\"q\":"}},{"index":1,"id":"call_b","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"two\"}"}},{"index":0,"function":{"arguments":"\"one\"}"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	blockingBody := `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{"tool_calls":[{"id":"call_a","function":{"name":"lookup","arguments":"{\"q\":\"one\"}"}},{"id":"call_b","function":{"name":"lookup","arguments":"{\"q\":\"two\"}"}}]}}]}`
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}
	streamProvider := newExchangeProvider(t, replayDialect{}, streamExchange(streamBody))
	var events []llm.Event
	for event, err := range streamProvider.ChatStream(context.Background(), req) {
		if err != nil {
			t.Fatalf("ChatStream returned error: %v", err)
		}
		events = append(events, event)
	}
	startsBeforeFinish := map[int]bool{}
	deltasBeforeFinish := map[int]bool{}
	for _, event := range events {
		switch event := event.(type) {
		case llm.ToolCallStart:
			startsBeforeFinish[event.Index] = true
		case llm.ToolCallDelta:
			deltasBeforeFinish[event.Index] = true
		case llm.ToolCallEnd:
			if !startsBeforeFinish[event.Index] || !deltasBeforeFinish[event.Index] {
				t.Fatalf("tool %d ended before incremental start/delta: %#v", event.Index, events)
			}
		}
	}
	if !startsBeforeFinish[3] || !startsBeforeFinish[4] || !deltasBeforeFinish[3] || !deltasBeforeFinish[4] {
		t.Fatalf("parallel calls were not both observable: %#v", events)
	}
	streamed, err := llm.Collect(func(yield func(llm.Event, error) bool) {
		for _, event := range events {
			if !yield(event, nil) {
				return
			}
		}
	})
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	blockingProvider := newExchangeProvider(t, replayDialect{}, e2e.RecordedExchange{
		Status:          http.StatusOK,
		ResponseHeaders: http.Header{"Content-Type": []string{"application/json"}},
		ResponseBody:    blockingBody,
	})
	blocking, err := blockingProvider.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if !reflect.DeepEqual(streamed.Parts, blocking.Parts) || !reflect.DeepEqual(streamed.DroppedToolCalls, blocking.DroppedToolCalls) {
		t.Fatalf("stream/blocking differ: %#v %#v / %#v %#v", streamed.Parts, streamed.DroppedToolCalls, blocking.Parts, blocking.DroppedToolCalls)
	}
}

func TestReversedArrivalDuplicateToolIDStableOwnership(t *testing.T) {
	firstChunk := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"dup","function":{"name":"second","arguments":"{\"n\":1}"}}]}}]}` + "\n\n"
	secondChunk := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"dup","function":{"name":"first","arguments":"{\"n\":0}"}}]}}]}` + "\n\n"
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}

	t.Run("complete", func(t *testing.T) {
		streamBody := firstChunk + secondChunk +
			`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		blockingBody := `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{"tool_calls":[{"index":1,"id":"dup","function":{"name":"second","arguments":"{\"n\":1}"}},{"index":0,"id":"dup","function":{"name":"first","arguments":"{\"n\":0}"}}]}}]}`
		streamProvider := newExchangeProvider(t, replayDialect{}, streamExchange(streamBody))
		events := collectStreamEvents(t, streamProvider.ChatStream(context.Background(), req))
		starts := map[int]string{}
		for _, event := range events {
			if start, ok := event.(llm.ToolCallStart); ok {
				starts[start.Index] = start.ID
			}
		}
		wantStarts := map[int]string{3: "dup", 4: "call_4"}
		if !reflect.DeepEqual(starts, wantStarts) {
			t.Fatalf("stream starts = %#v, want lowest-index ownership %#v", starts, wantStarts)
		}
		streamed, err := llm.Collect(testutil.EventSeq(events...))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		blockingProvider := newExchangeProvider(t, replayDialect{}, e2e.RecordedExchange{
			Status:          http.StatusOK,
			ResponseHeaders: http.Header{"Content-Type": []string{"application/json"}},
			ResponseBody:    blockingBody,
		})
		blocking, err := blockingProvider.Chat(context.Background(), req)
		if err != nil {
			t.Fatalf("Chat returned error: %v", err)
		}
		if !reflect.DeepEqual(streamed.Parts, blocking.Parts) {
			t.Fatalf("stream/blocking parts differ: %#v / %#v", streamed.Parts, blocking.Parts)
		}
	})

	t.Run("partial_error", func(t *testing.T) {
		p := newExchangeProvider(t, replayDialect{}, streamExchange(firstChunk+secondChunk+"data: {not-json}\n\n"))
		resp, err := llm.Collect(p.ChatStream(context.Background(), req))
		if !errors.Is(err, llm.ErrServer) {
			t.Fatalf("Collect error = %v, want ErrServer", err)
		}
		calls := resp.ToolCalls()
		if len(calls) != 2 || calls[0].ID != "dup" || calls[0].Name != "first" || calls[1].ID != "call_4" || calls[1].Name != "second" {
			t.Fatalf("partial calls = %#v, want stable duplicate ownership", calls)
		}
	})
}

func TestLateLowerDuplicateToolIDStableOwnership(t *testing.T) {
	firstChunk := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"first","arguments":"{\"n\":0}"}}]}}]}` + "\n\n"
	higherChunk := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"dup","function":{"name":"second","arguments":"{\"n\":1}"}}]}}]}` + "\n\n"
	lowerIDChunk := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"dup","function":{}}]}}]}` + "\n\n"
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}

	assertOwnership := func(t *testing.T, resp *llm.Response) {
		t.Helper()
		calls := resp.ToolCalls()
		if len(calls) != 2 || calls[0].ID != "dup" || calls[0].Name != "first" || calls[1].ID != "call_4" || calls[1].Name != "second" {
			t.Fatalf("calls = %#v, want lowest-index explicit ownership", calls)
		}
	}

	t.Run("complete", func(t *testing.T) {
		streamBody := firstChunk + higherChunk + lowerIDChunk +
			`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
			"data: [DONE]\n\n"
		blockingBody := `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{"tool_calls":[{"index":1,"id":"dup","function":{"name":"second","arguments":"{\"n\":1}"}},{"index":0,"id":"dup","function":{"name":"first","arguments":"{\"n\":0}"}}]}}]}`
		streamProvider := newExchangeProvider(t, replayDialect{}, streamExchange(streamBody))
		streamed, err := llm.Collect(streamProvider.ChatStream(context.Background(), req))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		assertOwnership(t, streamed)
		blockingProvider := newExchangeProvider(t, replayDialect{}, e2e.RecordedExchange{
			Status:          http.StatusOK,
			ResponseHeaders: http.Header{"Content-Type": []string{"application/json"}},
			ResponseBody:    blockingBody,
		})
		blocking, err := blockingProvider.Chat(context.Background(), req)
		if err != nil {
			t.Fatalf("Chat returned error: %v", err)
		}
		if !reflect.DeepEqual(streamed.Parts, blocking.Parts) {
			t.Fatalf("stream/blocking parts differ: %#v / %#v", streamed.Parts, blocking.Parts)
		}
	})

	t.Run("partial_error", func(t *testing.T) {
		p := newExchangeProvider(t, replayDialect{}, streamExchange(firstChunk+higherChunk+lowerIDChunk+"data: {not-json}\n\n"))
		resp, err := llm.Collect(p.ChatStream(context.Background(), req))
		if !errors.Is(err, llm.ErrServer) {
			t.Fatalf("Collect error = %v, want ErrServer", err)
		}
		assertOwnership(t, resp)
	})
}

func TestSparseHugeToolIndexIsBounded(t *testing.T) {
	const position = 1_000_000_000
	streamBody := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1000000000,"id":"huge","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	blockingBody := `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{"tool_calls":[{"index":1000000000,"id":"huge","function":{"name":"lookup","arguments":"{}"}}]}}]}`
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}
	streamProvider := newExchangeProvider(t, replayDialect{}, streamExchange(streamBody))
	events := collectStreamEvents(t, streamProvider.ChatStream(context.Background(), req))
	var startIndex = -1
	for _, event := range events {
		if start, ok := event.(llm.ToolCallStart); ok {
			startIndex = start.Index
		}
	}
	if startIndex != position+3 {
		t.Fatalf("start index = %d, want sparse stable index %d", startIndex, position+3)
	}
	streamed, err := llm.Collect(testutil.EventSeq(events...))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	blockingProvider := newExchangeProvider(t, replayDialect{}, e2e.RecordedExchange{
		Status:          http.StatusOK,
		ResponseHeaders: http.Header{"Content-Type": []string{"application/json"}},
		ResponseBody:    blockingBody,
	})
	blocking, err := blockingProvider.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if !reflect.DeepEqual(streamed.Parts, blocking.Parts) {
		t.Fatalf("stream/blocking parts differ: %#v / %#v", streamed.Parts, blocking.Parts)
	}
}

func TestParallelToolCallsRemainObservableWhenStreamFails(t *testing.T) {
	body := "" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"lookup","arguments":"{\"q\":"}},{"index":1,"id":"call_b","function":{"name":"lookup","arguments":"{\"q\":"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\"partial"}}]}}]}` + "\n\n" +
		"data: {not-json}\n\n"
	p := newExchangeProvider(t, replayDialect{}, streamExchange(body))
	resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}))
	if !errors.Is(err, llm.ErrServer) {
		t.Fatalf("Collect error = %v, want ErrServer", err)
	}
	if resp == nil {
		t.Fatal("partial response is nil")
	}
	calls := resp.ToolCalls()
	if len(calls) != 2 {
		t.Fatalf("partial tool calls = %#v, want both calls", calls)
	}
	if calls[0].ID != "call_a" || string(calls[0].Args) != `{"q":` {
		t.Fatalf("first partial call = %#v", calls[0])
	}
	if calls[1].ID != "call_b" || string(calls[1].Args) != `{"q":"partial` {
		t.Fatalf("second partial call = %#v", calls[1])
	}
}

func TestPendingToolIDsAreRescuedBeforeStreamErrors(t *testing.T) {
	partial := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"dup","function":{"name":"lookup","arguments":"{\"a\":"}},{"index":1,"id":"dup","function":{"name":"lookup","arguments":"{\"b\":"}},{"index":2,"function":{"name":"lookup","arguments":"{\"c\":"}}]}}]}` + "\n\n"
	tests := []struct {
		name string
		new  func(*testing.T) *chatcompletions.Provider
	}{
		{
			name: "decode",
			new: func(t *testing.T) *chatcompletions.Provider {
				return newExchangeProvider(t, replayDialect{}, streamExchange(partial+"data: {not-json}\n\n"))
			},
		},
		{
			name: "truncated",
			new: func(t *testing.T) *chatcompletions.Provider {
				return newExchangeProvider(t, replayDialect{}, streamExchange(partial))
			},
		},
		{
			name: "transport",
			new: func(t *testing.T) *chatcompletions.Provider {
				p, err := chatcompletions.NewWithDialect(chatcompletions.Config{
					Dialect: replayDialect{},
					APIKey:  "replay-key",
					HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
						return &http.Response{
							StatusCode: http.StatusOK,
							Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
							Body:       &readErrorBody{reader: strings.NewReader(partial), err: errors.New("stream read failed")},
						}, nil
					})},
					MaxRetries: new(int),
				})
				if err != nil {
					t.Fatalf("NewWithDialect returned error: %v", err)
				}
				return p
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := tc.new(t)
			var observed []llm.Event
			seq := p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}})
			resp, err := llm.Collect(func(yield func(llm.Event, error) bool) {
				for event, streamErr := range seq {
					if streamErr == nil {
						observed = append(observed, event)
					}
					if !yield(event, streamErr) {
						return
					}
				}
			})
			if !errors.Is(err, llm.ErrServer) {
				t.Fatalf("Collect error = %v, want ErrServer", err)
			}
			var providerErr *llm.ProviderError
			if !errors.As(err, &providerErr) || providerErr.Provider != replayDialectName {
				t.Fatalf("Collect error = %T %v, want replay ProviderError", err, err)
			}
			if resp == nil || len(resp.ToolCalls()) != 3 {
				t.Fatalf("partial response calls = %#v, want all three", resp)
			}
			wantIDs := map[int]string{3: "dup", 4: "call_4", 5: "call_5"}
			starts := map[int]string{}
			deltas := map[int]bool{}
			for _, event := range observed {
				switch event := event.(type) {
				case llm.ToolCallStart:
					starts[event.Index] = event.ID
				case llm.ToolCallDelta:
					deltas[event.Index] = true
				}
			}
			if !reflect.DeepEqual(starts, wantIDs) || !deltas[3] || !deltas[4] || !deltas[5] {
				t.Fatalf("rescued starts/deltas = %#v/%#v, want %#v/all", starts, deltas, wantIDs)
			}
		})
	}
}

func TestMissingNameToolIsDroppedBeforeStreamErrors(t *testing.T) {
	partial := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]}}]}` + "\n\n"
	tests := []struct {
		name string
		new  func(*testing.T) *chatcompletions.Provider
	}{
		{
			name: "semantic",
			new: func(t *testing.T) *chatcompletions.Provider {
				return newExchangeProvider(t, replayDialect{}, streamExchange(partial+
					`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"error"}]}`+"\n\n"))
			},
		},
		{
			name: "decode",
			new: func(t *testing.T) *chatcompletions.Provider {
				return newExchangeProvider(t, replayDialect{}, streamExchange(partial+"data: {not-json}\n\n"))
			},
		},
		{
			name: "truncated",
			new: func(t *testing.T) *chatcompletions.Provider {
				return newExchangeProvider(t, replayDialect{}, streamExchange(partial))
			},
		},
		{
			name: "transport",
			new: func(t *testing.T) *chatcompletions.Provider {
				p, err := chatcompletions.NewWithDialect(chatcompletions.Config{
					Dialect: replayDialect{},
					APIKey:  "replay-key",
					HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
						return &http.Response{
							StatusCode: http.StatusOK,
							Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
							Body:       &readErrorBody{reader: strings.NewReader(partial), err: errors.New("stream read failed")},
						}, nil
					})},
					MaxRetries: new(int),
				})
				if err != nil {
					t.Fatalf("NewWithDialect returned error: %v", err)
				}
				return p
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := tc.new(t)
			var observedDrops []llm.ToolCallDropped
			seq := p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}})
			resp, err := llm.Collect(func(yield func(llm.Event, error) bool) {
				for event, streamErr := range seq {
					if dropped, ok := event.(llm.ToolCallDropped); ok {
						observedDrops = append(observedDrops, dropped)
					}
					if !yield(event, streamErr) {
						return
					}
				}
			})
			if !errors.Is(err, llm.ErrServer) {
				t.Fatalf("Collect error = %v, want ErrServer", err)
			}
			if len(observedDrops) != 1 || observedDrops[0].Index != 3 || observedDrops[0].Reason != "missing tool name" {
				t.Fatalf("observed drops = %#v, want visible stable-index drop", observedDrops)
			}
			if resp == nil || len(resp.Parts) != 0 || !reflect.DeepEqual(resp.DroppedToolCalls, []llm.DroppedToolCall{{Index: 3, Reason: "missing tool name"}}) {
				t.Fatalf("partial response = %#v, want dropped malformed call", resp)
			}
		})
	}
}

func TestCompleteReasoningRawIsSettledBeforeStreamErrors(t *testing.T) {
	const wantRaw = `[{"type":"reasoning.text","text":"think","signature":"sig"}]`
	partial := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"reasoning":"think","reasoning_details":` + wantRaw + `}}]}` + "\n\n"
	tests := []struct {
		name string
		new  func(*testing.T) *chatcompletions.Provider
	}{
		{
			name: "semantic",
			new: func(t *testing.T) *chatcompletions.Provider {
				return newExchangeProvider(t, replayDialect{}, streamExchange(partial+
					`data: {"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"error","error":{"message":"boom","code":500},"delta":{}}]}`+"\n\n"))
			},
		},
		{
			name: "decode",
			new: func(t *testing.T) *chatcompletions.Provider {
				return newExchangeProvider(t, replayDialect{}, streamExchange(partial+"data: {not-json}\n\n"))
			},
		},
		{
			name: "truncated",
			new: func(t *testing.T) *chatcompletions.Provider {
				return newExchangeProvider(t, replayDialect{}, streamExchange(partial))
			},
		},
		{
			name: "transport",
			new: func(t *testing.T) *chatcompletions.Provider {
				p, err := chatcompletions.NewWithDialect(chatcompletions.Config{
					Dialect: replayDialect{},
					APIKey:  "replay-key",
					HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
						return &http.Response{
							StatusCode: http.StatusOK,
							Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
							Body:       &readErrorBody{reader: strings.NewReader(partial), err: errors.New("stream read failed")},
						}, nil
					})},
					MaxRetries: new(int),
				})
				if err != nil {
					t.Fatalf("NewWithDialect returned error: %v", err)
				}
				return p
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := tc.new(t)
			resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}))
			if !errors.Is(err, llm.ErrServer) {
				t.Fatalf("Collect error = %v, want ErrServer", err)
			}
			if resp == nil || len(resp.Parts) != 1 {
				t.Fatalf("partial response = %#v, want one reasoning part", resp)
			}
			reasoning, ok := resp.Parts[0].(llm.ReasoningPart)
			if !ok || reasoning.Text != "think" || string(reasoning.Raw) != wantRaw || reasoning.Provider != replayDialectName {
				t.Fatalf("reasoning = %#v, want complete replayable Raw", resp.Parts[0])
			}
		})
	}
}

func TestIncompleteReasoningDoesNotFabricateRawOnError(t *testing.T) {
	body := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"reasoning":"partial"}}]}` + "\n\n" +
		"data: {not-json}\n\n"
	p := newExchangeProvider(t, replayDialect{}, streamExchange(body))
	resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}))
	if !errors.Is(err, llm.ErrServer) {
		t.Fatalf("Collect error = %v, want ErrServer", err)
	}
	reasoning, ok := resp.Parts[0].(llm.ReasoningPart)
	if !ok || reasoning.Text != "partial" || len(reasoning.Raw) != 0 || reasoning.Provider != "" {
		t.Fatalf("reasoning = %#v, want text-only partial without fabricated Raw", resp.Parts[0])
	}
}

func TestStreamInvalidToolCallDropsAfterPartialDelta(t *testing.T) {
	body := "" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_bad","function":{"name":"lookup","arguments":"{"}}]}}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n" +
		"data: [DONE]\n\n"
	p := newExchangeProvider(t, replayDialect{}, streamExchange(body))
	var toolEvents []llm.Event
	for event, err := range p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}) {
		if err != nil {
			t.Fatalf("ChatStream returned error: %v", err)
		}
		switch event.(type) {
		case llm.ToolCallStart, llm.ToolCallDelta, llm.ToolCallEnd, llm.ToolCallDropped:
			toolEvents = append(toolEvents, event)
		}
	}
	if len(toolEvents) != 3 {
		t.Fatalf("tool events = %#v, want start/delta/drop", toolEvents)
	}
	if _, ok := toolEvents[0].(llm.ToolCallStart); !ok {
		t.Fatalf("first tool event = %#v, want ToolCallStart", toolEvents[0])
	}
	if delta, ok := toolEvents[1].(llm.ToolCallDelta); !ok || delta.ArgsFragment != "{" {
		t.Fatalf("second tool event = %#v, want partial ToolCallDelta", toolEvents[1])
	}
	if dropped, ok := toolEvents[2].(llm.ToolCallDropped); !ok || dropped.Index != 3 || dropped.Reason != "invalid tool arguments JSON" {
		t.Fatalf("third tool event = %#v, want ToolCallDropped", toolEvents[2])
	}
}

func TestToolIDReservationMatchesBlockingForInvalidPredecessors(t *testing.T) {
	tests := []struct {
		name         string
		first        string
		second       string
		wantValidID  string
		wantDropKind string
	}{
		{
			name:         "missing_name_duplicate_id",
			first:        `{"id":"dup","function":{"arguments":"{}"}}`,
			second:       `{"id":"dup","function":{"name":"good","arguments":"{}"}}`,
			wantValidID:  "dup",
			wantDropKind: "missing tool name",
		},
		{
			name:         "invalid_args_duplicate_id",
			first:        `{"id":"dup","function":{"name":"bad","arguments":"{"}}`,
			second:       `{"id":"dup","function":{"name":"good","arguments":"{}"}}`,
			wantValidID:  "call_4",
			wantDropKind: "invalid tool arguments JSON",
		},
		{
			name:         "missing_name_missing_id",
			first:        `{"function":{"arguments":"{}"}}`,
			second:       `{"function":{"name":"good","arguments":"{}"}}`,
			wantValidID:  "call_4",
			wantDropKind: "missing tool name",
		},
		{
			name:         "invalid_args_missing_id",
			first:        `{"function":{"name":"bad","arguments":"{"}}`,
			second:       `{"function":{"name":"good","arguments":"{}"}}`,
			wantValidID:  "call_4",
			wantDropKind: "invalid tool arguments JSON",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			blockingBody := `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"tool_calls","message":{"tool_calls":[` + tc.first + `,` + tc.second + `]}}]}`
			streamBody := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"tool_calls":[` +
				addToolIndex(tc.first, 0) + `,` + addToolIndex(tc.second, 1) + `]},"finish_reason":"tool_calls"}]}` + "\n\ndata: [DONE]\n\n"
			req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}
			blockingProvider := newExchangeProvider(t, replayDialect{}, e2e.RecordedExchange{
				Status:          http.StatusOK,
				ResponseHeaders: http.Header{"Content-Type": []string{"application/json"}},
				ResponseBody:    blockingBody,
			})
			streamProvider := newExchangeProvider(t, replayDialect{}, streamExchange(streamBody))
			blocking, err := blockingProvider.Chat(context.Background(), req)
			if err != nil {
				t.Fatalf("Chat returned error: %v", err)
			}
			streamed, err := llm.Collect(streamProvider.ChatStream(context.Background(), req))
			if err != nil {
				t.Fatalf("Collect returned error: %v", err)
			}
			if !reflect.DeepEqual(streamed.Parts, blocking.Parts) || !reflect.DeepEqual(streamed.DroppedToolCalls, blocking.DroppedToolCalls) {
				t.Fatalf("stream/blocking differ:\nstream=%#v %#v\nblocking=%#v %#v", streamed.Parts, streamed.DroppedToolCalls, blocking.Parts, blocking.DroppedToolCalls)
			}
			calls := streamed.ToolCalls()
			if len(calls) != 1 || calls[0].ID != tc.wantValidID {
				t.Fatalf("tool calls = %#v, want ID %q", calls, tc.wantValidID)
			}
			if len(streamed.DroppedToolCalls) != 1 || streamed.DroppedToolCalls[0].Index != 3 || streamed.DroppedToolCalls[0].Reason != tc.wantDropKind {
				t.Fatalf("drops = %#v", streamed.DroppedToolCalls)
			}
		})
	}
}

func TestStreamRejectsOutputAfterFinishReason(t *testing.T) {
	tests := []struct {
		name  string
		delta string
	}{
		{name: "text", delta: `{"content":"late"}`},
		{name: "tool", delta: `{"tool_calls":[{"index":0,"id":"late","function":{"name":"lookup","arguments":"{}"}}]}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}` + "\n\n" +
				`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":` + tc.delta + `}]}` + "\n\n" +
				"data: [DONE]\n\n"
			p := newExchangeProvider(t, replayDialect{}, streamExchange(body))
			resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}))
			if !errors.Is(err, llm.ErrServer) {
				t.Fatalf("Collect error = %v, want ErrServer", err)
			}
			if resp == nil || resp.Text() != "done" {
				t.Fatalf("partial response = %#v, want pre-finish content", resp)
			}
		})
	}
}

func TestStreamAllowsTrailingUsageOnlyChunk(t *testing.T) {
	body := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}` + "\n\n" +
		`data: {"id":"c1","model":"m","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}` + "\n\n" +
		"data: [DONE]\n\n"
	p := newExchangeProvider(t, replayDialect{}, streamExchange(body))
	resp, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if resp.Text() != "done" || resp.Usage.TotalTokens != 3 {
		t.Fatalf("response = %#v, want content and trailing usage", resp)
	}
}

func addToolIndex(raw string, index int) string {
	return `{"index":` + strconv.Itoa(index) + `,` + strings.TrimPrefix(raw, `{`)
}

func TestStreamRefusalMatchesBlocking(t *testing.T) {
	req := &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}
	blockingProvider := newExchangeProvider(t, replayDialect{}, e2e.RecordedExchange{
		Status:          http.StatusOK,
		ResponseHeaders: http.Header{"Content-Type": []string{"application/json"}},
		ResponseBody:    `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"refusal":"cannot comply"}}]}`,
	})
	streamProvider := newExchangeProvider(t, replayDialect{}, streamExchange(""+
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"refusal":"cannot "}}]}`+"\n\n"+
		`data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"refusal":"comply"},"finish_reason":"stop"}]}`+"\n\n"+
		"data: [DONE]\n\n"))
	blocking, err := blockingProvider.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	streamed, err := llm.Collect(streamProvider.ChatStream(context.Background(), req))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if !reflect.DeepEqual(streamed.Parts, blocking.Parts) || streamed.Text() != "cannot comply" {
		t.Fatalf("stream/blocking refusal = %#v / %#v", streamed.Parts, blocking.Parts)
	}
}

func TestStreamEarlyBreakDoesNotSynthesizeError(t *testing.T) {
	body := `data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"first"}}]}` + "\n\n"
	p := newExchangeProvider(t, replayDialect{}, streamExchange(body))
	seen := 0
	for _, err := range p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}) {
		if err != nil {
			t.Fatalf("early-break stream error = %v", err)
		}
		seen++
		break
	}
	if seen != 1 {
		t.Fatalf("events seen = %d, want 1", seen)
	}
}

func TestStreamNormalizesDecoderAndTransportErrors(t *testing.T) {
	t.Run("decoder", func(t *testing.T) {
		p := newExchangeProvider(t, replayDialect{}, streamExchange("data: {not-json}\n\n"))
		_, err := llm.Collect(p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}))
		assertProviderServerError(t, err)
	})

	t.Run("transport", func(t *testing.T) {
		p, err := chatcompletions.NewWithDialect(chatcompletions.Config{
			Dialect: replayDialect{},
			APIKey:  "replay-key",
			HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				return nil, errors.New("remote transport failed")
			})},
			MaxRetries: new(int),
		})
		if err != nil {
			t.Fatalf("NewWithDialect returned error: %v", err)
		}
		_, err = llm.Collect(p.ChatStream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}}))
		assertProviderServerError(t, err)
	})
}

func TestBlockingNormalizesTransportAndDecodeErrors(t *testing.T) {
	t.Run("chat_transport", func(t *testing.T) {
		p := newTransportErrorProvider(t, errors.New("chat transport failed"))
		_, err := p.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}})
		assertProviderServerError(t, err)
	})

	t.Run("models_transport", func(t *testing.T) {
		p := newTransportErrorProvider(t, errors.New("models transport failed"))
		_, err := p.Models(context.Background())
		assertProviderServerError(t, err)
	})

	t.Run("chat_decode", func(t *testing.T) {
		p := newExchangeProvider(t, replayDialect{}, e2e.RecordedExchange{
			Status:          http.StatusOK,
			ResponseHeaders: http.Header{"Content-Type": []string{"application/json"}},
			ResponseBody:    `{"id":"c1","model":"m","choices":[{"index":0,"finish_reason":"stop","message":{"content":"hi"}}],"usage":{"cost":"not-a-number"}}`,
		})
		_, err := p.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}})
		assertProviderServerError(t, err)
	})

	t.Run("models_decode", func(t *testing.T) {
		p := newExchangeProvider(t, replayDialect{}, e2e.RecordedExchange{
			Status:          http.StatusOK,
			ResponseHeaders: http.Header{"Content-Type": []string{"application/json"}},
			ResponseBody:    `{"data":[1]}`,
		})
		_, err := p.Models(context.Background())
		assertProviderServerError(t, err)
	})
}

func TestBlockingPreservesCancellationDeadlineAndProviderErrors(t *testing.T) {
	typed := &llm.ProviderError{Provider: replayDialectName, Message: "typed", Kind: llm.ErrRateLimited}
	for name, remoteErr := range map[string]error{
		"canceled": context.Canceled,
		"deadline": context.DeadlineExceeded,
		"typed":    typed,
	} {
		t.Run(name, func(t *testing.T) {
			p := newTransportErrorProvider(t, remoteErr)
			_, err := p.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{llm.UserText("hi")}})
			if !errors.Is(err, remoteErr) {
				t.Fatalf("Chat error = %v, want errors.Is(_, %v)", err, remoteErr)
			}
			if name == "typed" {
				var providerErr *llm.ProviderError
				if !errors.As(err, &providerErr) || providerErr != typed {
					t.Fatalf("Chat error = %T %v, want original ProviderError", err, err)
				}
			}
		})
	}
}

func newTransportErrorProvider(t *testing.T, remoteErr error) *chatcompletions.Provider {
	t.Helper()
	p, err := chatcompletions.NewWithDialect(chatcompletions.Config{
		Dialect: replayDialect{},
		APIKey:  "replay-key",
		HTTPClient: &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, remoteErr
		})},
		MaxRetries: new(int),
	})
	if err != nil {
		t.Fatalf("NewWithDialect returned error: %v", err)
	}
	return p
}

func assertProviderServerError(t *testing.T, err error) {
	t.Helper()
	if !errors.Is(err, llm.ErrServer) {
		t.Fatalf("error = %v, want ErrServer", err)
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Provider != replayDialectName {
		t.Fatalf("error = %T %v, want replay ProviderError", err, err)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

type readErrorBody struct {
	reader *strings.Reader
	err    error
}

func (b *readErrorBody) Read(p []byte) (int, error) {
	n, err := b.reader.Read(p)
	if errors.Is(err, io.EOF) {
		return 0, b.err
	}
	return n, err
}

func (*readErrorBody) Close() error { return nil }

func collectStreamEvents(t *testing.T, seq iter.Seq2[llm.Event, error]) []llm.Event {
	t.Helper()
	var events []llm.Event
	for event, err := range seq {
		if err != nil {
			t.Fatalf("stream returned error: %v", err)
		}
		events = append(events, event)
	}
	return events
}
