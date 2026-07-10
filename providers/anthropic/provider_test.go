package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

func TestAnthropicBuildRequestGolden(t *testing.T) {
	temp := 0.2
	topP := 0.8
	topK := int64(5)
	disableParallel := true
	p := &Provider{defaultMaxTokens: 99}
	params, opts, err := p.buildParams(&llm.Request{
		Model:       "claude-test",
		System:      "You are terse.",
		SystemCache: &llm.CacheHint{TTL: time.Hour},
		Messages: []llm.Message{
			llm.UserParts(llm.TextPart{Text: "hello", Cache: &llm.CacheHint{}}),
		},
		Temperature:   &temp,
		TopP:          &topP,
		StopSequences: []string{"END"},
		Tools: []llm.Tool{{
			Name:        "lookup",
			Description: "Look up a value.",
			Strict:      true,
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}`),
		}},
		ToolChoice: llm.ToolChoice{Mode: llm.ToolChoiceTool, Name: "lookup"},
		ResponseFormat: &llm.ResponseFormat{
			Type:   llm.FormatJSONSchema,
			Schema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`),
		},
		Effort: llm.EffortHigh,
		ProviderOptions: Options{
			BetaHeaders:            []string{"structured-outputs-2025-11-13"},
			ServiceTier:            "standard_only",
			Container:              "container_1",
			MetadataUserID:         "user_1",
			TopK:                   &topK,
			DisableParallelToolUse: &disableParallel,
		},
	})
	if err != nil {
		t.Fatalf("buildParams returned error: %v", err)
	}
	if len(opts) != 1 {
		t.Fatalf("request options len = %d, want beta header option", len(opts))
	}

	got := testutil.MustCompactJSON(t, params)
	want := `{
	"max_tokens": 99,
	"messages": [
		{
			"content": [
				{
					"text": "hello",
					"cache_control": {
						"ttl": "5m",
						"type": "ephemeral"
					},
					"type": "text"
				}
			],
			"role": "user"
		}
	],
	"model": "claude-test",
	"temperature": 0.2,
	"top_p": 0.8,
	"container": "container_1",
	"metadata": {
		"user_id": "user_1"
	},
	"output_config": {
		"effort": "high",
		"format": {
			"schema": {
				"properties": {
					"answer": {
						"type": "string"
					}
				},
				"required": [
					"answer"
				],
				"type": "object"
			},
			"type": "json_schema"
		}
	},
	"service_tier": "standard_only",
	"stop_sequences": [
		"END"
	],
	"system": [
		{
			"text": "You are terse.",
			"cache_control": {
				"ttl": "1h",
				"type": "ephemeral"
			},
			"type": "text"
		}
	],
	"thinking": {
		"display": "summarized",
		"type": "adaptive"
	},
	"tool_choice": {
		"name": "lookup",
		"disable_parallel_tool_use": true,
		"type": "tool"
	},
	"tools": [
		{
			"input_schema": {
				"properties": {
					"q": {
						"type": "string"
					}
				},
				"required": [
					"q"
				],
				"additionalProperties": false,
				"type": "object"
			},
			"name": "lookup",
			"description": "Look up a value.",
			"strict": true,
			"type": "custom"
		}
	],
	"top_k": 5
}`
	testutil.AssertJSONEqual(t, got, want)
}

func TestAnthropicRejectsSystemMessageRole(t *testing.T) {
	p := &Provider{defaultMaxTokens: 99}
	_, _, err := p.buildParams(&llm.Request{
		Model: "claude-test",
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Parts: []llm.Part{llm.Text("Use the request System field instead.")}},
			llm.UserText("hello"),
		},
	})
	if !errors.Is(err, llm.ErrUnsupported) {
		t.Fatalf("buildParams error = %v, want ErrUnsupported", err)
	}
}

func TestAnthropicReasoningReplayGolden(t *testing.T) {
	raw := json.RawMessage(`{"type":"thinking","thinking":"because","signature":"sig-with-bytes"}`)
	foreign := json.RawMessage(`{"type":"reasoning","encrypted_content":"opaque"}`)
	blocks, err := buildContentBlocks([]llm.Part{
		llm.Text("before"),
		llm.ReasoningPart{Provider: providerName, Raw: raw},
		llm.ReasoningPart{Provider: "openai", Raw: foreign},
		llm.ReasoningPart{Provider: providerName, Text: "summary only"},
	})
	if err != nil {
		t.Fatalf("buildContentBlocks returned error: %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("blocks len = %d, want text + same-provider reasoning", len(blocks))
	}
	got, err := json.Marshal(blocks[1])
	if err != nil {
		t.Fatalf("Marshal replay block returned error: %v", err)
	}
	if !bytes.Equal(got, raw) {
		t.Fatalf("replay raw changed\ngot:  %s\nwant: %s", got, raw)
	}
	all, err := json.Marshal(blocks)
	if err != nil {
		t.Fatalf("Marshal blocks returned error: %v", err)
	}
	if bytes.Contains(all, foreign) || bytes.Contains(all, []byte("summary only")) {
		t.Fatalf("foreign or summary-only reasoning was replayed: %s", all)
	}
}

func TestAnthropicEffortGoldens(t *testing.T) {
	tests := []struct {
		effort llm.Effort
		want   string
	}{
		{effort: llm.EffortNone, want: `{"max_tokens":0,"thinking":{"type":"disabled"}}`},
		{effort: llm.EffortMinimal, want: `{"max_tokens":0,"output_config":{"effort":"low"},"thinking":{"display":"summarized","type":"adaptive"}}`},
		{effort: llm.EffortLow, want: `{"max_tokens":0,"output_config":{"effort":"low"},"thinking":{"display":"summarized","type":"adaptive"}}`},
		{effort: llm.EffortMedium, want: `{"max_tokens":0,"output_config":{"effort":"medium"},"thinking":{"display":"summarized","type":"adaptive"}}`},
		{effort: llm.EffortHigh, want: `{"max_tokens":0,"output_config":{"effort":"high"},"thinking":{"display":"summarized","type":"adaptive"}}`},
		{effort: llm.EffortXHigh, want: `{"max_tokens":0,"output_config":{"effort":"xhigh"},"thinking":{"display":"summarized","type":"adaptive"}}`},
		{effort: llm.EffortMax, want: `{"max_tokens":0,"output_config":{"effort":"max"},"thinking":{"display":"summarized","type":"adaptive"}}`},
	}
	for _, tt := range tests {
		t.Run(string(tt.effort), func(t *testing.T) {
			var params sdk.MessageNewParams
			applyEffort(tt.effort, &params)
			testutil.AssertJSONEqual(t, testutil.MustCompactJSON(t, params), tt.want)
		})
	}
}

func TestAnthropicMapResponseFixtures(t *testing.T) {
	var thinking sdk.ContentBlockUnion
	if err := json.Unmarshal([]byte(`{"type":"thinking","thinking":"because","signature":"sig"}`), &thinking); err != nil {
		t.Fatalf("unmarshal thinking block: %v", err)
	}
	p := &Provider{}
	msg := &sdk.Message{
		ID:         "msg_1",
		Model:      sdk.Model("claude-test"),
		StopReason: sdk.StopReasonToolUse,
		Usage: sdk.Usage{
			InputTokens:              10,
			CacheReadInputTokens:     2,
			CacheCreationInputTokens: 3,
			OutputTokens:             5,
			OutputTokensDetails:      sdk.OutputTokensDetails{ThinkingTokens: 1},
		},
		Content: []sdk.ContentBlockUnion{
			{Type: "text", Text: "hi"},
			thinking,
			{Type: "tool_use", ID: "", Name: "lookup", Input: json.RawMessage(`{"q":"go"}`)},
			{Type: "tool_use", ID: "dup", Name: "lookup", Input: json.RawMessage(`{"q":"one"}`)},
			{Type: "tool_use", ID: "dup", Name: "lookup", Input: json.RawMessage(`{"q":"two"}`)},
			{Type: "tool_use", ID: "call_5", Name: "lookup", Input: json.RawMessage(`{"q":"preexisting"}`)},
			{Type: "tool_use", ID: "dup", Name: "lookup", Input: json.RawMessage(`{"q":"three"}`)},
			{Type: "tool_use", ID: "bad", Input: json.RawMessage(`{"q":"missing name"}`)},
		},
	}

	resp, err := p.mapMessage(msg)
	if err != nil {
		t.Fatalf("mapMessage returned error: %v", err)
	}
	if resp.Provider != providerName || resp.Model != "claude-test" || resp.StopReason != llm.StopReasonToolUse {
		t.Fatalf("response identity/stop = %+v", resp)
	}
	if resp.Text() != "hi" || resp.Reasoning() != "because" {
		t.Fatalf("text/reasoning = %q/%q", resp.Text(), resp.Reasoning())
	}
	if !bytes.Contains(resp.Parts[1].(llm.ReasoningPart).Raw, []byte(`"signature":"sig"`)) {
		t.Fatalf("reasoning raw = %s", resp.Parts[1].(llm.ReasoningPart).Raw)
	}
	calls := resp.ToolCalls()
	if len(calls) != 5 {
		t.Fatalf("tool calls len = %d, want 5", len(calls))
	}
	if calls[0].ID != "call_2" || calls[2].ID != "call_4" || calls[4].ID != "call_6" {
		t.Fatalf("rescued tool call ids = %+v", calls)
	}
	if len(resp.DroppedToolCalls) != 1 || resp.DroppedToolCalls[0].Reason != "missing tool name" {
		t.Fatalf("dropped calls = %+v", resp.DroppedToolCalls)
	}
	if resp.Usage.TotalTokens != 20 || resp.Usage.ReasoningTokens != 1 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestAnthropicMapRefusalStopReason(t *testing.T) {
	p := &Provider{}
	msg := &sdk.Message{
		ID:         "msg_refusal",
		Model:      sdk.Model("claude-test"),
		StopReason: sdk.StopReason("refusal"),
		Content:    []sdk.ContentBlockUnion{{Type: "text", Text: "I cannot help with that."}},
		Usage:      sdk.Usage{InputTokens: 1, OutputTokens: 2},
	}
	resp, err := p.mapMessage(msg)
	if err != nil {
		t.Fatalf("mapMessage returned error: %v", err)
	}
	if resp.StopReason != llm.StopReasonRefusal || resp.StopReasonRaw != "refusal" {
		t.Fatalf("stop reason = %q/%q, want refusal", resp.StopReason, resp.StopReasonRaw)
	}
	if resp.Text() != "I cannot help with that." {
		t.Fatalf("text = %q", resp.Text())
	}
}

func TestAnthropicStreamFixturesCollectEquivalent(t *testing.T) {
	rawEvents := []string{
		`{"type":"message_start","message":{"id":"msg_1","model":"claude-test","usage":{"input_tokens":7,"cache_read_input_tokens":2,"cache_creation_input_tokens":3,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"think "}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"more"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_1"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":"hel"}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"lo"}}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"go\"}"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_bad","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`,
		`{"type":"content_block_stop","index":3}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}`,
		`{"type":"message_stop"}`,
	}

	resp := collectAnthropicStream(t, rawEvents)
	if resp.ID != "msg_1" || resp.Text() != "hello" || resp.Reasoning() != "think more" {
		t.Fatalf("collected response = %+v text=%q reasoning=%q", resp, resp.Text(), resp.Reasoning())
	}
	reasoning, ok := resp.Parts[0].(llm.ReasoningPart)
	if !ok {
		t.Fatalf("part 0 = %T, want ReasoningPart", resp.Parts[0])
	}
	if reasoning.Provider != providerName || !bytes.Contains(reasoning.Raw, []byte(`"signature":"sig_1"`)) || !bytes.Contains(reasoning.Raw, []byte(`"thinking":"think more"`)) {
		t.Fatalf("stream reasoning raw/provider = %s/%q", reasoning.Raw, reasoning.Provider)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "toolu_1" || string(calls[0].Args) != `{"q":"go"}` {
		t.Fatalf("tool calls = %+v", calls)
	}
	if len(resp.DroppedToolCalls) != 1 || resp.DroppedToolCalls[0].Index != 3 {
		t.Fatalf("dropped calls = %+v", resp.DroppedToolCalls)
	}
	if resp.StopReason != llm.StopReasonToolUse {
		t.Fatalf("stop/usage = %q/%+v", resp.StopReason, resp.Usage)
	}
	if resp.Usage.InputTokens != 7 || resp.Usage.CacheReadTokens != 2 || resp.Usage.CacheWriteTokens != 3 || resp.Usage.OutputTokens != 2 || resp.Usage.TotalTokens != 14 {
		t.Fatalf("stream usage = %+v", resp.Usage)
	}

	var thinking sdk.ContentBlockUnion
	expectedRaw := `{"type":"thinking","thinking":"think more","signature":"sig_1"}`
	if err := json.Unmarshal([]byte(expectedRaw), &thinking); err != nil {
		t.Fatalf("unmarshal thinking block: %v", err)
	}
	nonStreaming, err := (&Provider{}).mapMessage(&sdk.Message{
		ID:         "msg_1",
		Model:      sdk.Model("claude-test"),
		StopReason: sdk.StopReasonToolUse,
		Usage: sdk.Usage{
			InputTokens:              7,
			CacheReadInputTokens:     2,
			CacheCreationInputTokens: 3,
			OutputTokens:             2,
		},
		Content: []sdk.ContentBlockUnion{
			thinking,
			{Type: "text", Text: "hello"},
			{Type: "tool_use", ID: "toolu_1", Name: "lookup", Input: json.RawMessage(`{"q":"go"}`)},
			{Type: "tool_use", ID: "toolu_bad", Name: "lookup", Input: json.RawMessage(`{"q":`)},
		},
	})
	if err != nil {
		t.Fatalf("non-streaming mapMessage returned error: %v", err)
	}
	streamRaw := resp.Parts[0].(llm.ReasoningPart).Raw
	nonStreamRaw := nonStreaming.Parts[0].(llm.ReasoningPart).Raw
	if !bytes.Equal(streamRaw, nonStreamRaw) || string(streamRaw) != expectedRaw {
		t.Fatalf("reasoning raw mismatch\nstream:     %s\nnon-stream: %s", streamRaw, nonStreamRaw)
	}
	if resp.Text() != nonStreaming.Text() || resp.Reasoning() != nonStreaming.Reasoning() || resp.StopReason != nonStreaming.StopReason {
		t.Fatalf("stream/non-stream mismatch: stream=%+v non-stream=%+v", resp, nonStreaming)
	}
}

func TestAnthropicStreamRedactedThinkingRaw(t *testing.T) {
	rawEvents := []string{
		`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1,"output_tokens":1}}`,
	}
	resp := collectAnthropicStream(t, rawEvents)
	reasoning, ok := resp.Parts[0].(llm.ReasoningPart)
	if !ok {
		t.Fatalf("part 0 = %T, want ReasoningPart", resp.Parts[0])
	}
	if reasoning.Text != "" || reasoning.Provider != providerName || string(reasoning.Raw) != `{"type":"redacted_thinking","data":"opaque"}` {
		t.Fatalf("redacted reasoning = text %q raw %s provider %q", reasoning.Text, reasoning.Raw, reasoning.Provider)
	}
}

func TestAnthropicStreamMessageStopFinalizesOpenReasoning(t *testing.T) {
	t.Run("canonical reconstruction", func(t *testing.T) {
		rawEvents := []string{
			`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"think "}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"more"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_1"}}`,
			`{"type":"message_stop"}`,
		}
		state := newStreamState(&Provider{})
		events := testutil.MapRawEvents(t, rawEvents, state.mapEvent)
		assertReasoningTerminalOrder(t, events, 0)
		if len(state.thinking) != 0 {
			t.Fatalf("open reasoning after message_stop = %#v", state.thinking)
		}

		resp, err := llm.Collect(providerutil.StreamContract(providerName, testutil.EventSeq(events...)))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		const expectedRaw = `{"type":"thinking","thinking":"think more","signature":"sig_1"}`
		assertAnthropicReasoning(t, resp, "think more", expectedRaw)
		assertAnthropicReasoningMatchesBlocking(t, resp, expectedRaw)
	})

	t.Run("provider verbatim", func(t *testing.T) {
		const contentRaw = `{ "type": "thinking", "thinking": "complete", "signature": "sig-v", "future": "kept" }`
		rawEvents := []string{
			`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
			`{"type":"content_block_start","index":2,"content_block":` + contentRaw + `}`,
			`{"type":"message_stop"}`,
		}
		state := newStreamState(&Provider{})
		events := testutil.MapRawEvents(t, rawEvents, state.mapEvent)
		assertReasoningTerminalOrder(t, events, 2)

		resp, err := llm.Collect(providerutil.StreamContract(providerName, testutil.EventSeq(events...)))
		if err != nil {
			t.Fatalf("Collect returned error: %v", err)
		}
		assertAnthropicReasoning(t, resp, "complete", contentRaw)
		assertAnthropicReasoningMatchesBlocking(t, resp, contentRaw)
	})
}

func TestAnthropicStreamMessageStopRejectsMalformedOpenReasoning(t *testing.T) {
	provider := newAnthropicStreamFixtureProvider(t, func(*http.Request) (*http.Response, error) {
		return anthropicStreamResponse(anthropicSSE(
			`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
			`{"type":"content_block_start","index":4,"content_block":{"type":"thinking","thinking":"partial"}}`,
			`{"type":"message_stop"}`,
		)), nil
	})
	items := observeAnthropicStream(provider.ChatStream(context.Background(), anthropicStreamFixtureRequest()))
	assertAnthropicObservedKinds(t, items, []string{"MessageStart", "ReasoningDelta", "error"})

	resp, err := collectAnthropicObserved(items)
	assertAnthropicProviderServerError(t, err)
	if !strings.Contains(err.Error(), "missing signature") {
		t.Fatalf("error = %v, want missing-signature detail", err)
	}
	if resp == nil || len(resp.Parts) != 1 {
		t.Fatalf("partial response = %#v, want one reasoning part", resp)
	}
	reasoning, ok := resp.Parts[0].(llm.ReasoningPart)
	if !ok || reasoning.Text != "partial" || len(reasoning.Raw) != 0 {
		t.Fatalf("partial reasoning = %#v", resp.Parts[0])
	}
}

func TestAnthropicStreamErrorsSettleCompleteReasoning(t *testing.T) {
	const messageStart = `{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`
	blocks := []struct {
		name      string
		start     string
		conflict  string
		text      string
		raw       string
		wantKinds []string
	}{
		{
			name:      "thinking",
			start:     `{"type":"content_block_start","index":4,"content_block":{"type":"thinking","thinking":"because","signature":"sig-v","future":"kept"}}`,
			conflict:  `{"type":"content_block_start","index":4,"content_block":{"type":"redacted_thinking","data":"conflict"}}`,
			text:      "because",
			raw:       `{"type":"thinking","thinking":"because","signature":"sig-v","future":"kept"}`,
			wantKinds: []string{"MessageStart", "ReasoningDelta", "ReasoningDelta", "error"},
		},
		{
			name:      "redacted thinking",
			start:     `{"type":"content_block_start","index":4,"content_block":{"type":"redacted_thinking","data":"opaque","future":"kept"}}`,
			conflict:  `{"type":"content_block_start","index":4,"content_block":{"type":"thinking","thinking":"conflict","signature":"sig"}}`,
			raw:       `{"type":"redacted_thinking","data":"opaque","future":"kept"}`,
			wantKinds: []string{"MessageStart", "ReasoningDelta", "error"},
		},
	}
	errorClasses := []string{"semantic", "decode", "transport", "truncated EOF"}

	for _, block := range blocks {
		for _, errorClass := range errorClasses {
			t.Run(block.name+"/"+errorClass, func(t *testing.T) {
				provider := newAnthropicStreamFixtureProvider(t, func(*http.Request) (*http.Response, error) {
					body := anthropicSSE(messageStart, block.start)
					switch errorClass {
					case "semantic":
						body += anthropicSSE(block.conflict)
						return anthropicStreamResponse(body), nil
					case "decode":
						body += "event: content_block_delta\ndata: {not-json}\n\n"
						return anthropicStreamResponse(body), nil
					case "transport":
						resp := anthropicStreamResponse("")
						resp.Body = &anthropicReadErrorBody{
							reader: strings.NewReader(body),
							err:    errors.New("stream read failed"),
						}
						return resp, nil
					default:
						return anthropicStreamResponse(body), nil
					}
				})

				items := observeAnthropicStream(provider.ChatStream(context.Background(), anthropicStreamFixtureRequest()))
				assertAnthropicObservedKinds(t, items, block.wantKinds)
				assertReasoningImmediatelyBeforeError(t, items, 4, block.raw)

				resp, err := collectAnthropicObserved(items)
				assertAnthropicProviderServerError(t, err)
				assertAnthropicReasoning(t, resp, block.text, block.raw)
			})
		}
	}
}

func TestAnthropicStreamErrorDoesNotFabricateIncompleteReasoning(t *testing.T) {
	provider := newAnthropicStreamFixtureProvider(t, func(*http.Request) (*http.Response, error) {
		body := anthropicSSE(
			`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
			`{"type":"content_block_start","index":4,"content_block":{"type":"thinking","thinking":"partial"}}`,
		) + "event: content_block_delta\ndata: {not-json}\n\n"
		return anthropicStreamResponse(body), nil
	})
	items := observeAnthropicStream(provider.ChatStream(context.Background(), anthropicStreamFixtureRequest()))
	assertAnthropicObservedKinds(t, items, []string{"MessageStart", "ReasoningDelta", "error"})

	resp, err := collectAnthropicObserved(items)
	assertAnthropicProviderServerError(t, err)
	if resp == nil || len(resp.Parts) != 1 {
		t.Fatalf("partial response = %#v, want one reasoning part", resp)
	}
	reasoning, ok := resp.Parts[0].(llm.ReasoningPart)
	if !ok || reasoning.Text != "partial" || len(reasoning.Raw) != 0 || reasoning.Provider != "" {
		t.Fatalf("partial reasoning = %#v, want text without fabricated raw", resp.Parts[0])
	}
}

func TestAnthropicStreamDuplicateSyntheticIDUnique(t *testing.T) {
	rawEvents := []string{
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_4","name":"lookup","input":{"q":"existing"}}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"dup","name":"lookup","input":{"q":"one"}}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":4,"content_block":{"type":"tool_use","id":"dup","name":"lookup","input":{"q":"two"}}}`,
		`{"type":"content_block_stop","index":4}`,
	}
	events := mapAnthropicStream(t, rawEvents)
	var ids []string
	for _, event := range events {
		if start, ok := event.(llm.ToolCallStart); ok {
			ids = append(ids, start.ID)
		}
	}
	if !reflect.DeepEqual(ids, []string{"call_4", "dup", "call_5"}) {
		t.Fatalf("stream tool call ids = %+v, want stable-position IDs [call_4 dup call_5]", ids)
	}
}

func TestAnthropicStreamLateToolMetadataPreservesArguments(t *testing.T) {
	state := newStreamState(&Provider{})
	var events []llm.Event
	events = append(events, mapOneAnthropicEvent(t, state,
		`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`)...)
	if got := mapOneAnthropicEvent(t, state,
		`{"type":"content_block_delta","index":7,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"go\"}"}}`); len(got) != 0 {
		t.Fatalf("pre-metadata events = %#v, want buffered", got)
	}
	metadata := mapOneAnthropicEvent(t, state,
		`{"type":"content_block_start","index":7,"content_block":{"type":"tool_use","id":"toolu_late","name":"lookup","input":{}}}`)
	if len(metadata) != 2 {
		t.Fatalf("late-metadata events = %#v, want start and buffered delta", metadata)
	}
	start, ok := providerutil.DerefEvent(metadata[0]).(llm.ToolCallStart)
	if !ok || start.Index != 7 || start.ID != "toolu_late" || start.Name != "lookup" {
		t.Fatalf("late-metadata start = %#v", metadata[0])
	}
	delta, ok := providerutil.DerefEvent(metadata[1]).(llm.ToolCallDelta)
	if !ok || delta.Index != 7 || delta.ArgsFragment != `{"q":"go"}` {
		t.Fatalf("late-metadata delta = %#v", metadata[1])
	}
	events = append(events, metadata...)
	events = append(events, mapOneAnthropicEvent(t, state, `{"type":"content_block_stop","index":7}`)...)
	events = append(events, mapOneAnthropicEvent(t, state, `{"type":"message_stop"}`)...)

	resp, err := llm.Collect(providerutil.StreamContract(providerName, testutil.EventSeq(events...)))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "toolu_late" || calls[0].Name != "lookup" || string(calls[0].Args) != `{"q":"go"}` {
		t.Fatalf("late-metadata tool calls = %#v", calls)
	}
}

func TestAnthropicStreamToolCallIsIncremental(t *testing.T) {
	state := newStreamState(&Provider{})
	start := mapOneAnthropicEvent(t, state,
		`{"type":"content_block_start","index":4,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}`)
	if len(start) != 1 {
		t.Fatalf("tool start events = %#v, want one", start)
	}
	toolStart, ok := providerutil.DerefEvent(start[0]).(llm.ToolCallStart)
	if !ok || toolStart.Index != 4 || toolStart.ID != "toolu_1" || toolStart.Name != "lookup" {
		t.Fatalf("tool start = %#v", start[0])
	}

	first := mapOneAnthropicEvent(t, state,
		`{"type":"content_block_delta","index":4,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`)
	second := mapOneAnthropicEvent(t, state,
		`{"type":"content_block_delta","index":4,"delta":{"type":"input_json_delta","partial_json":"\"go\"}"}}`)
	for i, want := range []string{`{"q":`, `"go"}`} {
		mapped := [][]llm.Event{first, second}[i]
		if len(mapped) != 1 {
			t.Fatalf("argument fragment %d events = %#v, want one", i, mapped)
		}
		delta, ok := providerutil.DerefEvent(mapped[0]).(llm.ToolCallDelta)
		if !ok || delta.Index != 4 || delta.ArgsFragment != want {
			t.Fatalf("argument fragment %d = %#v, want %q", i, mapped[0], want)
		}
	}
	stop := mapOneAnthropicEvent(t, state, `{"type":"content_block_stop","index":4}`)
	if len(stop) != 1 {
		t.Fatalf("tool stop events = %#v, want one", stop)
	}
	if end, ok := providerutil.DerefEvent(stop[0]).(llm.ToolCallEnd); !ok || end.Index != 4 {
		t.Fatalf("tool stop = %#v", stop[0])
	}
}

func TestAnthropicStreamMalformedToolCallDropsPartialState(t *testing.T) {
	rawEvents := []string{
		`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
		`{"type":"content_block_start","index":5,"content_block":{"type":"tool_use","id":"bad","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":5,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`,
		`{"type":"content_block_stop","index":5}`,
		`{"type":"message_stop"}`,
	}
	state := newStreamState(&Provider{})
	events := testutil.MapRawEvents(t, rawEvents, state.mapEvent)
	var sawStart, sawDelta, sawDrop bool
	for _, event := range events {
		switch event := providerutil.DerefEvent(event).(type) {
		case llm.ToolCallStart:
			sawStart = event.Index == 5
		case llm.ToolCallDelta:
			sawDelta = event.Index == 5 && event.ArgsFragment == `{"q":`
		case llm.ToolCallDropped:
			sawDrop = event.Index == 5 && event.Reason == "invalid tool arguments JSON"
		case llm.ToolCallEnd:
			t.Fatalf("malformed tool call emitted ToolCallEnd: %#v", event)
		}
	}
	if !sawStart || !sawDelta || !sawDrop {
		t.Fatalf("malformed tool events start/delta/drop = %v/%v/%v: %#v", sawStart, sawDelta, sawDrop, events)
	}
	resp, err := llm.Collect(providerutil.StreamContract(providerName, testutil.EventSeq(events...)))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if len(resp.Parts) != 0 || len(resp.DroppedToolCalls) != 1 {
		t.Fatalf("collected malformed tool response = %#v", resp)
	}
}

func TestAnthropicStreamMessageStopRescuesOpenValidToolCall(t *testing.T) {
	rawEvents := []string{
		`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
		`{"type":"content_block_delta","index":9,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"go\"}"}}`,
		`{"type":"content_block_start","index":9,"content_block":{"type":"tool_use","id":"toolu_late","name":"lookup","input":{}}}`,
		`{"type":"message_stop"}`,
	}
	state := newStreamState(&Provider{})
	events := testutil.MapRawEvents(t, rawEvents, state.mapEvent)
	if len(events) < 2 {
		t.Fatalf("events = %#v, want terminal tool and message events", events)
	}
	toolEnd, ok := providerutil.DerefEvent(events[len(events)-2]).(llm.ToolCallEnd)
	if !ok || toolEnd.Index != 9 {
		t.Fatalf("penultimate event = %#v, want ToolCallEnd index 9", events[len(events)-2])
	}
	if _, ok := providerutil.DerefEvent(events[len(events)-1]).(llm.MessageEnd); !ok {
		t.Fatalf("last event = %#v, want MessageEnd", events[len(events)-1])
	}
	if len(state.tools) != 0 {
		t.Fatalf("open tools after message_stop = %#v", state.tools)
	}

	resp, err := llm.Collect(providerutil.StreamContract(providerName, testutil.EventSeq(events...)))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "toolu_late" || calls[0].Name != "lookup" || string(calls[0].Args) != `{"q":"go"}` {
		t.Fatalf("rescued tool calls = %#v", calls)
	}
}

func TestAnthropicStreamMessageStopDropsOpenMalformedToolCall(t *testing.T) {
	rawEvents := []string{
		`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
		`{"type":"content_block_start","index":6,"content_block":{"type":"tool_use","id":"bad","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":6,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`,
		`{"type":"message_stop"}`,
	}
	state := newStreamState(&Provider{})
	events := testutil.MapRawEvents(t, rawEvents, state.mapEvent)
	if len(events) < 2 {
		t.Fatalf("events = %#v, want terminal drop and message events", events)
	}
	drop, ok := providerutil.DerefEvent(events[len(events)-2]).(llm.ToolCallDropped)
	if !ok || drop.Index != 6 || drop.Reason != "invalid tool arguments JSON" {
		t.Fatalf("penultimate event = %#v, want malformed ToolCallDropped", events[len(events)-2])
	}
	if _, ok := providerutil.DerefEvent(events[len(events)-1]).(llm.MessageEnd); !ok {
		t.Fatalf("last event = %#v, want MessageEnd", events[len(events)-1])
	}
	if len(state.tools) != 0 {
		t.Fatalf("open tools after message_stop = %#v", state.tools)
	}

	resp, err := llm.Collect(providerutil.StreamContract(providerName, testutil.EventSeq(events...)))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if len(resp.Parts) != 0 || len(resp.DroppedToolCalls) != 1 {
		t.Fatalf("collected malformed terminal tool = %#v", resp)
	}
}

func TestAnthropicStreamMessageStopWithoutDeltaEnds(t *testing.T) {
	resp := collectContractAnthropicStream(t, []string{
		`{"type":"message_start","message":{"id":"msg_1","model":"claude-test","usage":{"input_tokens":3,"output_tokens":0}}}`,
		`{"type":"message_stop"}`,
	})
	if resp.ID != "msg_1" || resp.Model != "claude-test" || resp.Usage.InputTokens != 3 {
		t.Fatalf("message-stop-only response = %#v", resp)
	}
}

func TestAnthropicDroppedToolIndexesMatchBlocking(t *testing.T) {
	provider := &Provider{}
	blocking, err := provider.mapMessage(&sdk.Message{
		ID:         "msg_1",
		Model:      sdk.Model("claude-test"),
		StopReason: sdk.StopReasonEndTurn,
		Content: []sdk.ContentBlockUnion{
			{Type: "server_tool_use"},
			{Type: "tool_use", ID: "bad", Name: "lookup", Input: json.RawMessage(`{"q":`)},
			{Type: "text", Text: "after"},
			{Type: "tool_use", ID: "good", Name: "lookup", Input: json.RawMessage(`{"q":"go"}`)},
		},
	})
	if err != nil {
		t.Fatalf("mapMessage returned error: %v", err)
	}

	rawEvents := []string{
		`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"bad","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"text","text":"after"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"good","name":"lookup","input":{"q":"go"}}}`,
		`{"type":"content_block_stop","index":3}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
		`{"type":"message_stop"}`,
	}
	state := newStreamState(provider)
	events := testutil.MapRawEvents(t, rawEvents, state.mapEvent)
	streamed, err := llm.Collect(providerutil.StreamContract(providerName, testutil.EventSeq(events...)))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	if !reflect.DeepEqual(streamed.Parts, blocking.Parts) {
		t.Fatalf("stream/blocking parts differ:\nstream:   %#v\nblocking: %#v", streamed.Parts, blocking.Parts)
	}
	if !reflect.DeepEqual(streamed.DroppedToolCalls, blocking.DroppedToolCalls) {
		t.Fatalf("stream/blocking drops differ: %#v / %#v", streamed.DroppedToolCalls, blocking.DroppedToolCalls)
	}
	if len(streamed.DroppedToolCalls) != 1 || streamed.DroppedToolCalls[0].Index != 1 {
		t.Fatalf("dropped calls = %#v, want stable provider index 1", streamed.DroppedToolCalls)
	}

	wantIndexes := []int{1, 1, 1, 2, 3, 3, 3}
	var gotIndexes []int
	for _, event := range events {
		switch event := providerutil.DerefEvent(event).(type) {
		case llm.ToolCallDropped:
			gotIndexes = append(gotIndexes, event.Index)
		case llm.TextDelta:
			gotIndexes = append(gotIndexes, event.Index)
		case llm.ToolCallStart:
			gotIndexes = append(gotIndexes, event.Index)
		case llm.ToolCallDelta:
			gotIndexes = append(gotIndexes, event.Index)
		case llm.ToolCallEnd:
			gotIndexes = append(gotIndexes, event.Index)
		}
	}
	if !reflect.DeepEqual(gotIndexes, wantIndexes) {
		t.Fatalf("stable provider event indexes = %v, want %v", gotIndexes, wantIndexes)
	}
}

func TestAnthropicErrorMapping(t *testing.T) {
	tests := []struct {
		name   string
		status int
		typ    string
		msg    string
		want   error
	}{
		{name: "auth", status: 401, typ: string(sdk.ErrorTypeAuthenticationError), msg: "authentication failed", want: llm.ErrAuth},
		{name: "rate", status: 429, typ: string(sdk.ErrorTypeRateLimitError), msg: "rate limited", want: llm.ErrRateLimited},
		{name: "overload", status: 529, typ: string(sdk.ErrorTypeOverloadedError), msg: "overloaded", want: llm.ErrOverloaded},
		{name: "context", status: 400, typ: string(sdk.ErrorTypeInvalidRequestError), msg: "context window exceeded", want: llm.ErrContextTooLong},
		{name: "context length", status: 400, typ: string(sdk.ErrorTypeInvalidRequestError), msg: "maximum context length exceeded", want: llm.ErrContextTooLong},
		{name: "prompt too long", status: 400, typ: string(sdk.ErrorTypeInvalidRequestError), msg: "prompt is too long: 205001 tokens > 200000 maximum", want: llm.ErrContextTooLong},
		{name: "bare context", status: 400, typ: string(sdk.ErrorTypeInvalidRequestError), msg: "context parameter is invalid", want: llm.ErrBadRequest},
		{name: "unscoped token count", status: 400, typ: string(sdk.ErrorTypeInvalidRequestError), msg: "205001 tokens > 200000 maximum", want: llm.ErrBadRequest},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var apiErr sdk.Error
			raw := `{"error":{"type":"` + tt.typ + `","message":` + strconv.Quote(tt.msg) + `}}`
			if err := json.Unmarshal([]byte(raw), &apiErr); err != nil {
				t.Fatalf("unmarshal api error: %v", err)
			}
			apiErr.StatusCode = tt.status
			apiErr.Response = &http.Response{Header: http.Header{"Retry-After": []string{"2"}}}
			err := mapError(&apiErr)
			if !errors.Is(err, tt.want) {
				t.Fatalf("error = %v, want %v", err, tt.want)
			}
			var providerErr *llm.ProviderError
			if !errors.As(err, &providerErr) {
				t.Fatalf("error does not unwrap to ProviderError: %v", err)
			}
			if providerErr.RetryAfter != 2*time.Second || providerErr.Code != tt.typ {
				t.Fatalf("provider error = %+v", providerErr)
			}
		})
	}
}

func TestAnthropicBuildMessagesOmitsFilteredEmptyAssistant(t *testing.T) {
	messages, err := buildMessages([]llm.Message{
		{
			Role: llm.RoleAssistant,
			Parts: []llm.Part{
				llm.ReasoningPart{Provider: "openai", Raw: json.RawMessage(`{"type":"reasoning"}`)},
				llm.UnknownPart{Type: "future", Data: json.RawMessage(`{"type":"future"}`)},
			},
		},
		llm.UserText("continue"),
	})
	if err != nil {
		t.Fatalf("buildMessages returned error: %v", err)
	}
	if len(messages) != 1 || messages[0].Role != sdk.MessageParamRoleUser {
		t.Fatalf("messages = %#v, want only user message", messages)
	}

	empty, err := buildMessages([]llm.Message{{Role: llm.RoleAssistant}})
	if err != nil {
		t.Fatalf("buildMessages(empty assistant) returned error: %v", err)
	}
	if len(empty) != 1 {
		t.Fatalf("unfiltered empty assistant was omitted: %#v", empty)
	}
}

func TestAnthropicStreamProviderContractEOF(t *testing.T) {
	tests := []struct {
		name        string
		body        string
		wantPartial bool
	}{
		{name: "empty EOF"},
		{
			name: "truncated EOF",
			body: anthropicSSE(
				`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
				`{"type":"content_block_delta","index":4,"delta":{"type":"text_delta","text":"partial"}}`,
			),
			wantPartial: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := newAnthropicStreamFixtureProvider(t, func(*http.Request) (*http.Response, error) {
				return anthropicStreamResponse(tt.body), nil
			})
			resp, err := llm.Collect(provider.ChatStream(context.Background(), anthropicStreamFixtureRequest()))
			assertAnthropicProviderServerError(t, err)
			if tt.wantPartial {
				if resp == nil || resp.Text() != "partial" || resp.Model != "claude-test" {
					t.Fatalf("partial response = %#v", resp)
				}
			} else if resp != nil {
				t.Fatalf("empty stream response = %#v, want nil", resp)
			}
		})
	}
}

func TestAnthropicStreamProviderMessageStopOnlyAndEarlyBreak(t *testing.T) {
	body := anthropicSSE(
		`{"type":"message_start","message":{"id":"msg_1","model":"claude-test","usage":{"input_tokens":2,"output_tokens":0}}}`,
		`{"type":"message_stop"}`,
	)
	provider := newAnthropicStreamFixtureProvider(t, func(*http.Request) (*http.Response, error) {
		return anthropicStreamResponse(body), nil
	})
	var events []llm.Event
	starts, ends := 0, 0
	for event, err := range provider.ChatStream(context.Background(), anthropicStreamFixtureRequest()) {
		if err != nil {
			t.Fatalf("ChatStream returned error: %v", err)
		}
		events = append(events, event)
		switch providerutil.DerefEvent(event).(type) {
		case llm.MessageStart:
			starts++
		case llm.MessageEnd:
			ends++
		}
	}
	if starts != 1 || ends != 1 {
		t.Fatalf("start/end counts = %d/%d, want 1/1", starts, ends)
	}
	if _, ok := providerutil.DerefEvent(events[len(events)-1]).(llm.MessageEnd); !ok {
		t.Fatalf("last event = %T, want MessageEnd", events[len(events)-1])
	}
	resp, err := llm.Collect(testutil.EventSeq(events...))
	if err != nil {
		t.Fatalf("Collect mapped events returned error: %v", err)
	}
	if resp.ID != "msg_1" || resp.Usage.InputTokens != 2 {
		t.Fatalf("message-stop-only response = %#v", resp)
	}

	provider = newAnthropicStreamFixtureProvider(t, func(*http.Request) (*http.Response, error) {
		return anthropicStreamResponse(anthropicSSE(
			`{"type":"message_start","message":{"id":"msg_2","model":"claude-test"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"never consumed"}}`,
		)), nil
	})
	count := 0
	for _, err := range provider.ChatStream(context.Background(), anthropicStreamFixtureRequest()) {
		if err != nil {
			t.Fatalf("early-break stream returned error: %v", err)
		}
		count++
		break
	}
	if count != 1 {
		t.Fatalf("early-break event count = %d, want 1", count)
	}
}

func TestAnthropicStreamUnknownRemoteErrorsAreNormalized(t *testing.T) {
	t.Run("transport", func(t *testing.T) {
		provider := newAnthropicStreamFixtureProvider(t, func(*http.Request) (*http.Response, error) {
			return nil, errors.New("transport exploded")
		})
		resp, err := llm.Collect(provider.ChatStream(context.Background(), anthropicStreamFixtureRequest()))
		assertAnthropicProviderServerError(t, err)
		if resp != nil {
			t.Fatalf("transport error response = %#v, want nil", resp)
		}
	})

	t.Run("decoder", func(t *testing.T) {
		provider := newAnthropicStreamFixtureProvider(t, func(*http.Request) (*http.Response, error) {
			body := anthropicSSE(`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`) +
				"event: content_block_delta\ndata: {not-json}\n\n"
			return anthropicStreamResponse(body), nil
		})
		resp, err := llm.Collect(provider.ChatStream(context.Background(), anthropicStreamFixtureRequest()))
		assertAnthropicProviderServerError(t, err)
		if resp == nil || resp.ID != "msg_1" {
			t.Fatalf("decoder partial response = %#v", resp)
		}
	})
}

func TestAnthropicStreamFailurePreservesPartialToolCall(t *testing.T) {
	provider := newAnthropicStreamFixtureProvider(t, func(*http.Request) (*http.Response, error) {
		body := anthropicSSE(
			`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
			`{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}`,
			`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`,
		) + "event: content_block_delta\ndata: {not-json}\n\n"
		return anthropicStreamResponse(body), nil
	})
	resp, err := llm.Collect(provider.ChatStream(context.Background(), anthropicStreamFixtureRequest()))
	assertAnthropicProviderServerError(t, err)
	if resp == nil || len(resp.Parts) != 1 {
		t.Fatalf("partial response = %#v, want one tool call", resp)
	}
	call, ok := resp.Parts[0].(llm.ToolCallPart)
	if !ok || call.ID != "toolu_1" || call.Name != "lookup" || string(call.Args) != `{"q":` {
		t.Fatalf("partial tool call = %#v", resp.Parts[0])
	}
}

func TestAnthropicStreamSemanticErrorSettlesLateMetadata(t *testing.T) {
	state := newStreamState(&Provider{})
	var items []anthropicObservedItem
	for _, event := range mapOneAnthropicEvent(t, state,
		`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`) {
		items = append(items, anthropicObservedItem{event: event})
	}
	if events := mapOneAnthropicEvent(t, state,
		`{"type":"content_block_delta","index":7,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"go\"}"}}`); len(events) != 0 {
		t.Fatalf("pre-metadata events = %#v, want buffered", events)
	}

	_, semanticErr := state.mapEvent(sdk.MessageStreamEventUnion{
		Type:  "content_block_start",
		Index: 7,
		ContentBlock: sdk.ContentBlockStartEventContentBlockUnion{
			Type:  "tool_use",
			ID:    "toolu_late",
			Name:  "lookup",
			Input: make(chan int),
		},
	})
	if semanticErr == nil {
		t.Fatal("semantic map error = nil, want unsupported input error")
	}
	for _, event := range state.settleBlocksOnError() {
		items = append(items, anthropicObservedItem{event: event})
	}
	items = append(items, anthropicObservedItem{err: mapError(semanticErr)})
	assertAnthropicObservedKinds(t, items, []string{"MessageStart", "ToolCallStart", "ToolCallDelta", "error"})

	resp, err := collectAnthropicObserved(items)
	assertAnthropicProviderServerError(t, err)
	assertAnthropicPartialToolCall(t, resp, "toolu_late", `{"q":"go"}`)
}

func TestAnthropicStreamSemanticErrorDropsUnresolvedTool(t *testing.T) {
	state := newStreamState(&Provider{})
	var items []anthropicObservedItem
	for _, event := range mapOneAnthropicEvent(t, state,
		`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`) {
		items = append(items, anthropicObservedItem{event: event})
	}
	if events := mapOneAnthropicEvent(t, state,
		`{"type":"content_block_delta","index":5,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`); len(events) != 0 {
		t.Fatalf("pre-metadata events = %#v, want buffered", events)
	}

	_, semanticErr := state.mapEvent(sdk.MessageStreamEventUnion{
		Type:  "content_block_start",
		Index: 6,
		ContentBlock: sdk.ContentBlockStartEventContentBlockUnion{
			Type:  "tool_use",
			Input: make(chan int),
		},
	})
	if semanticErr == nil {
		t.Fatal("semantic map error = nil, want unsupported input error")
	}
	for _, event := range state.settleBlocksOnError() {
		items = append(items, anthropicObservedItem{event: event})
	}
	items = append(items, anthropicObservedItem{err: mapError(semanticErr)})
	assertAnthropicObservedKinds(t, items, []string{"MessageStart", "ToolCallDropped", "ToolCallDropped", "error"})

	resp, err := collectAnthropicObserved(items)
	assertAnthropicProviderServerError(t, err)
	if resp == nil || len(resp.Parts) != 0 || len(resp.DroppedToolCalls) != 2 || resp.DroppedToolCalls[0].Index != 5 {
		t.Fatalf("partial response = %#v, want unresolved provider-index tool dropped before error", resp)
	}
}

func TestAnthropicStreamDecodeErrorDropsUnresolvedArguments(t *testing.T) {
	provider := newAnthropicStreamFixtureProvider(t, func(*http.Request) (*http.Response, error) {
		body := anthropicSSE(
			`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
			`{"type":"content_block_delta","index":5,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`,
		) + "event: content_block_delta\ndata: {not-json}\n\n"
		return anthropicStreamResponse(body), nil
	})
	items := observeAnthropicStream(provider.ChatStream(context.Background(), anthropicStreamFixtureRequest()))
	assertAnthropicObservedKinds(t, items, []string{"MessageStart", "ToolCallDropped", "error"})

	resp, err := collectAnthropicObserved(items)
	assertAnthropicProviderServerError(t, err)
	assertAnthropicDroppedOnly(t, resp, 5, "missing tool name before stream error")
}

func TestAnthropicStreamReadErrorPreservesLateMetadataTool(t *testing.T) {
	provider := newAnthropicStreamFixtureProvider(t, func(*http.Request) (*http.Response, error) {
		body := anthropicSSE(
			`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
			`{"type":"content_block_delta","index":8,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"go\"}"}}`,
			`{"type":"content_block_start","index":8,"content_block":{"type":"tool_use","id":"toolu_late","name":"lookup","input":{}}}`,
		)
		resp := anthropicStreamResponse("")
		resp.Body = &anthropicReadErrorBody{
			reader: strings.NewReader(body),
			err:    errors.New("stream read failed"),
		}
		return resp, nil
	})
	items := observeAnthropicStream(provider.ChatStream(context.Background(), anthropicStreamFixtureRequest()))
	assertAnthropicObservedKinds(t, items, []string{"MessageStart", "ToolCallStart", "ToolCallDelta", "error"})

	resp, err := collectAnthropicObserved(items)
	assertAnthropicProviderServerError(t, err)
	assertAnthropicPartialToolCall(t, resp, "toolu_late", `{"q":"go"}`)
}

func TestAnthropicStreamReadErrorDropsUnresolvedTool(t *testing.T) {
	provider := newAnthropicStreamFixtureProvider(t, func(*http.Request) (*http.Response, error) {
		body := anthropicSSE(
			`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
			`{"type":"content_block_delta","index":5,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`,
		)
		resp := anthropicStreamResponse("")
		resp.Body = &anthropicReadErrorBody{
			reader: strings.NewReader(body),
			err:    errors.New("stream read failed"),
		}
		return resp, nil
	})
	items := observeAnthropicStream(provider.ChatStream(context.Background(), anthropicStreamFixtureRequest()))
	assertAnthropicObservedKinds(t, items, []string{"MessageStart", "ToolCallDropped", "error"})

	resp, err := collectAnthropicObserved(items)
	assertAnthropicProviderServerError(t, err)
	assertAnthropicDroppedOnly(t, resp, 5, "missing tool name before stream error")
}

func TestAnthropicStreamTruncatedEOFSettlesPreMetadataArguments(t *testing.T) {
	tests := []struct {
		name      string
		body      string
		wantKinds []string
		wantCall  bool
	}{
		{
			name: "unresolved arguments are dropped",
			body: anthropicSSE(
				`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
				`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}`,
			),
			wantKinds: []string{"MessageStart", "ToolCallDropped", "error"},
		},
		{
			name: "late metadata rescues partial call",
			body: anthropicSSE(
				`{"type":"message_start","message":{"id":"msg_1","model":"claude-test"}}`,
				`{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"go\"}"}}`,
				`{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_late","name":"lookup","input":{}}}`,
			),
			wantKinds: []string{"MessageStart", "ToolCallStart", "ToolCallDelta", "error"},
			wantCall:  true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := newAnthropicStreamFixtureProvider(t, func(*http.Request) (*http.Response, error) {
				return anthropicStreamResponse(tt.body), nil
			})
			items := observeAnthropicStream(provider.ChatStream(context.Background(), anthropicStreamFixtureRequest()))
			assertAnthropicObservedKinds(t, items, tt.wantKinds)

			resp, err := collectAnthropicObserved(items)
			assertAnthropicProviderServerError(t, err)
			if tt.wantCall {
				assertAnthropicPartialToolCall(t, resp, "toolu_late", `{"q":"go"}`)
			} else {
				assertAnthropicDroppedOnly(t, resp, 3, "missing tool name before stream error")
			}
		})
	}
}

// TestAnthropicStatusFallbackTable asserts the FS §16 canonical status
// mapping on anthropic's status fallback path (no native error code) —
// identical rows to the responsesapi and chatcompletions classifier tables,
// proving the sentinels can't drift between engines.
func TestAnthropicStatusFallbackTable(t *testing.T) {
	cases := []struct {
		status int
		want   error
	}{
		{401, llm.ErrAuth},
		{402, llm.ErrInsufficientCredits},
		{403, llm.ErrPermission},
		{404, llm.ErrNotFound},
		{408, llm.ErrTimeout},
		{429, llm.ErrRateLimited},
		{503, llm.ErrOverloaded},
		{529, llm.ErrOverloaded},
		{500, llm.ErrServer},
		{502, llm.ErrServer},
		{400, llm.ErrBadRequest},
	}
	for _, tc := range cases {
		if got := errorKind(tc.status, "", ""); !errors.Is(got, tc.want) {
			t.Fatalf("errorKind(%d) = %v, want %v", tc.status, got, tc.want)
		}
	}
}

func TestAnthropicOptionsAndDebugCapture(t *testing.T) {
	var sawKey string
	var sawUserAgent string
	var sawXApp string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawKey = r.Header.Get("X-Api-Key")
		sawUserAgent = r.Header.Get("User-Agent")
		sawXApp = r.Header.Get("x-app")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"pong"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer server.Close()

	var captures []llm.WireCapture
	p, err := New(
		WithAPIKeyFunc(func(context.Context) (string, error) { return "dynamic-secret", nil }),
		WithBaseURL(server.URL),
		WithHTTPClient(server.Client()),
		WithDefaultMaxTokens(7),
		WithWireCapture(func(c llm.WireCapture) { captures = append(captures, c) }),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if p.Client() == nil {
		t.Fatalf("Client returned nil")
	}

	resp, err := p.Chat(context.Background(), &llm.Request{
		Model:    "claude-test",
		Messages: []llm.Message{llm.UserText("ping")},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Text() != "pong" || sawKey != "dynamic-secret" {
		t.Fatalf("response/key = %q/%q", resp.Text(), sawKey)
	}
	// Claude Code OAuth identity headers must NOT leak into api-key mode.
	if strings.HasPrefix(sawUserAgent, "claude-cli/") || sawXApp != "" {
		t.Fatalf("api-key request carried OAuth identity headers: ua=%q x-app=%q", sawUserAgent, sawXApp)
	}
	if len(captures) != 1 {
		t.Fatalf("captures len = %d, want 1", len(captures))
	}
	if got := captures[0].RequestHeaders.Get("X-Api-Key"); got != "[REDACTED]" {
		t.Fatalf("captured X-Api-Key = %q, want redacted", got)
	}
	if bytes.Contains(captures[0].RequestBody, []byte("dynamic-secret")) || bytes.Contains(captures[0].ResponseBody, []byte("dynamic-secret")) {
		t.Fatalf("capture leaked API key: %+v", captures[0])
	}
}

func TestAnthropicOAuthHeadersAndRetry(t *testing.T) {
	var messageAuth []string
	var messageAPIKeys []string
	var messageBetas []string
	var messageUserAgents []string
	var messageXApps []string
	var messageBodies []string
	var refreshBody string
	var refreshed llm.AuthCredential
	var persistenceHadDeadline bool
	messageCalls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/messages":
			messageCalls++
			body, _ := io.ReadAll(r.Body)
			messageBodies = append(messageBodies, string(body))
			messageAuth = append(messageAuth, r.Header.Get("Authorization"))
			messageAPIKeys = append(messageAPIKeys, r.Header.Get("X-Api-Key"))
			messageBetas = append(messageBetas, r.Header.Get("anthropic-beta"))
			messageUserAgents = append(messageUserAgents, r.Header.Get("User-Agent"))
			messageXApps = append(messageXApps, r.Header.Get("x-app"))
			if messageCalls == 1 {
				http.Error(w, `{"error":{"type":"authentication_error","message":"expired"}}`, http.StatusUnauthorized)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"pong"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		case "/oauth/token":
			body, _ := io.ReadAll(r.Body)
			refreshBody = string(body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"access_token":"new-access","refresh_token":"new-refresh","expires_in":3600}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	p, err := New(
		WithOAuth(llm.AuthCredential{Type: "oauth", Access: "old-access", Refresh: "old-refresh"}, func(ctx context.Context, cred llm.AuthCredential) error {
			_, persistenceHadDeadline = ctx.Deadline()
			refreshed = cred
			return nil
		}),
		withOAuthTokenURL(server.URL+"/oauth/token"),
		WithBaseURL(server.URL),
		WithHTTPClient(server.Client()),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	resp, err := p.Chat(context.Background(), &llm.Request{
		Model:     "claude-test",
		System:    "You are terse.",
		MaxTokens: 8,
		Messages:  []llm.Message{llm.UserText("ping")},
		ProviderOptions: Options{
			BetaHeaders: []string{"structured-outputs-2025-11-13"},
		},
	})
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	if resp.Text() != "pong" {
		t.Fatalf("response text = %q", resp.Text())
	}
	if !reflect.DeepEqual(messageAuth, []string{"Bearer old-access", "Bearer new-access"}) {
		t.Fatalf("Authorization headers = %+v", messageAuth)
	}
	if !reflect.DeepEqual(messageAPIKeys, []string{"", ""}) {
		t.Fatalf("X-Api-Key headers = %+v", messageAPIKeys)
	}
	wantBeta := anthropicClaudeCodeBeta + "," + anthropicOAuthBeta + ",structured-outputs-2025-11-13"
	if !reflect.DeepEqual(messageBetas, []string{wantBeta, wantBeta}) {
		t.Fatalf("anthropic-beta headers = %+v, want %q", messageBetas, wantBeta)
	}
	if !reflect.DeepEqual(messageUserAgents, []string{anthropicOAuthUserAgent, anthropicOAuthUserAgent}) {
		t.Fatalf("User-Agent headers = %+v", messageUserAgents)
	}
	if !reflect.DeepEqual(messageXApps, []string{"cli", "cli"}) {
		t.Fatalf("x-app headers = %+v", messageXApps)
	}
	for _, body := range messageBodies {
		var payload struct {
			System []struct {
				Text string `json:"text"`
			} `json:"system"`
		}
		if err := json.Unmarshal([]byte(body), &payload); err != nil {
			t.Fatalf("request body is invalid JSON: %v\n%s", err, body)
		}
		if len(payload.System) != 2 || payload.System[0].Text != claudeCodeSystemPrompt || payload.System[1].Text != "You are terse." {
			t.Fatalf("OAuth system blocks = %+v, want Claude Code identity first", payload.System)
		}
	}
	if !strings.Contains(refreshBody, `"refresh_token":"old-refresh"`) || !strings.Contains(refreshBody, `"client_id":"`+anthropicOAuthClientID+`"`) {
		t.Fatalf("refresh body missing expected non-secret fields: %s", refreshBody)
	}
	if refreshed.Access != "new-access" || refreshed.Refresh != "new-refresh" {
		t.Fatalf("refreshed credential = %+v", refreshed)
	}
	if !persistenceHadDeadline {
		t.Fatal("persistence callback did not receive generation deadline")
	}
}

func TestAnthropicOAuthPersistenceContract(t *testing.T) {
	networkCalls := 0
	client := &http.Client{Transport: testutil.RoundTripFunc(func(*http.Request) (*http.Response, error) {
		networkCalls++
		return nil, errors.New("unexpected network request")
	})}

	p, err := New(
		WithOAuth(llm.AuthCredential{Type: "oauth", Access: "access", Refresh: "refresh"}, nil),
		WithHTTPClient(client),
	)
	if !errors.Is(err, llm.ErrBadRequest) {
		t.Fatalf("refreshable New error = %v, want ErrBadRequest", err)
	}
	if p != nil || networkCalls != 0 {
		t.Fatalf("refreshable New provider/network calls = %+v/%d, want nil/0", p, networkCalls)
	}

	p, err = New(
		WithOAuth(llm.AuthCredential{Type: "oauth", Access: "access-only"}, nil),
		WithHTTPClient(client),
	)
	if err != nil {
		t.Fatalf("access-only New returned error: %v", err)
	}
	if p == nil || networkCalls != 0 {
		t.Fatalf("access-only New provider/network calls = %+v/%d, want non-nil/0", p, networkCalls)
	}
}

func TestAnthropicOAuthPersistenceErrorStopsRetry(t *testing.T) {
	persistErr := errors.New("persist anthropic credential")
	messageCalls := 0
	persistenceCalls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/messages":
			messageCalls++
			http.Error(w, `{"error":{"type":"authentication_error","message":"expired"}}`, http.StatusUnauthorized)
		case "/oauth/token":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"access_token":"new-access","refresh_token":"new-refresh","expires_in":3600}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	p, err := New(
		WithOAuth(llm.AuthCredential{Type: "oauth", Access: "old-access", Refresh: "old-refresh"}, func(ctx context.Context, _ llm.AuthCredential) error {
			persistenceCalls++
			if _, ok := ctx.Deadline(); !ok {
				t.Error("persistence context has no generation deadline")
			}
			return persistErr
		}),
		withOAuthTokenURL(server.URL+"/oauth/token"),
		WithBaseURL(server.URL),
		WithHTTPClient(server.Client()),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	resp, err := p.Chat(context.Background(), &llm.Request{
		Model:     "claude-test",
		MaxTokens: 8,
		Messages:  []llm.Message{llm.UserText("ping")},
	})
	if !errors.Is(err, persistErr) {
		t.Fatalf("Chat error = %v, want persistence error", err)
	}
	if resp != nil {
		t.Fatalf("Chat response = %+v, want nil", resp)
	}
	if messageCalls != 1 || persistenceCalls != 1 {
		t.Fatalf("message/persistence calls = %d/%d, want 1/1", messageCalls, persistenceCalls)
	}
}

func TestAnthropicOAuthSystemBlockGoldens(t *testing.T) {
	oauthProvider := &Provider{defaultMaxTokens: 8, oauth: true}
	params, _, err := oauthProvider.buildParams(&llm.Request{
		Model:       "claude-test",
		System:      "You are terse.",
		SystemCache: &llm.CacheHint{TTL: time.Hour},
		Messages:    []llm.Message{llm.UserText("hello")},
	})
	if err != nil {
		t.Fatalf("buildParams returned error: %v", err)
	}
	got := testutil.MustCompactJSON(t, params.System)
	want := `[{"text":"You are Claude Code, Anthropic's official CLI for Claude.","cache_control":{"ttl":"1h","type":"ephemeral"},"type":"text"},{"text":"You are terse.","cache_control":{"ttl":"1h","type":"ephemeral"},"type":"text"}]`
	testutil.AssertJSONEqual(t, got, want)

	// Without user System text the identity block still leads (and is alone).
	params, _, err = oauthProvider.buildParams(&llm.Request{
		Model:    "claude-test",
		Messages: []llm.Message{llm.UserText("hello")},
	})
	if err != nil {
		t.Fatalf("buildParams returned error: %v", err)
	}
	if len(params.System) != 1 || params.System[0].Text != claudeCodeSystemPrompt {
		t.Fatalf("OAuth system blocks without user system = %+v", params.System)
	}

	// Api-key mode must NOT inject the identity block.
	apiKeyProvider := &Provider{defaultMaxTokens: 8}
	params, _, err = apiKeyProvider.buildParams(&llm.Request{
		Model:    "claude-test",
		System:   "You are terse.",
		Messages: []llm.Message{llm.UserText("hello")},
	})
	if err != nil {
		t.Fatalf("buildParams returned error: %v", err)
	}
	if len(params.System) != 1 || params.System[0].Text != "You are terse." {
		t.Fatalf("api-key system blocks = %+v", params.System)
	}
	raw := testutil.MustCompactJSON(t, params)
	if strings.Contains(raw, "Claude Code") {
		t.Fatalf("api-key request contains Claude Code identity: %s", raw)
	}
}

func TestAnthropicStopReasonTable(t *testing.T) {
	tests := []struct {
		raw  string
		want llm.StopReason
	}{
		{raw: "end_turn", want: llm.StopReasonEndTurn},
		{raw: "max_tokens", want: llm.StopReasonMaxTokens},
		{raw: "stop_sequence", want: llm.StopReasonStopSequence},
		{raw: "tool_use", want: llm.StopReasonToolUse},
		{raw: "refusal", want: llm.StopReasonRefusal},
		{raw: "pause_turn", want: llm.StopReasonPaused},
		{raw: "model_context_window_exceeded", want: llm.StopReasonContextOverflow},
		{raw: "", want: ""},
		{raw: "something_new", want: llm.StopReasonOther},
	}
	for _, tt := range tests {
		name := tt.raw
		if name == "" {
			name = "empty"
		}
		t.Run(name, func(t *testing.T) {
			if got := mapStopReason(tt.raw); got != tt.want {
				t.Fatalf("mapStopReason(%q) = %q, want %q", tt.raw, got, tt.want)
			}
		})
	}
}

func TestAnthropicRefreshErrorMapping(t *testing.T) {
	tests := []struct {
		status int
		want   error
	}{
		{status: http.StatusBadRequest, want: llm.ErrAuth},
		{status: http.StatusUnauthorized, want: llm.ErrAuth},
		{status: http.StatusForbidden, want: llm.ErrAuth},
		{status: http.StatusRequestTimeout, want: llm.ErrTimeout},
		{status: http.StatusTooManyRequests, want: llm.ErrRateLimited},
		{status: http.StatusInternalServerError, want: llm.ErrServer},
	}
	for _, tt := range tests {
		t.Run(http.StatusText(tt.status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, `{"error":"invalid_grant"}`, tt.status)
			}))
			defer server.Close()
			_, err := refreshAnthropicOAuth(context.Background(), server.Client(), server.URL, llm.AuthCredential{Refresh: "stale"})
			if !errors.Is(err, tt.want) {
				t.Fatalf("refresh error = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestAnthropicModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"data":[{"id":"claude-test","display_name":"Claude Test","max_input_tokens":200000,"max_tokens":8192}],"has_more":false}`)
	}))
	defer server.Close()

	p, err := New(WithAPIKey("test-key"), WithBaseURL(server.URL), WithHTTPClient(server.Client()))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatalf("Models returned error: %v", err)
	}
	if len(models) != 1 || models[0].ID != "claude-test" || models[0].ContextWindow != 200000 || models[0].MaxOutputTokens != 8192 {
		t.Fatalf("models = %+v", models)
	}
}

func TestAnthropicNewRequiresAPIKey(t *testing.T) {
	t.Setenv(apiKeyEnv, "")
	_, err := New()
	if !errors.Is(err, llm.ErrAuth) {
		t.Fatalf("New error = %v, want ErrAuth", err)
	}
}

func mapAnthropicStream(t *testing.T, rawEvents []string) []llm.Event {
	t.Helper()
	state := newStreamState(&Provider{})
	return testutil.MapRawEvents(t, rawEvents, state.mapEvent)
}

func mapOneAnthropicEvent(t *testing.T, state *streamState, raw string) []llm.Event {
	t.Helper()
	var event sdk.MessageStreamEventUnion
	if err := json.Unmarshal([]byte(raw), &event); err != nil {
		t.Fatalf("unmarshal stream event %s: %v", raw, err)
	}
	events, err := state.mapEvent(event)
	if err != nil {
		t.Fatalf("mapEvent returned error: %v", err)
	}
	return events
}

func collectAnthropicStream(t *testing.T, rawEvents []string) *llm.Response {
	t.Helper()
	state := newStreamState(&Provider{})
	return testutil.CollectRawEvents(t, rawEvents, state.mapEvent)
}

func collectContractAnthropicStream(t *testing.T, rawEvents []string) *llm.Response {
	t.Helper()
	state := newStreamState(&Provider{})
	events := testutil.MapRawEvents(t, rawEvents, state.mapEvent)
	resp, err := llm.Collect(providerutil.StreamContract(providerName, testutil.EventSeq(events...)))
	if err != nil {
		t.Fatalf("Collect returned error: %v", err)
	}
	return resp
}

func newAnthropicStreamFixtureProvider(t *testing.T, roundTrip func(*http.Request) (*http.Response, error)) *Provider {
	t.Helper()
	provider, err := New(
		WithAPIKey("test-key"),
		WithBaseURL("https://anthropic.test"),
		WithHTTPClient(&http.Client{Transport: testutil.RoundTripFunc(roundTrip)}),
		WithMaxRetries(0),
	)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	return provider
}

func anthropicStreamFixtureRequest() *llm.Request {
	return &llm.Request{
		Model:     "claude-test",
		MaxTokens: 8,
		Messages:  []llm.Message{llm.UserText("hello")},
	}
}

func anthropicStreamResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func anthropicSSE(events ...string) string {
	var out strings.Builder
	for _, event := range events {
		var envelope struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal([]byte(event), &envelope)
		out.WriteString("event: ")
		out.WriteString(envelope.Type)
		out.WriteString("\ndata: ")
		out.WriteString(event)
		out.WriteString("\n\n")
	}
	return out.String()
}

func assertAnthropicProviderServerError(t *testing.T, err error) {
	t.Helper()
	if !errors.Is(err, llm.ErrServer) {
		t.Fatalf("error = %v, want ErrServer", err)
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.Provider != providerName {
		t.Fatalf("error = %v, want Anthropic ProviderError", err)
	}
}

type anthropicObservedItem struct {
	event llm.Event
	err   error
}

func observeAnthropicStream(seq iter.Seq2[llm.Event, error]) []anthropicObservedItem {
	var items []anthropicObservedItem
	for event, err := range seq {
		items = append(items, anthropicObservedItem{event: event, err: err})
	}
	return items
}

func collectAnthropicObserved(items []anthropicObservedItem) (*llm.Response, error) {
	return llm.Collect(func(yield func(llm.Event, error) bool) {
		for _, item := range items {
			if !yield(item.event, item.err) {
				return
			}
		}
	})
}

func assertAnthropicObservedKinds(t *testing.T, items []anthropicObservedItem, want []string) {
	t.Helper()
	got := make([]string, 0, len(items))
	for _, item := range items {
		if item.err != nil {
			got = append(got, "error")
			continue
		}
		event := providerutil.DerefEvent(item.event)
		if event == nil {
			got = append(got, "nil")
			continue
		}
		got = append(got, reflect.TypeOf(event).Name())
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("observed stream kinds = %v, want %v", got, want)
	}
}

func assertAnthropicPartialToolCall(t *testing.T, resp *llm.Response, id, args string) {
	t.Helper()
	if resp == nil || len(resp.Parts) != 1 {
		t.Fatalf("partial response = %#v, want one tool call", resp)
	}
	call, ok := resp.Parts[0].(llm.ToolCallPart)
	if !ok || call.ID != id || call.Name != "lookup" || string(call.Args) != args {
		t.Fatalf("partial tool call = %#v, want id %q args %q", resp.Parts[0], id, args)
	}
}

func assertAnthropicDroppedOnly(t *testing.T, resp *llm.Response, index int, reason string) {
	t.Helper()
	if resp == nil || len(resp.Parts) != 0 || len(resp.DroppedToolCalls) != 1 {
		t.Fatalf("dropped response = %#v", resp)
	}
	drop := resp.DroppedToolCalls[0]
	if drop.Index != index || drop.Reason != reason {
		t.Fatalf("dropped tool call = %#v, want index %d reason %q", drop, index, reason)
	}
}

func assertReasoningTerminalOrder(t *testing.T, events []llm.Event, index int) {
	t.Helper()
	if len(events) < 2 {
		t.Fatalf("events = %#v, want terminal reasoning and message events", events)
	}
	reasoning, ok := providerutil.DerefEvent(events[len(events)-2]).(llm.ReasoningDelta)
	if !ok || reasoning.Index != index || len(reasoning.Raw) == 0 || reasoning.Provider != providerName {
		t.Fatalf("penultimate event = %#v, want final reasoning index %d", events[len(events)-2], index)
	}
	if _, ok := providerutil.DerefEvent(events[len(events)-1]).(llm.MessageEnd); !ok {
		t.Fatalf("last event = %#v, want MessageEnd", events[len(events)-1])
	}
}

func assertReasoningImmediatelyBeforeError(t *testing.T, items []anthropicObservedItem, index int, raw string) {
	t.Helper()
	if len(items) < 2 || items[len(items)-1].err == nil {
		t.Fatalf("observed items = %#v, want terminal error", items)
	}
	reasoning, ok := providerutil.DerefEvent(items[len(items)-2].event).(llm.ReasoningDelta)
	if !ok || reasoning.Index != index || reasoning.Provider != providerName || !bytes.Equal(reasoning.Raw, []byte(raw)) {
		t.Fatalf("pre-error reasoning = %#v, want index %d raw %s", items[len(items)-2].event, index, raw)
	}
}

func assertAnthropicReasoning(t *testing.T, resp *llm.Response, text, raw string) {
	t.Helper()
	if resp == nil || len(resp.Parts) != 1 {
		t.Fatalf("response = %#v, want one reasoning part", resp)
	}
	reasoning, ok := resp.Parts[0].(llm.ReasoningPart)
	if !ok || reasoning.Text != text || reasoning.Provider != providerName || !bytes.Equal(reasoning.Raw, []byte(raw)) {
		t.Fatalf("reasoning = %#v, want text %q raw %s", resp.Parts[0], text, raw)
	}
}

func assertAnthropicReasoningMatchesBlocking(t *testing.T, streamed *llm.Response, raw string) {
	t.Helper()
	var block sdk.ContentBlockUnion
	if err := json.Unmarshal([]byte(raw), &block); err != nil {
		t.Fatalf("unmarshal reasoning block: %v", err)
	}
	blocking, err := (&Provider{}).mapMessage(&sdk.Message{
		ID:      "msg_1",
		Model:   sdk.Model("claude-test"),
		Content: []sdk.ContentBlockUnion{block},
	})
	if err != nil {
		t.Fatalf("mapMessage returned error: %v", err)
	}
	if !reflect.DeepEqual(streamed.Parts, blocking.Parts) {
		t.Fatalf("stream/blocking reasoning differ:\nstream:   %#v\nblocking: %#v", streamed.Parts, blocking.Parts)
	}
}

type anthropicReadErrorBody struct {
	reader *strings.Reader
	err    error
}

func (b *anthropicReadErrorBody) Read(p []byte) (int, error) {
	n, err := b.reader.Read(p)
	if errors.Is(err, io.EOF) {
		return 0, b.err
	}
	return n, err
}

func (*anthropicReadErrorBody) Close() error { return nil }
