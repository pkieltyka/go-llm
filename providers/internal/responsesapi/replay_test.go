package responsesapi_test

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/e2e"
	"github.com/pkieltyka/go-llm/internal/testutil"
	"github.com/pkieltyka/go-llm/providers/internal/responsesapi"
)

const replayAdapterName = "responsesapi-replay"

func replayAdapter() responsesapi.Adapter {
	return responsesapi.Adapter{
		ProviderName: replayAdapterName,
		Capabilities: []llm.Capability{
			llm.CapabilityStreaming,
			llm.CapabilityTools,
			llm.CapabilityToolChoiceRequired,
			llm.CapabilityToolStreaming,
			llm.CapabilityParallelTools,
			llm.CapabilityStrictTools,
			llm.CapabilityJSONSchema,
			llm.CapabilityReasoning,
			llm.CapabilityImageInput,
			llm.CapabilityPDFInput,
		},
	}
}

// TestReplayRecordedFixtures drives the shared Responses mapping directly:
// each recorded openai-codex SSE stream (a genuine Responses wire stream) is
// decoded through StreamState.MapEvent and collected, and its terminal
// response object is mapped through MapResponse — asserting both paths meet
// the normalized invariants and agree with each other.
func TestReplayRecordedFixtures(t *testing.T) {
	root, err := e2e.RepoRoot(".")
	if err != nil {
		t.Fatalf("RepoRoot returned error: %v", err)
	}
	fixture := filepath.Join(root, "internal", "e2e", "fixtures", "openai-codex", "live.json")
	exchanges, err := e2e.LoadRecordedExchanges(fixture)
	if err != nil {
		t.Fatalf("LoadRecordedExchanges returned error: %v", err)
	}
	if len(exchanges) == 0 {
		t.Fatalf("fixture %s contains no exchanges", fixture)
	}
	profile := e2e.ReplayProfile{
		Provider:         replayAdapterName,
		ToolCallMarkers:  []string{`"type":"function_call"`},
		ReasoningMarkers: []string{`"type":"reasoning"`},
	}
	for i, exchange := range exchanges {
		if exchange.Err != "" || exchange.Status == 0 {
			continue
		}
		if exchange.Status >= 400 {
			t.Run(fmt.Sprintf("%02d_error_%d", i, exchange.Status), func(t *testing.T) {
				replayHTTPError(t, exchange)
			})
			continue
		}
		t.Run(fmt.Sprintf("%02d_stream", i), func(t *testing.T) {
			replayStreamExchange(t, exchange, profile)
		})
	}
}

func replayHTTPError(t *testing.T, exchange e2e.RecordedExchange) {
	t.Helper()
	resp := &http.Response{
		StatusCode: exchange.Status,
		Header:     http.Header{},
		Body:       io.NopCloser(strings.NewReader(exchange.ResponseBody)),
	}
	err := replayAdapter().MapHTTPResponseError(resp)
	if err == nil {
		t.Fatalf("MapHTTPResponseError returned nil for status %d", exchange.Status)
	}
	if !errors.Is(err, llm.ErrBadRequest) && !errors.Is(err, llm.ErrNotFound) {
		t.Fatalf("MapHTTPResponseError = %v, want ErrBadRequest or ErrNotFound", err)
	}
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) || providerErr.HTTPStatus != exchange.Status {
		t.Fatalf("MapHTTPResponseError provider error = %+v, want HTTP status %d", providerErr, exchange.Status)
	}
}

func replayStreamExchange(t *testing.T, exchange e2e.RecordedExchange, profile e2e.ReplayProfile) {
	t.Helper()
	adapter := replayAdapter()
	state := adapter.NewStreamState()
	var events []llm.Event
	var terminalRaw json.RawMessage
	var doneItems []json.RawMessage
	for payload := range ssePayloads(t, exchange.ResponseBody) {
		if strings.TrimSpace(string(payload)) == "[DONE]" {
			continue
		}
		var event responses.ResponseStreamEventUnion
		if err := json.Unmarshal(payload, &event); err != nil {
			t.Fatalf("malformed recorded stream event: %v: %.120s", err, payload)
		}
		switch event.Type {
		case "response.completed", "response.incomplete":
			terminalRaw = terminalResponseJSON(t, payload)
		case "response.output_item.done":
			doneItems = append(doneItems, outputItemJSON(t, payload))
		}
		mapped, err := state.MapEvent(event)
		if err != nil {
			t.Fatalf("MapEvent returned error: %v", err)
		}
		events = append(events, mapped...)
	}

	streamed, err := llm.Collect(testutil.EventSeq(events...))
	if err != nil {
		t.Fatalf("Collect over mapped events returned error: %v", err)
	}
	e2e.AssertReplayResponse(t, exchange, profile, streamed)

	if len(terminalRaw) == 0 {
		t.Fatalf("recorded stream has no terminal response event")
	}
	// The codex backend sends response.completed with an EMPTY output array;
	// rebuild the terminal output from the recorded output_item.done items so
	// the blocking MapResponse path maps the same wire content.
	terminalRaw = withOutputItems(t, terminalRaw, doneItems)
	var terminal responses.Response
	if err := json.Unmarshal(terminalRaw, &terminal); err != nil {
		t.Fatalf("terminal response unmarshal returned error: %v", err)
	}
	mapped, err := adapter.MapResponse(&terminal)
	if err != nil {
		t.Fatalf("MapResponse returned error: %v", err)
	}
	e2e.AssertReplayResponse(t, exchange, profile, mapped)

	// Collect-equivalence (ARCH §9): the streamed and blocking mappings must
	// agree on the response essentials.
	if streamed.Text() != mapped.Text() {
		t.Fatalf("streamed text %q != mapped text %q", streamed.Text(), mapped.Text())
	}
	if streamed.StopReason != mapped.StopReason {
		t.Fatalf("streamed stop reason %q != mapped stop reason %q", streamed.StopReason, mapped.StopReason)
	}
	if len(streamed.ToolCalls()) != len(mapped.ToolCalls()) {
		t.Fatalf("streamed tool calls %d != mapped tool calls %d", len(streamed.ToolCalls()), len(mapped.ToolCalls()))
	}
}

func terminalResponseJSON(t *testing.T, payload []byte) json.RawMessage {
	t.Helper()
	var wire struct {
		Response json.RawMessage `json:"response"`
	}
	if err := json.Unmarshal(payload, &wire); err != nil {
		t.Fatalf("terminal event unmarshal returned error: %v", err)
	}
	return wire.Response
}

func outputItemJSON(t *testing.T, payload []byte) json.RawMessage {
	t.Helper()
	var wire struct {
		Item json.RawMessage `json:"item"`
	}
	if err := json.Unmarshal(payload, &wire); err != nil {
		t.Fatalf("output item event unmarshal returned error: %v", err)
	}
	return wire.Item
}

// withOutputItems splices the recorded output_item.done items into a terminal
// response whose output array is empty.
func withOutputItems(t *testing.T, terminal json.RawMessage, items []json.RawMessage) json.RawMessage {
	t.Helper()
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(terminal, &fields); err != nil {
		t.Fatalf("terminal response decode returned error: %v", err)
	}
	var output []json.RawMessage
	_ = json.Unmarshal(fields["output"], &output)
	if len(output) > 0 || len(items) == 0 {
		return terminal
	}
	joined, err := json.Marshal(items)
	if err != nil {
		t.Fatalf("output items marshal returned error: %v", err)
	}
	fields["output"] = joined
	rebuilt, err := json.Marshal(fields)
	if err != nil {
		t.Fatalf("terminal response rebuild returned error: %v", err)
	}
	return rebuilt
}

// ssePayloads yields the data payloads of a recorded SSE body.
func ssePayloads(t *testing.T, body string) iter.Seq[[]byte] {
	return func(yield func([]byte) bool) {
		scanner := bufio.NewScanner(strings.NewReader(body))
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		var data strings.Builder
		flush := func() bool {
			if data.Len() == 0 {
				return true
			}
			payload := strings.TrimSpace(data.String())
			data.Reset()
			if payload == "" {
				return true
			}
			return yield([]byte(payload))
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
			if value, ok := strings.CutPrefix(line, "data:"); ok {
				if data.Len() > 0 {
					data.WriteByte('\n')
				}
				data.WriteString(strings.TrimSpace(value))
			}
		}
		if err := scanner.Err(); err != nil {
			t.Fatalf("scan recorded SSE body: %v", err)
		}
		flush()
	}
}
