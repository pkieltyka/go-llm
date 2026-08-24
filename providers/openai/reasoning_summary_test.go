package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/testutil"
)

func TestOpenAIReasoningSummaryBuildParamsWire(t *testing.T) {
	tests := []struct {
		name      string
		effort    llm.Effort
		summary   ReasoningSummary
		reasoning string
	}{
		{name: "empty option and effort"},
		{name: "empty option with low effort", effort: llm.EffortLow, reasoning: `{"effort":"low","summary":"auto"}`},
		{name: "empty option with high effort", effort: llm.EffortHigh, reasoning: `{"effort":"high","summary":"auto"}`},
		{name: "auto without effort", summary: ReasoningSummaryAuto, reasoning: `{"summary":"auto"}`},
		{name: "concise without effort", summary: ReasoningSummaryConcise, reasoning: `{"summary":"concise"}`},
		{name: "detailed without effort", summary: ReasoningSummaryDetailed, reasoning: `{"summary":"detailed"}`},
		{name: "concise overrides effort auto", effort: llm.EffortLow, summary: ReasoningSummaryConcise, reasoning: `{"effort":"low","summary":"concise"}`},
		{name: "detailed overrides effort auto", effort: llm.EffortHigh, summary: ReasoningSummaryDetailed, reasoning: `{"effort":"high","summary":"detailed"}`},
		{name: "none effort with summary", effort: llm.EffortNone, summary: ReasoningSummaryConcise, reasoning: `{"effort":"none","summary":"concise"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var blockingWire string
			for _, stream := range []bool{false, true} {
				req := &llm.Request{
					Model:    "gpt-test",
					Messages: []llm.Message{llm.UserText("hello")},
					Effort:   tt.effort,
					ProviderOptions: Options{
						ReasoningSummary: tt.summary,
					},
				}
				params, err := (&Provider{}).adapter().BuildParams(req, stream)
				if err != nil {
					t.Fatalf("BuildParams(stream=%v) returned error: %v", stream, err)
				}
				got := testutil.MustCompactJSON(t, params)
				want := `{"store":false,"include":["reasoning.encrypted_content"],"input":[{"content":[{"text":"hello","type":"input_text"}],"role":"user"}],"model":"gpt-test"`
				if tt.reasoning != "" {
					want += `,"reasoning":` + tt.reasoning
				}
				want += `}`
				if got != want {
					t.Fatalf("BuildParams(stream=%v) wire\ngot:  %s\nwant: %s", stream, got, want)
				}
				assertReasoningSummaryWire(t, []byte(got), tt.reasoning)
				if !stream {
					blockingWire = got
				} else if got != blockingWire {
					t.Fatalf("blocking and streaming params differ:\nblocking: %s\nstream:   %s", blockingWire, got)
				}
			}
		})
	}
}

func TestOpenAIReasoningSummaryDoesNotMutateCallerValues(t *testing.T) {
	options := &Options{ReasoningSummary: ReasoningSummaryConcise}
	wantOptions := *options
	req := &llm.Request{
		Model:           "gpt-test",
		Messages:        []llm.Message{llm.UserText("hello")},
		Effort:          llm.EffortHigh,
		ProviderOptions: options,
	}
	wantMessages := append([]llm.Message(nil), req.Messages...)

	for _, stream := range []bool{false, true} {
		if _, err := (&Provider{}).adapter().BuildParams(req, stream); err != nil {
			t.Fatalf("BuildParams(stream=%v) returned error: %v", stream, err)
		}
	}
	if !reflect.DeepEqual(*options, wantOptions) {
		t.Fatalf("options mutated: got %+v, want %+v", *options, wantOptions)
	}
	if req.Model != "gpt-test" || req.Effort != llm.EffortHigh || req.ProviderOptions != options || !reflect.DeepEqual(req.Messages, wantMessages) {
		t.Fatalf("request mutated: %+v", req)
	}
}

type spoofedOpenAIOptions struct{}

func (spoofedOpenAIOptions) ForProvider() string { return providerName }

func TestOpenAIReasoningSummaryPreservesWrongOptionsTypeError(t *testing.T) {
	for _, stream := range []bool{false, true} {
		_, err := (&Provider{}).adapter().BuildParams(&llm.Request{
			Model:           "gpt-test",
			Messages:        []llm.Message{llm.UserText("hello")},
			ProviderOptions: spoofedOpenAIOptions{},
		}, stream)
		if !errors.Is(err, llm.ErrBadRequest) {
			t.Fatalf("BuildParams(stream=%v) error = %v, want ErrBadRequest", stream, err)
		}
	}
}

func TestOpenAIReasoningSummaryBlockingAndStreamingRequestsAndResponses(t *testing.T) {
	const reasoningItem = `{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"visible summary"}],"encrypted_content":"enc","status":"completed"}`
	const finalResponse = `{"id":"resp_1","model":"gpt-test","status":"completed","output":[` + reasoningItem + `],"usage":{"input_tokens":2,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":1},"total_tokens":3}}`
	var requestBodies [][]byte
	client := &http.Client{Transport: testutil.RoundTripFunc(func(req *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		requestBodies = append(requestBodies, append([]byte(nil), body...))
		contentType := "application/json"
		responseBody := finalResponse
		if bytes.Contains(body, []byte(`"stream":true`)) {
			contentType = "text/event-stream"
			responseBody = "event: response.created\n" +
				`data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","model":"gpt-test","status":"in_progress","output":[]}}` + "\n\n" +
				"event: response.reasoning_summary_text.delta\n" +
				`data: {"type":"response.reasoning_summary_text.delta","sequence_number":1,"item_id":"rs_1","output_index":0,"summary_index":0,"delta":"visible summary"}` + "\n\n" +
				"event: response.output_item.done\n" +
				`data: {"type":"response.output_item.done","sequence_number":2,"output_index":0,"item":` + reasoningItem + `}` + "\n\n" +
				"event: response.completed\n" +
				`data: {"type":"response.completed","sequence_number":3,"response":` + finalResponse + `}` + "\n\n"
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{contentType}},
			Body:       io.NopCloser(strings.NewReader(responseBody)),
			Request:    req,
		}, nil
	})}
	p, err := New(WithAPIKey("test-key"), WithHTTPClient(client), WithMaxRetries(0))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	newRequest := func() *llm.Request {
		return &llm.Request{
			Model:    "gpt-test",
			Messages: []llm.Message{llm.UserText("hello")},
			Effort:   llm.EffortHigh,
			ProviderOptions: Options{
				ReasoningSummary: ReasoningSummaryDetailed,
			},
		}
	}

	blocking, err := p.Chat(context.Background(), newRequest())
	if err != nil {
		t.Fatalf("Chat returned error: %v", err)
	}
	streaming, err := llm.Collect(p.ChatStream(context.Background(), newRequest()))
	if err != nil {
		t.Fatalf("ChatStream returned error: %v", err)
	}
	if len(requestBodies) != 2 {
		t.Fatalf("fixture requests = %d, want 2", len(requestBodies))
	}
	for i, body := range requestBodies {
		assertReasoningSummaryWire(t, body, `{"effort":"high","summary":"detailed"}`)
		var payload map[string]json.RawMessage
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("request %d JSON: %v", i, err)
		}
		_, streams := payload["stream"]
		if streams != (i == 1) {
			t.Fatalf("request %d stream presence = %v, want %v: %s", i, streams, i == 1, body)
		}
	}
	for name, resp := range map[string]*llm.Response{"blocking": blocking, "streaming": streaming} {
		if resp.Reasoning() != "visible summary" {
			t.Fatalf("%s reasoning = %q", name, resp.Reasoning())
		}
		parts := reasoningParts(resp.Parts)
		if len(parts) != 1 || !bytes.Equal(parts[0].Raw, []byte(reasoningItem)) {
			t.Fatalf("%s reasoning parts = %+v", name, parts)
		}
		raw, ok := resp.Raw.(*responses.Response)
		if !ok {
			t.Fatalf("%s terminal raw = %T, want *responses.Response", name, resp.Raw)
		}
		if raw.RawJSON() != finalResponse {
			t.Fatalf("%s terminal raw changed\ngot:  %s\nwant: %s", name, raw.RawJSON(), finalResponse)
		}
	}
}

func TestOpenAIReasoningSummaryInvalidValuesFailBeforeNetwork(t *testing.T) {
	var requests atomic.Int64
	client := &http.Client{Transport: testutil.RoundTripFunc(func(*http.Request) (*http.Response, error) {
		requests.Add(1)
		return nil, errors.New("unexpected network request")
	})}
	p, err := New(WithAPIKey("test-key"), WithHTTPClient(client), WithMaxRetries(0))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	for _, summary := range []ReasoningSummary{"brief", " concise"} {
		t.Run(string(summary), func(t *testing.T) {
			newRequest := func() *llm.Request {
				return &llm.Request{
					Model:    "gpt-test",
					Messages: []llm.Message{llm.UserText("hello")},
					ProviderOptions: Options{
						ReasoningSummary: summary,
					},
				}
			}
			if _, err := p.Chat(context.Background(), newRequest()); !errors.Is(err, llm.ErrBadRequest) {
				t.Fatalf("Chat error = %v, want ErrBadRequest", err)
			}
			streamed, err := llm.Collect(p.ChatStream(context.Background(), newRequest()))
			if !errors.Is(err, llm.ErrBadRequest) {
				t.Fatalf("ChatStream error = %v, want ErrBadRequest", err)
			}
			if streamed != nil {
				t.Fatalf("ChatStream response = %+v, want nil", streamed)
			}
		})
	}
	if got := requests.Load(); got != 0 {
		t.Fatalf("fixture requests = %d, want 0", got)
	}
}

func assertReasoningSummaryWire(t *testing.T, body []byte, want string) {
	t.Helper()
	if bytes.Contains(body, []byte(`"generate_summary"`)) {
		t.Fatalf("request uses deprecated reasoning.generate_summary: %s", body)
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("request JSON: %v\n%s", err, body)
	}
	got, ok := payload["reasoning"]
	if want == "" {
		if ok {
			t.Fatalf("reasoning = %s, want omitted", got)
		}
		return
	}
	if !ok {
		t.Fatalf("reasoning omitted, want %s: %s", want, body)
	}
	if string(got) != want {
		t.Fatalf("reasoning = %s, want %s", got, want)
	}
}
