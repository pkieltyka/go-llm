package chatcompletions_test

import (
	"context"
	"errors"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	sdk "github.com/openai/openai-go/v3"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/internal/e2e"
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
)

// replayDialect is a plain OpenAI-compatible dialect with no quirks: it
// defers part extraction to the adapter's default chat-completions mapping
// and uses the shared error-kind table, so replaying the recorded OpenRouter
// corpus (a standard chat-completions wire shape) exercises the adapter's own
// convert/response/stream/error paths directly.
type replayDialect struct{}

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
