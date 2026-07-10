package chatcompletions

import (
	"context"
	"encoding/json"
	"net/http"

	sdk "github.com/openai/openai-go/v3"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// genericDialect is the quirk-free dialect behind New: standard
// chat-completions mappings everywhere, with the caller's Compat as the only
// server-specific configuration.
type genericDialect struct {
	name         string
	capabilities []llm.Capability
	compat       Compat
}

func (d genericDialect) Name() string           { return d.name }
func (d genericDialect) DefaultBaseURL() string { return "" }
func (d genericDialect) APIKeyEnv() string      { return "" }

func (d genericDialect) Capabilities() []llm.Capability {
	return append([]llm.Capability(nil), d.capabilities...)
}

func (d genericDialect) Compat() Compat { return d.compat }

func (genericDialect) ApplyRequest(*llm.Request, *sdk.ChatCompletionNewParams, JSONObject) error {
	return nil
}

// MapStopReason maps the standard chat-completions finish_reason vocabulary.
func (genericDialect) MapStopReason(raw string) llm.StopReason {
	return DefaultStopReason(raw)
}

func (genericDialect) MapErrorStatus(status int, code, message string) error {
	return DefaultErrorKind(status, code, message)
}

func (genericDialect) ExtractParts(JSONObject, RawMessage) ([]llm.Part, []llm.DroppedToolCall, error) {
	return nil, nil, nil
}

// ExtractExtras exposes the raw response root (choice-stripped on the
// streaming path) as Response.Raw so unmapped server fields stay reachable.
func (genericDialect) ExtractExtras(raw JSONObject, _ RawChoice) any {
	if len(raw) == 0 {
		return nil
	}
	return raw
}

func (d genericDialect) MapUsage(model string, raw RawUsage, table llm.PriceTable) llm.Usage {
	return DefaultUsage(d.name, model, raw, table)
}

// Models lists the server's models via GET {baseURL}/models, mapping the
// standard OpenAI list shape. Rows carry their full decoded payload in Raw
// for server-specific fields (e.g. vLLM's max_model_len and LoRA parent).
func (genericDialect) Models(ctx context.Context, p *Provider) ([]llm.ModelInfo, error) {
	return DefaultModels(ctx, p)
}

// DefaultStopReason is the standard chat-completions finish_reason mapping
// shared by the generic dialect and presets without extra vocabulary.
func DefaultStopReason(raw string) llm.StopReason {
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

// DefaultUsage is the standard chat-completions usage normalization (FS §11):
// InputTokens excludes cache reads, ReasoningTokens is the informational
// subset from completion_tokens_details, and TotalTokens is recomputed when
// the wire omits it. Costs come from the price table when provided (falling
// back to the embedded table); self-hosted servers typically match neither
// and report no cost.
func DefaultUsage(provider, model string, raw RawUsage, table llm.PriceTable) llm.Usage {
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
	if table != nil {
		return llm.EstimateCostWithTable(table, provider, model, out)
	}
	return llm.EstimateCostForModel(provider, model, out)
}

// DefaultModels lists models via GET {baseURL}/models using the standard
// OpenAI list shape, keeping each row's decoded payload in ModelInfo.Raw.
func DefaultModels(ctx context.Context, p *Provider) ([]llm.ModelInfo, error) {
	var payload struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := p.DoJSON(ctx, http.MethodGet, "/models", nil, &payload); err != nil {
		return nil, err
	}
	models := make([]llm.ModelInfo, 0, len(payload.Data))
	for _, rawRow := range payload.Data {
		var row struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(rawRow, &row); err != nil {
			return nil, providerutil.NormalizeRemoteError(p.Name(), err)
		}
		models = append(models, llm.ModelInfo{
			ID:  row.ID,
			Raw: append(json.RawMessage(nil), rawRow...),
		})
	}
	return models, nil
}
