package chatcompletions

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	sdk "github.com/openai/openai-go/v3"
	sdkparam "github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// BuildParams converts a go-llm request to chat-completions parameters. It is
// an advanced, vendor-coupled escape hatch and is exempt from pre-v1 API
// stability; ordinary callers should use Chat or ChatStream.
func (p *Provider) BuildParams(req *llm.Request, stream bool) (sdk.ChatCompletionNewParams, error) {
	if stream {
		if err := llm.ValidateStreamRequest(p.Capabilities(), req); err != nil {
			return sdk.ChatCompletionNewParams{}, err
		}
	} else if err := llm.ValidateRequest(p.Capabilities(), req); err != nil {
		return sdk.ChatCompletionNewParams{}, err
	}
	if err := llm.ValidateProviderOptions(p.Name(), req); err != nil {
		return sdk.ChatCompletionNewParams{}, err
	}
	messages, err := p.buildMessages(req)
	if err != nil {
		return sdk.ChatCompletionNewParams{}, err
	}
	params := sdk.ChatCompletionNewParams{
		Model:    shared.ChatModel(req.Model),
		Messages: messages,
		N:        sdk.Int(1),
	}
	if req.MaxTokens > 0 {
		params.MaxTokens = sdk.Int(int64(req.MaxTokens))
	}
	if req.Temperature != nil {
		params.Temperature = sdk.Float(*req.Temperature)
	}
	if req.TopP != nil {
		params.TopP = sdk.Float(*req.TopP)
	}
	if len(req.StopSequences) > 0 {
		params.Stop = sdkparam.Override[sdk.ChatCompletionNewParamsStopUnion](req.StopSequences)
	}
	if len(req.Tools) > 0 {
		params.Tools, err = buildTools(req.Tools, p.Name())
		if err != nil {
			return sdk.ChatCompletionNewParams{}, err
		}
		if slices.Contains(p.Capabilities(), llm.CapabilityParallelTools) {
			params.ParallelToolCalls = sdk.Bool(true)
		}
	}
	if len(req.Tools) > 0 || req.ToolChoice.Mode != "" {
		params.ToolChoice = buildToolChoice(req.ToolChoice)
	}
	if req.ResponseFormat != nil {
		params.ResponseFormat, err = buildResponseFormat(*req.ResponseFormat, p.Name())
		if err != nil {
			return sdk.ChatCompletionNewParams{}, err
		}
	}
	if stream && p.compat.StreamIncludeUsage {
		params.StreamOptions.IncludeUsage = sdk.Bool(true)
	}
	extras := jsonObject{}
	if req.Effort != "" {
		for key, value := range p.mapEffort(req.Effort) {
			extras[key] = value
		}
	}
	if err := p.dialect.ApplyRequest(req, &params, extras); err != nil {
		return sdk.ChatCompletionNewParams{}, err
	}
	if !slices.Contains(p.Capabilities(), llm.CapabilityParallelTools) {
		params.ParallelToolCalls = sdkparam.Opt[bool]{}
		delete(extras, "parallel_tool_calls")
	}
	if len(extras) > 0 {
		params.SetExtraFields(extras)
	}
	return params, nil
}

// mapEffort translates the unified Effort into top-level wire request fields,
// preferring the dialect's Compat mapping (FS §9 per-provider effort table).
// The default is the plain chat-completions spelling shared by OpenAI's Chat
// Completions surface and vLLM: {"reasoning_effort": "<level>"}.
func (p *Provider) mapEffort(effort llm.Effort) map[string]any {
	if p.compat.MapEffort != nil {
		return p.compat.MapEffort(effort)
	}
	return map[string]any{"reasoning_effort": string(effort)}
}

func (p *Provider) buildMessages(req *llm.Request) ([]sdk.ChatCompletionMessageParamUnion, error) {
	var messages []sdk.ChatCompletionMessageParamUnion
	if req.System != "" {
		msg := jsonObject{"role": "system", "content": []any{textBlock(req.System, req.SystemCache)}}
		messages = append(messages, sdkparam.Override[sdk.ChatCompletionMessageParamUnion](msg))
	}
	for _, message := range req.Messages {
		if message.Role == llm.RoleTool {
			toolMessages, err := p.buildToolMessages(message)
			if err != nil {
				return nil, err
			}
			messages = append(messages, toolMessages...)
			continue
		}
		msg, ok, err := p.buildMessage(message)
		if err != nil {
			return nil, err
		}
		if ok {
			messages = append(messages, sdkparam.Override[sdk.ChatCompletionMessageParamUnion](msg))
		}
	}
	return messages, nil
}

func (p *Provider) buildMessage(message llm.Message) (jsonObject, bool, error) {
	switch message.Role {
	case llm.RoleUser, llm.RoleSystem:
		var content []any
		for _, part := range message.Parts {
			blocks, err := p.inputBlocks(part)
			if err != nil {
				return nil, false, err
			}
			content = append(content, blocks...)
		}
		return jsonObject{"role": string(message.Role), "content": content}, len(content) > 0, nil
	case llm.RoleAssistant:
		var text strings.Builder
		var reasoningText strings.Builder
		var calls []any
		var reasoningDetails []any
		for _, part := range message.Parts {
			switch value := providerutil.DerefPart(part).(type) {
			case llm.TextPart:
				text.WriteString(value.Text)
			case llm.ToolCallPart:
				call, err := toolCallObject(value, p.Name())
				if err != nil {
					return nil, false, err
				}
				calls = append(calls, call)
			case llm.ReasoningPart:
				if value.Provider != p.Name() {
					continue // foreign reasoning is dropped silently (FS §18)
				}
				switch {
				case len(value.Raw) > 0:
					details, err := reasoningDetailElements(value.Raw)
					if err != nil {
						return nil, false, fmt.Errorf("%w: %s reasoning replay: invalid JSON", llm.ErrBadRequest, p.Name())
					}
					reasoningDetails = append(reasoningDetails, details...)
				case p.compat.ReasoningReplayField != "" && value.Text != "":
					// Plain-text reasoning replays under the Compat-declared
					// field name (e.g. vLLM's "reasoning"); without one it is
					// dropped, matching chat templates that discard prior
					// thinking.
					reasoningText.WriteString(value.Text)
				}
			case llm.UnknownPart:
				continue
			default:
				return nil, false, fmt.Errorf("%w: %s cannot send assistant part %T", llm.ErrUnsupported, p.Name(), part)
			}
		}
		msg := jsonObject{"role": "assistant", "content": text.String()}
		if len(calls) > 0 {
			msg["tool_calls"] = calls
		}
		if len(reasoningDetails) > 0 {
			msg["reasoning_details"] = reasoningDetails
		}
		if reasoningText.Len() > 0 {
			msg[p.compat.ReasoningReplayField] = reasoningText.String()
		}
		return msg, text.Len() > 0 || len(calls) > 0 || len(reasoningDetails) > 0 || reasoningText.Len() > 0, nil
	case llm.RoleTool:
		return nil, false, fmt.Errorf("%w: internal tool role expansion failure", llm.ErrBadRequest)
	default:
		return nil, false, fmt.Errorf("%w: unknown message role %q", llm.ErrBadRequest, message.Role)
	}
}

// reasoningDetailElements normalizes ReasoningPart.Raw for replay. The
// adapter stores the wire `reasoning_details` ARRAY verbatim in Raw, so an
// array is spliced element-by-element into the outgoing reasoning_details —
// appending it whole would nest it ("reasoning_details": [[...]]) and break
// same-provider replay. A bare object (hand-built part) is kept as a single
// detail. Numbers are decoded as json.Number (parity with
// withStreamEnabled) so giant integers replay verbatim instead of being
// perturbed through float64.
func reasoningDetailElements(raw json.RawMessage) ([]any, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var decoded any
	if err := dec.Decode(&decoded); err != nil {
		return nil, err
	}
	if dec.More() {
		return nil, fmt.Errorf("trailing data after reasoning details JSON")
	}
	if elements, ok := decoded.([]any); ok {
		return elements, nil
	}
	return []any{decoded}, nil
}

func (p *Provider) buildToolMessages(message llm.Message) ([]sdk.ChatCompletionMessageParamUnion, error) {
	if len(message.Parts) == 0 {
		return nil, nil
	}
	messages := make([]sdk.ChatCompletionMessageParamUnion, 0, len(message.Parts))
	for _, part := range message.Parts {
		result, ok := providerutil.DerefPart(part).(llm.ToolResultPart)
		if !ok {
			return nil, fmt.Errorf("%w: %s tool messages must contain ToolResultPart", llm.ErrBadRequest, p.Name())
		}
		// The chat-completions wire genuinely accepts only string content for
		// role:"tool" messages, so images/files in tool results stay
		// ErrUnsupported here — unlike the Responses adapter, which maps them
		// to its content-array output form.
		text, err := providerutil.ToolResultText(result, p.Name())
		if err != nil {
			return nil, err
		}
		msg := jsonObject{"role": "tool", "tool_call_id": result.ToolCallID, "content": text}
		messages = append(messages, sdkparam.Override[sdk.ChatCompletionMessageParamUnion](msg))
	}
	return messages, nil
}

func (p *Provider) inputBlocks(part llm.Part) ([]any, error) {
	switch value := providerutil.DerefPart(part).(type) {
	case llm.TextPart:
		return []any{textBlock(value.Text, value.Cache)}, nil
	case llm.ImagePart:
		block, err := imageBlock(value, p.Name())
		return []any{block}, err
	case llm.UnknownPart:
		return nil, nil
	default:
		return nil, fmt.Errorf("%w: %s cannot send input part %T", llm.ErrUnsupported, p.Name(), part)
	}
}

func textBlock(text string, cache *llm.CacheHint) jsonObject {
	block := jsonObject{"type": "text", "text": text}
	if cache != nil {
		block["cache_control"] = cacheControlValue(cache)
	}
	return block
}

// cacheControlValue maps the unified CacheHint onto the Anthropic-style
// cache_control block that OpenRouter forwards upstream (FS §15). The
// passthrough accepts Anthropic's optional TTL — cross-checked against pi's
// openai-completions cacheControl, which sends {"type":"ephemeral",
// "ttl":"1h"} through OpenRouter for long retention — so, mirroring the
// anthropic adapter's mapping, hints above five minutes request the one-hour
// tier and anything else relies on the upstream default (5m); no hint data
// is silently dropped.
func cacheControlValue(cache *llm.CacheHint) map[string]any {
	if cache.TTL > 5*time.Minute {
		return map[string]any{"type": "ephemeral", "ttl": "1h"}
	}
	return map[string]any{"type": "ephemeral"}
}

func imageBlock(part llm.ImagePart, provider string) (jsonObject, error) {
	url := part.URL
	if url == "" && len(part.Data) > 0 {
		if part.MediaType == "" {
			return nil, fmt.Errorf("%w: %s image data requires media type", llm.ErrBadRequest, provider)
		}
		url = "data:" + part.MediaType + ";base64," + base64.StdEncoding.EncodeToString(part.Data)
	}
	if url == "" {
		return nil, fmt.Errorf("%w: %s image requires URL or data", llm.ErrBadRequest, provider)
	}
	block := jsonObject{"type": "image_url", "image_url": map[string]any{"url": url}}
	if part.Cache != nil {
		block["cache_control"] = cacheControlValue(part.Cache)
	}
	return block, nil
}

func toolCallObject(part llm.ToolCallPart, provider string) (jsonObject, error) {
	if part.ID == "" || part.Name == "" {
		return nil, fmt.Errorf("%w: %s tool call requires id and name", llm.ErrBadRequest, provider)
	}
	args := strings.TrimSpace(string(part.Args))
	if args == "" {
		args = "{}"
	}
	if !json.Valid([]byte(args)) {
		return nil, fmt.Errorf("%w: %s tool call args must be valid JSON", llm.ErrBadRequest, provider)
	}
	return jsonObject{"id": part.ID, "type": "function", "function": map[string]any{"name": part.Name, "arguments": args}}, nil
}

func buildTools(tools []llm.Tool, provider string) ([]sdk.ChatCompletionToolUnionParam, error) {
	out := make([]sdk.ChatCompletionToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		schema, err := providerutil.SchemaAsMap(tool.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("%w: %s tool %q schema: %v", llm.ErrBadRequest, provider, tool.Name, err)
		}
		strict := tool.Strict && providerutil.StrictSchemaSupported(schema)
		raw := jsonObject{
			"type": "function",
			"function": map[string]any{
				"name":        tool.Name,
				"description": tool.Description,
				"parameters":  schema,
				"strict":      strict,
			},
		}
		out = append(out, sdkparam.Override[sdk.ChatCompletionToolUnionParam](raw))
	}
	return out, nil
}

func buildToolChoice(choice llm.ToolChoice) sdk.ChatCompletionToolChoiceOptionUnionParam {
	switch choice.Mode {
	case "", llm.ToolChoiceAuto:
		return sdkparam.Override[sdk.ChatCompletionToolChoiceOptionUnionParam]("auto")
	case llm.ToolChoiceNone:
		return sdkparam.Override[sdk.ChatCompletionToolChoiceOptionUnionParam]("none")
	case llm.ToolChoiceRequired:
		return sdkparam.Override[sdk.ChatCompletionToolChoiceOptionUnionParam]("required")
	case llm.ToolChoiceTool:
		return sdkparam.Override[sdk.ChatCompletionToolChoiceOptionUnionParam](jsonObject{"type": "function", "function": map[string]any{"name": choice.Name}})
	default:
		return sdk.ChatCompletionToolChoiceOptionUnionParam{}
	}
}

func buildResponseFormat(format llm.ResponseFormat, provider string) (sdk.ChatCompletionNewParamsResponseFormatUnion, error) {
	switch format.Type {
	case llm.FormatJSONSchema:
		schema, err := providerutil.SchemaAsMap(format.Schema)
		if err != nil {
			return sdk.ChatCompletionNewParamsResponseFormatUnion{}, fmt.Errorf("%w: %s response schema: %v", llm.ErrBadRequest, provider, err)
		}
		name := format.Name
		if name == "" {
			name = "response"
		}
		raw := jsonObject{"type": "json_schema", "json_schema": map[string]any{
			"name":   name,
			"schema": schema,
			"strict": format.Strict && providerutil.StrictSchemaSupported(schema),
		}}
		return sdkparam.Override[sdk.ChatCompletionNewParamsResponseFormatUnion](raw), nil
	case llm.FormatJSONMode:
		return sdkparam.Override[sdk.ChatCompletionNewParamsResponseFormatUnion](jsonObject{"type": "json_object"}), nil
	default:
		return sdk.ChatCompletionNewParamsResponseFormatUnion{}, fmt.Errorf("%w: unknown response format %q", llm.ErrBadRequest, format.Type)
	}
}
