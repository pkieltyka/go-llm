package anthropic

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"time"

	sdk "github.com/anthropics/anthropic-sdk-go"
	sdkoption "github.com/anthropics/anthropic-sdk-go/option"
	sdkparam "github.com/anthropics/anthropic-sdk-go/packages/param"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

func (p *Provider) buildParams(req *llm.Request) (sdk.MessageNewParams, []sdkoption.RequestOption, error) {
	if err := validateRequest(req, false); err != nil {
		return sdk.MessageNewParams{}, nil, err
	}
	return p.buildParamsAfterValidation(req)
}

func (p *Provider) buildStreamParams(req *llm.Request) (sdk.MessageNewParams, []sdkoption.RequestOption, error) {
	if err := validateRequest(req, true); err != nil {
		return sdk.MessageNewParams{}, nil, err
	}
	return p.buildParamsAfterValidation(req)
}

func validateRequest(req *llm.Request, stream bool) error {
	var err error
	if stream {
		err = llm.ValidateStreamRequest(capabilities, req)
	} else {
		err = llm.ValidateRequest(capabilities, req)
	}
	if err != nil {
		return err
	}
	return llm.ValidateProviderOptions(providerName, req)
}

func (p *Provider) buildParamsAfterValidation(req *llm.Request) (sdk.MessageNewParams, []sdkoption.RequestOption, error) {
	options, err := requestOptions(req)
	if err != nil {
		return sdk.MessageNewParams{}, nil, err
	}
	messages, err := buildMessages(req.Messages)
	if err != nil {
		return sdk.MessageNewParams{}, nil, err
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = p.defaultMaxTokens
	}
	params := sdk.MessageNewParams{
		Model:     sdk.Model(req.Model),
		MaxTokens: int64(maxTokens),
		Messages:  messages,
	}

	params.System = p.systemBlocks(req)
	if req.Temperature != nil {
		params.Temperature = sdk.Float(*req.Temperature)
	}
	if req.TopP != nil {
		params.TopP = sdk.Float(*req.TopP)
	}
	params.StopSequences = append([]string(nil), req.StopSequences...)

	if len(req.Tools) > 0 {
		params.Tools, err = buildTools(req.Tools)
		if err != nil {
			return sdk.MessageNewParams{}, nil, err
		}
	}
	if len(req.Tools) > 0 || req.ToolChoice.Mode != "" {
		params.ToolChoice = buildToolChoice(req.ToolChoice, options)
	}

	if req.ResponseFormat != nil {
		schema, err := providerutil.SchemaAsMap(req.ResponseFormat.Schema)
		if err != nil {
			return sdk.MessageNewParams{}, nil, fmt.Errorf("%w: anthropic response schema: %v", llm.ErrBadRequest, err)
		}
		params.OutputConfig.Format = sdk.JSONOutputFormatParam{Schema: schema}
	}
	if req.Effort != "" {
		applyEffort(req.Effort, &params)
	}

	opts := requestSDKOptions(options)
	applyOptions(options, &params)
	return params, opts, nil
}

// systemBlocks builds the request's system blocks. In OAuth mode ONLY
// (FS §17C / ARCH §3.1), the Claude Code identity line is injected as the
// FIRST system block — subscription tokens are rejected without it — and the
// caller's System text becomes the second block, mirroring pi's
// api/anthropic-messages.ts buildParams. Api-key requests carry only the
// caller's System text.
func (p *Provider) systemBlocks(req *llm.Request) []sdk.TextBlockParam {
	blocks := make([]sdk.TextBlockParam, 0, 2)
	if p.oauth {
		block := sdk.TextBlockParam{Text: claudeCodeSystemPrompt}
		if req.SystemCache != nil {
			block.CacheControl = cacheControl(req.SystemCache)
		}
		blocks = append(blocks, block)
	}
	if req.System != "" {
		block := sdk.TextBlockParam{Text: req.System}
		if req.SystemCache != nil {
			block.CacheControl = cacheControl(req.SystemCache)
		}
		blocks = append(blocks, block)
	}
	if len(blocks) == 0 {
		return nil
	}
	return blocks
}

func requestOptions(req *llm.Request) (*Options, error) {
	options, ok, err := providerutil.OptionsOf[Options](req)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, nil
	}
	return &options, nil
}

func requestSDKOptions(options *Options) []sdkoption.RequestOption {
	if options == nil {
		return nil
	}
	out := make([]sdkoption.RequestOption, 0, len(options.BetaHeaders))
	for _, beta := range options.BetaHeaders {
		if beta != "" {
			out = append(out, sdkoption.WithHeaderAdd("anthropic-beta", beta))
		}
	}
	return out
}

func applyOptions(options *Options, params *sdk.MessageNewParams) {
	if options == nil {
		return
	}
	if options.ServiceTier != "" {
		params.ServiceTier = sdk.MessageNewParamsServiceTier(options.ServiceTier)
	}
	if options.Container != "" {
		params.Container = sdk.String(options.Container)
	}
	if options.MetadataUserID != "" {
		params.Metadata.UserID = sdk.String(options.MetadataUserID)
	}
	if options.TopK != nil {
		params.TopK = sdk.Int(*options.TopK)
	}
	if options.DisableParallelToolUse != nil {
		disable := sdk.Bool(*options.DisableParallelToolUse)
		switch {
		case params.ToolChoice.OfAuto != nil:
			params.ToolChoice.OfAuto.DisableParallelToolUse = disable
		case params.ToolChoice.OfAny != nil:
			params.ToolChoice.OfAny.DisableParallelToolUse = disable
		case params.ToolChoice.OfTool != nil:
			params.ToolChoice.OfTool.DisableParallelToolUse = disable
		default:
			params.ToolChoice.OfAuto = &sdk.ToolChoiceAutoParam{DisableParallelToolUse: disable}
		}
	}
}

func buildMessages(messages []llm.Message) ([]sdk.MessageParam, error) {
	out := make([]sdk.MessageParam, 0, len(messages))
	for _, msg := range messages {
		blocks, err := buildContentBlocks(msg.Parts)
		if err != nil {
			return nil, err
		}
		role, err := messageRole(msg.Role)
		if err != nil {
			return nil, err
		}
		out = append(out, sdk.MessageParam{Role: role, Content: blocks})
	}
	return out, nil
}

func messageRole(role llm.Role) (sdk.MessageParamRole, error) {
	switch role {
	case llm.RoleUser, llm.RoleTool:
		return sdk.MessageParamRoleUser, nil
	case llm.RoleAssistant:
		return sdk.MessageParamRoleAssistant, nil
	case llm.RoleSystem:
		return "", fmt.Errorf("%w: anthropic system messages must use Request.System", llm.ErrUnsupported)
	default:
		return "", fmt.Errorf("%w: unknown message role %q", llm.ErrBadRequest, role)
	}
}

func buildContentBlocks(parts []llm.Part) ([]sdk.ContentBlockParamUnion, error) {
	blocks := make([]sdk.ContentBlockParamUnion, 0, len(parts))
	for _, part := range parts {
		switch p := providerutil.DerefPart(part).(type) {
		case llm.TextPart:
			blocks = append(blocks, textBlock(p.Text, p.Cache))
		case llm.ImagePart:
			block, err := imageBlock(p)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, block)
		case llm.FilePart:
			block, err := fileBlock(p)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, block)
		case llm.ToolCallPart:
			block, err := toolUseBlock(p)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, block)
		case llm.ToolResultPart:
			block, err := toolResultBlock(p)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, block)
		case llm.ReasoningPart:
			block, ok, err := reasoningReplayBlock(p)
			if err != nil {
				return nil, err
			}
			if ok {
				blocks = append(blocks, block)
			}
		case llm.UnknownPart:
			continue
		default:
			return nil, fmt.Errorf("%w: anthropic cannot send part %T", llm.ErrUnsupported, part)
		}
	}
	return blocks, nil
}

func textBlock(text string, cache *llm.CacheHint) sdk.ContentBlockParamUnion {
	block := sdk.TextBlockParam{Text: text}
	if cache != nil {
		block.CacheControl = cacheControl(cache)
	}
	return sdk.ContentBlockParamUnion{OfText: &block}
}

func imageBlock(part llm.ImagePart) (sdk.ContentBlockParamUnion, error) {
	var block sdk.ImageBlockParam
	if part.Cache != nil {
		block.CacheControl = cacheControl(part.Cache)
	}
	switch {
	case part.URL != "":
		block.Source.OfURL = &sdk.URLImageSourceParam{URL: part.URL}
	case len(part.Data) > 0:
		block.Source.OfBase64 = &sdk.Base64ImageSourceParam{
			Data:      base64.StdEncoding.EncodeToString(part.Data),
			MediaType: sdk.Base64ImageSourceMediaType(part.MediaType),
		}
	default:
		return sdk.ContentBlockParamUnion{}, fmt.Errorf("%w: anthropic image requires URL or data", llm.ErrBadRequest)
	}
	return sdk.ContentBlockParamUnion{OfImage: &block}, nil
}

func fileBlock(part llm.FilePart) (sdk.ContentBlockParamUnion, error) {
	if part.MediaType != "" && part.MediaType != "application/pdf" {
		return sdk.ContentBlockParamUnion{}, fmt.Errorf("%w: anthropic file input supports application/pdf", llm.ErrUnsupported)
	}
	var block sdk.DocumentBlockParam
	if part.Name != "" {
		block.Title = sdk.String(part.Name)
	}
	if part.Cache != nil {
		block.CacheControl = cacheControl(part.Cache)
	}
	switch {
	case part.URL != "":
		block.Source.OfURL = &sdk.URLPDFSourceParam{URL: part.URL}
	case len(part.Data) > 0:
		block.Source.OfBase64 = &sdk.Base64PDFSourceParam{Data: base64.StdEncoding.EncodeToString(part.Data)}
	default:
		return sdk.ContentBlockParamUnion{}, fmt.Errorf("%w: anthropic file requires URL or data", llm.ErrBadRequest)
	}
	return sdk.ContentBlockParamUnion{OfDocument: &block}, nil
}

func toolUseBlock(part llm.ToolCallPart) (sdk.ContentBlockParamUnion, error) {
	input, err := rawJSONAsAny(part.Args)
	if err != nil {
		return sdk.ContentBlockParamUnion{}, fmt.Errorf("%w: anthropic tool call args: %v", llm.ErrBadRequest, err)
	}
	return sdk.NewToolUseBlock(part.ID, input, part.Name), nil
}

func toolResultBlock(part llm.ToolResultPart) (sdk.ContentBlockParamUnion, error) {
	content := make([]sdk.ToolResultBlockParamContentUnion, 0, len(part.Content))
	for _, nested := range part.Content {
		switch p := providerutil.DerefPart(nested).(type) {
		case llm.TextPart:
			content = append(content, sdk.ToolResultBlockParamContentUnion{OfText: &sdk.TextBlockParam{Text: p.Text}})
		case llm.ImagePart:
			block, err := imageBlock(p)
			if err != nil {
				return sdk.ContentBlockParamUnion{}, err
			}
			content = append(content, sdk.ToolResultBlockParamContentUnion{OfImage: block.OfImage})
		case llm.FilePart:
			block, err := fileBlock(p)
			if err != nil {
				return sdk.ContentBlockParamUnion{}, err
			}
			content = append(content, sdk.ToolResultBlockParamContentUnion{OfDocument: block.OfDocument})
		case llm.UnknownPart:
			continue
		default:
			return sdk.ContentBlockParamUnion{}, fmt.Errorf("%w: anthropic cannot send tool result part %T", llm.ErrUnsupported, nested)
		}
	}
	block := sdk.ToolResultBlockParam{
		ToolUseID: part.ToolCallID,
		Content:   content,
		IsError:   sdk.Bool(part.IsError),
	}
	return sdk.ContentBlockParamUnion{OfToolResult: &block}, nil
}

func reasoningReplayBlock(part llm.ReasoningPart) (sdk.ContentBlockParamUnion, bool, error) {
	if part.Provider != providerName || len(part.Raw) == 0 {
		return sdk.ContentBlockParamUnion{}, false, nil
	}
	if !json.Valid(part.Raw) {
		return sdk.ContentBlockParamUnion{}, false, fmt.Errorf("%w: anthropic reasoning replay: invalid JSON", llm.ErrBadRequest)
	}
	return sdkparam.Override[sdk.ContentBlockParamUnion](json.RawMessage(part.Raw)), true, nil
}

func buildTools(tools []llm.Tool) ([]sdk.ToolUnionParam, error) {
	out := make([]sdk.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		schema, err := toolSchema(tool.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("%w: anthropic tool %q schema: %v", llm.ErrBadRequest, tool.Name, err)
		}
		param := sdk.ToolParam{
			Name:        tool.Name,
			InputSchema: schema,
			Type:        sdk.ToolTypeCustom,
			Strict:      sdk.Bool(tool.Strict),
		}
		if tool.Description != "" {
			param.Description = sdk.String(tool.Description)
		}
		out = append(out, sdk.ToolUnionParam{OfTool: &param})
	}
	return out, nil
}

func toolSchema(value any) (sdk.ToolInputSchemaParam, error) {
	schema, err := providerutil.SchemaAsMap(value)
	if err != nil {
		return sdk.ToolInputSchemaParam{}, err
	}
	out := sdk.ToolInputSchemaParam{ExtraFields: map[string]any{}}
	for key, value := range schema {
		switch key {
		case "properties":
			out.Properties = value
		case "required":
			raw, _ := json.Marshal(value)
			if err := json.Unmarshal(raw, &out.Required); err != nil {
				return sdk.ToolInputSchemaParam{}, err
			}
		case "type":
			if typ, ok := value.(string); ok && typ != "object" {
				out.ExtraFields[key] = value
			}
		default:
			out.ExtraFields[key] = value
		}
	}
	if len(out.ExtraFields) == 0 {
		out.ExtraFields = nil
	}
	return out, nil
}

func buildToolChoice(choice llm.ToolChoice, options *Options) sdk.ToolChoiceUnionParam {
	var out sdk.ToolChoiceUnionParam
	switch choice.Mode {
	case "", llm.ToolChoiceAuto:
		out.OfAuto = &sdk.ToolChoiceAutoParam{}
	case llm.ToolChoiceNone:
		none := sdk.NewToolChoiceNoneParam()
		out.OfNone = &none
	case llm.ToolChoiceRequired:
		out.OfAny = &sdk.ToolChoiceAnyParam{}
	case llm.ToolChoiceTool:
		out = sdk.ToolChoiceParamOfTool(choice.Name)
	}
	if options != nil && options.DisableParallelToolUse != nil {
		disable := sdk.Bool(*options.DisableParallelToolUse)
		if out.OfAuto != nil {
			out.OfAuto.DisableParallelToolUse = disable
		}
		if out.OfAny != nil {
			out.OfAny.DisableParallelToolUse = disable
		}
		if out.OfTool != nil {
			out.OfTool.DisableParallelToolUse = disable
		}
	}
	return out
}

func applyEffort(effort llm.Effort, params *sdk.MessageNewParams) {
	switch effort {
	case llm.EffortNone:
		disabled := sdk.NewThinkingConfigDisabledParam()
		params.Thinking.OfDisabled = &disabled
		return
	case llm.EffortMinimal, llm.EffortLow:
		params.OutputConfig.Effort = sdk.OutputConfigEffortLow
	case llm.EffortMedium:
		params.OutputConfig.Effort = sdk.OutputConfigEffortMedium
	case llm.EffortHigh:
		params.OutputConfig.Effort = sdk.OutputConfigEffortHigh
	case llm.EffortXHigh:
		params.OutputConfig.Effort = sdk.OutputConfigEffortXhigh
	case llm.EffortMax:
		params.OutputConfig.Effort = sdk.OutputConfigEffortMax
	}
	params.Thinking.OfAdaptive = &sdk.ThinkingConfigAdaptiveParam{
		Display: sdk.ThinkingConfigAdaptiveDisplaySummarized,
	}
}

func cacheControl(h *llm.CacheHint) sdk.CacheControlEphemeralParam {
	out := sdk.NewCacheControlEphemeralParam()
	if h != nil && h.TTL > 5*time.Minute {
		out.TTL = sdk.CacheControlEphemeralTTLTTL1h
	} else {
		out.TTL = sdk.CacheControlEphemeralTTLTTL5m
	}
	return out
}

func rawJSONAsAny(raw json.RawMessage) (any, error) {
	if len(raw) == 0 {
		return map[string]any{}, nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var out any
	if err := dec.Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}
