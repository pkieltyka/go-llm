package anthropic

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
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
	req = llm.FoldPositional(req, providerutil.PositionalSupport(providerName, req.Model))
	req = providerutil.ApplyCacheRetention(req)
	options, err := requestOptions(req)
	if err != nil {
		return sdk.MessageNewParams{}, nil, err
	}
	messages, err := buildMessages(providerutil.ReplayableHistory(req.Messages, req.Model))
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

	params.System = systemBlocks(req)
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
	if llm.HasToolsDelta(req.Messages) && !hasBeta(options, inlineToolsBeta) {
		opts = append(opts, sdkoption.WithHeaderAdd("anthropic-beta", inlineToolsBeta))
	}
	applyOptions(options, &params)
	return params, opts, nil
}

// inlineToolsBeta enables tool_addition and tool_removal blocks in system
// messages. The header may first appear mid-conversation: the cached prefix
// survives it (probed 2026-10-22, plan 11 Part A).
const inlineToolsBeta = "inline-tools-2026-09-15"

func hasBeta(options *Options, beta string) bool {
	return options != nil && slices.Contains(options.BetaHeaders, beta)
}

func systemBlocks(req *llm.Request) []sdk.TextBlockParam {
	blocks := make([]sdk.TextBlockParam, 0, 1)
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
		params.Container = sdk.MessageCreateParamsContainerUnion{
			OfString: sdk.String(options.Container),
		}
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

// buildMessages converts history. FoldPositional has already folded every
// RoleSystem message this model cannot take in place, so the rest are sent
// as role "system". Anthropic accepts one only after a user message (or an
// assistant turn ending in a server tool result), and a tool_use must be
// followed directly by its tool_result, so system messages are held back
// until just before the next assistant turn that is sent, as pi does, and
// adjacent ones are merged. Blank assistant text is dropped, since Anthropic
// rejects empty text blocks.
func buildMessages(messages []llm.Message) ([]sdk.MessageParam, error) {
	out := make([]sdk.MessageParam, 0, len(messages))
	var pending []sdk.ContentBlockParamUnion
	flush := func() error {
		if len(pending) == 0 {
			return nil
		}
		if len(out) > 0 && out[len(out)-1].Role == sdk.MessageParamRoleAssistant {
			return fmt.Errorf("%w: anthropic accepts a system message only after a user message, not directly after an assistant turn", llm.ErrBadRequest)
		}
		out = append(out, sdk.MessageParam{Role: sdk.MessageParamRoleSystem, Content: pending})
		pending = nil
		return nil
	}
	for _, msg := range messages {
		if msg.Role == llm.RoleSystem {
			blocks, err := systemMessageBlocks(msg.Parts)
			if err != nil {
				return nil, err
			}
			pending = append(pending, blocks...)
			continue
		}
		parts, blank := msg.Parts, false
		if msg.Role == llm.RoleAssistant {
			parts, blank = withoutBlankText(parts)
		}
		blocks, filtered, err := buildContentBlocksDetailed(parts)
		if err != nil {
			return nil, err
		}
		filtered = filtered || blank
		role, err := messageRole(msg.Role)
		if err != nil {
			return nil, err
		}
		if msg.Role == llm.RoleAssistant {
			// A dropped assistant turn is not sent, so held system
			// messages wait for the next one that is (pi #11091).
			if len(blocks) == 0 && filtered {
				continue
			}
			if err := flush(); err != nil {
				return nil, err
			}
		}
		out = append(out, sdk.MessageParam{Role: role, Content: blocks})
	}
	if err := flush(); err != nil {
		return nil, err
	}
	return out, nil
}

// withoutBlankText drops whitespace-only text parts (another provider's
// empty reply, say) and reports whether it dropped any.
func withoutBlankText(parts []llm.Part) ([]llm.Part, bool) {
	var out []llm.Part
	for i, part := range parts {
		if text, ok := providerutil.DerefPart(part).(llm.TextPart); ok && strings.TrimSpace(text.Text) == "" {
			if out == nil {
				out = append(make([]llm.Part, 0, len(parts)), parts[:i]...)
			}
			continue
		}
		if out != nil {
			out = append(out, part)
		}
	}
	if out == nil {
		return parts, false
	}
	return out, true
}

// systemMessageBlocks converts a system message sent in place: text as is,
// section updates framed by name, and tool changes as inline-tools blocks
// (removals first, skipping a name the same delta redefines).
func systemMessageBlocks(parts []llm.Part) ([]sdk.ContentBlockParamUnion, error) {
	var blocks []sdk.ContentBlockParamUnion
	for _, part := range parts {
		switch p := providerutil.DerefPart(part).(type) {
		case llm.TextPart:
			if p.Text != "" {
				blocks = append(blocks, textBlock(p.Text, p.Cache))
			}
		case llm.SystemDeltaPart:
			blocks = append(blocks, textBlock(providerutil.RenderSystemDelta(p), p.Cache))
		case llm.ToolsDeltaPart:
			changes, err := toolChangeBlocks(p)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, changes...)
		case llm.UnknownPart:
		default:
			return nil, fmt.Errorf("%w: anthropic system messages cannot carry %T", llm.ErrBadRequest, part)
		}
	}
	return blocks, nil
}

func toolChangeBlocks(delta llm.ToolsDeltaPart) ([]sdk.ContentBlockParamUnion, error) {
	redefined := make(map[string]bool, len(delta.Added))
	for _, tool := range delta.Added {
		redefined[tool.Name] = true
	}
	var changes []map[string]any
	for _, name := range delta.Removed {
		if !redefined[name] {
			changes = append(changes, map[string]any{"type": "tool_removal", "tool": map[string]any{"type": "tool_reference", "name": name}})
		}
	}
	definitions, err := buildTools(delta.Added)
	if err != nil {
		return nil, err
	}
	for _, definition := range definitions {
		changes = append(changes, map[string]any{"type": "tool_addition", "tool": map[string]any{"type": "tool_definition", "definition": definition}})
	}
	if delta.Cache != nil && len(changes) > 0 {
		changes[len(changes)-1]["cache_control"] = cacheControl(delta.Cache)
	}
	blocks := make([]sdk.ContentBlockParamUnion, len(changes))
	for i, change := range changes {
		raw, err := json.Marshal(change)
		if err != nil {
			return nil, fmt.Errorf("%w: anthropic tool change: %v", llm.ErrBadRequest, err)
		}
		blocks[i] = sdkparam.Override[sdk.ContentBlockParamUnion](json.RawMessage(raw))
	}
	return blocks, nil
}

func messageRole(role llm.Role) (sdk.MessageParamRole, error) {
	switch role {
	case llm.RoleUser, llm.RoleTool:
		return sdk.MessageParamRoleUser, nil
	case llm.RoleAssistant:
		return sdk.MessageParamRoleAssistant, nil
	default:
		return "", fmt.Errorf("%w: unknown message role %q", llm.ErrBadRequest, role)
	}
}

func buildContentBlocks(parts []llm.Part) ([]sdk.ContentBlockParamUnion, error) {
	blocks, _, err := buildContentBlocksDetailed(parts)
	return blocks, err
}

func buildContentBlocksDetailed(parts []llm.Part) ([]sdk.ContentBlockParamUnion, bool, error) {
	blocks := make([]sdk.ContentBlockParamUnion, 0, len(parts))
	filtered := false
	for _, part := range parts {
		switch p := providerutil.DerefPart(part).(type) {
		case llm.TextPart:
			blocks = append(blocks, textBlock(p.Text, p.Cache))
		case llm.ImagePart:
			block, err := imageBlock(p)
			if err != nil {
				return nil, false, err
			}
			blocks = append(blocks, block)
		case llm.FilePart:
			block, err := fileBlock(p)
			if err != nil {
				return nil, false, err
			}
			blocks = append(blocks, block)
		case llm.ToolCallPart:
			block, err := toolUseBlock(p)
			if err != nil {
				return nil, false, err
			}
			blocks = append(blocks, block)
		case llm.ToolResultPart:
			block, err := toolResultBlock(p)
			if err != nil {
				return nil, false, err
			}
			blocks = append(blocks, block)
		case llm.ReasoningPart:
			block, ok, err := reasoningReplayBlock(p)
			if err != nil {
				return nil, false, err
			}
			if ok {
				blocks = append(blocks, block)
			} else {
				filtered = true
			}
		case llm.UnknownPart:
			filtered = true
			continue
		default:
			return nil, false, fmt.Errorf("%w: anthropic cannot send part %T", llm.ErrUnsupported, part)
		}
	}
	return blocks, filtered, nil
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
			block := textBlock(p.Text, p.Cache)
			content = append(content, sdk.ToolResultBlockParamContentUnion{OfText: block.OfText})
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
	if out.Properties == nil {
		// The SDK omits a zero input schema entirely, and the Messages API
		// requires input_schema, so a no-argument tool ({"type":"object"})
		// sends an explicit empty object schema.
		out.Properties = map[string]any{}
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
