package responsesapi

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"mime"
	"strings"

	sdk "github.com/openai/openai-go/v3"
	sdkparam "github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// ApplyOptionsFunc applies provider-specific Responses request extensions.
// A non-nil error aborts the request build (e.g. options claiming the
// provider's name with an unexpected concrete type → ErrBadRequest).
type ApplyOptionsFunc func(req *llm.Request, params *responses.ResponseNewParams) error

// Adapter contains the provider-specific values needed by the shared Responses
// mapping.
type Adapter struct {
	ProviderName string
	Capabilities []llm.Capability
	PriceTable   llm.PriceTable
	ApplyOptions ApplyOptionsFunc
}

// BuildParams converts a go-llm request into OpenAI Responses parameters.
func (a Adapter) BuildParams(req *llm.Request, stream bool) (responses.ResponseNewParams, error) {
	if err := a.validateRequest(req, stream); err != nil {
		return responses.ResponseNewParams{}, err
	}
	return a.buildParamsAfterValidation(req)
}

// BuildInput converts messages into Responses input items.
func (a Adapter) BuildInput(messages []llm.Message) (responses.ResponseInputParam, error) {
	return a.buildInput(messages)
}

func (a Adapter) validateRequest(req *llm.Request, stream bool) error {
	var err error
	if stream {
		err = llm.ValidateStreamRequest(a.Capabilities, req)
	} else {
		err = llm.ValidateRequest(a.Capabilities, req)
	}
	if err != nil {
		return err
	}
	return llm.ValidateProviderOptions(a.ProviderName, req)
}

func (a Adapter) buildParamsAfterValidation(req *llm.Request) (responses.ResponseNewParams, error) {
	input, err := a.buildInput(req.Messages)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}

	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(req.Model),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
		Store: sdk.Bool(false),
		Include: []responses.ResponseIncludable{
			responses.ResponseIncludableReasoningEncryptedContent,
		},
	}
	if req.System != "" {
		params.Instructions = sdk.String(req.System)
	}
	if req.MaxTokens > 0 {
		params.MaxOutputTokens = sdk.Int(int64(req.MaxTokens))
	}
	if req.Temperature != nil {
		params.Temperature = sdk.Float(*req.Temperature)
	}
	if req.TopP != nil {
		params.TopP = sdk.Float(*req.TopP)
	}
	if req.SessionID != "" {
		params.PromptCacheKey = sdk.String(req.SessionID)
	}
	if len(req.Tools) > 0 {
		params.Tools, err = buildTools(req.Tools, a.ProviderName)
		if err != nil {
			return responses.ResponseNewParams{}, err
		}
	}
	if len(req.Tools) > 0 || req.ToolChoice.Mode != "" {
		params.ToolChoice = buildToolChoice(req.ToolChoice)
	}
	if req.ResponseFormat != nil {
		format, err := buildResponseFormat(*req.ResponseFormat, a.ProviderName)
		if err != nil {
			return responses.ResponseNewParams{}, err
		}
		params.Text.Format = format
	}
	if req.Effort != "" {
		params.Reasoning = buildReasoning(req.Effort)
	}

	if a.ApplyOptions != nil {
		if err := a.ApplyOptions(req, &params); err != nil {
			return responses.ResponseNewParams{}, err
		}
	}
	params.Include = normalizeIncludes(params.Include, responsesRequestIsStateless(params))
	return params, nil
}

func responsesRequestIsStateless(params responses.ResponseNewParams) bool {
	if params.Store.Valid() && params.Store.Value {
		return false
	}
	if params.PreviousResponseID.Valid() && params.PreviousResponseID.Value != "" {
		return false
	}
	return sdkparam.IsOmitted(params.Conversation)
}

func normalizeIncludes(includes []responses.ResponseIncludable, stateless bool) []responses.ResponseIncludable {
	out := includes[:0]
	seen := map[responses.ResponseIncludable]struct{}{}
	for _, include := range includes {
		if include == "" {
			continue
		}
		if include == responses.ResponseIncludableReasoningEncryptedContent && !stateless {
			continue
		}
		if _, ok := seen[include]; ok {
			continue
		}
		seen[include] = struct{}{}
		out = append(out, include)
	}
	return out
}

func (a Adapter) buildInput(messages []llm.Message) (responses.ResponseInputParam, error) {
	items := make(responses.ResponseInputParam, 0, len(messages))
	for _, msg := range messages {
		role, err := easyRole(msg.Role)
		if err != nil {
			return nil, err
		}
		var inputContent responses.ResponseInputMessageContentListParam
		var outputContent []responses.ResponseOutputMessageContentUnionParam
		flushContent := func() {
			if len(inputContent) > 0 {
				items = append(items, responses.ResponseInputItemParamOfMessage(inputContent, role))
				inputContent = nil
			}
			if len(outputContent) > 0 {
				items = append(items, responses.ResponseInputItemParamOfOutputMessage(outputContent, "", responses.ResponseOutputMessageStatusCompleted))
				outputContent = nil
			}
		}

		for _, part := range msg.Parts {
			switch p := providerutil.DerefPart(part).(type) {
			case llm.TextPart:
				if msg.Role == llm.RoleTool {
					return nil, fmt.Errorf("%w: %s tool messages must contain ToolResultPart", llm.ErrBadRequest, a.ProviderName)
				}
				if msg.Role == llm.RoleAssistant {
					outputContent = append(outputContent, outputTextContent(p.Text))
				} else {
					inputContent = append(inputContent, responses.ResponseInputContentParamOfInputText(p.Text))
				}
			case llm.ImagePart:
				if msg.Role == llm.RoleAssistant {
					return nil, fmt.Errorf("%w: %s assistant messages cannot replay image input parts", llm.ErrUnsupported, a.ProviderName)
				}
				block, err := imageInput(p, a.ProviderName)
				if err != nil {
					return nil, err
				}
				inputContent = append(inputContent, block)
			case llm.FilePart:
				if msg.Role == llm.RoleAssistant {
					return nil, fmt.Errorf("%w: %s assistant messages cannot replay file input parts", llm.ErrUnsupported, a.ProviderName)
				}
				block, err := fileInput(p, a.ProviderName)
				if err != nil {
					return nil, err
				}
				inputContent = append(inputContent, block)
			case llm.ToolCallPart:
				flushContent()
				item, err := functionCallInput(p, a.ProviderName)
				if err != nil {
					return nil, err
				}
				items = append(items, item)
			case llm.ToolResultPart:
				flushContent()
				item, err := functionCallOutputInput(p, a.ProviderName)
				if err != nil {
					return nil, err
				}
				items = append(items, item)
			case llm.ReasoningPart:
				flushContent()
				item, ok, err := a.reasoningInput(p)
				if err != nil {
					return nil, err
				}
				if ok {
					items = append(items, item)
				}
			case llm.UnknownPart:
				continue
			default:
				return nil, fmt.Errorf("%w: %s cannot send part %T", llm.ErrUnsupported, a.ProviderName, part)
			}
		}
		flushContent()
	}
	return items, nil
}

func outputTextContent(text string) responses.ResponseOutputMessageContentUnionParam {
	output := responses.ResponseOutputTextParam{Text: text}
	return responses.ResponseOutputMessageContentUnionParam{OfOutputText: &output}
}

func easyRole(role llm.Role) (responses.EasyInputMessageRole, error) {
	switch role {
	case llm.RoleUser:
		return responses.EasyInputMessageRoleUser, nil
	case llm.RoleAssistant:
		return responses.EasyInputMessageRoleAssistant, nil
	case llm.RoleSystem:
		return responses.EasyInputMessageRoleSystem, nil
	case llm.RoleTool:
		return responses.EasyInputMessageRoleUser, nil
	default:
		return "", fmt.Errorf("%w: unknown message role %q", llm.ErrBadRequest, role)
	}
}

func imageInput(part llm.ImagePart, providerName string) (responses.ResponseInputContentUnionParam, error) {
	var image responses.ResponseInputImageParam
	image.Detail = responses.ResponseInputImageDetailAuto
	switch {
	case part.URL != "":
		image.ImageURL = sdk.String(part.URL)
	case len(part.Data) > 0:
		if part.MediaType == "" {
			return responses.ResponseInputContentUnionParam{}, fmt.Errorf("%w: %s image data requires media type", llm.ErrBadRequest, providerName)
		}
		image.ImageURL = sdk.String("data:" + part.MediaType + ";base64," + base64.StdEncoding.EncodeToString(part.Data))
	default:
		return responses.ResponseInputContentUnionParam{}, fmt.Errorf("%w: %s image requires URL or data", llm.ErrBadRequest, providerName)
	}
	return responses.ResponseInputContentUnionParam{OfInputImage: &image}, nil
}

func fileInput(part llm.FilePart, providerName string) (responses.ResponseInputContentUnionParam, error) {
	if part.MediaType != "" && part.MediaType != "application/pdf" {
		return responses.ResponseInputContentUnionParam{}, fmt.Errorf("%w: %s file input supports application/pdf", llm.ErrUnsupported, providerName)
	}
	var file responses.ResponseInputFileParam
	if part.Name != "" {
		file.Filename = sdk.String(part.Name)
	}
	switch {
	case part.URL != "":
		file.FileURL = sdk.String(part.URL)
	case len(part.Data) > 0:
		file.FileData = sdk.String(base64.StdEncoding.EncodeToString(part.Data))
		if file.Filename.Value == "" {
			file.Filename = sdk.String(defaultPDFName(part.MediaType))
		}
	default:
		return responses.ResponseInputContentUnionParam{}, fmt.Errorf("%w: %s file requires URL or data", llm.ErrBadRequest, providerName)
	}
	return responses.ResponseInputContentUnionParam{OfInputFile: &file}, nil
}

func defaultPDFName(mediaType string) string {
	if ext, _ := mime.ExtensionsByType(mediaType); len(ext) > 0 {
		return "file" + ext[0]
	}
	return "file.pdf"
}

func functionCallInput(part llm.ToolCallPart, providerName string) (responses.ResponseInputItemUnionParam, error) {
	if part.ID == "" || part.Name == "" {
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("%w: %s tool call requires id and name", llm.ErrBadRequest, providerName)
	}
	args := string(part.Args)
	if strings.TrimSpace(args) == "" {
		args = "{}"
	}
	if !json.Valid([]byte(args)) {
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("%w: %s tool call args must be valid JSON", llm.ErrBadRequest, providerName)
	}
	return responses.ResponseInputItemParamOfFunctionCall(args, part.ID, part.Name), nil
}

// functionCallOutputInput maps a tool result onto function_call_output.
// Text-only results use the plain string output form; results carrying
// images or files use the Responses content-array form (input_text /
// input_image / input_file), which the API accepts for tool outputs — so
// screenshot-returning agent tools work here, not only on Anthropic.
func functionCallOutputInput(part llm.ToolResultPart, providerName string) (responses.ResponseInputItemUnionParam, error) {
	if part.ToolCallID == "" {
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("%w: %s tool result requires tool call id", llm.ErrBadRequest, providerName)
	}
	text, content, err := toolResultOutput(part, providerName)
	if err != nil {
		return responses.ResponseInputItemUnionParam{}, err
	}
	if content != nil {
		return responses.ResponseInputItemParamOfFunctionCallOutput(part.ToolCallID, content), nil
	}
	return responses.ResponseInputItemParamOfFunctionCallOutput(part.ToolCallID, text), nil
}

// toolResultOutput flattens tool-result content. It returns either a plain
// string (content == nil, text-only results — the wire shape the backends
// have always accepted) or a content-item list when images/files are present.
func toolResultOutput(part llm.ToolResultPart, providerName string) (string, responses.ResponseFunctionCallOutputItemListParam, error) {
	multimodal := false
	for _, nested := range part.Content {
		switch providerutil.DerefPart(nested).(type) {
		case llm.ImagePart, llm.FilePart:
			multimodal = true
		}
	}
	if !multimodal {
		text, err := providerutil.ToolResultText(part, providerName)
		if err != nil {
			return "", nil, err
		}
		return text, nil, nil
	}

	content := make(responses.ResponseFunctionCallOutputItemListParam, 0, len(part.Content))
	for _, nested := range part.Content {
		switch p := providerutil.DerefPart(nested).(type) {
		case llm.TextPart:
			content = append(content, responses.ResponseFunctionCallOutputItemUnionParam{
				OfInputText: &responses.ResponseInputTextContentParam{Text: p.Text},
			})
		case llm.ImagePart:
			image, err := imageOutputContent(p, providerName)
			if err != nil {
				return "", nil, err
			}
			content = append(content, responses.ResponseFunctionCallOutputItemUnionParam{OfInputImage: &image})
		case llm.FilePart:
			file, err := fileOutputContent(p, providerName)
			if err != nil {
				return "", nil, err
			}
			content = append(content, responses.ResponseFunctionCallOutputItemUnionParam{OfInputFile: &file})
		case llm.UnknownPart:
			continue
		default:
			return "", nil, fmt.Errorf("%w: %s tool result cannot send part %T", llm.ErrUnsupported, providerName, nested)
		}
	}
	return "", content, nil
}

func imageOutputContent(part llm.ImagePart, providerName string) (responses.ResponseInputImageContentParam, error) {
	image := responses.ResponseInputImageContentParam{Detail: responses.ResponseInputImageContentDetailAuto}
	switch {
	case part.URL != "":
		image.ImageURL = sdk.String(part.URL)
	case len(part.Data) > 0:
		if part.MediaType == "" {
			return responses.ResponseInputImageContentParam{}, fmt.Errorf("%w: %s image data requires media type", llm.ErrBadRequest, providerName)
		}
		image.ImageURL = sdk.String("data:" + part.MediaType + ";base64," + base64.StdEncoding.EncodeToString(part.Data))
	default:
		return responses.ResponseInputImageContentParam{}, fmt.Errorf("%w: %s image requires URL or data", llm.ErrBadRequest, providerName)
	}
	return image, nil
}

func fileOutputContent(part llm.FilePart, providerName string) (responses.ResponseInputFileContentParam, error) {
	if part.MediaType != "" && part.MediaType != "application/pdf" {
		return responses.ResponseInputFileContentParam{}, fmt.Errorf("%w: %s file input supports application/pdf", llm.ErrUnsupported, providerName)
	}
	var file responses.ResponseInputFileContentParam
	if part.Name != "" {
		file.Filename = sdk.String(part.Name)
	}
	switch {
	case part.URL != "":
		file.FileURL = sdk.String(part.URL)
	case len(part.Data) > 0:
		file.FileData = sdk.String(base64.StdEncoding.EncodeToString(part.Data))
		if file.Filename.Value == "" {
			file.Filename = sdk.String(defaultPDFName(part.MediaType))
		}
	default:
		return responses.ResponseInputFileContentParam{}, fmt.Errorf("%w: %s file requires URL or data", llm.ErrBadRequest, providerName)
	}
	return file, nil
}

func (a Adapter) reasoningInput(part llm.ReasoningPart) (responses.ResponseInputItemUnionParam, bool, error) {
	if part.Provider != a.ProviderName || len(part.Raw) == 0 {
		return responses.ResponseInputItemUnionParam{}, false, nil
	}
	if !json.Valid(part.Raw) {
		return responses.ResponseInputItemUnionParam{}, false, fmt.Errorf("%w: %s reasoning replay: invalid JSON", llm.ErrBadRequest, a.ProviderName)
	}
	return sdkparam.Override[responses.ResponseInputItemUnionParam](json.RawMessage(part.Raw)), true, nil
}

func buildTools(tools []llm.Tool, providerName string) ([]responses.ToolUnionParam, error) {
	out := make([]responses.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		schema, err := providerutil.SchemaAsMap(tool.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("%w: %s tool %q schema: %v", llm.ErrBadRequest, providerName, tool.Name, err)
		}
		strict := tool.Strict && providerutil.StrictSchemaSupported(schema)
		param := responses.FunctionToolParam{
			Name:       tool.Name,
			Parameters: schema,
			Strict:     sdk.Bool(strict),
		}
		if tool.Description != "" {
			param.Description = sdk.String(tool.Description)
		}
		out = append(out, responses.ToolUnionParam{OfFunction: &param})
	}
	return out, nil
}

func buildToolChoice(choice llm.ToolChoice) responses.ResponseNewParamsToolChoiceUnion {
	var out responses.ResponseNewParamsToolChoiceUnion
	switch choice.Mode {
	case "", llm.ToolChoiceAuto:
		out.OfToolChoiceMode = sdkparam.NewOpt(responses.ToolChoiceOptionsAuto)
	case llm.ToolChoiceNone:
		out.OfToolChoiceMode = sdkparam.NewOpt(responses.ToolChoiceOptionsNone)
	case llm.ToolChoiceRequired:
		out.OfToolChoiceMode = sdkparam.NewOpt(responses.ToolChoiceOptionsRequired)
	case llm.ToolChoiceTool:
		out.OfFunctionTool = &responses.ToolChoiceFunctionParam{Name: choice.Name}
	}
	return out
}

func buildResponseFormat(format llm.ResponseFormat, providerName string) (responses.ResponseFormatTextConfigUnionParam, error) {
	schema, err := providerutil.SchemaAsMap(format.Schema)
	if err != nil {
		return responses.ResponseFormatTextConfigUnionParam{}, fmt.Errorf("%w: %s response schema: %v", llm.ErrBadRequest, providerName, err)
	}
	name := format.Name
	if name == "" {
		name = "response"
	}
	strict := format.Strict && providerutil.StrictSchemaSupported(schema)
	return responses.ResponseFormatTextConfigUnionParam{
		OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
			Name:   name,
			Schema: schema,
			Strict: sdk.Bool(strict),
		},
	}, nil
}

func buildReasoning(effort llm.Effort) shared.ReasoningParam {
	out := shared.ReasoningParam{Summary: shared.ReasoningSummaryAuto}
	switch effort {
	case llm.EffortNone:
		out.Effort = shared.ReasoningEffortNone
	case llm.EffortMinimal:
		out.Effort = shared.ReasoningEffortMinimal
	case llm.EffortLow:
		out.Effort = shared.ReasoningEffortLow
	case llm.EffortMedium:
		out.Effort = shared.ReasoningEffortMedium
	case llm.EffortHigh:
		out.Effort = shared.ReasoningEffortHigh
	case llm.EffortXHigh, llm.EffortMax:
		out.Effort = shared.ReasoningEffortXhigh
	}
	return out
}
