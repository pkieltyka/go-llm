package openai

import (
	"encoding/json"
	"fmt"

	sdk "github.com/openai/openai-go/v3"
	sdkparam "github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	llm "github.com/pkieltyka/go-llm"
	"github.com/pkieltyka/go-llm/providers/internal/providerutil"
)

// Include selects optional data to return in an OpenAI response. Values are
// strings on the wire so callers can use a future OpenAI value by converting
// it to Include without waiting for an SDK upgrade.
type Include string

const (
	IncludeFileSearchCallResults            Include = "file_search_call.results"
	IncludeWebSearchCallResults             Include = "web_search_call.results"
	IncludeWebSearchCallActionSources       Include = "web_search_call.action.sources"
	IncludeMessageInputImageImageURL        Include = "message.input_image.image_url"
	IncludeComputerCallOutputOutputImageURL Include = "computer_call_output.output.image_url"
	IncludeCodeInterpreterCallOutputs       Include = "code_interpreter_call.outputs"
	IncludeReasoningEncryptedContent        Include = "reasoning.encrypted_content"
	IncludeMessageOutputTextLogprobs        Include = "message.output_text.logprobs"
)

// Verbosity controls the detail level of generated text.
type Verbosity string

const (
	VerbosityLow    Verbosity = "low"
	VerbosityMedium Verbosity = "medium"
	VerbosityHigh   Verbosity = "high"
)

// ServiceTier selects OpenAI's request processing tier.
type ServiceTier string

const (
	ServiceTierAuto     ServiceTier = "auto"
	ServiceTierDefault  ServiceTier = "default"
	ServiceTierFlex     ServiceTier = "flex"
	ServiceTierScale    ServiceTier = "scale"
	ServiceTierPriority ServiceTier = "priority"
)

// PromptCacheRetention selects how long OpenAI retains prompt cache entries.
type PromptCacheRetention string

const (
	PromptCacheRetentionInMemory PromptCacheRetention = "in_memory"
	PromptCacheRetention24h      PromptCacheRetention = "24h"
)

// Metadata contains OpenAI response metadata. OpenAI currently permits up to
// 16 string key-value pairs.
type Metadata map[string]string

// Conversation identifies an OpenAI conversation object. ConversationID is
// the simpler string form; Conversation represents the object wire form.
type Conversation struct {
	ID string `json:"id"`
}

// Options carries OpenAI Responses-specific request extensions. Its public
// fields use go-llm or standard-library types so ordinary callers are not
// coupled to the version of openai-go used by this package.
type Options struct {
	// Store opts into or out of OpenAI response storage. The provider defaults
	// to false for stateless encrypted-reasoning replay.
	Store *bool
	// PreviousResponseID continues a stored response chain.
	PreviousResponseID string
	// ConversationID attaches the request to a conversation by string ID.
	ConversationID string
	// Conversation attaches the request using OpenAI's object form. It takes
	// precedence over ConversationID when both are set.
	Conversation *Conversation
	// Include selects additional response fields.
	Include []Include
	// Background requests asynchronous background execution.
	Background *bool
	// HostedTools appends provider-hosted tool definitions to unified function
	// tools. Each entry must be a JSON object in the OpenAI Responses tool
	// shape. Raw JSON keeps this open-ended surface independent of openai-go.
	HostedTools []json.RawMessage
	// Verbosity controls generated text detail.
	Verbosity Verbosity
	// Metadata attaches string metadata to the response.
	Metadata Metadata
	// ServiceTier selects OpenAI's processing tier.
	ServiceTier ServiceTier
	// SafetyIdentifier is a stable, preferably hashed end-user identifier.
	SafetyIdentifier string
	// PromptCacheRetention controls extended prompt-cache retention.
	PromptCacheRetention PromptCacheRetention
}

// ForProvider identifies these options as OpenAI-specific.
func (Options) ForProvider() string { return providerName }

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

func applyOptions(options *Options, params *responses.ResponseNewParams) error {
	if options == nil {
		return nil
	}
	if options.Store != nil {
		params.Store = sdk.Bool(*options.Store)
	}
	if options.PreviousResponseID != "" {
		params.PreviousResponseID = sdk.String(options.PreviousResponseID)
	}
	if options.Conversation != nil {
		params.Conversation.OfConversationObject = &responses.ResponseConversationParam{ID: options.Conversation.ID}
	} else if options.ConversationID != "" {
		params.Conversation.OfString = sdkparam.NewOpt(options.ConversationID)
	}
	if len(options.Include) > 0 {
		params.Include = mergeIncludes(params.Include, options.Include)
	}
	if options.Background != nil {
		params.Background = sdk.Bool(*options.Background)
	}
	for i, raw := range options.HostedTools {
		var object map[string]json.RawMessage
		if err := json.Unmarshal(raw, &object); err != nil || object == nil {
			if err == nil {
				err = fmt.Errorf("expected a JSON object")
			}
			return fmt.Errorf("%w: OpenAI hosted tool %d: %v", llm.ErrBadRequest, i, err)
		}
		copyRaw := append(json.RawMessage(nil), raw...)
		params.Tools = append(params.Tools, sdkparam.Override[responses.ToolUnionParam](copyRaw))
	}
	if options.Verbosity != "" {
		params.Text.Verbosity = responses.ResponseTextConfigVerbosity(options.Verbosity)
	}
	if len(options.Metadata) > 0 {
		params.Metadata = make(shared.Metadata, len(options.Metadata))
		for key, value := range options.Metadata {
			params.Metadata[key] = value
		}
	}
	if options.ServiceTier != "" {
		params.ServiceTier = responses.ResponseNewParamsServiceTier(options.ServiceTier)
	}
	if options.SafetyIdentifier != "" {
		params.SafetyIdentifier = sdk.String(options.SafetyIdentifier)
	}
	if options.PromptCacheRetention != "" {
		params.PromptCacheRetention = responses.ResponseNewParamsPromptCacheRetention(options.PromptCacheRetention)
	}
	return nil
}

func mergeIncludes(base []responses.ResponseIncludable, extra []Include) []responses.ResponseIncludable {
	out := append([]responses.ResponseIncludable(nil), base...)
	seen := make(map[responses.ResponseIncludable]struct{}, len(out)+len(extra))
	for _, include := range out {
		seen[include] = struct{}{}
	}
	for _, include := range extra {
		if include == "" {
			continue
		}
		sdkInclude := responses.ResponseIncludable(include)
		if _, ok := seen[sdkInclude]; ok {
			continue
		}
		seen[sdkInclude] = struct{}{}
		out = append(out, sdkInclude)
	}
	return out
}
