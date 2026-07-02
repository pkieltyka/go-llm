package llm

import (
	"encoding/json"
	"time"
)

// Role identifies the speaker for a message.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
	RoleSystem    Role = "system"
)

// Message is a role-tagged, ordered list of content parts.
type Message struct {
	Role     Role
	Parts    []Part
	Provider string
	Model    string
}

// Part is the marker interface for content parts.
type Part interface {
	part()
}

// ExtensionPart is implemented by provider-specific parts.
type ExtensionPart interface {
	Part
	ExtensionProvider() string
}

// ExtensionPartBase can be embedded by provider-specific part types to
// satisfy Part's sealed marker method. Embedders MUST also implement
// ExtensionProvider() string (the full ExtensionPart interface): adapters
// and canonical serialization identify extension parts only through
// ExtensionPart, and cannot route or encode a Part that embeds this base
// without it.
type ExtensionPartBase struct{}

func (ExtensionPartBase) part() {}

// CacheHint marks content as cacheable when a provider supports explicit
// prompt-cache breakpoints. A zero TTL means the provider default.
type CacheHint struct {
	TTL time.Duration
}

// TextPart contains plain text content.
type TextPart struct {
	Text  string
	Cache *CacheHint
}

func (TextPart) part() {}

// ImagePart contains image input by URL or inline bytes.
type ImagePart struct {
	URL       string
	Data      []byte
	MediaType string
	Cache     *CacheHint
}

func (ImagePart) part() {}

// FilePart contains file input by URL or inline bytes.
type FilePart struct {
	URL       string
	Data      []byte
	MediaType string
	Name      string
	Cache     *CacheHint
}

func (FilePart) part() {}

// ToolCallPart is an assistant-issued tool invocation.
type ToolCallPart struct {
	ID   string
	Name string
	Args json.RawMessage
}

func (ToolCallPart) part() {}

// ToolResultPart is the caller-provided result for a tool call.
type ToolResultPart struct {
	ToolCallID string
	Name       string
	Parts      []Part
	IsError    bool
}

func (ToolResultPart) part() {}

// ReasoningPart contains normalized reasoning plus an optional opaque provider
// payload for same-provider replay.
type ReasoningPart struct {
	Text     string
	Raw      json.RawMessage
	Provider string
}

func (ReasoningPart) part() {}

// Text creates a text content part.
func Text(s string) TextPart {
	return TextPart{Text: s}
}

// ImageURL creates an image part backed by a URL.
func ImageURL(url string) ImagePart {
	return ImagePart{URL: url}
}

// ImageData creates an image part backed by inline bytes.
func ImageData(data []byte, mediaType string) ImagePart {
	return ImagePart{Data: append([]byte(nil), data...), MediaType: mediaType}
}

// FileURL creates a file part backed by a URL.
func FileURL(url, mediaType string) FilePart {
	return FilePart{URL: url, MediaType: mediaType}
}

// FileData creates a file part backed by inline bytes.
func FileData(data []byte, mediaType, name string) FilePart {
	return FilePart{Data: append([]byte(nil), data...), MediaType: mediaType, Name: name}
}

// ToolCall creates a tool-call part with copied JSON arguments.
func ToolCall(id, name string, args json.RawMessage) ToolCallPart {
	return ToolCallPart{ID: id, Name: name, Args: append(json.RawMessage(nil), args...)}
}

// ToolResult creates a text tool-result part.
func ToolResult(id, content string) ToolResultPart {
	return ToolResultPart{ToolCallID: id, Parts: []Part{Text(content)}}
}

// ToolResultWithName creates a named text tool-result part.
func ToolResultWithName(id, name, content string) ToolResultPart {
	return ToolResultPart{ToolCallID: id, Name: name, Parts: []Part{Text(content)}}
}

// ToolResultParts creates a tool-result part from explicit content parts.
func ToolResultParts(id string, parts ...Part) ToolResultPart {
	return ToolResultPart{ToolCallID: id, Parts: cloneParts(parts)}
}

// ToolResultPartsWithName creates a named tool-result part from explicit parts.
func ToolResultPartsWithName(id, name string, parts ...Part) ToolResultPart {
	return ToolResultPart{ToolCallID: id, Name: name, Parts: cloneParts(parts)}
}

// UserText creates a user message containing one text part.
func UserText(s string) Message {
	return Message{Role: RoleUser, Parts: []Part{Text(s)}}
}

// AssistantText creates an assistant message containing one text part.
func AssistantText(s string) Message {
	return Message{Role: RoleAssistant, Parts: []Part{Text(s)}}
}

// UserParts creates a user message from explicit parts.
func UserParts(parts ...Part) Message {
	return Message{Role: RoleUser, Parts: cloneParts(parts)}
}

// AssistantParts creates an assistant message from explicit parts.
func AssistantParts(parts ...Part) Message {
	return Message{Role: RoleAssistant, Parts: cloneParts(parts)}
}

func cloneMessages(msgs []Message) []Message {
	if len(msgs) == 0 {
		return nil
	}
	out := make([]Message, len(msgs))
	for i, msg := range msgs {
		out[i] = Message{
			Role:     msg.Role,
			Parts:    cloneParts(msg.Parts),
			Provider: msg.Provider,
			Model:    msg.Model,
		}
	}
	return out
}

func cloneParts(parts []Part) []Part {
	if len(parts) == 0 {
		return nil
	}
	out := make([]Part, len(parts))
	for i, part := range parts {
		out[i] = clonePart(part)
	}
	return out
}

func clonePart(part Part) Part {
	switch p := part.(type) {
	case TextPart:
		p.Cache = cloneCacheHint(p.Cache)
		return p
	case *TextPart:
		if p == nil {
			return p
		}
		copied := *p
		copied.Cache = cloneCacheHint(copied.Cache)
		return &copied
	case ImagePart:
		p.Data = append([]byte(nil), p.Data...)
		p.Cache = cloneCacheHint(p.Cache)
		return p
	case *ImagePart:
		if p == nil {
			return p
		}
		copied := *p
		copied.Data = append([]byte(nil), copied.Data...)
		copied.Cache = cloneCacheHint(copied.Cache)
		return &copied
	case FilePart:
		p.Data = append([]byte(nil), p.Data...)
		p.Cache = cloneCacheHint(p.Cache)
		return p
	case *FilePart:
		if p == nil {
			return p
		}
		copied := *p
		copied.Data = append([]byte(nil), copied.Data...)
		copied.Cache = cloneCacheHint(copied.Cache)
		return &copied
	case ToolCallPart:
		p.Args = append(json.RawMessage(nil), p.Args...)
		return p
	case *ToolCallPart:
		if p == nil {
			return p
		}
		copied := *p
		copied.Args = append(json.RawMessage(nil), copied.Args...)
		return &copied
	case ToolResultPart:
		p.Parts = cloneParts(p.Parts)
		return p
	case *ToolResultPart:
		if p == nil {
			return p
		}
		copied := *p
		copied.Parts = cloneParts(copied.Parts)
		return &copied
	case ReasoningPart:
		p.Raw = append(json.RawMessage(nil), p.Raw...)
		return p
	case *ReasoningPart:
		if p == nil {
			return p
		}
		copied := *p
		copied.Raw = append(json.RawMessage(nil), copied.Raw...)
		return &copied
	default:
		// Extension parts (and any other unknown Part implementations) are
		// shared by reference: their concrete types live outside this
		// package, so a deep copy is not possible here. History.Messages
		// documents this shallow-copy behavior.
		return part
	}
}

func cloneCacheHint(h *CacheHint) *CacheHint {
	if h == nil {
		return nil
	}
	copied := *h
	return &copied
}
