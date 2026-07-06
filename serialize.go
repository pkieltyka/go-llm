package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	messageEnvelopeVersion = 1

	partTypeText       = "text"
	partTypeImage      = "image"
	partTypeFile       = "file"
	partTypeToolCall   = "tool_call"
	partTypeToolResult = "tool_result"
	partTypeReasoning  = "reasoning"
)

var builtinPartTypes = map[string]struct{}{
	partTypeText:       {},
	partTypeImage:      {},
	partTypeFile:       {},
	partTypeToolCall:   {},
	partTypeToolResult: {},
	partTypeReasoning:  {},
}

var partTypeRegistry = struct {
	sync.RWMutex
	decoders map[string]func(json.RawMessage) (Part, error)
}{
	decoders: make(map[string]func(json.RawMessage) (Part, error)),
}

// UnknownPart preserves unrecognized serialized part types for
// forward-compatible history reloads.
type UnknownPart struct {
	Type string
	Data json.RawMessage
}

func (UnknownPart) part() {}

// MarshalJSON re-emits the original serialized payload when called directly.
// Use MarshalMessage, MarshalMessages, or MarshalResponse for canonical
// persistence; encoding/json compacts Marshaler output before returning it.
func (p UnknownPart) MarshalJSON() ([]byte, error) {
	return marshalUnknownPart(p)
}

// RegisterPartType registers a decoder for a provider extension part type.
// Provider packages normally call it from init.
func RegisterPartType(name string, decode func(json.RawMessage) (Part, error)) error {
	if name == "" {
		return fmt.Errorf("%w: cannot register empty part type", ErrBadRequest)
	}
	if decode == nil {
		return fmt.Errorf("%w: cannot register nil part decoder", ErrBadRequest)
	}
	if _, ok := builtinPartTypes[name]; ok {
		return fmt.Errorf("%w: cannot register built-in part type %q", ErrBadRequest, name)
	}
	if err := validateExtensionPartTypeName(name); err != nil {
		return err
	}

	partTypeRegistry.Lock()
	defer partTypeRegistry.Unlock()
	if _, exists := partTypeRegistry.decoders[name]; exists {
		return fmt.Errorf("%w: part type %q already registered", ErrBadRequest, name)
	}
	partTypeRegistry.decoders[name] = decode
	return nil
}

// MarshalMessages serializes messages in a versioned persistence envelope.
func MarshalMessages(msgs []Message) ([]byte, error) {
	var b bytes.Buffer
	b.WriteString(`{"version":`)
	b.WriteString(strconv.Itoa(messageEnvelopeVersion))
	b.WriteString(`,"messages":`)
	rawMessages, err := marshalMessagesArray(msgs)
	if err != nil {
		return nil, err
	}
	b.Write(rawMessages)
	b.WriteByte('}')
	return b.Bytes(), nil
}

// MarshalMessage serializes one message without encoding/json post-processing,
// preserving raw replay payload bytes inside parts.
func MarshalMessage(msg Message) ([]byte, error) {
	return marshalMessage(msg)
}

// UnmarshalMessages decodes a versioned persistence envelope.
func UnmarshalMessages(data []byte) ([]Message, error) {
	var header struct {
		Version int `json:"version"`
	}
	if err := json.Unmarshal(data, &header); err != nil {
		return nil, err
	}
	if header.Version != messageEnvelopeVersion {
		return nil, fmt.Errorf("%w: unsupported message envelope version %d", ErrBadRequest, header.Version)
	}

	var env messageEnvelope
	if err := json.Unmarshal(data, &env); err != nil {
		return nil, err
	}
	return cloneMessages(env.Messages), nil
}

// UnmarshalMessage decodes one canonical message.
func UnmarshalMessage(data []byte) (Message, error) {
	var msg Message
	if err := json.Unmarshal(data, &msg); err != nil {
		return Message{}, err
	}
	return cloneMessages([]Message{msg})[0], nil
}

// MarshalResponse serializes a response without encoding/json post-processing,
// preserving raw replay payload bytes inside parts. Response.Raw and Usage.Raw
// are intentionally excluded.
func MarshalResponse(resp *Response) ([]byte, error) {
	if resp == nil {
		return nil, fmt.Errorf("%w: nil response", ErrBadRequest)
	}
	return marshalResponse(*resp)
}

// UnmarshalResponse decodes one canonical response.
func UnmarshalResponse(data []byte) (*Response, error) {
	var resp Response
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

type messageEnvelope struct {
	Version  int       `json:"version"`
	Messages []Message `json:"messages"`
}

// MarshalJSON supports ordinary encoding/json use. Use MarshalMessage or
// MarshalMessages when raw replay payload bytes must be preserved exactly;
// encoding/json compacts and HTML-escapes Marshaler output.
func (m Message) MarshalJSON() ([]byte, error) {
	return marshalMessage(m)
}

// UnmarshalJSON decodes Message parts through the part registry.
func (m *Message) UnmarshalJSON(data []byte) error {
	var wire messageJSON
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	parts, err := unmarshalParts(wire.Parts)
	if err != nil {
		return err
	}
	*m = Message{
		Role:     wire.Role,
		Parts:    parts,
		Provider: wire.Provider,
		Model:    wire.Model,
	}
	return nil
}

type messageJSON struct {
	Role     Role              `json:"role"`
	Parts    []json.RawMessage `json:"parts"`
	Provider string            `json:"provider,omitempty"`
	Model    string            `json:"model,omitempty"`
}

func (p TextPart) MarshalJSON() ([]byte, error) {
	return json.Marshal(textPartJSON{
		Type:  partTypeText,
		Text:  p.Text,
		Cache: cacheToJSON(p.Cache),
	})
}

func (p *TextPart) UnmarshalJSON(data []byte) error {
	var wire textPartJSON
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*p = TextPart{Text: wire.Text, Cache: cacheFromJSON(wire.Cache)}
	return nil
}

type textPartJSON struct {
	Type  string         `json:"type"`
	Text  string         `json:"text"`
	Cache *cacheHintJSON `json:"cache,omitempty"`
}

func (p ImagePart) MarshalJSON() ([]byte, error) {
	return json.Marshal(imagePartJSON{
		Type:      partTypeImage,
		URL:       p.URL,
		Data:      p.Data,
		MediaType: p.MediaType,
		Cache:     cacheToJSON(p.Cache),
	})
}

func (p *ImagePart) UnmarshalJSON(data []byte) error {
	var wire imagePartJSON
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*p = ImagePart{
		URL:       wire.URL,
		Data:      append([]byte(nil), wire.Data...),
		MediaType: wire.MediaType,
		Cache:     cacheFromJSON(wire.Cache),
	}
	return nil
}

type imagePartJSON struct {
	Type      string         `json:"type"`
	URL       string         `json:"url,omitempty"`
	Data      []byte         `json:"data,omitempty"`
	MediaType string         `json:"media_type,omitempty"`
	Cache     *cacheHintJSON `json:"cache,omitempty"`
}

func (p FilePart) MarshalJSON() ([]byte, error) {
	return json.Marshal(filePartJSON{
		Type:      partTypeFile,
		URL:       p.URL,
		Data:      p.Data,
		MediaType: p.MediaType,
		Name:      p.Name,
		Cache:     cacheToJSON(p.Cache),
	})
}

func (p *FilePart) UnmarshalJSON(data []byte) error {
	var wire filePartJSON
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*p = FilePart{
		URL:       wire.URL,
		Data:      append([]byte(nil), wire.Data...),
		MediaType: wire.MediaType,
		Name:      wire.Name,
		Cache:     cacheFromJSON(wire.Cache),
	}
	return nil
}

type filePartJSON struct {
	Type      string         `json:"type"`
	URL       string         `json:"url,omitempty"`
	Data      []byte         `json:"data,omitempty"`
	MediaType string         `json:"media_type,omitempty"`
	Name      string         `json:"name,omitempty"`
	Cache     *cacheHintJSON `json:"cache,omitempty"`
}

func (p ToolCallPart) MarshalJSON() ([]byte, error) {
	return marshalToolCallPart(p)
}

func (p *ToolCallPart) UnmarshalJSON(data []byte) error {
	var wire toolCallPartJSON
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*p = ToolCallPart{
		ID:   wire.ID,
		Name: wire.Name,
		Args: cloneRaw(wire.Args),
	}
	return nil
}

type toolCallPartJSON struct {
	Type string          `json:"type"`
	ID   string          `json:"id"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

func (p ToolResultPart) MarshalJSON() ([]byte, error) {
	return marshalToolResultPart(p)
}

func (p *ToolResultPart) UnmarshalJSON(data []byte) error {
	var wire toolResultPartJSON
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	parts, err := unmarshalParts(wire.Parts)
	if err != nil {
		return err
	}
	*p = ToolResultPart{
		ToolCallID: wire.ToolCallID,
		Name:       wire.Name,
		Content:    parts,
		IsError:    wire.IsError,
	}
	return nil
}

// toolResultPartJSON keeps the serialized key "parts" for the Go-level
// Content field: the v0.2 field rename is source-only, and the canonical
// envelope's wire shape stays stable (no version bump).
type toolResultPartJSON struct {
	Type       string            `json:"type"`
	ToolCallID string            `json:"tool_call_id"`
	Name       string            `json:"name,omitempty"`
	Parts      []json.RawMessage `json:"parts"`
	IsError    bool              `json:"is_error,omitempty"`
}

func (p ReasoningPart) MarshalJSON() ([]byte, error) {
	return marshalReasoningPart(p)
}

func (p *ReasoningPart) UnmarshalJSON(data []byte) error {
	var wire reasoningPartJSON
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*p = ReasoningPart{
		Text:     wire.Text,
		Raw:      cloneRaw(wire.Raw),
		Provider: wire.Provider,
	}
	return nil
}

type reasoningPartJSON struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	Raw      json.RawMessage `json:"raw,omitempty"`
	Provider string          `json:"provider,omitempty"`
}

// MarshalJSON supports ordinary encoding/json use. Use MarshalResponse when
// raw replay payload bytes must be preserved exactly; encoding/json compacts
// and HTML-escapes Marshaler output.
func (r Response) MarshalJSON() ([]byte, error) {
	return marshalResponse(r)
}

func (r *Response) UnmarshalJSON(data []byte) error {
	var wire responseJSON
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	parts, err := unmarshalParts(wire.Parts)
	if err != nil {
		return err
	}
	*r = Response{
		ID:               wire.ID,
		Provider:         wire.Provider,
		Model:            wire.Model,
		Parts:            parts,
		StopReason:       wire.StopReason,
		StopReasonRaw:    wire.StopReasonRaw,
		Usage:            wire.Usage,
		DroppedToolCalls: append([]DroppedToolCall(nil), wire.DroppedToolCalls...),
	}
	return nil
}

type responseJSON struct {
	ID               string            `json:"id,omitempty"`
	Provider         string            `json:"provider,omitempty"`
	Model            string            `json:"model,omitempty"`
	Parts            []json.RawMessage `json:"parts"`
	StopReason       StopReason        `json:"stop_reason,omitempty"`
	StopReasonRaw    string            `json:"stop_reason_raw,omitempty"`
	Usage            Usage             `json:"usage"`
	DroppedToolCalls []DroppedToolCall `json:"dropped_tool_calls,omitempty"`
}

func (u Usage) MarshalJSON() ([]byte, error) {
	return json.Marshal(usageJSON{
		InputTokens:      u.InputTokens,
		OutputTokens:     u.OutputTokens,
		TotalTokens:      u.TotalTokens,
		CacheReadTokens:  u.CacheReadTokens,
		CacheWriteTokens: u.CacheWriteTokens,
		ReasoningTokens:  u.ReasoningTokens,
		CostUSD:          u.CostUSD,
		CostSource:       u.CostSource,
	})
}

func (u *Usage) UnmarshalJSON(data []byte) error {
	var wire usageJSON
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*u = Usage{
		InputTokens:      wire.InputTokens,
		OutputTokens:     wire.OutputTokens,
		TotalTokens:      wire.TotalTokens,
		CacheReadTokens:  wire.CacheReadTokens,
		CacheWriteTokens: wire.CacheWriteTokens,
		ReasoningTokens:  wire.ReasoningTokens,
		CostUSD:          wire.CostUSD,
		CostSource:       wire.CostSource,
	}
	return nil
}

// usageJSON adds cost_source with omitempty: the canonical envelope is
// forward-compatible for additive optional keys, so no version bump.
type usageJSON struct {
	InputTokens      int64    `json:"input_tokens"`
	OutputTokens     int64    `json:"output_tokens"`
	TotalTokens      int64    `json:"total_tokens"`
	CacheReadTokens  int64    `json:"cache_read_tokens"`
	CacheWriteTokens int64    `json:"cache_write_tokens"`
	ReasoningTokens  int64    `json:"reasoning_tokens"`
	CostUSD          *float64 `json:"cost_usd,omitempty"`
	CostSource       string   `json:"cost_source,omitempty"`
}

type cacheHintJSON struct {
	TTLNanos int64 `json:"ttl_ns,omitempty"`
}

func cacheToJSON(h *CacheHint) *cacheHintJSON {
	if h == nil {
		return nil
	}
	return &cacheHintJSON{TTLNanos: int64(h.TTL)}
}

func cacheFromJSON(h *cacheHintJSON) *CacheHint {
	if h == nil {
		return nil
	}
	return &CacheHint{TTL: time.Duration(h.TTLNanos)}
}

func marshalMessagesArray(msgs []Message) (json.RawMessage, error) {
	var b bytes.Buffer
	b.WriteByte('[')
	for i, msg := range msgs {
		if i > 0 {
			b.WriteByte(',')
		}
		raw, err := marshalMessage(msg)
		if err != nil {
			return nil, err
		}
		b.Write(raw)
	}
	b.WriteByte(']')
	return b.Bytes(), nil
}

func marshalMessage(m Message) (json.RawMessage, error) {
	parts, err := marshalParts(m.Parts)
	if err != nil {
		return nil, err
	}

	var b bytes.Buffer
	b.WriteString(`{"role":`)
	writeJSONString(&b, string(m.Role))
	b.WriteString(`,"parts":`)
	b.Write(parts)
	if m.Provider != "" {
		b.WriteString(`,"provider":`)
		writeJSONString(&b, m.Provider)
	}
	if m.Model != "" {
		b.WriteString(`,"model":`)
		writeJSONString(&b, m.Model)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

func marshalResponse(r Response) (json.RawMessage, error) {
	parts, err := marshalParts(r.Parts)
	if err != nil {
		return nil, err
	}
	usage, err := r.Usage.MarshalJSON()
	if err != nil {
		return nil, err
	}

	var b bytes.Buffer
	b.WriteByte('{')
	first := true
	writeStringField := func(name, value string) {
		if value == "" {
			return
		}
		writeComma(&b, &first)
		writeJSONString(&b, name)
		b.WriteByte(':')
		writeJSONString(&b, value)
	}

	writeStringField("id", r.ID)
	writeStringField("provider", r.Provider)
	writeStringField("model", r.Model)
	writeComma(&b, &first)
	b.WriteString(`"parts":`)
	b.Write(parts)
	writeStringField("stop_reason", string(r.StopReason))
	writeStringField("stop_reason_raw", r.StopReasonRaw)
	writeComma(&b, &first)
	b.WriteString(`"usage":`)
	b.Write(usage)
	if len(r.DroppedToolCalls) > 0 {
		dropped, err := json.Marshal(r.DroppedToolCalls)
		if err != nil {
			return nil, err
		}
		writeComma(&b, &first)
		b.WriteString(`"dropped_tool_calls":`)
		b.Write(dropped)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

func marshalToolCallPart(p ToolCallPart) (json.RawMessage, error) {
	var b bytes.Buffer
	b.WriteString(`{"type":"tool_call","id":`)
	writeJSONString(&b, p.ID)
	b.WriteString(`,"name":`)
	writeJSONString(&b, p.Name)
	if len(p.Args) != 0 {
		if !json.Valid(p.Args) {
			return nil, fmt.Errorf("%w: invalid tool call args JSON", ErrBadRequest)
		}
		b.WriteString(`,"args":`)
		b.Write(p.Args)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// marshalToolResultPart writes ToolResultPart.Content under the stable wire
// key "parts" (see toolResultPartJSON).
func marshalToolResultPart(p ToolResultPart) (json.RawMessage, error) {
	parts, err := marshalParts(p.Content)
	if err != nil {
		return nil, err
	}

	var b bytes.Buffer
	b.WriteString(`{"type":"tool_result","tool_call_id":`)
	writeJSONString(&b, p.ToolCallID)
	if p.Name != "" {
		b.WriteString(`,"name":`)
		writeJSONString(&b, p.Name)
	}
	b.WriteString(`,"parts":`)
	b.Write(parts)
	if p.IsError {
		b.WriteString(`,"is_error":true`)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

func marshalReasoningPart(p ReasoningPart) (json.RawMessage, error) {
	var b bytes.Buffer
	b.WriteString(`{"type":"reasoning"`)
	if p.Text != "" {
		b.WriteString(`,"text":`)
		writeJSONString(&b, p.Text)
	}
	if len(p.Raw) != 0 {
		if !json.Valid(p.Raw) {
			return nil, fmt.Errorf("%w: invalid reasoning raw JSON", ErrBadRequest)
		}
		b.WriteString(`,"raw":`)
		b.Write(p.Raw)
	}
	if p.Provider != "" {
		b.WriteString(`,"provider":`)
		writeJSONString(&b, p.Provider)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

func marshalUnknownPart(p UnknownPart) (json.RawMessage, error) {
	if len(p.Data) != 0 {
		if !json.Valid(p.Data) {
			return nil, fmt.Errorf("%w: invalid unknown part JSON", ErrBadRequest)
		}
		typ, err := partTypeFromRaw(p.Data)
		if err != nil {
			return nil, err
		}
		if p.Type != "" && p.Type != typ {
			return nil, fmt.Errorf("%w: unknown part type %q does not match raw type %q", ErrBadRequest, p.Type, typ)
		}
		return cloneRaw(p.Data), nil
	}
	if p.Type == "" {
		return nil, fmt.Errorf("%w: unknown part type is required", ErrBadRequest)
	}
	var b bytes.Buffer
	b.WriteString(`{"type":`)
	writeJSONString(&b, p.Type)
	b.WriteByte('}')
	return b.Bytes(), nil
}

func marshalParts(parts []Part) (json.RawMessage, error) {
	if len(parts) == 0 {
		return json.RawMessage("[]"), nil
	}

	var b bytes.Buffer
	b.WriteByte('[')
	for i, part := range parts {
		if i > 0 {
			b.WriteByte(',')
		}
		raw, err := marshalPart(part)
		if err != nil {
			return nil, err
		}
		b.Write(raw)
	}
	b.WriteByte(']')
	return b.Bytes(), nil
}

func marshalPart(part Part) (json.RawMessage, error) {
	part = derefPart(part)
	if part == nil {
		return nil, fmt.Errorf("%w: nil part", ErrBadRequest)
	}

	switch p := part.(type) {
	case TextPart:
		return p.MarshalJSON()
	case ImagePart:
		return p.MarshalJSON()
	case FilePart:
		return p.MarshalJSON()
	case ToolCallPart:
		return p.MarshalJSON()
	case ToolResultPart:
		return p.MarshalJSON()
	case ReasoningPart:
		return p.MarshalJSON()
	case UnknownPart:
		return p.MarshalJSON()
	}

	if isNilPart(part) {
		return nil, fmt.Errorf("%w: nil extension part", ErrBadRequest)
	}
	if marshaler, ok := part.(json.Marshaler); ok {
		extension, ok := part.(ExtensionPart)
		if !ok {
			return nil, fmt.Errorf("%w: extension part %T must implement ExtensionPart", ErrBadRequest, part)
		}
		raw, err := marshaler.MarshalJSON()
		if err != nil {
			return nil, err
		}
		if !json.Valid(raw) {
			return nil, fmt.Errorf("%w: part %T marshaled invalid JSON", ErrBadRequest, part)
		}
		if err := validateMarshaledExtensionPart(extension, raw); err != nil {
			return nil, err
		}
		return cloneRaw(raw), nil
	}
	return nil, fmt.Errorf("%w: part %T does not implement json.Marshaler", ErrBadRequest, part)
}

func validateExtensionPartTypeName(name string) error {
	if strings.TrimSpace(name) != name || strings.Count(name, "/") != 1 {
		return fmt.Errorf("%w: extension part type %q must use provider/kind namespace", ErrBadRequest, name)
	}
	provider, kind, _ := strings.Cut(name, "/")
	if provider == "" || kind == "" {
		return fmt.Errorf("%w: extension part type %q must use provider/kind namespace", ErrBadRequest, name)
	}
	return nil
}

func validateMarshaledExtensionPart(part ExtensionPart, raw json.RawMessage) error {
	provider := part.ExtensionProvider()
	if provider == "" {
		return fmt.Errorf("%w: extension part %T returned empty provider", ErrBadRequest, part)
	}
	typ, err := partTypeFromRaw(raw)
	if err != nil {
		return err
	}
	if err := validateExtensionPartTypeName(typ); err != nil {
		return err
	}
	typeProvider, _, _ := strings.Cut(typ, "/")
	if typeProvider != provider {
		return fmt.Errorf("%w: extension part %T type %q does not match provider %q", ErrBadRequest, part, typ, provider)
	}
	return nil
}

func partTypeFromRaw(raw json.RawMessage) (string, error) {
	var header struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &header); err != nil {
		return "", fmt.Errorf("%w: part payload must be an object with a type", ErrBadRequest)
	}
	if header.Type == "" {
		return "", fmt.Errorf("%w: part type is required", ErrBadRequest)
	}
	return header.Type, nil
}

func isNilPart(part Part) bool {
	value := reflect.ValueOf(part)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

func writeComma(b *bytes.Buffer, first *bool) {
	if *first {
		*first = false
		return
	}
	b.WriteByte(',')
}

func writeJSONString(b *bytes.Buffer, s string) {
	raw, _ := json.Marshal(s)
	b.Write(raw)
}

func unmarshalParts(rawParts []json.RawMessage) ([]Part, error) {
	if len(rawParts) == 0 {
		return nil, nil
	}
	parts := make([]Part, len(rawParts))
	for i, raw := range rawParts {
		part, err := unmarshalPart(raw)
		if err != nil {
			return nil, err
		}
		parts[i] = part
	}
	return parts, nil
}

func unmarshalPart(raw json.RawMessage) (Part, error) {
	var header struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &header); err != nil {
		return nil, err
	}
	if header.Type == "" {
		return nil, fmt.Errorf("%w: part type is required", ErrBadRequest)
	}

	switch header.Type {
	case partTypeText:
		var part TextPart
		if err := json.Unmarshal(raw, &part); err != nil {
			return nil, err
		}
		return part, nil
	case partTypeImage:
		var part ImagePart
		if err := json.Unmarshal(raw, &part); err != nil {
			return nil, err
		}
		return part, nil
	case partTypeFile:
		var part FilePart
		if err := json.Unmarshal(raw, &part); err != nil {
			return nil, err
		}
		return part, nil
	case partTypeToolCall:
		var part ToolCallPart
		if err := json.Unmarshal(raw, &part); err != nil {
			return nil, err
		}
		return part, nil
	case partTypeToolResult:
		var part ToolResultPart
		if err := json.Unmarshal(raw, &part); err != nil {
			return nil, err
		}
		return part, nil
	case partTypeReasoning:
		var part ReasoningPart
		if err := json.Unmarshal(raw, &part); err != nil {
			return nil, err
		}
		return part, nil
	}

	partTypeRegistry.RLock()
	decoder := partTypeRegistry.decoders[header.Type]
	partTypeRegistry.RUnlock()
	if decoder != nil {
		part, err := decoder(cloneRaw(raw))
		if err != nil {
			return nil, err
		}
		if part == nil {
			return nil, fmt.Errorf("%w: decoder for part type %q returned nil", ErrBadRequest, header.Type)
		}
		return part, nil
	}

	return UnknownPart{
		Type: header.Type,
		Data: cloneRaw(raw),
	}, nil
}

func cloneRaw(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	return append(json.RawMessage(nil), raw...)
}
