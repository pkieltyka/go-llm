package chatcompletions

import "encoding/json"

// JSONObject is a decoded JSON object. Dialects receive raw wire payloads as
// JSONObject and attach extra outbound request fields through the same type.
type JSONObject map[string]any

// jsonObject is the package-internal shorthand for JSONObject.
type jsonObject = JSONObject

// Internal aliases retained for readability inside the adapter; the exported
// names became canonical when the package went public.

type rawChoice = RawChoice
type rawMessage = RawMessage
type rawToolCall = RawToolCall
type rawUsage = RawUsage
type rawError = RawError

// rawChatCompletion is the decoded chat-completion (or stream chunk) payload.
type rawChatCompletion struct {
	ID      string      `json:"id"`
	Model   string      `json:"model"`
	Choices []RawChoice `json:"choices"`
	Usage   RawUsage    `json:"usage"`
	// Raw carries the full decoded payload, including fields not modeled
	// above.
	Raw JSONObject `json:"-"`
}

// RawChoice is one decoded response choice (blocking message or stream delta).
type RawChoice struct {
	Index        int        `json:"index"`
	FinishReason string     `json:"finish_reason"`
	Message      RawMessage `json:"message"`
	Delta        RawMessage `json:"delta"`
	Error        *RawError  `json:"error"`
	Raw          JSONObject `json:"-"`
}

// RawMessage is a decoded wire message or stream delta. Reasoning carries the
// current `reasoning` field name; ReasoningContent carries the legacy
// `reasoning_content` spelling still emitted by older servers (pre-rename
// vLLM, DeepSeek-convention dialects) — the adapter's default mapping reads
// both, preferring Reasoning.
type RawMessage struct {
	Role             string          `json:"role"`
	Content          string          `json:"content"`
	Refusal          string          `json:"refusal"`
	ToolCalls        []RawToolCall   `json:"tool_calls"`
	Reasoning        string          `json:"reasoning"`
	ReasoningContent string          `json:"reasoning_content"`
	ReasoningDetails json.RawMessage `json:"reasoning_details"`
	Annotations      json.RawMessage `json:"annotations"`
	Raw              JSONObject      `json:"-"`
}

// reasoningText returns the reasoning text, tolerating both field spellings.
func (m RawMessage) reasoningText() string {
	if m.Reasoning != "" {
		return m.Reasoning
	}
	return m.ReasoningContent
}

// RawToolCall is a decoded wire tool call (complete or delta fragment).
type RawToolCall struct {
	Index    *int            `json:"index,omitempty"`
	ID       string          `json:"id"`
	Type     string          `json:"type"`
	Function RawFunctionCall `json:"function"`
}

// RawFunctionCall carries a tool call's function name and JSON arguments.
type RawFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// RawUsage is the decoded wire usage object.
type RawUsage struct {
	PromptTokens            int64          `json:"prompt_tokens"`
	CompletionTokens        int64          `json:"completion_tokens"`
	TotalTokens             int64          `json:"total_tokens"`
	PromptTokensDetails     RawPromptUsage `json:"prompt_tokens_details"`
	CompletionTokensDetails RawOutputUsage `json:"completion_tokens_details"`
	Cost                    *float64       `json:"cost"`
	Raw                     JSONObject     `json:"-"`
}

// RawPromptUsage is the prompt_tokens_details wire object.
type RawPromptUsage struct {
	CachedTokens int64 `json:"cached_tokens"`
}

// RawOutputUsage is the completion_tokens_details wire object.
type RawOutputUsage struct {
	ReasoningTokens int64 `json:"reasoning_tokens"`
}

// RawError is a decoded wire error payload (nested error object, flat legacy
// object, or mid-stream error chunk).
type RawError struct {
	Code     any        `json:"code"`
	Message  string     `json:"message"`
	Metadata JSONObject `json:"metadata"`
}

func decodeObject(data []byte) jsonObject {
	var out jsonObject
	_ = json.Unmarshal(data, &out)
	return out
}

func (r *rawChatCompletion) UnmarshalJSON(data []byte) error {
	type alias rawChatCompletion
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*r = rawChatCompletion(a)
	r.Raw = decodeObject(data)
	return nil
}

func (r *RawChoice) UnmarshalJSON(data []byte) error {
	type alias RawChoice
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*r = RawChoice(a)
	r.Raw = decodeObject(data)
	return nil
}

func (r *RawMessage) UnmarshalJSON(data []byte) error {
	type alias RawMessage
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*r = RawMessage(a)
	r.Raw = decodeObject(data)
	return nil
}

func (r *RawUsage) UnmarshalJSON(data []byte) error {
	type alias RawUsage
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*r = RawUsage(a)
	r.Raw = decodeObject(data)
	return nil
}
