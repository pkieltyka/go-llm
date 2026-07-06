// Package llm provides one clean, provider-neutral Go interface to chat
// LLMs — Anthropic, OpenAI, ChatGPT/Claude subscription plans, OpenRouter —
// with normalized streaming, tool calling, structured output, reasoning,
// usage, cost, and errors.
//
// go-llm is a low-level provider client library: it does one layer well and
// stops there. No agent loops, no prompt frameworks. This core package has
// zero third-party dependencies; official vendor SDKs are used where they
// exist and are pulled in only by the provider subpackage you import.
//
// # Providers
//
// Every provider implements the small Provider interface: Name,
// Capabilities, Models, blocking Chat, and streaming ChatStream. Shipped
// providers live under providers/: anthropic (API key or Claude Pro/Max
// OAuth), openai (Responses API), openaicodex (ChatGPT Plus/Pro
// subscription OAuth), and openrouter. Each exposes its concrete type so a
// type assertion reaches the raw SDK client, and each accepts typed
// extension options through Request.ProviderOptions. Subscription (OAuth)
// credentials minted by existing CLIs are consumed and auto-refreshed;
// LoadAuthFile reads the pi-compatible credential file format.
//
// # Requests, messages, and streaming
//
// A Request carries Messages built from value-typed content parts (TextPart,
// ImagePart, FilePart, ToolCallPart, ToolResultPart, ReasoningPart) plus
// unified knobs: System, MaxTokens, Temperature/TopP (use Ptr for the
// optional pointers), Tools, ToolChoice, ResponseFormat, the Effort
// reasoning dial, and SessionID routing affinity.
//
// ChatStream returns an iter.Seq2[Event, error] of unified events. Collect
// drains any stream into a *Response — and on an in-stream error it returns
// the partial Response accumulated so far alongside the error, so aborted
// turns can be persisted. StreamText filters a stream to plain text deltas.
//
// # Tools
//
// Tools are declared with JSON Schema input (hand-written or generated from
// Go types by the schema subpackage). Malformed model tool calls never fail
// the turn and are never silently swallowed: adapters rescue what is
// rescuable and drop the rest visibly (ToolCallDropped events and
// Response.DroppedToolCalls), with the opt-in RetryDroppedToolCalls
// middleware for bounded automatic correction.
//
// # Structured output
//
// Parse[T] derives a JSON Schema from T, picks the best supported strategy
// (native json-schema, forced-tool extraction, or JSON mode plus
// validation), calls the provider, and decodes into T — with optional
// bounded retries and a semantic validator.
//
// # Sessions, history, and persistence
//
// History accumulates a conversation; Session wraps a provider, model,
// defaults, tools, and a History into a multi-turn convenience (Chat,
// ChatStream, AddToolResults, Continue). Messages and Responses have a
// canonical, versioned JSON encoding (MarshalMessages/UnmarshalMessages and
// friends) that round-trips reasoning payloads byte-exactly, so persisted
// conversations replay correctly — including across providers.
//
// # Usage, cost, and observability
//
// Every Response and MessageEnd carries normalized Usage (tokens including
// cache reads/writes and reasoning; CostUSD native or estimated from the
// embedded price table, with provenance in CostSource). Middleware composes
// via Wrap; UsageTracker aggregates per provider/model; WithLogger enables
// slog logging and WithWireCapture taps the redacted HTTP exchange — all
// silent by default.
//
// # Errors
//
// Failures normalize into two layers: sentinel errors (ErrAuth,
// ErrRateLimited, ErrOverloaded, ErrContextTooLong, ...) matched with
// errors.Is, and the full provider detail in *ProviderError extracted with
// errors.As. Capability mismatches fail fast with ErrUnsupported before any
// network call.
//
// The llmtest package provides a scriptable fake Provider for offline
// testing of code built on this interface.
package llm
