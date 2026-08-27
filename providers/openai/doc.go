// Package openai implements the llm.Provider interface for OpenAI's
// Responses API.
//
// Ordinary configuration and per-request Options use go-llm and standard
// library types; applications do not need to import openai-go. Provider.Client
// remains an explicitly advanced, vendor-coupled escape hatch.
//
// Options.ReasoningSummary selects auto, concise, or detailed user-visible
// reasoning summaries. Empty preserves the existing effort-driven automatic
// summary behavior, and other values fail locally. OpenAI remains authoritative
// about model support. The option is specific to this provider and is not
// supported by the separate openaicodex subscription provider.
package openai
