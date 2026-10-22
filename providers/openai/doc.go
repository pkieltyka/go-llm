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
// about model support.
//
// Authentication: an API key (WithAPIKey, OPENAI_API_KEY), or a credential
// store (WithCredentials) holding an API key or a Sign in with ChatGPT OAuth
// credential minted by NewLoginFlow. OAuth tokens are refreshed under the
// store's lock and retried once after a 401. A Sign in with ChatGPT token
// reaches only the subscription's models through a streaming-only endpoint,
// so on that path Chat drains a stream and MaxTokens, Temperature, TopP, and
// prompt-cache retention are not sent; Models lists the subscription catalog.
package openai
