// Package openaicodex implements the llm.Provider interface for the ChatGPT
// subscription Codex backend using OAuth credentials. Refreshable
// credentials require a context-aware persistence callback; renewed
// credentials become visible only after persistence succeeds.
package openaicodex
