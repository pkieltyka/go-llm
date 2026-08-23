// Package openrouter implements the llm.Provider interface for OpenRouter's
// OpenAI-compatible chat completions API. Model discovery is explicit
// caller-triggered network I/O; it performs one OpenRouter catalog request and
// does not use a fallback catalog.
package openrouter
