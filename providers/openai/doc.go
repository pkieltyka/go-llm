// Package openai implements the llm.Provider interface for OpenAI's
// Responses API.
//
// Ordinary configuration and per-request Options use go-llm and standard
// library types; applications do not need to import openai-go. Provider.Client
// remains an explicitly advanced, vendor-coupled escape hatch.
package openai
