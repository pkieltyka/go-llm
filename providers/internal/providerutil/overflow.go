package providerutil

import "strings"

// contextOverflowCodes are provider error codes/types that name a
// context-window or request-size overflow.
var contextOverflowCodes = []string{
	"context_length",                // OpenAI-family context_length_exceeded
	"request_too_large",             // Anthropic HTTP 413 error type
	"model_context_window_exceeded", // Anthropic/z.ai stop-reason-shaped code
}

// contextOverflowPhrases are scoped message phrases real providers emit for
// prompt overflow, with the provider attribution of each first occurrence. A
// bare "context" or "tokens" substring is deliberately absent: unrelated
// errors mention both words.
var contextOverflowPhrases = []string{
	"context window",                    // OpenAI "exceeds the context window", MiniMax
	"context length",                    // OpenRouter, Together, LM Studio, Ollama
	"context limit",                     // generic
	"maximum context",                   // vLLM, OpenAI-compatible proxies, Mistral
	"context size",                      // llama.cpp "exceeds the available context size", DS4
	"prompt is too long",                // Anthropic
	"prompt too long",                   // z.ai, Ollama
	"prompt exceeds max length",         // z.ai CN endpoint
	"request_too_large",                 // Anthropic 413 body
	"maximum prompt length",             // xAI
	"reduce the length of the messages", // Groq
	"maximum allowed input length",      // OpenRouter/Poolside
	"exceeded model token limit",        // Kimi
	"range of input length should be",   // DashScope/Qwen
	"maximum model length",              // vLLM decoder-prompt variant
	"input token count",                 // Gemini "input token count ... exceeds the maximum"
}

// nonOverflowPhrases veto overflow classification for throttling texts that
// happen to contain an overflow phrase, such as "Too many tokens, please
// wait" or token-per-minute rate limits.
var nonOverflowPhrases = []string{
	"rate limit",
	"too many requests",
	"throttl",
	"per minute",
	"tokens per min",
}

// IsContextOverflow reports whether a provider error code or message
// describes a context-window or request-size overflow (FS §16). Matching is
// case-insensitive and scoped: a code from contextOverflowCodes always
// matches, while message phrases match only when no throttling phrase is
// present.
func IsContextOverflow(code, message string) bool {
	lowerCode := strings.ToLower(code)
	for _, candidate := range contextOverflowCodes {
		if strings.Contains(lowerCode, candidate) {
			return true
		}
	}
	lowerMessage := strings.ToLower(message)
	for _, veto := range nonOverflowPhrases {
		if strings.Contains(lowerMessage, veto) {
			return false
		}
	}
	for _, phrase := range contextOverflowPhrases {
		if strings.Contains(lowerMessage, phrase) {
			return true
		}
	}
	return false
}
