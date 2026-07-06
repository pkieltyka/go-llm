// Package ollama is a data-only convenience preset for local Ollama servers
// (https://ollama.com) over their OpenAI-compatible /v1 surface, riding the
// public providers/chatcompletions engine.
//
// The preset is COMMUNITY-VERIFIED, not live-tested in this repository's e2e
// matrix: it contributes only conventional data (base URL, provider name,
// usage-in-stream flag) on top of the engine's standard behavior and adds no
// code paths of its own. Ollama's OpenAI compatibility layer is documented at
// https://docs.ollama.com/api/openai-compatibility; features beyond it (model
// pulling, keep_alive, Ollama-native options) are out of scope — use
// Ollama's native API or chatcompletions.New with custom options directly.
//
//	p, err := ollama.New("") // "" = the http://localhost:11434/v1 convention
//	resp, err := p.Chat(ctx, &llm.Request{Model: "qwen3:8b", ...})
package ollama

import (
	"github.com/pkieltyka/go-llm/providers/chatcompletions"
)

// DefaultBaseURL is Ollama's conventional local OpenAI-compatible endpoint.
const DefaultBaseURL = "http://localhost:11434/v1"

// Compat declares Ollama's engine quirks: usage arrives on the final stream
// chunk when stream_options.include_usage is set (community-verified against
// Ollama's OpenAI compatibility layer). Everything else is the engine's
// standard chat-completions behavior.
func Compat() chatcompletions.Compat {
	return chatcompletions.Compat{StreamIncludeUsage: true}
}

// New constructs a provider for the Ollama server at baseURL; an empty
// baseURL uses DefaultBaseURL. Ollama is keyless — pass
// chatcompletions.WithAPIKey only when a proxy in front of it requires one.
// Later options may override the preset's name and compat data.
func New(baseURL string, opts ...chatcompletions.Option) (*chatcompletions.Provider, error) {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	preset := []chatcompletions.Option{
		chatcompletions.WithName("ollama"),
		chatcompletions.WithCompat(Compat()),
	}
	return chatcompletions.New(baseURL, append(preset, opts...)...)
}
