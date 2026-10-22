package providerutil

import (
	"errors"
	"testing"

	llm "github.com/pkieltyka/go-llm"
)

// TestIsContextOverflowCorpus pins the scoped overflow matcher against real
// provider messages (attributed) and against look-alike non-overflow texts.
func TestIsContextOverflowCorpus(t *testing.T) {
	overflow := []struct{ source, code, message string }{
		{"Anthropic", "invalid_request_error", "prompt is too long: 213462 tokens > 200000 maximum"},
		{"Anthropic 413", "request_too_large", "Request exceeds the maximum size"},
		{"OpenAI", "context_length_exceeded", "Your input exceeds the context window of this model."},
		{"OpenAI-compatible", "", "Requested token count exceeds the model's maximum context length of 131072 tokens"},
		{"vLLM (live 2026-10-22)", "400", "This model's maximum context length is 262144 tokens. However, you requested 16 output tokens and your prompt contains at least 262129 input tokens, for a total of at least 262145 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=262129)"},
		{"vLLM decoder", "", "The decoder prompt (length 300000) is longer than the maximum model length of 262144."},
		{"OpenRouter", "", "This endpoint's maximum context length is 200000 tokens. However, you requested about 250000 tokens."},
		{"OpenRouter/Poolside", "", "Input length 300000 exceeds the maximum allowed input length of 262144 tokens."},
		{"Together", "", "The input (300000 tokens) is longer than the model's context length (131072 tokens)."},
		{"Gemini", "", "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)"},
		{"xAI", "", "This model's maximum prompt length is 131072 but the request contains 537812 tokens"},
		{"Groq", "", "Please reduce the length of the messages or completion."},
		{"llama.cpp", "", "the request exceeds the available context size, try increasing it"},
		{"LM Studio", "", "tokens to keep from the initial prompt is greater than the context length"},
		{"Ollama", "", "prompt too long; exceeded max context length by 1200 tokens"},
		{"z.ai", "1261", "Prompt exceeds max length"},
		{"Kimi", "", "Your request exceeded model token limit: 262144 (requested: 300000)"},
		{"Mistral", "", "Prompt contains 300000 tokens and 0 draft tokens, too large for model with 131072 maximum context length"},
		{"Qwen/DashScope", "invalid_parameter_error", "Range of input length should be [1, 129024]"},
		{"stop-reason-shaped", "model_context_window_exceeded", ""},
	}
	for _, tc := range overflow {
		if !IsContextOverflow(tc.code, tc.message) {
			t.Errorf("%s: IsContextOverflow(%q, %q) = false, want true", tc.source, tc.code, tc.message)
		}
		if got := ErrorKind(400, tc.code, tc.message); !errors.Is(got, llm.ErrContextTooLong) {
			t.Errorf("%s: ErrorKind = %v, want ErrContextTooLong", tc.source, got)
		}
	}

	notOverflow := []struct{ source, code, message string }{
		{"Bedrock throttling", "", "ThrottlingException: Too many tokens, please wait before trying again."},
		{"TPM rate limit", "", "Rate limit reached for tokens per min (TPM): limit 30000, used 29000. Context window usage high."},
		{"429 text", "", "Too many requests; maximum context window throughput exceeded"},
		{"vLLM max_tokens (live 2026-10-22)", "400", "max_tokens=300000 cannot be greater than max_model_len=max_total_tokens=262144. Please request fewer output tokens. (parameter=max_tokens, value=300000)"},
		{"bare context word", "context_error", "invalid context id"},
		{"tokens word", "", "invalid tokens in request"},
		{"quota", "insufficient_quota", "You exceeded your current quota"},
	}
	for _, tc := range notOverflow {
		if IsContextOverflow(tc.code, tc.message) {
			t.Errorf("%s: IsContextOverflow(%q, %q) = true, want false", tc.source, tc.code, tc.message)
		}
	}
}
