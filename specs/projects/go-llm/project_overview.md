---
status: complete
---

# go-llm

A Go library providing a unified interface to LLM providers.

It is a low-level provider client with one request, response, stream, tool,
usage, and error vocabulary. The shipped presets support Anthropic, OpenAI,
OpenAI Codex subscription OAuth, OpenRouter, vLLM, and Ollama; the public
`chatcompletions` engine covers other OpenAI-compatible servers. ZAI is
future work and is not part of the current provider set.

The root `llm` package is standard-library-only. Provider packages wrap the
official Anthropic and OpenAI SDKs where appropriate while keeping ordinary
application-facing options library-owned.
