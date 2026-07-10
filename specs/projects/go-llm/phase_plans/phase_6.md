---
status: complete
---

# Phase 6: Subscription Auth and OpenAI Codex

> Historical, non-normative execution record. It preserves phase-time OAuth
> wording; the current fail-closed persistence contract is in the main specs.

## Overview

This phase adds explicit credential-file loading, shared OAuth token handling,
Anthropic subscription auth, and the `openai-codex` provider. It also extracts
the OpenAI Responses request/response/stream mapping so `openai` and
`openai-codex` share behavior while keeping provider-tagged encrypted reasoning
replay isolated per provider id.

## Steps

1. Add core auth types in `auth.go`:
   - `type AuthFile map[string]AuthCredential`
   - `type AuthCredential struct { Type, Key, Access, Refresh, AccountID, Model, BaseURL string; Expires int64 }`
   - `func LoadAuthFile(path string) (AuthFile, error)`
   - `func ParseAuthFile(data []byte) (AuthFile, error)`
   - accept both bare provider maps and nested `{"providers": ...}`, tolerate unknown fields, and avoid any secret-printing helpers.
2. Add shared OAuth primitives in provider-owned internal code:
   - implement goroutine-safe token sources with refresh-before-expiry and single-flight behavior.
   - support `Token(ctx)` and an internal forced refresh used for one 401 retry.
   - expose only provider package options as public API: `WithOAuth(cred llm.AuthCredential, persist llm.OAuthPersistenceFunc)` where the callback honors context and returns after durable persistence; require it for credentials with refresh tokens while permitting nil for access-only credentials.
3. Wire Anthropic OAuth mode in `providers/anthropic`:
   - extend config with OAuth token source.
   - set bearer `Authorization`, remove `X-Api-Key`, and add `anthropic-beta: oauth-2025-04-20`.
   - refresh via Anthropic OAuth endpoint and retry once after a 401.
   - keep existing API-key/API-key-func paths unchanged.
4. Extract OpenAI Responses mapping into `providers/internal/responsesapi`:
   - move request build helpers, options application, response mapping, usage mapping, stream state, and error mapping behind a small `Config`.
   - parameterize provider id, capabilities, price table, and accepted reasoning provider id.
   - preserve OpenAI statelessness behavior: include `reasoning.encrypted_content` only when effective request options remain stateless.
   - keep `openai` provider behavior and public `openai.Options` intact.
5. Add `providers/openaicodex`:
   - provider id `openai-codex`.
   - construct an OpenAI SDK Responses client against the codex backend base URL.
   - require OAuth credentials, set bearer auth, `chatgpt-account-id`, and `originator` headers per request, re-reading account id from the token source after refresh.
   - expose static curated `Models()` and mirror OpenAI Responses capabilities except live models listing.
   - reuse shared Responses mapping with accepted reasoning provider id `openai-codex`.
6. Reconcile live e2e config:
   - point `internal/e2e.LoadConfig` at `llm.LoadAuthFile`.
   - support `api_key` and `oauth` entries, env fallback for API-key providers, placeholder scrubbing, and no secret logging.
   - update `gollm-test.json.sample` with `openai-codex`.
   - add live tests that run Anthropic API-key and OpenAI Codex OAuth paths when valid credentials are present, skipping visibly when missing.
7. Add focused tests:
   - auth-file parse/load tests for bare and nested formats plus placeholders.
   - OAuth token refresh state-machine tests against `httptest`: expiry, 401 forced refresh retry, durable persistence, and single-flight concurrency.
   - Anthropic OAuth header/retry tests.
   - OpenAI Codex request golden tests for endpoint, bearer, account id, originator, and provider-specific reasoning replay.
   - shared Responses fixture tests reused for OpenAI and Codex mapping.

## Tests

- `TestParseAuthFileNestedAndBare`: verifies `LoadAuthFile`/`ParseAuthFile` accept both supported credential file shapes.
- `TestTokenSourceRefreshesExpiredCredential`: verifies expired OAuth credentials refresh and invoke durable persistence.
- `TestTokenSourceSingleFlight`: verifies concurrent token calls issue one refresh under `-race`.
- `TestAnthropicOAuthHeadersAndRetry`: verifies bearer auth, OAuth beta header, one forced 401 refresh retry, and no API-key header.
- `TestOpenAICodexBuildRequestGolden`: verifies codex request JSON, headers, and stateless Responses mapping.
- `TestOpenAICodexReasoningReplayIsolation`: verifies `openai` and `openai-codex` encrypted reasoning are not mutually replayed.
- `TestProviderConfigUsesLoadAuthFile`: verifies the e2e harness consumes pi-compatible auth through `llm.LoadAuthFile`.
- Existing provider tests plus `go vet`, `golangci-lint`, and `go test -race ./...`.

## Live verification status

- **openai-codex**: ✅ green — 10/10 scenarios against a real ChatGPT
  subscription credential (2026-07-03); re-verified 12/12 after the phase-6
  review pass added `error_mapping` and `prompt_cache` scenarios. Live
  findings recorded in the provider: the backend rejects `temperature`
  (400 "Unsupported parameter") and does not accept server-reported dated
  model snapshots as request models.
- **anthropic (api_key)**: ✅ green — 12/12 scenarios (2026-07-03).
- **anthropic OAuth**: ✅ green — 12/12 scenarios against a real Claude
  Pro/Max subscription credential (2026-07-03), first live run after the
  Claude Code identity-header/system-block fix. FS §17C's live-verification
  gate for this mode is satisfied; **phase 6 is fully closed**.
