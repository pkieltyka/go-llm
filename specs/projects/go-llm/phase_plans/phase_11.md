---
status: complete
---

# Phase 11 (post-v0.3.1): vLLM `/tokenize` extensions + native `structured_outputs` modes

## Overview

Deliver the two vLLM increments queued at the end of phase 10 (plan Future
Work): typed tokenizer extension methods for exact context accounting, and
the native v0.12+ `structured_outputs` constraint modes (regex/choice/
grammar/structural-tag) as typed provider options. Live-verified against the
checkpoint host `http://pax.local:8000` (vLLM 0.23.1rc1.dev786,
`Qwen/Qwen3.6-27B-FP8`, qwen3 reasoning + tool parsers).

## Steps (as executed)

1. Engine: `chatcompletions.Provider.DoJSONURL` — `DoJSON` against an
   absolute URL (same auth/headers/error mapping; `DoJSON` now delegates to
   it). Needed because vLLM parks the tokenizer endpoints at the SERVER
   ROOT while the OpenAI surface hangs off `/v1`. Probe answer recorded
   below. Only engine touch; full OpenRouter live suite re-run green.
2. `providers/vllm/tokenize.go`: extension methods on the concrete
   `*vllm.Provider` (escape-hatch style, NOT on `llm.Provider`):
   - `Tokenize(ctx, *llm.Request) (TokenizeResult, error)` — builds the
     chat-shaped `/tokenize` body `{model, messages, add_generation_prompt,
     tools?, chat_template_kwargs?}` by marshaling the engine's
     `BuildParams` output (message/tool conversion reused verbatim, and the
     request inherits Chat's validation + provider-option conflict rules).
     Mirrors vLLM's server-side `reasoning_effort` → `enable_thinking`
     chat_template_kwargs injection (none→false, other efforts→true;
     explicit kwargs win) so counts match a real request.
   - `Detokenize(ctx, []int) (string, error)`; `TokenizerInfo(ctx)
     (json.RawMessage, error)` — raw because the upstream schema is
     `extra="allow"` and varies by version.
   - `TokenizeResult{Count, MaxModelLen, Tokens}` with a
     `ContextUsage() llm.ContextUsage` bridge: exact occupancy (server
     count + server max_model_len) vs the estimate path (§13's prior-usage
     + price-table window). Godoc example added.
   - The provider derives the server root by trimming one trailing `/v1`
     segment from the base URL.
3. `vllm.Options.StructuredOutputs *vllm.StructuredOutputs` — typed native
   constraint modes per `vllm_research.md` §3: `Regex`, `Choice`,
   `Grammar`, `StructuralTag json.RawMessage`, `WhitespacePattern`. The
   dialect injects `structured_outputs: {...}` extras. JSON-schema/JSON
   mode stay on the unified `ResponseFormat` (era-portable spelling).
   Conflict rules, fail-loud at build:
   - `Request.ResponseFormat` set AND `StructuredOutputs` set →
     `ErrBadRequest` (two constraint systems, ambiguous; unified owns the
     slot per FS §14).
   - `WithLegacyEra()` AND `StructuredOutputs` → `ErrUnsupported`: the
     param exists only v0.12+, and the removed pre-v0.12 `guided_*`
     spelling is not emitted as a fallback because modern servers silently
     ignore it (phase-10 probe: `guided_json` free-forms with no error —
     the #1 wrong-era footgun).
   - Exactly one mode field, else `ErrBadRequest` (the server 400s on
     multiple modes; the client fails first with a clearer message).
     `WhitespacePattern` is a modifier, not a mode. Invalid
     `StructuralTag` JSON → `ErrBadRequest`.
4. Offline tests: `/tokenize` body golden (system + user + tools +
   chat_template_kwargs incl. the EnableThinking merge), root-vs-/v1 path
   pin, effort-injection table, validation inheritance, detokenize
   (incl. nil-token normalization), tokenizer_info raw passthrough + 404 →
   `ErrNotFound` mapping; `structured_outputs` extras goldens for all four
   modes (+ whitespace_pattern), conflict-rule table, era-gate test.
5. e2e (`TestLiveVLLM`): `tokenize` (count/max_model_len/tokens asserts,
   ContextUsage bridge, count-vs-prompt_tokens parity with tolerance 8,
   detokenize round trip, tolerant tokenizer_info), `structured_choice`
   (grass → exactly "green" of red/green/blue), `structured_regex`
   (`^[0-9]{4}$` → "1945"). Both structured scenarios pin
   `Effort: none` (see findings).
6. Docs/specs: package doc sections for both features; FS §14 vLLM entry
   grown; README self-hosted section: exact token counting paragraph +
   snippet, structured modes mention; plan Phase 10 note + Future Work
   updated; coverage floor `providers/vllm` ratcheted 76→84
   (check-coverage.sh + docs/release.md).

## Live findings (vLLM 0.23.1rc1.dev786, Qwen/Qwen3.6-27B-FP8, qwen3 parsers)

- **Tokenize path answer: root-relative.** `POST /tokenize` and
  `POST /detokenize` work at the server root; `POST /v1/tokenize` is 404.
  `GET /tokenizer_info` is 404 on this host (flag-gated upstream —
  `--enable-tokenizer-info-endpoint`); the extension maps it to
  `ErrNotFound` and the e2e scenario tolerates+logs it.
- **Count accuracy: exact.** Chat-style tokenize of the suite request
  (user text + one tool, effort none) = 288; the identical real chat
  reported `prompt_tokens` 288 — diff 0. Probes also matched at 20/20
  (no effort) and 22/22 (effort none → `enable_thinking:false` renders an
  empty think block worth +2 tokens), confirming the effort-injection
  mirror is required for parity. `max_model_len` = 131072.
- **structured_outputs × thinking: corrupts, doesn't 500.** Unlike the
  named-tool-choice 500 from phase 10, constraint modes with thinking
  active return HTTP 200 but corrupted constrained output: a
  `{"choice":["red","green","blue"]}` probe answered `"greengreen"` (not a
  member) with a full reasoning block attached. With thinking off
  (`reasoning_effort: "none"`), choice returns exactly `"green"` and
  regex `^[0-9]{4}$` returns `"1945"` (anchored form verified). Documented
  on the `StructuredOutputs` type; scenarios pin Effort none.
- Sending `structured_outputs` with two modes → server 400 (client now
  rejects earlier with `ErrBadRequest`).

## Tests

- `gofmt -l` clean; `go vet ./...` and `go vet -tags live ./...` clean.
- `go build ./...` (incl. examples); `go test -race -count=1 ./...`.
- Fuzz targets (4 × 5s): FuzzUnmarshalMessages, FuzzCollectEventSequences,
  FuzzValidateArgs, FuzzFor.
- `scripts/check-coverage.sh` green with the vllm floor at 84.
- Live: full `TestLiveVLLM` (16 scenarios incl. the three new ones) PASS
  against `http://pax.local:8000/v1`; full `TestLiveOpenRouter`
  (15 scenarios) PASS because the engine gained `DoJSONURL`.
