---
status: implemented
ratified_date: 2026-10-24
implemented_date: 2026-10-24
planned_at: e077bd6
planned_at_date: 2026-10-24
---

# Classifier Models (Jev, Clef, GPT-6 Luna)

**Ratified 2026-10-24.** The maintainer chose pi's naming (decision 1) and
the full scope: every System One service pi supports, Cloudflare Workers AI
included, plus OpenAI Decisions (decision 2). Decisions 3–8 were ratified
as recommended. Phase 1 is a set of live wire probes: if a probe
contradicts the wire shapes recorded here, STOP and report rather than
improvising.

## Why

A new class of models answers typed questions about structured state instead
of generating text. TypeSafe's **Jev**, Cloudflare's **Clef** models (Clef,
Clef Flash, Clef Omni) and OpenAI's **GPT-6 Luna** are the current ones;
llama.cpp also serves local ones such as Julia-1 and Kev. The ecosystem has
no single name for them: models.dev types them `"decision"`, OpenRouter lists
them with the `decisions` output modality, OpenAI's endpoint is the
"Decisions API", and TypeSafe's protocol is "System One". pi calls them
**classifier models** with a `classify()` operation, and so does go-llm
(decision 1); the docs say "classifier (decision) models" so all the names
lead here.

You send JSON state (and optionally images) plus named questions, and get
back calibrated answers:

- `choice`: pick one labelled option; the answer carries every option's
  probability and a confidence;
- `score`: place the state on an ordered scale of levels; the answer carries
  the score and a confidence;
- `bool`: true or false against stated meanings; the answer is a probability.

Typical uses are routing, moderation, triage, approval detection and
evaluation: decisions an application now makes by prompting a chat model and
parsing its text. A classifier answers in one call, with probabilities
instead of prose, at a fraction of the cost (Jev lists $0.042 per million
input tokens, Clef Flash $0.038, GPT-6 Luna $0.10).

go-llm has no such operation, and its catalog sources do not list these
models: the captured models.dev `api.json` (2026-10-23) has no
`type: "decision"` rows, and OpenRouter's default `/models` excludes
`decisions` models (its only Jev row, `typesafe/jev-router`, is a chat
router).

## Reference: pi

Studied at `/home/peter/Dev/other/pi` HEAD `f321e5bdd`, package
`packages/ai`.

### Contract (`src/types.ts`)

- `ClassifierQuestion`: `choice` (`criteria: Record<value, description>`),
  `score` (`criteria: string[]`, ordered levels), `bool`
  (`criteria: { true, false }`), each with `instructions`.
- `ClassifierContext`: `state: JsonObject`, optional `images`,
  `questions: Record<name, ClassifierQuestion>`.
- `ClassifierAnswer`: choice `{ choice, probabilities, confidence }`, score
  `{ score, confidence }`, bool `{ probability }`.
- `ClassifierResult`: `answers`, optional `usage`, `stopReason`,
  `errorMessage`. pi returns errors in the result instead of throwing.
- `ClassifierModel`: `type: "classifier"`, `contextWindow`, optional System
  One `compat`. `ClassifierOptions.temperature` divides answer logits; System
  One and Decisions ignore it.

### The services pi supports

| pi provider | Classifier API | Endpoint | Models | Auth |
|---|---|---|---|---|
| `typesafe` | `typesafe-system-one` | `https://api.typesafe.ai/v1/systemone` | `jev-latest` | `TYPESAFE_API_KEY` |
| `openrouter` | `typesafe-system-one` | `https://openrouter.ai/api/v1/systemone` | `typesafe/jev-1.13`, `typesafe/jev-1.14`, `~typesafe/jev-latest`, `cloudflare/clef`, `cloudflare/clef-flash`, `cloudflare/clef-omni` | `OPENROUTER_API_KEY` |
| `cloudflare-workers-ai` | `cloudflare-workers-ai-system-one` | `https://api.cloudflare.com/client/v4/accounts/{account}/ai/run` | `@cf/cloudflare/clef`, `@cf/cloudflare/clef-flash`, `@cf/cloudflare/clef-omni`, `typesafe/jev`, `typesafe/jev-1.14.0` | `CLOUDFLARE_API_KEY` and `CLOUDFLARE_ACCOUNT_ID` |
| `vercel-ai-gateway` | `typesafe-system-one` | `https://ai-gateway.vercel.sh/typesafe/v1/systemone` | `typesafe-ai/jev` | `AI_GATEWAY_API_KEY` |
| `opencode` | `typesafe-system-one` | OpenCode Zen (pi's code says `/zen/v1/systemone`, its catalog `https://opencode.ai/inference/v1`; Phase 1 settles it) | `jev-1.13`, `jev-1.13-free` | `OPENCODE_API_KEY` |
| custom (llama.cpp ≥ 0.6.0) | `typesafe-system-one` | `<server>/v1/systemone` | whatever it serves (Julia-1, Kev) | optional key |
| `openai` | `openai-decisions` | `https://api.openai.com/v1/decisions` | `gpt-6-luna` | `OPENAI_API_KEY` only |

### System One protocol (`src/api/system-one-shared.ts`, `typesafe-system-one.ts`)

- Request: `{ model, state, questions: { <name>: { type, instructions,
  criteria } }, assets?, images? }`, bearer auth. Public `bool` is wire
  `noul` (answer field `noul`); `choice` and `score` pass through.
- Response: `{ answers: { <name>: … }, usage: { input_tokens,
  output_tokens } }`.
- Images, in one of three forms (`compat.imageInput`):
  - `assets` (TypeSafe; Jev on Workers AI): `{ "<image-1>": { type:
    "image", format: "png"|"jpeg", data_base64 } }`, the names placed before
    the state, `state: ["<image-1>", …, state]`. On TypeSafe's own API this
    is experimental and needs the `TypeSafe-Danger-Zone-Images: true` header
    (`compat.sendImageOptInHeader`); Workers AI does not.
  - `images` (Clef on Workers AI): up to four PNG/JPEG/WebP data URLs,
    beside the unchanged state.
  - `state-image-urls` (OpenRouter): `{ type: "image_url", image_url: { url
    } }` parts before the state. OpenRouter drops `assets` silently.
  - Detection when a model sets no `compat`: OpenRouter by provider or an
    `openrouter.ai` base URL; Clef by a model ID starting with
    `@cf/cloudflare/clef`; otherwise `assets` with the opt-in header.
- Vercel AI Gateway and OpenCode list their Jev models as text-only.

### Cloudflare Workers AI (`src/api/cloudflare-workers-ai-system-one.ts`)

- `POST <base>/run` with `{ model, input: <System One request> }`, where
  `<base>` is `https://api.cloudflare.com/client/v4/accounts/{account}/ai`.
- Response envelope: `{ success, errors, result }`. Cloudflare-hosted models
  (Clef) return `result: { model, answers, usage }`; third-party models (Jev)
  add a run record, `result: { state: "Completed", result: { answers,
  usage } }`. `success: false` carries `errors: [{ message }]`; a state other
  than `Completed` is an error.
- Auth (`src/providers/cloudflare-auth.ts`): the API key and the account ID
  resolve per field, the stored credential first (`key`, and
  `env.CLOUDFLARE_ACCOUNT_ID` in the auth-file entry), then the
  environment. pi's `/login` asks for both and stores `{ type: "api_key",
  key, env: { CLOUDFLARE_ACCOUNT_ID } }`.
- pi's Cloudflare provider also serves chat models (OpenAI-compatible at
  `<base>/v1`); that is outside this plan.

### OpenAI Decisions (`src/api/openai-decisions.ts`)

- `POST /v1/decisions` with `{ model, input, questions: [ … ] }`. `input` is
  the state as JSON text, or one user message with `input_text` plus up to
  128 `input_image` data URLs.
- Questions are an ordered list with `name`: `choice` → `choices: [{ value,
  description }]`, `score` → `levels: [{ label }]`, `bool` → `predicate`,
  whose meanings are appended to the instructions ("True means: …").
- Answers are a list keyed by `name`; choice probabilities are `[{ value,
  probability }]`; any `type: "refusal"` answer fails the result.
- Usage reports input tokens only, priced at the input rate with OpenAI's
  long-context multiplier. Only API keys work: Sign in with ChatGPT tokens
  are rejected, so pi hides `gpt-6-luna` under OAuth. A 504 from the
  Cloudflare gateway in front of api.openai.com (requests above roughly 600K
  tokens run past about five seconds) is not retried.

### Shared plumbing (`src/api/classifier-shared.ts`)

One bearer-auth JSON POST with provider retries; usage is recorded before
answers are parsed, so a billed request with malformed answers keeps its
usage.

### Catalog (`scripts/generate-models.ts`)

models.dev `models.json?type=decision` (TypeSafe's `jev-latest`), OpenRouter
`/models?output_modalities=decisions`, Vercel's evaluation-model catalog,
and hand-listed entries where no catalog exists (Workers AI, OpenCode,
`gpt-6-luna`).

### Not adopted

`llama-cpp-classify` (461 lines) turns a llama.cpp *chat* model into a
classifier through next-token log-probabilities. It is not System One and
stays deferred; llama.cpp's native `/v1/systemone` is covered.

## Evidence baseline and dispositions

| Finding | Disposition |
|---|---|
| Classification is a distinct operation: no messages, no streaming, typed answers. | **Implement** an optional `llm.Classifier` interface and an `llm.Classify` helper, discovered like `TokenCounter` (Phase 2). |
| FS §3 lists "embeddings, image generation, audio" as out of scope and says nothing about classification. | **Amend** FS §3 to admit classification as the one non-chat operation; the others stay out (Phase 0). |
| One protocol (System One) serves TypeSafe, OpenRouter, Vercel AI Gateway, OpenCode and llama.cpp. | **Implement** one engine with every service as a thin configuration (Phase 3). |
| Cloudflare Workers AI wraps System One in its own URL, envelope and two-part credential. | **Implement** a Cloudflare transport on the same engine (Phase 4). |
| OpenAI's Decisions API is a second protocol, API-key only. | **Implement** `Classify` on `providers/openai`; a Sign in with ChatGPT credential fails with `ErrUnsupported` (Phase 5). |
| `gpt-6-luna` appears in models.dev as an ordinary chat model under many providers, yet OpenAI also serves it as the Decisions classifier. | **Classification is a per-model capability**, not a model type: one model ID may serve both operations (decision 6). |
| The catalog sources omit decision models; pi hand-lists the ones no catalog carries. | **Extend** the snapshot with the catalogs that exist; no hand-listed rows (decision 7, Phase 6). |
| Three System One image forms. | **Implement** all three (Phases 3 and 4). |
| Cloudflare credentials need an account ID beside the key; pi keeps it in the entry's `env`. | **Expose** the entry's `env` map on `llm.AuthCredential` and resolve it per field, as pi does (Phase 4). |
| pi turns llama.cpp chat models into classifiers with logprobs. | **Defer.** |
| `temperature` is ignored by every System One service and by Decisions. | **Leave out** of the request until an implementation can apply it. |

## Decisions

1. **Naming: pi's.** *Ratified.* `llm.Classifier` (the optional provider
   interface), `Classify` (the operation), `ClassifyRequest` /
   `ClassifyResult`, `llm.CapabilityClassify`, `llm-cli classify`. Docs call
   them "classifier (decision) models" and mention models.dev's
   `type: "decision"`, OpenRouter's `decisions` modality, OpenAI's Decisions
   API and TypeSafe's System One.
2. **Scope: every System One service pi supports, plus OpenAI Decisions.**
   *Ratified.* TypeSafe, OpenRouter, Cloudflare Workers AI, Vercel AI
   Gateway, OpenCode Zen and any custom System One server (llama.cpp); all
   three image forms. FS §3 is amended to admit classification as the one
   non-chat operation; embeddings, image generation and audio stay out.
3. **Questions are an ordered slice**, `[]llm.Question` with a `Name` each,
   not a map. Decisions takes an ordered list and System One an object; a
   slice keeps requests deterministic (stable bytes and tests) and lets
   validation report duplicates. Answers come back as `map[string]Answer`
   keyed by name. *Ratified.*
4. **Errors are Go errors.** `Classify` returns `(*ClassifyResult, error)`
   with the usual sentinels (FS §16). When a billed response carries
   malformed answers, or Decisions refuses, the result still holds `Usage`
   (and any answers parsed) beside an error wrapping `ErrServer`
   (malformed) or `ErrContentFiltered` (refusal), rather than pi's
   error-in-result shape. *Ratified.*
5. **Package layout.** *Ratified:*
   - `providers/internal/systemone`: the engine (wire types, request
     building, answer parsing, the three image forms, an envelope hook);
   - `providers/systemone`: the public package. `New(baseURL, opts...)` is
     host-first, key optional and never reads the environment (llama.cpp,
     any System One server). Named constructors carry each service's base
     URL, provider name, image form, environment variable and credential
     entry: `NewTypeSafe`, `NewVercelAIGateway`, `NewOpenCode`. One package,
     because the protocol is one.
   - `providers/cloudflare`: Workers AI classification (its URL, envelope
     and two-part credential). Chat on Workers AI can join it later.
   - `Classify` methods on `openrouter` (System One, `state-image-urls`) and
     `openai` (Decisions).

   A classifier-only provider still implements `llm.Provider`:
   `Chat`/`ChatStream` fail with `ErrUnsupported`, and `Capabilities()` lists
   `CapabilityClassify` (plus `CapabilityImageInput` where images work).
   Not chosen: one package per service (`providers/typesafe`,
   `providers/vercel`, `providers/opencode`), as pi has a provider per
   service; that spreads one protocol over three near-empty packages.
6. **Capability and model metadata.** `llm.CapabilityClassify` is advertised
   by providers that implement `Classifier` and listed per model in
   `ModelInfo.Capabilities` from the catalog. There is no `ModelInfo.Kind`:
   a model ID can be chat-only, classifier-only or both. *Ratified.*
7. **Hand-listed catalog rows.** pi hand-lists the Workers AI, OpenCode and
   Decisions classifier models because no catalog carries them. go-llm's
   rule so far is that models.json is models.dev plus OpenRouter, with
   `overrides.json` empty. *Ratified:* keep the rule. Those models still
   work through `Classify`; their pricing and context window are unknown, so
   cost is reported as unknown. Not chosen: a small, source-linked
   classifier list in `overrides.json` (each row citing the provider's model
   page, as pi's comments do) so Clef and Luna costs are estimated.
8. **Middleware.** `llm.Classify(ctx, p, req)` finds a `Classifier` on `p`
   or the providers it wraps (following `Unwrap() Provider`), like
   `CountTokens`. In this plan chat middleware (`UsageTracker`, `otelllm`,
   `ContextGuard`) does not see classification calls; a classify middleware
   seam is future work. *Ratified*, to keep the first version small.
   *Follow-up (2026-10-24):* the seam landed after the plan —
   `Middleware.Classify`, with `UsageTracker` and `otelllm` handlers.

## Design

### Core types (package `llm`, new FS §13C "Classification")

```go
// Classifier is an optional Provider extension, discovered by type
// assertion (see Classify).
type Classifier interface {
	Classify(ctx context.Context, req *ClassifyRequest) (*ClassifyResult, error)
}

type ClassifyRequest struct {
	Model           string
	State           json.RawMessage // a JSON object
	Images          []ImagePart     // inline Data only; judged with State
	Questions       []Question
	ProviderOptions any
}

type QuestionType string // QuestionChoice, QuestionScore, QuestionBool

type Question struct {
	Name         string
	Type         QuestionType
	Instructions string
	Choices      []Choice // choice: at least two, unique values
	Levels       []string // score: at least two, ordered low to high
	True, False  string   // bool: what each answer means (optional)
}

type Choice struct{ Value, Description string }

type ClassifyResult struct {
	Provider, Model string
	Answers         map[string]Answer
	Usage           Usage
}

type Answer struct {
	Type          QuestionType
	Choice        string             // choice
	Probabilities map[string]float64 // choice: per value
	Score         float64            // score
	Confidence    float64            // choice and score
	Probability   float64            // bool: P(true)
}
```

Constructors keep call sites short: `llm.ChoiceQuestion(name, instructions,
choices...)`, `llm.ScoreQuestion(name, instructions, levels...)`,
`llm.BoolQuestion(name, instructions, ifTrue, ifFalse)`.

- **A bound model, `llm.ClassifierModel`**, the classifier counterpart of
  `llm.NewSession`, named after pi's `ClassifierModel` (the model object
  pi's `classify()` takes):

  ```go
  jev := llm.NewClassifierModel(p, "typesafe/jev-1.14",
      llm.WithClassifierQuestions(approved, category)) // optional defaults

  result, err := jev.Classify(ctx, state)            // the bound questions
  result, err = jev.Classify(ctx, state, sentiment)  // or these instead
  result, err = jev.ClassifyRequest(ctx, req)        // full control; Model filled in
  ```

  `state` is any value that marshals to a JSON object (a struct, a map, or
  `json.RawMessage`). Questions passed to `Classify` replace the bound
  defaults for that call. Options can also bind `ProviderOptions`; images
  are not bound, since they belong to each call (`ClassifyRequest` takes
  them). It works with any
  provider that implements `Classifier`, through `llm.Classify`, so it also
  finds the classifier through an `llm.Wrap` chain. Unlike `Session` it is
  immutable after construction and safe for concurrent use: classification
  keeps no history.
- **Validation** (`llm.ValidateClassifyRequest`, run before any network
  I/O): a model; `State` a JSON object; at least one question; unique,
  non-empty names; the per-type shape above; inline images with a media
  type. Failures wrap `ErrBadRequest`.
- **`llm.Classify`** follows `Unwrap()` like `CountTokens`; a provider
  without a classifier yields `ErrUnsupported`.
- **Answer checking:** every requested name answered with the requested
  type; extra answers ignored; probabilities and scores finite numbers.
- **Usage:** `InputTokens`/`OutputTokens` from the wire; `CostUSD`
  estimated from the model's catalog pricing (`CostSource` estimated), as
  for chat (FS §11), including OpenAI's long-context tier when the table has
  it; unknown pricing leaves the cost unknown.
- **Serialization:** canonical JSON for `ClassifyRequest`/`ClassifyResult`
  (FS §10A style) for logging, test fixtures and `llm-cli`. The CLI also
  accepts pi's object form of questions
  (`{"<name>": {"type", "instructions", "criteria"}}`), converted to the
  ordered slice.

### System One engine (`providers/internal/systemone`)

- Request building as recorded above; `bool` ↔ `noul`; `choice` criteria as
  an object of value → description; `score` criteria as the list of levels.
- Images in the configured form (`assets`, `images`, `state-image-urls`),
  with media-type and count checks before I/O, and the TypeSafe opt-in
  header where the service needs it. A model may override the form through
  provider options, as pi's `compat` allows.
- A service hook for the URL, request envelope and response envelope, so
  Cloudflare is a configuration rather than a fork.
- Response parsing: `answers` object, usage parsed before answers.
- Transport: the shared HTTP client, retries, error normalization
  (`providerutil`), wire capture and logging the chat engines use.

### Services (`providers/systemone`, `providers/cloudflare`, `openrouter`)

| Constructor | Provider name | Base URL | Images | Key |
|---|---|---|---|---|
| `systemone.New(baseURL)` | `systemone` (or `WithName`) | caller's | `assets`, no header (configurable) | optional, `WithAPIKey`/`WithCredentials`, never from the environment |
| `systemone.NewTypeSafe()` | `typesafe` | `https://api.typesafe.ai/v1` | `assets` + opt-in header | `TYPESAFE_API_KEY` or `WithCredentials` |
| `systemone.NewVercelAIGateway()` | `vercel-ai-gateway` | `https://ai-gateway.vercel.sh/typesafe/v1` | text only | `AI_GATEWAY_API_KEY` |
| `systemone.NewOpenCode()` | `opencode` | settled in Phase 1 | text only | `OPENCODE_API_KEY` |
| `cloudflare.New()` | `cloudflare-workers-ai` | `https://api.cloudflare.com/client/v4/accounts/{account}/ai` | `images` for Clef, `assets` for Jev | `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID`, or `WithAccountID` |
| `openrouter.Provider.Classify` | `openrouter` | `https://openrouter.ai/api/v1` | `state-image-urls` | existing |

- **OpenRouter serves both operations from one provider.** The existing
  `openrouter.Provider` keeps chatting and also implements `Classifier`,
  advertising `CapabilityClassify`, with the same key, `WithCredentials`
  entry, base URL and HTTP client. Its classifier models (Jev 1.13, Jev 1.14,
  `~typesafe/jev-latest`, Clef, Clef Flash, Clef Omni) appear in its live
  `Models()` listing with the `classify` capability, beside the chat models:

  ```go
  p, _ := openrouter.New() // OPENROUTER_API_KEY, or WithCredentials(store)
  resp, _ := p.Chat(ctx, chatReq) // chat models, as today

  jev := llm.NewClassifierModel(p, "typesafe/jev-1.14")
  result, _ := jev.Classify(ctx, state,
      llm.BoolQuestion("approved", "Does the user approve?", "Approval", "No approval"))
  ```

  `llm.Classify(ctx, p, req)` finds it through any `llm.Wrap` chain.
- **Credentials:** each service reads its auth-file entry by provider name,
  then its environment variable, as the chat providers do. For Cloudflare,
  `llm.AuthCredential` gains `Env map[string]string` (pi's entry `env`,
  already preserved by `authstore.File`); the account ID resolves from
  `WithAccountID`, then the entry's `env.CLOUDFLARE_ACCOUNT_ID`, then the
  environment, per field, as pi does.
- `TYPESAFE_API_KEY`, `AI_GATEWAY_API_KEY`, `OPENCODE_API_KEY`,
  `CLOUDFLARE_API_KEY` and `CLOUDFLARE_ACCOUNT_ID` join
  `llm.CredentialEnvNames`.

### OpenAI Decisions (`providers/openai`)

- `POST /v1/decisions` through the existing client, API-key credentials
  only; a stored Sign in with ChatGPT credential fails before I/O with
  `ErrUnsupported` and says why.
- Ordered questions; `bool` → `predicate` with the meanings appended exactly
  as pi formats them.
- Images: one user message, `input_text` state then up to 128
  `input_image` data URLs.
- `refusal` → usage plus `ErrContentFiltered`; 504 not retried, with an
  error that explains the large-input gateway limit.

### Catalog (`scripts/snapshot-models-table.ts`, `models.json`)

- Fetch models.dev `models.json?type=decision` (or the field `api.json`
  carries, as Phase 1 shows), OpenRouter
  `/models?output_modalities=decisions` and Vercel AI Gateway's model
  catalog; capture them under `scripts/model-sources/<date>/` like the
  existing sources.
- Rows gain the `classify` capability (and `image-input` where the source
  lists it), with pricing and context window. No hand-listed rows
  (decision 7).
- OpenRouter's live `Models()` also fetches the decisions listing, so
  classifier rows appear in `llm-cli models -p openrouter`.

### Testing support

- `llmtest.Provider` implements `Classifier`: `EnqueueClassifyResult`,
  `EnqueueClassifyError`, `ClassifyRequests()`.
- `RunConformance` gains a classify section for providers advertising
  `CapabilityClassify` (validation before I/O, answer-type checks, usage
  kept with a malformed-answer error).
- Golden request bytes and recorded responses per service from Phase 1.
- Live e2e scenarios (`internal/e2e`, `live` tag) per service whose
  credentials are in `gollm-test.json`; missing credentials skip visibly.

### Examples, CLI and docs

- `examples/classify`: dual-mode like the others (scripted `llmtest`
  offline; OpenRouter Jev with `OPENROUTER_API_KEY`).
- `llm-cli classify -p <provider> -m <model> --state state.json --questions
  questions.json [--image file]`, state from stdin when piped, answers as
  JSON, `--usage` for cost. Providers: `openrouter`, `typesafe`,
  `cloudflare`, `vercel-ai-gateway`, `opencode`, `systemone` (with
  `--base-url`), `openai`.
- `llm-cli login typesafe|cloudflare|vercel-ai-gateway|opencode` (API key;
  Cloudflare also prompts for the account ID, as pi does) and `llm-cli auth`
  rows for them.
- README: a "Classification" section and the providers table. Spec: FS §3,
  new §13C, §12 (capability), §13 (models listing), §17A (llmtest), §17C
  (credential `Env`), §19 (CLI); ARCH §1 (packages), §2 (types), §3
  (provider architecture).

## Phases

Each phase ends with the usual gates (`make check` lanes, coverage floors,
govulncheck), a clean-context review, and a signed commit.

### Phase 0: amend the spec

The decisions are ratified. Amend FS §3, add FS §13C and the ARCH entries
before code.

### Phase 1: live wire probes (STOP on contradiction)

In-process through the e2e credential loader; shapes printed, never
credentials. Recorded (redacted) in this plan.

1. OpenRouter `POST /api/v1/systemone` with `typesafe/jev-1.14` and with
   `cloudflare/clef-flash`, one question of each type: the `noul` mapping,
   criteria shapes, answer fields, `usage`.
2. OpenRouter image input with `state-image-urls`, and that `assets` is
   dropped without an error.
3. Catalogs: OpenRouter `?output_modalities=decisions`, models.dev
   `?type=decision`, Vercel AI Gateway's catalog: the fields the snapshot
   reads.
4. With credentials only, otherwise recorded as untested: TypeSafe (with and
   without the image opt-in header), Cloudflare Workers AI (`/ai/run`, both
   envelopes, Clef `images` and Jev `assets`), Vercel AI Gateway, OpenCode
   (which base URL), OpenAI Decisions (questions, predicate, refusal,
   input-only usage). A local llama.cpp ≥ 0.6.0 with a decision model, if
   one is available, for `systemone.New`.

Services without credentials ship offline-tested against pi's documented
shapes, and the FS says so. As of 2026-10-24, `gollm-test.json` holds an
OpenRouter API key (probes 1–3 run live, covering Jev and Clef through
OpenRouter), an OpenAI *Sign in with ChatGPT* credential (which Decisions
rejects, so probe 4's Decisions part needs an OpenAI API key), and no
TypeSafe, Cloudflare, Vercel or OpenCode credentials.

#### Phase 1 results (2026-10-24, through OpenRouter)

No probe contradicted the recorded shapes; several add to them.

- **Request.** `POST https://openrouter.ai/api/v1/systemone` with `{ model,
  state, questions }` as recorded. The wire's question types are `noul`,
  `choice`, `score` and `box`; `bool` is rejected (400, "Invalid
  discriminator value. Expected 'noul' | 'choice' | 'score' | 'box'"), which
  confirms the `bool` → `noul` mapping. A one-option `choice` is accepted
  (local validation still requires two).
- **Response.** `{ model, answers, usage, id, provider }`:
  - `model` is the resolved snapshot (`typesafe/jev-1.14-20261014`,
    `openai/gpt-6-luna-decisions-20261006`), `provider` the upstream
    (`TypeSafe`, `OpenAI`, `Perplexity`, `PrimeIntellect`, `SiliconFlow`),
    `id` a generation ID (`gen-dec-…`);
  - choice: `{ type, choice, probabilities: { value: p }, confidence }`;
  - **score adds to pi's shape**: `{ type, score, legend: { "0": level, … },
    probabilities: { "0": p, … }, confidence }`, and `score` is the
    expected level index, fractional when uncertain (`1.8352`);
  - noul: `{ type: "noul", noul: p }`;
  - **usage adds `cost`**: `{ input_tokens, output_tokens, cost }`, the
    native charge in USD.
- **Models.** Jev 1.14, Clef Flash, `openai/gpt-6-luna-decisions`,
  `perplexity/pplx-decider-flash-v1.2` and `jaredpalmer/kev-4b` all answered
  the same three-question request in the same shape. `liquid/d1` returned an
  upstream 502 (transient). A chat model ID is a 400 ("Model … does not
  exist").
- **Images.** `state-image-urls` works on Clef Flash (a red image classified
  red at 0.98). Jev 1.14 on OpenRouter rejects 2×2 and 256×256 PNGs with
  400 "Invalid image dimensions"; its size requirement is unknown and the
  error is passed through. `assets` is dropped silently, as pi documents.
  PrimeIntellect-served Clef rejects `image_url` parts ("does not accept
  \"image_url\" parts in state"), so image support also depends on the
  upstream OpenRouter routes to.
- **`box` (not in pi).** A fourth question type returning `{ type: "box",
  box: { top, bottom, left, right }, confidence }`, coordinates in [0, 1]: an
  image localization question. Answered for text state with near-zero
  confidence. *Deferred* (see Deferred): pi does not support it, and with
  Jev's image input failing here it cannot be validated end to end.
- **Catalogs.**
  - OpenRouter `GET /api/v1/models?output_modalities=decisions` lists 30
    classifier models with `context_length`, `architecture.input_modalities`
    (`image` where supported), and per-token `pricing.prompt`
    (`completion` 0, sometimes `input_cache_read`): Jev 1.13/1.14/latest,
    Clef, Clef Flash, Clef Omni, GPT-6 Luna Decisions, Perplexity deciders,
    Liquid d1 and d1-ultra, Upstage Solar Decide, Microsoft Decision-1,
    Inception Mercury Decide, Nace Drex, Respan Span, ScaleDown Classify,
    Baseten Elephant/Mammoth, Together tev1, Kev 4B.
  - models.dev `models.json?type=decision` lists 8 canonical rows
    (`typesafe/jev-latest`, `typesafe/jev-1.14.0`, `cloudflare/clef`,
    `cloudflare/clef-flash`, `liquid/d1`, …) with context windows and input
    modalities but **no pricing** and no serving provider.
  - Vercel AI Gateway `GET /v1/models` lists 14 `type: "evaluation"` models
    with per-token `pricing.input` and `context_window`
    (`typesafe-ai/jev`, `typesafe-ai/jev-1.14.0`, `liquid/d1`,
    `openai/gpt-6-luna-decisions`, `perplexity/pplx-decider-v1.2`, …); `vision`
    in `tags` marks image input.
- **Not probed** (no credentials): TypeSafe, Cloudflare Workers AI, Vercel
  AI Gateway and OpenCode classification, and OpenAI's own `/v1/decisions`
  (the stored OpenAI credential is a Sign in with ChatGPT login). They ship
  against pi's recorded shapes and the System One shapes above.

Consequences for the design:

- `Answer` gains `LevelProbabilities []float64` (score, by level index,
  from `probabilities` keyed `"0"…`); `Score` is a `float64` expected index.
- Usage takes OpenRouter's `usage.cost` as a native cost (`CostSource`
  native), as chat already does; elsewhere cost is estimated.
- `ClassifyResult` gains `ID` (the generation ID when the service returns
  one) and keeps the resolved `Model` from the response.
- The catalog's classifier rows come from OpenRouter's decisions listing
  (with pricing), Vercel's evaluation models (with pricing) and the
  `typesafe/*` rows of models.dev's decision listing for the `typesafe`
  provider (no pricing). These are upstream catalogs, not hand-listed rows,
  so decision 7 holds.

### Phase 2: core types, validation, `llm.Classify`, `llmtest`

Types, constructors, validation, `llm.Classify`, the bound
`llm.ClassifierModel`, `CapabilityClassify`, canonical JSON, usage and cost;
`llmtest` support. Tests: validation table,
`Unwrap` discovery, canonical round trip, the usage-with-error contract.

### Phase 3: System One engine, `providers/systemone`, OpenRouter

The engine with the `assets` and `state-image-urls` forms; `systemone.New`,
`NewTypeSafe`, `NewVercelAIGateway`, `NewOpenCode`;
`openrouter.Provider.Classify`. Offline golden tests, the conformance
section, live scenarios.

### Phase 4: Cloudflare Workers AI

`providers/cloudflare` with the envelope hook, the `images` form, the
two-part credential (`AuthCredential.Env`, `WithAccountID`), errors from
`success: false` and incomplete run states. Offline golden tests; live when
credentials exist.

### Phase 5: OpenAI Decisions

`openai.Provider.Classify` with API keys; OAuth rejection; predicate
formatting; images; refusal; 504. Offline golden tests; live when an API
key exists.

### Phase 6: catalog

Snapshot sources, `classify` capability rows, a refreshed `models.json` with
captured sources, OpenRouter's live listing, the capability in
`llm-cli models`.

### Phase 7: example, CLI, docs

`examples/classify` with its smoke test; `llm-cli classify`; login and auth
rows; README and spec sections.

## Implementation notes (2026-10-24)

All phases landed on `next`. Where the result differs from the text above:

- Conformance: a separate `llmtest.RunClassifierConformance` (Classifier
  found through `Wrap`, `CapabilityClassify`, validation before I/O,
  cancellation) instead of a section in `RunConformance`, whose chat
  contract classifier-only providers cannot meet. Answer-type checks and
  usage-with-malformed-answers are covered per provider.
- `ClassifyRequest.ProviderOptions` is carried but no provider defines
  classify options yet; a System One service's image form is provider-wide
  (`systemone.WithImageForm`), not per request.
- CLI and auth IDs follow pi: `cloudflare-workers-ai`, not `cloudflare`.
  `llm-cli login systemone` stores the optional key of a System One server.
- OpenCode's base URL is `https://opencode.ai/inference/v1`, as pi's model
  data and tests have it (pi's code comment says `/zen/v1`); unverified live,
  as no OpenCode credential was available.
- A Cloudflare base URL without `{account}` (a proxy) needs no account ID.
- OpenRouter's `Models` tolerates a failed decisions listing.
- An OpenAI Decisions refusal keeps the other questions' answers.
- `AuthCredential` gained `Env` (pi's entry `env`), so it is no longer
  comparable with `==`; OAuth refresh carries `Env` over.
- Live scenarios: `classify` on OpenRouter (Jev 1.14) and OpenAI (GPT-6
  Luna; skipped with a Sign in with ChatGPT credential). TypeSafe,
  Cloudflare, Vercel and OpenCode have no credentials in `gollm-test.json`
  and are covered offline by golden requests.

### After the Codex plan review (`reviews/review-of-plan-12-from-codex.md`)

Adopted: classifier cost only from classifier evidence (DC-1); answer range,
declared-choice and level checks that drop invalid answers, with refusal
precedence (DC-2/3); Decisions cache reads/writes split out of input tokens
(DC-8); collision-free asset names (DC-9); documented Decisions and Clef
limits as per-service preflight, the four-image limit moved out of the
shared encoder (DC-10/22); TypeSafe `max_tokens_exceeded` and Cloudflare
5007/3006/3007 codes (DC-11); one Cloudflare credential read per request
and a path-safe account ID (DC-12); the Cloudflare route recorded as pi's
observed shape, unverified (DC-13); CLI partial output and incomplete
Cloudflare auth status (DC-18/19); 504 and beta wording (DC-21).
`systemone.New` now defaults to llama.cpp's `images` data URLs, per its
server README. Already covered by the implementation: DC-4, DC-6, DC-7,
DC-14–DC-17. Not adopted: classify-specific `ProviderOptions` validation
(DC-5; no provider defines classify options) and a public
`ModelInfo.Classification` slot (DC-1's larger option; the narrower
evidence rule suffices while no catalog source gives one model separate
chat and classifier prices).

## Deferred

- The System One `box` question type (image localization), found in Phase 1;
  pi does not support it yet.
- `llama-cpp-classify`, pi's logprob classifier for chat models.
- `temperature`, until an implementation can apply it.
- Chat models on Cloudflare Workers AI.
- go-agent integration (a classifier-backed routing or guard tool), after
  this ships.

## Risks

- **Young APIs.** OpenAI Decisions is new and System One image
  input is experimental on TypeSafe; wire shapes may move. Golden fixtures
  make drift visible; the live scenarios catch it.
- **Overloaded model IDs.** `gpt-6-luna` is a chat model to most catalogs
  and a classifier to OpenAI's Decisions API; capability per model
  (decision 6) keeps this honest.
- **Credentials we may not have.** TypeSafe, Cloudflare, Vercel and OpenCode
  probes need accounts; without them those services rest on pi's shapes
  until someone runs the live scenarios.
- **Pricing gaps.** Without hand-listed rows (decision 7), Clef on Workers
  AI and Luna costs stay unknown; that is reported, not guessed.
