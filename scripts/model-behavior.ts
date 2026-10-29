// Per-model wire behavior, applied to every snapshot row like pi's compat
// rules in packages/ai/scripts/generate-models.ts. Upstream catalogs do not
// publish these facts, so each rule cites its evidence. go-llm reads the
// result as llm.ModelBehavior (FS §13B).

export type ModelBehavior = {
  // RoleSystem messages are accepted after the conversation has started.
  positional_instructions?: boolean;
  // Tool additions and removals are accepted in place.
  positional_tools?: boolean;
};

const dated = String.raw`(?:-\d{8})?`;

// Anthropic: pi's supportsAnthropicMidConvoSystemMessages, which also sets
// supportsMidConvoToolChanges (inline-tools-2026-09-15). Probed 2026-10-22
// (plan 11 Part A): all accept role "system" and tool_addition/tool_removal.
const anthropicPositional = [
  new RegExp(String.raw`^claude-opus-(?:4[.-]8|5(?:[.-]5)?)${dated}$`),
  new RegExp(String.raw`^claude-(?:sonnet|haiku)-5[.-]5${dated}$`),
  new RegExp(String.raw`^claude-(?:fable|mythos)-5(?:[.-]1)?${dated}$`),
];

// Anthropic models that take mid-conversation system text but reject
// tool_addition/tool_removal ("not supported on this model"). Probed
// 2026-10-22; pi's rule omits claude-sonnet-5 entirely.
const anthropicInstructionsOnly = [new RegExp(String.raw`^claude-sonnet-5${dated}$`)];

// OpenAI Responses: pi's OPENAI_MID_CONVO_SYSTEM_MESSAGE_MODEL_IDS and
// OPENAI_ADDITIONAL_TOOLS_MODEL_IDS. Probed 2026-10-22 on
// gpt-5.6-luna and gpt-6-luna under Sign in with ChatGPT.
// gpt-6.1-terra is included in both sets at pi commit 363ad78bf.
const openAIPositionalInstructions = new Set([
  "gpt-5.4",
  "gpt-5.4-mini",
  "gpt-5.4-pro",
  "gpt-5.5",
  "gpt-5.6-sol",
  "gpt-5.6-terra",
  "gpt-5.6-luna",
  "gpt-6-astra",
  "gpt-6-sol",
  "gpt-6-luna",
  "gpt-6.1-sol",
  "gpt-6.1-terra",
]);

// pi 81d6759fc removed GPT-5.4/5.5 from additional_tools support while
// retaining mid-conversation system messages. Replaying another model's
// tool history otherwise fails with "Missing namespace for function_call".
const openAIPositionalTools = new Set([
  "gpt-5.6-sol",
  "gpt-5.6-terra",
  "gpt-5.6-luna",
  "gpt-6-astra",
  "gpt-6-sol",
  "gpt-6-luna",
  "gpt-6.1-sol",
  "gpt-6.1-terra",
]);

export function modelBehavior(provider: string, id: string): ModelBehavior | undefined {
  switch (provider) {
    case "anthropic":
      if (anthropicPositional.some((rule) => rule.test(id))) {
        return { positional_instructions: true, positional_tools: true };
      }
      if (anthropicInstructionsOnly.some((rule) => rule.test(id))) return { positional_instructions: true };
      return undefined;
    case "openai":
      if (openAIPositionalInstructions.has(id)) {
        return openAIPositionalTools.has(id)
          ? { positional_instructions: true, positional_tools: true }
          : { positional_instructions: true };
      }
      return undefined;
    default:
      return undefined;
  }
}
