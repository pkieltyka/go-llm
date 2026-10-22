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
// OPENAI_ADDITIONAL_TOOLS_MODEL_IDS (the same set). Probed 2026-10-22 on
// gpt-5.6-luna and gpt-6-luna under Sign in with ChatGPT.
const openAIPositional = new Set([
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
      return openAIPositional.has(id) ? { positional_instructions: true, positional_tools: true } : undefined;
    default:
      return undefined;
  }
}
