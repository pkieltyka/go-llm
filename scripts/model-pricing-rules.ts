// Documented pricing rules the upstream catalogs do not carry, applied to
// every snapshot row like model-behavior.ts. go-llm reads the result as
// llm.ModelPricing.CacheWritesAsInput (FS §11).

// OpenAI charges no separate cache write before the GPT-5.6 family: a
// written token bills as uncached input. OpenAI's prompt-caching guide
// (https://developers.openai.com/api/docs/guides/prompt-caching, read
// 2026-10-28) lists "No additional cache-write charge" for GPT-5.5 and
// "Other earlier models", and OpenRouter's prompt-caching guide
// (https://openrouter.ai/docs/guides/best-practices/prompt-caching, same
// day) says cache writes have "no cost on models before the GPT-5.6
// family". GPT-5.6 and later charge 1.25x or 2.5x and are not listed. This
// is an exact allowlist of chat families plus dated snapshots: pro, codex,
// :batch, image/audio/realtime/embedding models and gpt-oss (third-party
// hosted on OpenRouter) are left out (plan 14 A0).
const openAIWritesAsInput = new Set([
  "gpt-4o",
  "gpt-4o-mini",
  "gpt-4.1",
  "gpt-4.1-mini",
  "gpt-4.1-nano",
  "gpt-5",
  "gpt-5-mini",
  "gpt-5-nano",
  "gpt-5.1",
  "gpt-5.2",
  "gpt-5.4",
  "gpt-5.4-mini",
  "gpt-5.4-nano",
  "gpt-5.5",
]);

const datedSnapshot = /-\d{4}-\d{2}-\d{2}$/;

export function cacheWritesAsInput(provider: string, id: string): boolean {
  let model: string;
  switch (provider) {
    case "openai":
      model = id;
      break;
    case "openrouter":
      if (!id.startsWith("openai/")) return false;
      model = id.slice("openai/".length);
      break;
    default:
      return false;
  }
  return openAIWritesAsInput.has(model.replace(datedSnapshot, ""));
}
