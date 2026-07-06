import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

type Pricing = {
  input_per_mtok?: number;
  output_per_mtok?: number;
  cache_read_per_mtok?: number;
  cache_write_per_mtok?: number;
};

type ModelRow = {
  provider: string;
  id: string;
  canonical_id?: string;
  display_name?: string;
  context_window?: number;
  max_output_tokens?: number;
  pricing?: Pricing;
};

const providers = new Set(["anthropic", "openai", "openrouter", "zai"]);
const root = resolve(dirname(new URL(import.meta.url).pathname), "..");
const outputPath = resolve(root, "models/models-table.json");
const overridesPath = resolve(root, "scripts/overrides.json");

async function main() {
  const [modelsDev, openRouter, overrides] = await Promise.all([
    fetchJSON("https://models.dev/api.json"),
    fetchJSON("https://openrouter.ai/api/v1/models"),
    readOverrides(),
  ]);

  const rows = new Map<string, ModelRow>();
  for (const row of rowsFromModelsDev(modelsDev)) {
    rows.set(key(row), row);
  }
  for (const row of rowsFromOpenRouter(openRouter)) {
    rows.set(key(row), mergeRow(rows.get(key(row)), row));
  }
  for (const row of overrides.models ?? []) {
    rows.set(key(row), mergeRow(rows.get(key(row)), row));
  }

  const document = {
    generated_at: new Date().toISOString(),
    models: [...rows.values()].sort((a, b) => key(a).localeCompare(key(b))),
  };

  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(document, null, 2)}\n`);
}

async function fetchJSON(url: string): Promise<unknown> {
  const response = await fetch(url, {
    headers: { "user-agent": "go-llm-model-snapshot" },
  });
  if (!response.ok) {
    throw new Error(`${url}: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function readOverrides(): Promise<{ models?: ModelRow[] }> {
  const raw = await readFile(overridesPath, "utf8");
  return JSON.parse(raw);
}

function rowsFromModelsDev(input: unknown): ModelRow[] {
  const rows: ModelRow[] = [];
  const root = input as Record<string, unknown>;
  for (const [provider, value] of Object.entries(root)) {
    if (!providers.has(provider)) continue;
    for (const [idHint, model] of modelEntries(value)) {
      const row = normalizeModel(provider, model, idHint);
      if (row) rows.push(row);
    }
  }
  return rows;
}

function rowsFromOpenRouter(input: unknown): ModelRow[] {
  const data = (input as { data?: unknown }).data;
  const rows: ModelRow[] = [];
  for (const model of Array.isArray(data) ? data : []) {
    const record = model as Record<string, unknown>;
    const id = stringValue(record.id);
    if (!id) continue;
    const pricingRecord = record.pricing as Record<string, unknown> | undefined;
    rows.push({
      provider: "openrouter",
      id,
      canonical_id: canonicalFromOpenRouterID(id),
      display_name: stringValue(record.name),
      context_window: numberValue(record.context_length),
      pricing: pricingFromOpenRouter(pricingRecord),
    });
  }
  return rows;
}

function modelEntries(value: unknown): Array<[string | undefined, unknown]> {
  if (Array.isArray(value)) return value.map((model) => [undefined, model]);
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    if (Array.isArray(record.models)) {
      return record.models.map((model) => [undefined, model]);
    }
    if (record.models && typeof record.models === "object") {
      return Object.entries(record.models as Record<string, unknown>);
    }
    return Object.entries(record);
  }
  return [];
}

function normalizeModel(provider: string, model: unknown, idHint?: string): ModelRow | undefined {
  if (!model || typeof model !== "object") return undefined;
  const record = model as Record<string, unknown>;
  const limit = record.limit as Record<string, unknown> | undefined;
  const id = stringValue(record.id) ?? idHint ?? stringValue(record.name) ?? stringValue(record.slug);
  if (!id) return undefined;
  return {
    provider,
    id,
    canonical_id: canonicalIDForProviderModel(provider, id, stringValue(record.canonical_id)),
    display_name: stringValue(record.display_name) ?? stringValue(record.name),
    context_window: numberValue(record.context_window ?? record.context_length ?? limit?.context ?? limit?.context_window),
    max_output_tokens: numberValue(record.max_output_tokens ?? record.max_output ?? limit?.output ?? limit?.max_output),
    pricing: pricingFromRecord((record.pricing ?? record.cost) as Record<string, unknown> | undefined),
  };
}

function pricingFromRecord(record?: Record<string, unknown>): Pricing | undefined {
  if (!record) return undefined;
  const pricing: Pricing = {
    input_per_mtok: numberValue(record.input_per_mtok ?? record.input ?? record.prompt),
    output_per_mtok: numberValue(record.output_per_mtok ?? record.output ?? record.completion),
    cache_read_per_mtok: numberValue(record.cache_read_per_mtok ?? record.cache_read ?? record.input_cache_read),
    cache_write_per_mtok: numberValue(record.cache_write_per_mtok ?? record.cache_write ?? record.input_cache_write),
  };
  return compactPricing(pricing);
}

function pricingFromOpenRouter(record?: Record<string, unknown>): Pricing | undefined {
  if (!record) return undefined;
  const perTokenToMTok = (value: unknown) => {
    const n = numberValue(value);
    return n === undefined ? undefined : n * 1_000_000;
  };
  return compactPricing({
    input_per_mtok: perTokenToMTok(record.prompt),
    output_per_mtok: perTokenToMTok(record.completion),
    cache_read_per_mtok: perTokenToMTok(record.input_cache_read),
    cache_write_per_mtok: perTokenToMTok(record.input_cache_write),
  });
}

function compactPricing(pricing: Pricing): Pricing | undefined {
  const out: Pricing = {};
  for (const [name, value] of Object.entries(pricing)) {
    if (typeof value === "number" && Number.isFinite(value)) {
      (out as Record<string, number>)[name] = value;
    }
  }
  return Object.keys(out).length === 0 ? undefined : out;
}

function mergeRow(base: ModelRow | undefined, override: ModelRow): ModelRow {
  const out: ModelRow = base ? { ...base } : { provider: override.provider, id: override.id };
  for (const [name, value] of Object.entries(override)) {
    if (name === "pricing") continue;
    if (value !== undefined) {
      (out as Record<string, unknown>)[name] = value;
    }
  }
  const pricing = mergePricing(base?.pricing, override.pricing);
  if (pricing) out.pricing = pricing;
  return out;
}

function mergePricing(base: Pricing | undefined, override: Pricing | undefined): Pricing | undefined {
  if (!base && !override) return undefined;
  const out: Pricing = base ? { ...base } : {};
  if (override) {
    for (const [name, value] of Object.entries(override)) {
      if (value !== undefined) {
        (out as Record<string, number>)[name] = value;
      }
    }
  }
  return compactPricing(out);
}

function key(row: ModelRow): string {
  return `${row.provider}/${row.id}`;
}

function canonicalFromOpenRouterID(id: string): string | undefined {
  return id.includes("/") ? id : undefined;
}

function canonicalIDForProviderModel(provider: string, id: string, canonicalID?: string): string | undefined {
  if (!canonicalID || canonicalID === `${provider}/${id}`) return undefined;
  return canonicalID;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

function numberValue(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value !== "") {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
