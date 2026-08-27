import { Buffer } from "node:buffer";
import { createHash, randomUUID } from "node:crypto";
import { chmod, mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync, gzipSync } from "node:zlib";

export type Pricing = {
  input_per_mtok?: number;
  output_per_mtok?: number;
  cache_read_per_mtok?: number;
  cache_write_per_mtok?: number;
  tiers?: PricingTier[];
};

export type PricingTier = {
  input_tokens_above: number;
  input_per_mtok?: number;
  output_per_mtok?: number;
  cache_read_per_mtok?: number;
  cache_write_per_mtok?: number;
};

export type ModelRow = {
  provider: string;
  id: string;
  canonical_id?: string;
  display_name?: string;
  context_window?: number;
  max_output_tokens?: number;
  pricing?: Pricing;
  supported_efforts?: string[];
};

export type SnapshotDocument = {
  schema_version: number;
  generator: string;
  generated_at: string;
  sources: SnapshotSource[];
  models: ModelRow[];
};

export type SnapshotSource = {
  id: string;
  url: string;
  sha256: string;
  etag?: string;
};

export type SourceMinimums = {
  modelsDev: Record<Provider, number>;
  openRouter: number;
};

export type ReadJSONOptions = {
  timeoutMs?: number;
  maxBytes?: number;
  fetchImpl?: typeof fetch;
};

type Provider = (typeof currentProviders)[number];
type JSONRecord = Record<string, unknown>;

const currentProviders = ["anthropic", "openai", "openrouter"] as const;
const currentProviderSet = new Set<string>(currentProviders);
const productionMinimums: SourceMinimums = {
  modelsDev: { anthropic: 5, openai: 10, openrouter: 50 },
  openRouter: 50,
};
const fixtureMinimums: SourceMinimums = {
  modelsDev: { anthropic: 1, openai: 1, openrouter: 1 },
  openRouter: 1,
};
const maximumIdentityLossRatio = 0.25;
const minimumMaterialIdentityLoss = 2;
const maximumMetadataLossRatio = 0.5;
const minimumMaterialFieldLoss = 2;
const minimumMaterialMetadataLoss = 3;
const remoteSourceTimeoutMs = 15_000;
const remoteSourceMaxBytes = 32 << 20;
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pricingRateNames = [
  "input_per_mtok",
  "output_per_mtok",
  "cache_read_per_mtok",
  "cache_write_per_mtok",
] as const;

type BuildOptions = {
  generatedAt?: string;
  minimums?: SourceMinimums;
  sources?: SnapshotSource[];
};

type CLIOptions = {
  modelsDevSource: string;
  openRouterSource: string;
  overridesSource: string;
  outputPath: string;
  allowDestructive: boolean;
  dryRun: boolean;
  verify: boolean;
  minimums?: SourceMinimums;
  generatedAt?: string;
  modelsDevURL: string;
  openRouterURL: string;
  overridesURL: string;
  captureDirectory?: string;
};

type LoadedJSON = {
  value: unknown;
  bytes: Uint8Array;
};

const snapshotSchemaVersion = 1;
const snapshotGenerator = "go-llm-model-snapshot/v1";
const modelsDevURL = "https://models.dev/api.json";
const openRouterURL = "https://openrouter.ai/api/v1/models";

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const [modelsDev, openRouter, overrides] = await Promise.all([
    loadJSON(options.modelsDevSource),
    loadJSON(options.openRouterSource),
    loadJSON(options.overridesSource),
  ]);
  const previous = await readSnapshot(options.outputPath);
  const document = buildSnapshot(modelsDev.value, openRouter.value, overrides.value, {
    generatedAt: options.generatedAt ?? (options.verify ? previous?.generated_at : undefined),
    minimums: options.minimums,
    sources: [
      provenance("models.dev", options.modelsDevURL, modelsDev),
      provenance("openrouter", options.openRouterURL, openRouter),
      provenance("overrides", options.overridesURL, overrides),
    ],
  });
  assertNonDestructive(previous, document, options.allowDestructive);
  if (options.captureDirectory) {
    await Promise.all([
      atomicWriteBytes(resolve(options.captureDirectory, "models-dev.json.gz"), gzipSync(modelsDev.bytes, { level: 9 })),
      atomicWriteBytes(resolve(options.captureDirectory, "openrouter-models.json.gz"), gzipSync(openRouter.bytes, { level: 9 })),
    ]);
  }

  if (options.verify) {
    const expected = serializeSnapshot(document);
    const actual = await readFile(options.outputPath);
    if (!actual.equals(expected)) throw new Error(`${options.outputPath} is not reproducible from its captured sources`);
    console.log(snapshotSummary(document, "verified"));
    return;
  }

  if (options.dryRun) {
    console.log(snapshotSummary(document, "validated"));
    return;
  }
  await atomicWriteSnapshot(options.outputPath, document);
  console.log(snapshotSummary(document, "wrote"));
}

export function buildSnapshot(
  modelsDev: unknown,
  openRouter: unknown,
  overrides: unknown,
  options: BuildOptions = {},
): SnapshotDocument {
  const minimums = options.minimums ?? productionMinimums;
  const rows = new Map<string, ModelRow>();
  for (const row of rowsFromModelsDev(modelsDev, minimums.modelsDev)) {
    rows.set(key(row), row);
  }
  for (const row of rowsFromOpenRouter(openRouter, minimums.openRouter)) {
    rows.set(key(row), mergeRow(rows.get(key(row)), row));
  }
  for (const row of rowsFromOverrides(overrides)) {
    rows.set(key(row), mergeRow(rows.get(key(row)), row));
  }

  const models = [...rows.values()].map(finalizePricing).sort((a, b) => compareStrings(key(a), key(b)));
  validateOutputProviders(models);
  const document = {
    schema_version: snapshotSchemaVersion,
    generator: snapshotGenerator,
    generated_at: options.generatedAt ?? new Date().toISOString(),
    sources: options.sources ?? inMemoryProvenance(modelsDev, openRouter, overrides),
    models,
  };
  validateSnapshotDocument(document);
  return document;
}

function provenance(id: string, url: string, loaded: LoadedJSON): SnapshotSource {
  return { id, url, sha256: sha256(loaded.bytes) };
}

function inMemoryProvenance(modelsDev: unknown, openRouter: unknown, overrides: unknown): SnapshotSource[] {
  return [
    memoryProvenance("models.dev", modelsDev),
    memoryProvenance("openrouter", openRouter),
    memoryProvenance("overrides", overrides),
  ];
}

function memoryProvenance(id: string, value: unknown): SnapshotSource {
  const bytes = Buffer.from(JSON.stringify(value));
  return { id, url: `memory:${id}`, sha256: sha256(bytes) };
}

function compactSource(source: SnapshotSource): SnapshotSource {
  return Object.fromEntries(Object.entries(source).filter(([, value]) => value !== undefined)) as SnapshotSource;
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

export function assertNonDestructive(
  previous: SnapshotDocument | undefined,
  next: SnapshotDocument,
  allowDestructive = false,
): void {
  if (!previous) return;

  const problems: string[] = [];
  for (const provider of currentProviders) {
    const before = rowsByID(previous.models, provider);
    const after = rowsByID(next.models, provider);
    if (before.size === 0) continue;

    const removed = [...before.keys()].filter((id) => !after.has(id));
    const added = [...after.keys()].filter((id) => !before.has(id));
    const retained = [...before.keys()].filter((id) => after.has(id));
    if (retained.length === 0) {
      problems.push(
        `${provider}: all ${before.size} previous model IDs were removed or replaced ` +
          `(${added.length} new; removed: ${sampleIDs(removed)})`,
      );
      continue;
    }

    const identityLossRatio = removed.length / before.size;
    if (removed.length >= minimumMaterialIdentityLoss && identityLossRatio > maximumIdentityLossRatio) {
      problems.push(
        `${provider}: removed or replaced ${removed.length}/${before.size} model IDs ` +
          `(${formatPercent(identityLossRatio)}; ${added.length} new; removed: ${sampleIDs(removed)})`,
      );
    }

    problems.push(...metadataLossProblems(provider, before, after, retained));
  }
  if (problems.length > 0 && !allowDestructive) {
    throw new Error(
      `refusing destructive model snapshot update:\n- ${problems.join("\n- ")}\n` +
        "rerun with --allow-destructive after reviewing the upstream change",
    );
  }
}

const metadataFields = [
  ["canonical_id", (row: ModelRow) => row.canonical_id !== undefined],
  ["display_name", (row: ModelRow) => row.display_name !== undefined],
  ["context_window", (row: ModelRow) => row.context_window !== undefined],
  ["max_output_tokens", (row: ModelRow) => row.max_output_tokens !== undefined],
  ["pricing.input_per_mtok", (row: ModelRow) => row.pricing?.input_per_mtok !== undefined],
  ["pricing.output_per_mtok", (row: ModelRow) => row.pricing?.output_per_mtok !== undefined],
  ["pricing.cache_read_per_mtok", (row: ModelRow) => row.pricing?.cache_read_per_mtok !== undefined],
  ["pricing.cache_write_per_mtok", (row: ModelRow) => row.pricing?.cache_write_per_mtok !== undefined],
  ["pricing.tiers", (row: ModelRow) => row.pricing?.tiers !== undefined],
  ["supported_efforts", (row: ModelRow) => row.supported_efforts !== undefined],
] as const;

function metadataLossProblems(
  provider: Provider,
  before: Map<string, ModelRow>,
  after: Map<string, ModelRow>,
  retained: string[],
): string[] {
  const fieldProblems: string[] = [];
  let totalBefore = 0;
  let totalLost = 0;
  for (const [name, present] of metadataFields) {
    let fieldBefore = 0;
    let fieldLost = 0;
    for (const id of retained) {
      const oldRow = before.get(id)!;
      const newRow = after.get(id)!;
      if (!present(oldRow)) continue;
      fieldBefore++;
      totalBefore++;
      if (!present(newRow)) {
        fieldLost++;
        totalLost++;
      }
    }
    const ratio = fieldBefore === 0 ? 0 : fieldLost / fieldBefore;
    if (fieldLost >= minimumMaterialFieldLoss && ratio > maximumMetadataLossRatio) {
      fieldProblems.push(
        `${provider}: lost ${name} for ${fieldLost}/${fieldBefore} retained models (${formatPercent(ratio)})`,
      );
    }
  }

  if (fieldProblems.length > 0) return fieldProblems;
  const aggregateRatio = totalBefore === 0 ? 0 : totalLost / totalBefore;
  if (totalLost >= minimumMaterialMetadataLoss && aggregateRatio > maximumMetadataLossRatio) {
    return [
      `${provider}: lost ${totalLost}/${totalBefore} metadata values across retained models ` +
        `(${formatPercent(aggregateRatio)})`,
    ];
  }
  return [];
}

function rowsByID(rows: ModelRow[], provider: Provider): Map<string, ModelRow> {
  return new Map(rows.filter((row) => row.provider === provider).map((row) => [row.id, row]));
}

function sampleIDs(ids: string[]): string {
  const sorted = [...ids].sort();
  const sample = sorted.slice(0, 5).join(", ");
  return sorted.length > 5 ? `${sample}, ...` : sample;
}

function formatPercent(ratio: number): string {
  return `${(ratio * 100).toFixed(1)}%`;
}

export async function persistSnapshot(
  outputPath: string,
  document: SnapshotDocument,
  allowDestructive = false,
): Promise<void> {
  assertNonDestructive(await readSnapshot(outputPath), document, allowDestructive);
  await atomicWriteSnapshot(outputPath, document);
}

export function mergeRow(base: ModelRow | undefined, override: ModelRow): ModelRow {
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

function rowsFromModelsDev(input: unknown, minimums: Record<Provider, number>): ModelRow[] {
  const source = objectValue(input, "models.dev root");
  const rows: ModelRow[] = [];
  for (const provider of currentProviders) {
    const providerRecord = objectValue(source[provider], `models.dev provider ${provider}`);
    const models = objectValue(providerRecord.models, `models.dev provider ${provider}.models`);
    const entries = Object.entries(models);
    if (entries.length < minimums[provider]) {
      throw new Error(
        `models.dev provider ${provider}: ${entries.length} models, want at least ${minimums[provider]}`,
      );
    }
    for (const [idHint, value] of entries) {
      rows.push(normalizeModelsDevModel(provider, idHint, value));
    }
  }
  return rows;
}

function normalizeModelsDevModel(provider: Provider, idHint: string, value: unknown): ModelRow {
  const record = objectValue(value, `models.dev ${provider}/${idHint}`);
  const id = optionalString(record, "id", `models.dev ${provider}/${idHint}`) ?? idHint;
  if (id === "") throw new Error(`models.dev ${provider}: empty model id`);
  const limit = optionalObject(record, "limit", `models.dev ${provider}/${id}`);
  const cost = optionalObject(record, "cost", `models.dev ${provider}/${id}`);
  return compactRow({
    provider,
    id,
    canonical_id: canonicalIDForProviderModel(
      provider,
      id,
      optionalString(record, "canonical_id", `models.dev ${provider}/${id}`),
    ),
    display_name:
      optionalString(record, "display_name", `models.dev ${provider}/${id}`) ??
      optionalString(record, "name", `models.dev ${provider}/${id}`),
    context_window:
      optionalAvailablePositiveNumber(record, ["context_window", "context_length"]) ??
      optionalAvailablePositiveNumber(limit, ["context", "context_window"]),
    max_output_tokens:
      optionalAvailablePositiveNumber(record, ["max_output_tokens", "max_output"]) ??
      optionalAvailablePositiveNumber(limit, ["output", "max_output"]),
    pricing: pricingFromUpstreamRecord(
      cost ?? optionalObject(record, "pricing", `models.dev ${provider}/${id}`),
      `models.dev ${provider}/${id}.cost`,
    ),
    supported_efforts: reasoningEfforts(record, `models.dev ${provider}/${id}`),
  });
}

function rowsFromOpenRouter(input: unknown, minimumRows: number): ModelRow[] {
  const source = objectValue(input, "OpenRouter root");
  if (!Array.isArray(source.data)) {
    throw new Error("OpenRouter root.data must be an array");
  }
  if (source.data.length < minimumRows) {
    throw new Error(`OpenRouter: ${source.data.length} models, want at least ${minimumRows}`);
  }
  return source.data.map((value, index) => {
    const label = `OpenRouter data[${index}]`;
    const record = objectValue(value, label);
    const id = requiredString(record, "id", label);
    const pricing = optionalObject(record, "pricing", label);
    const topProvider = optionalObject(record, "top_provider", label);
    optionalString(record, "canonical_slug", label);
    return compactRow({
      provider: "openrouter",
      id,
      canonical_id: canonicalFromOpenRouterID(id),
      display_name: optionalString(record, "name", label),
      context_window:
        optionalAvailablePositiveNumber(record, ["context_length"]) ??
        optionalAvailablePositiveNumber(topProvider, ["context_length"]),
      max_output_tokens: optionalAvailablePositiveNumber(topProvider, ["max_completion_tokens"]),
      pricing: pricingFromOpenRouter(pricing),
    });
  });
}

function rowsFromOverrides(input: unknown): ModelRow[] {
  const source = objectValue(input, "overrides root");
  if (source.models === undefined) return [];
  if (!Array.isArray(source.models)) throw new Error("overrides.models must be an array");
  return source.models.map((value, index) => {
    const label = `overrides.models[${index}]`;
    const record = objectValue(value, label);
    const provider = requiredString(record, "provider", label);
    if (!currentProviderSet.has(provider)) {
      throw new Error(`${label}.provider ${JSON.stringify(provider)} is not a current provider`);
    }
    const pricing = optionalObject(record, "pricing", label);
    return compactRow({
      provider,
      id: requiredString(record, "id", label),
      canonical_id: optionalString(record, "canonical_id", label),
      display_name: optionalString(record, "display_name", label),
      context_window: optionalPositiveNumber(record, ["context_window"], label),
      max_output_tokens: optionalPositiveNumber(record, ["max_output_tokens"], label),
      pricing: pricingFromRecord(pricing, label),
      supported_efforts: optionalEffortList(record, "supported_efforts", label),
    });
  });
}

function pricingFromRecord(record: JSONRecord | undefined, label: string): Pricing | undefined {
  if (!record) return undefined;
  const pricing = compactPricing({
    input_per_mtok: optionalNonNegativeNumber(record, ["input_per_mtok", "input", "prompt"], `${label}.cost`),
    output_per_mtok: optionalNonNegativeNumber(record, ["output_per_mtok", "output", "completion"], `${label}.cost`),
    cache_read_per_mtok: optionalNonNegativeNumber(
      record,
      ["cache_read_per_mtok", "cache_read", "input_cache_read"],
      `${label}.cost`,
    ),
    cache_write_per_mtok: optionalNonNegativeNumber(
      record,
      ["cache_write_per_mtok", "cache_write", "input_cache_write"],
      `${label}.cost`,
    ),
    tiers: overridePricingTiers(record, `${label}.pricing`),
  });
  return pricing;
}

function pricingFromUpstreamRecord(record: JSONRecord | undefined, label: string): Pricing | undefined {
  if (!record) return undefined;
  return compactPricing({
    input_per_mtok: optionalAvailableNonNegativeNumber(record, ["input_per_mtok", "input", "prompt"]),
    output_per_mtok: optionalAvailableNonNegativeNumber(record, ["output_per_mtok", "output", "completion"]),
    cache_read_per_mtok: optionalAvailableNonNegativeNumber(record, [
      "cache_read_per_mtok",
      "cache_read",
      "input_cache_read",
    ]),
    cache_write_per_mtok: optionalAvailableNonNegativeNumber(record, [
      "cache_write_per_mtok",
      "cache_write",
      "input_cache_write",
    ]),
    tiers: upstreamContextPricingTiers(record, label),
  });
}

function upstreamContextPricingTiers(record: JSONRecord, label: string): PricingTier[] | undefined {
  const value = record.tiers;
  if (value === undefined || value === null) return undefined;
  if (!Array.isArray(value)) throw new Error(`${label}.tiers must be an array`);
  const tiers: PricingTier[] = [];
  for (let index = 0; index < value.length; index++) {
    const tierLabel = `${label}.tiers[${index}]`;
    const source = objectValue(value[index], tierLabel);
    const discriminator = optionalObject(source, "tier", tierLabel);
    if (optionalString(discriminator ?? {}, "type", `${tierLabel}.tier`) !== "context") continue;
    const threshold = optionalPositiveNumber(discriminator, ["size"], `${tierLabel}.tier`);
    if (threshold === undefined || !Number.isSafeInteger(threshold)) {
      throw new Error(`${tierLabel}.tier.size must be a positive safe integer`);
    }
    tiers.push({
      input_tokens_above: threshold,
      input_per_mtok: optionalNonNegativeNumber(source, ["input_per_mtok", "input", "prompt"], tierLabel),
      output_per_mtok: optionalNonNegativeNumber(source, ["output_per_mtok", "output", "completion"], tierLabel),
      cache_read_per_mtok: optionalNonNegativeNumber(
        source,
        ["cache_read_per_mtok", "cache_read", "input_cache_read"],
        tierLabel,
      ),
      cache_write_per_mtok: optionalNonNegativeNumber(
        source,
        ["cache_write_per_mtok", "cache_write", "input_cache_write"],
        tierLabel,
      ),
    });
  }
  return tiers.length === 0 ? undefined : tiers;
}

function overridePricingTiers(record: JSONRecord, label: string): PricingTier[] | undefined {
  const value = record.tiers;
  if (value === undefined) return undefined;
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${label}.tiers must be a non-empty array`);
  }
  return value.map((entry, index) => {
    const tierLabel = `${label}.tiers[${index}]`;
    const source = objectValue(entry, tierLabel);
    const threshold = optionalPositiveNumber(source, ["input_tokens_above"], tierLabel);
    if (threshold === undefined || !Number.isSafeInteger(threshold)) {
      throw new Error(`${tierLabel}.input_tokens_above must be a positive safe integer`);
    }
    const tier: PricingTier = { input_tokens_above: threshold };
    for (const name of pricingRateNames) {
      const rate = optionalNonNegativeNumber(source, [name], tierLabel);
      if (rate === undefined) throw new Error(`${tierLabel}.${name} is required`);
      tier[name] = rate;
    }
    return tier;
  });
}

function reasoningEfforts(record: JSONRecord, label: string): string[] | undefined {
  const value = record.reasoning_options;
  if (value === undefined || value === null) return undefined;
  if (!Array.isArray(value)) throw new Error(`${label}.reasoning_options must be an array`);
  const found = new Set<string>();
  for (let index = 0; index < value.length; index++) {
    const optionLabel = `${label}.reasoning_options[${index}]`;
    const option = objectValue(value[index], optionLabel);
    if (optionalString(option, "type", optionLabel) !== "effort") continue;
    const values = option.values;
    if (values === undefined || values === null) continue;
    if (!Array.isArray(values)) throw new Error(`${optionLabel}.values must be an array`);
    for (const effort of values) {
      if (effort === null || effort === "default") continue;
      if (typeof effort !== "string" || !effortScale.includes(effort as (typeof effortScale)[number])) {
        throw new Error(`${label} has unknown reasoning effort ${JSON.stringify(effort)}`);
      }
      found.add(effort);
    }
  }
  const efforts = effortScale.filter((effort) => found.has(effort));
  return efforts.length === 0 ? undefined : [...efforts];
}

function pricingFromOpenRouter(record: JSONRecord | undefined): Pricing | undefined {
  if (!record) return undefined;
  const perTokenToMTok = (names: string[]) => {
    const value = optionalAvailableNonNegativeNumber(record, names);
    if (value === undefined) return undefined;
    const perMillion = value * 1_000_000;
    return Number.isFinite(perMillion) ? perMillion : undefined;
  };
  return compactPricing({
    input_per_mtok: perTokenToMTok(["prompt"]),
    output_per_mtok: perTokenToMTok(["completion"]),
    cache_read_per_mtok: perTokenToMTok(["input_cache_read"]),
    cache_write_per_mtok: perTokenToMTok(["input_cache_write"]),
  });
}

function compactRow(row: ModelRow): ModelRow {
  return Object.fromEntries(Object.entries(row).filter(([, value]) => value !== undefined)) as ModelRow;
}

function compactPricing(pricing: Pricing): Pricing | undefined {
  const out = Object.fromEntries(Object.entries(pricing).filter(([, value]) => value !== undefined)) as Pricing;
  return Object.keys(out).length === 0 ? undefined : out;
}

function mergePricing(base: Pricing | undefined, override: Pricing | undefined): Pricing | undefined {
  if (!base && !override) return undefined;
  const out: Pricing = base ? { ...base } : {};
  for (const [name, value] of Object.entries(override ?? {})) {
    if (value !== undefined) {
      if (name === "tiers") out.tiers = structuredClone(value as PricingTier[]);
      else (out as Record<string, number>)[name] = value as number;
    }
  }
  return compactPricing(out);
}

function finalizePricing(row: ModelRow): ModelRow {
  if (!row.pricing?.tiers) return row;
  const base = row.pricing;
  const tiers = base.tiers
    .map((tier) => {
      const complete: PricingTier = { input_tokens_above: tier.input_tokens_above };
      for (const name of pricingRateNames) {
        const value = tier[name] ?? base[name];
        if (value === undefined) {
          throw new Error(`${key(row)} pricing tier ${tier.input_tokens_above} has no ${name} in the tier or base pricing`);
        }
        complete[name] = value;
      }
      return complete;
    })
    .sort((a, b) => a.input_tokens_above - b.input_tokens_above);
  for (let index = 0; index < tiers.length; index++) {
    const tier = tiers[index];
    if (!Number.isSafeInteger(tier.input_tokens_above) || tier.input_tokens_above <= 0) {
      throw new Error(`${key(row)} pricing tier threshold must be a positive safe integer`);
    }
    if (index > 0 && tier.input_tokens_above === tiers[index - 1].input_tokens_above) {
      throw new Error(`${key(row)} has duplicate pricing tier threshold ${tier.input_tokens_above}`);
    }
    for (const name of pricingRateNames) {
      const value = tier[name];
      if (value === undefined || !Number.isFinite(value) || value < 0) {
        throw new Error(`${key(row)} pricing tier ${tier.input_tokens_above} ${name} must be finite and non-negative`);
      }
    }
  }
  return { ...row, pricing: { ...base, tiers } };
}

function compareStrings(left: string, right: string): number {
  return Buffer.compare(Buffer.from(left), Buffer.from(right));
}

function validateOutputProviders(rows: ModelRow[]): void {
  const counts = providerCounts(rows);
  for (const provider of currentProviders) {
    if ((counts.get(provider) ?? 0) === 0) {
      throw new Error(`snapshot has no rows for expected provider ${provider}`);
    }
  }
  for (const provider of counts.keys()) {
    if (!currentProviderSet.has(provider)) {
      throw new Error(`snapshot contains deferred or unknown provider ${provider}`);
    }
  }
  for (const row of rows) {
    for (const name of pricingRateNames) {
      const value = row.pricing?.[name];
      if (value !== undefined && (!Number.isFinite(value) || value < 0)) {
        throw new Error(`${key(row)} pricing.${name} must be finite and non-negative`);
      }
    }
  }
}

function providerCounts(rows: ModelRow[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const row of rows) counts.set(row.provider, (counts.get(row.provider) ?? 0) + 1);
  return counts;
}

function key(row: ModelRow): string {
  return `${row.provider}/${row.id}`;
}

function canonicalFromOpenRouterID(id: string): string | undefined {
  const slash = id.indexOf("/");
  return slash > 0 && slash < id.length - 1 ? id : undefined;
}

function canonicalIDForProviderModel(provider: string, id: string, canonicalID?: string): string | undefined {
  if (!canonicalID || canonicalID === `${provider}/${id}`) return undefined;
  return canonicalID;
}

function objectValue(value: unknown, label: string): JSONRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as JSONRecord;
}

function optionalObject(record: JSONRecord | undefined, name: string, label: string): JSONRecord | undefined {
  if (!record || record[name] === undefined || record[name] === null) return undefined;
  return objectValue(record[name], `${label}.${name}`);
}

function requiredString(record: JSONRecord, name: string, label: string): string {
  const value = optionalString(record, name, label);
  if (!value) throw new Error(`${label}.${name} must be a non-empty string`);
  return value;
}

// Ordered weakest → strongest; mirrors go-llm's Effort constants. Curated
// supported_efforts lists must use these values in ascending order.
const effortScale = ["none", "minimal", "low", "medium", "high", "xhigh", "max"] as const;

function optionalEffortList(record: JSONRecord, name: string, label: string): string[] | undefined {
  const value = record[name];
  if (value === undefined) return undefined;
  if (!Array.isArray(value)) {
    throw new Error(`${label}.${name} must be an array`);
  }
  let previous = -1;
  for (const entry of value) {
    const index = effortScale.indexOf(entry as (typeof effortScale)[number]);
    if (typeof entry !== "string" || index < 0) {
      throw new Error(`${label}.${name} entry ${JSON.stringify(entry)} is not a known effort`);
    }
    if (index <= previous) {
      throw new Error(`${label}.${name} must be ordered weakest to strongest without duplicates`);
    }
    previous = index;
  }
  return value.map(String);
}

function optionalString(record: JSONRecord, name: string, label: string): string | undefined {
  const value = record[name];
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "string" || value === "") {
    throw new Error(`${label}.${name} must be a non-empty string`);
  }
  return value;
}

function optionalPositiveNumber(record: JSONRecord | undefined, names: string[], label: string): number | undefined {
  return optionalNumber(record, names, label, false);
}

function optionalNonNegativeNumber(record: JSONRecord | undefined, names: string[], label: string): number | undefined {
  return optionalNumber(record, names, label, true);
}

function optionalAvailablePositiveNumber(record: JSONRecord | undefined, names: string[]): number | undefined {
  return optionalAvailableNumber(record, names, false);
}

function optionalAvailableNonNegativeNumber(record: JSONRecord | undefined, names: string[]): number | undefined {
  return optionalAvailableNumber(record, names, true);
}

function optionalAvailableNumber(
  record: JSONRecord | undefined,
  names: string[],
  allowZero: boolean,
): number | undefined {
  if (!record) return undefined;
  for (const name of names) {
    const value = record[name];
    if (value === undefined || value === null || value === "") continue;
    const number = strictDecimalNumber(value);
    if (number === undefined || (allowZero ? number < 0 : number <= 0)) continue;
    return number;
  }
  return undefined;
}

function optionalNumber(
  record: JSONRecord | undefined,
  names: string[],
  label: string,
  allowZero: boolean,
): number | undefined {
  if (!record) return undefined;
  for (const name of names) {
    const value = record[name];
    if (value === undefined || value === null || value === "") continue;
    const number = strictDecimalNumber(value);
    if (number === undefined || (allowZero ? number < 0 : number <= 0)) {
      throw new Error(`${label}.${name} must be a ${allowZero ? "non-negative" : "positive"} finite number`);
    }
    return number;
  }
  return undefined;
}

const decimalNumberPattern = /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$/;

function strictDecimalNumber(value: unknown): number | undefined {
  if (typeof value === "number") return Number.isFinite(value) ? value : undefined;
  if (typeof value !== "string") return undefined;
  const spelling = value.trim();
  if (!decimalNumberPattern.test(spelling)) return undefined;
  const number = Number(spelling);
  return Number.isFinite(number) ? number : undefined;
}

export async function readJSON(source: string, options: ReadJSONOptions = {}): Promise<unknown> {
  return (await loadJSON(source, options)).value;
}

async function loadJSON(source: string, options: ReadJSONOptions = {}): Promise<LoadedJSON> {
  if (/^https?:\/\//.test(source)) {
    const timeoutMs = options.timeoutMs ?? remoteSourceTimeoutMs;
    const maxBytes = options.maxBytes ?? remoteSourceMaxBytes;
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) throw new Error("remote JSON timeout must be a positive integer");
    if (!Number.isSafeInteger(maxBytes) || maxBytes <= 0) throw new Error("remote JSON byte limit must be a positive integer");

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
    let completed = false;
    try {
      const response = await (options.fetchImpl ?? fetch)(source, {
        headers: { "user-agent": "go-llm-model-snapshot" },
        signal: controller.signal,
      });
      if (!response.ok) {
        if (response.body) await response.body.cancel().catch(() => undefined);
        throw new Error(`${source}: ${response.status} ${response.statusText}`);
      }

      const declaredLength = Number(response.headers.get("content-length"));
      if (Number.isFinite(declaredLength) && declaredLength > maxBytes) {
        if (response.body) await response.body.cancel().catch(() => undefined);
        throw new Error(`${source}: response exceeds ${maxBytes} byte limit`);
      }
      if (!response.body) throw new Error(`${source}: response body is missing`);

      reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      let total = 0;
      for (;;) {
        const { done, value } = await readStreamChunk(reader, controller.signal);
        if (done) {
          completed = true;
          break;
        }
        total += value.byteLength;
        if (total > maxBytes) {
          await reader.cancel().catch(() => undefined);
          throw new Error(`${source}: response exceeds ${maxBytes} byte limit`);
        }
        chunks.push(value);
      }
      const bytes = Buffer.concat(chunks, total);
      return {
        value: JSON.parse(bytes.toString("utf8")),
        bytes,
      };
    } catch (error) {
      if (controller.signal.aborted) throw new Error(`${source}: timed out after ${timeoutMs}ms`, { cause: error });
      throw error;
    } finally {
      if (reader) {
        if (!completed) await reader.cancel().catch(() => undefined);
        reader.releaseLock();
      }
      clearTimeout(timeout);
    }
  }
  const stored = await readFile(resolve(source));
  const bytes = source.endsWith(".gz") ? gunzipSync(stored) : stored;
  return { value: JSON.parse(bytes.toString("utf8")), bytes };
}

function readStreamChunk(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  signal: AbortSignal,
): Promise<ReadableStreamReadResult<Uint8Array>> {
  if (signal.aborted) return Promise.reject(signal.reason);
  return new Promise((resolveRead, rejectRead) => {
    const onAbort = () => rejectRead(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    reader.read().then(resolveRead, rejectRead).finally(() => signal.removeEventListener("abort", onAbort));
  });
}

async function readSnapshot(path: string): Promise<SnapshotDocument | undefined> {
  let raw: string;
  try {
    raw = await readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
  const source = objectValue(JSON.parse(raw), `${path} root`);
  if (typeof source.generated_at !== "string" || !Array.isArray(source.models)) {
    throw new Error(`${path} is not a model snapshot document`);
  }
  const models = source.models.map((value, index) => {
    const label = `${path}.models[${index}]`;
    const record = objectValue(value, label);
    const pricing = optionalObject(record, "pricing", label);
    return compactRow({
      provider: requiredString(record, "provider", label),
      id: requiredString(record, "id", label),
      canonical_id: optionalString(record, "canonical_id", label),
      display_name: optionalString(record, "display_name", label),
      context_window: optionalPositiveNumber(record, ["context_window"], label),
      max_output_tokens: optionalPositiveNumber(record, ["max_output_tokens"], label),
      pricing: pricingFromRecord(pricing, label),
      supported_efforts: optionalEffortList(record, "supported_efforts", label),
    });
  });
  const sources = Array.isArray(source.sources)
    ? source.sources.map((value, index) => parseSnapshotSource(value, `${path}.sources[${index}]`))
    : legacySnapshotSources();
  return {
    schema_version: typeof source.schema_version === "number" ? source.schema_version : 0,
    generator: typeof source.generator === "string" ? source.generator : "legacy",
    generated_at: source.generated_at,
    sources,
    models,
  };
}

function parseSnapshotSource(value: unknown, label: string): SnapshotSource {
  const source = objectValue(value, label);
  return compactSource({
    id: requiredString(source, "id", label),
    url: requiredString(source, "url", label),
    sha256: requiredString(source, "sha256", label),
    etag: optionalString(source, "etag", label),
  });
}

function legacySnapshotSources(): SnapshotSource[] {
  const digest = "0".repeat(64);
  return [
    { id: "models.dev", url: "legacy:unknown", sha256: digest },
    { id: "openrouter", url: "legacy:unknown", sha256: digest },
    { id: "overrides", url: "legacy:unknown", sha256: digest },
  ];
}

function validateSnapshotDocument(document: SnapshotDocument): void {
  if (document.schema_version !== snapshotSchemaVersion) {
    throw new Error(`snapshot schema_version must be ${snapshotSchemaVersion}`);
  }
  if (document.generator !== snapshotGenerator) {
    throw new Error(`snapshot generator must be ${snapshotGenerator}`);
  }
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(document.generated_at) ||
      !Number.isFinite(Date.parse(document.generated_at))) {
    throw new Error("snapshot generated_at must be an RFC3339 timestamp");
  }
  const expectedIDs = ["models.dev", "openrouter", "overrides"];
  if (document.sources.length !== expectedIDs.length) {
    throw new Error(`snapshot sources must contain ${expectedIDs.join(", ")}`);
  }
  for (let index = 0; index < expectedIDs.length; index++) {
    const source = document.sources[index];
    if (source.id !== expectedIDs[index] || source.url.trim() === "" || !/^[0-9a-f]{64}$/.test(source.sha256)) {
      throw new Error(`snapshot source ${index} must identify ${expectedIDs[index]} with a URL and SHA-256 digest`);
    }
  }
}

async function atomicWriteSnapshot(path: string, document: SnapshotDocument): Promise<void> {
  await atomicWriteBytes(path, serializeSnapshot(document), 0o644);
}

function serializeSnapshot(document: SnapshotDocument): Buffer {
  validateSnapshotDocument(document);
  return Buffer.from(`${JSON.stringify(document, null, 2)}\n`);
}

async function atomicWriteBytes(path: string, bytes: Uint8Array, mode = 0o644): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}-${randomUUID()}`;
  const handle = await open(temporary, "wx", 0o600);
  try {
    await handle.writeFile(bytes);
    await handle.sync();
    await handle.close();
    await chmod(temporary, mode);
    await rename(temporary, path);
  } catch (error) {
    await handle.close().catch(() => undefined);
    await rm(temporary, { force: true }).catch(() => undefined);
    throw error;
  }
}

function parseArgs(args: string[]): CLIOptions {
  const options: CLIOptions = {
    modelsDevSource: modelsDevURL,
    openRouterSource: openRouterURL,
    overridesSource: resolve(root, "scripts/overrides.json"),
    outputPath: resolve(root, "models.json"),
    allowDestructive: false,
    dryRun: false,
    verify: false,
    modelsDevURL,
    openRouterURL,
    overridesURL: "file:scripts/overrides.json",
  };
  for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (arg === "--allow-destructive") options.allowDestructive = true;
    else if (arg === "--dry-run") options.dryRun = true;
    else if (arg === "--verify") options.verify = true;
    else if (arg === "--fixture-minimums") options.minimums = fixtureMinimums;
    else if (arg === "--models-dev") options.modelsDevSource = sourcePath(requiredArg(args, ++index, arg));
    else if (arg === "--openrouter") options.openRouterSource = sourcePath(requiredArg(args, ++index, arg));
    else if (arg === "--overrides") options.overridesSource = sourcePath(requiredArg(args, ++index, arg));
    else if (arg === "--models-dev-url") options.modelsDevURL = requiredArg(args, ++index, arg);
    else if (arg === "--openrouter-url") options.openRouterURL = requiredArg(args, ++index, arg);
    else if (arg === "--overrides-url") options.overridesURL = requiredArg(args, ++index, arg);
    else if (arg === "--generated-at") options.generatedAt = requiredArg(args, ++index, arg);
    else if (arg === "--capture-dir") options.captureDirectory = resolve(root, requiredArg(args, ++index, arg));
    else if (arg === "--output") options.outputPath = resolve(root, requiredArg(args, ++index, arg));
    else throw new Error(`unknown argument ${arg}`);
  }
  return options;
}

function sourcePath(value: string): string {
  return /^https?:\/\//.test(value) ? value : resolve(root, value);
}

function requiredArg(args: string[], index: number, flag: string): string {
  const value = args[index];
  if (!value || value.startsWith("--")) throw new Error(`${flag} requires a value`);
  return value;
}

function snapshotSummary(document: SnapshotDocument, verb: string): string {
  const counts = providerCounts(document.models);
  return `${verb} ${document.models.length} models (${currentProviders
    .map((provider) => `${provider}: ${counts.get(provider) ?? 0}`)
    .join(", ")})`;
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : "";
if (invokedPath === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
