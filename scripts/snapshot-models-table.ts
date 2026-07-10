import { randomUUID } from "node:crypto";
import { chmod, mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export type Pricing = {
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
};

export type SnapshotDocument = {
  generated_at: string;
  models: ModelRow[];
};

export type SourceMinimums = {
  modelsDev: Record<Provider, number>;
  openRouter: number;
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
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

type BuildOptions = {
  generatedAt?: string;
  minimums?: SourceMinimums;
};

type CLIOptions = {
  modelsDevSource: string;
  openRouterSource: string;
  overridesSource: string;
  outputPath: string;
  allowDestructive: boolean;
  dryRun: boolean;
  minimums?: SourceMinimums;
};

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const [modelsDev, openRouter, overrides] = await Promise.all([
    readJSON(options.modelsDevSource),
    readJSON(options.openRouterSource),
    readJSON(options.overridesSource),
  ]);
  const document = buildSnapshot(modelsDev, openRouter, overrides, { minimums: options.minimums });
  const previous = await readSnapshot(options.outputPath);
  assertNonDestructive(previous, document, options.allowDestructive);

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

  const models = [...rows.values()].sort((a, b) => key(a).localeCompare(key(b)));
  validateOutputProviders(models);
  return {
    generated_at: options.generatedAt ?? new Date().toISOString(),
    models,
  };
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
    pricing: pricingFromUpstreamRecord(cost ?? optionalObject(record, "pricing", `models.dev ${provider}/${id}`)),
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
    });
  });
}

function pricingFromRecord(record: JSONRecord | undefined, label: string): Pricing | undefined {
  if (!record) return undefined;
  return compactPricing({
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
  });
}

function pricingFromUpstreamRecord(record: JSONRecord | undefined): Pricing | undefined {
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
  });
}

function pricingFromOpenRouter(record: JSONRecord | undefined): Pricing | undefined {
  if (!record) return undefined;
  const perTokenToMTok = (names: string[]) => {
    const value = optionalAvailableNonNegativeNumber(record, names);
    return value === undefined ? undefined : value * 1_000_000;
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
    if (value !== undefined) (out as Record<string, number>)[name] = value;
  }
  return compactPricing(out);
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
    for (const [name, value] of Object.entries(row.pricing ?? {})) {
      if (value < 0) throw new Error(`${key(row)} pricing.${name} must not be negative`);
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
    const number = typeof value === "number" ? value : typeof value === "string" ? Number(value) : Number.NaN;
    if (!Number.isFinite(number) || (allowZero ? number < 0 : number <= 0)) continue;
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
    const number = typeof value === "number" ? value : typeof value === "string" ? Number(value) : Number.NaN;
    if (!Number.isFinite(number) || (allowZero ? number < 0 : number <= 0)) {
      throw new Error(`${label}.${name} must be a ${allowZero ? "non-negative" : "positive"} finite number`);
    }
    return number;
  }
  return undefined;
}

async function readJSON(source: string): Promise<unknown> {
  if (/^https?:\/\//.test(source)) {
    const response = await fetch(source, { headers: { "user-agent": "go-llm-model-snapshot" } });
    if (!response.ok) throw new Error(`${source}: ${response.status} ${response.statusText}`);
    return response.json();
  }
  return JSON.parse(await readFile(resolve(source), "utf8"));
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
    });
  });
  return { generated_at: source.generated_at, models };
}

async function atomicWriteSnapshot(path: string, document: SnapshotDocument): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}-${randomUUID()}`;
  const handle = await open(temporary, "wx", 0o600);
  try {
    await handle.writeFile(`${JSON.stringify(document, null, 2)}\n`);
    await handle.sync();
    await handle.close();
    await chmod(temporary, 0o644);
    await rename(temporary, path);
  } catch (error) {
    await handle.close().catch(() => undefined);
    await rm(temporary, { force: true }).catch(() => undefined);
    throw error;
  }
}

function parseArgs(args: string[]): CLIOptions {
  const options: CLIOptions = {
    modelsDevSource: "https://models.dev/api.json",
    openRouterSource: "https://openrouter.ai/api/v1/models",
    overridesSource: resolve(root, "scripts/overrides.json"),
    outputPath: resolve(root, "models.json"),
    allowDestructive: false,
    dryRun: false,
  };
  for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (arg === "--allow-destructive") options.allowDestructive = true;
    else if (arg === "--dry-run") options.dryRun = true;
    else if (arg === "--fixture-minimums") options.minimums = fixtureMinimums;
    else if (arg === "--models-dev") options.modelsDevSource = requiredArg(args, ++index, arg);
    else if (arg === "--openrouter") options.openRouterSource = requiredArg(args, ++index, arg);
    else if (arg === "--overrides") options.overridesSource = requiredArg(args, ++index, arg);
    else if (arg === "--output") options.outputPath = resolve(requiredArg(args, ++index, arg));
    else throw new Error(`unknown argument ${arg}`);
  }
  return options;
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
