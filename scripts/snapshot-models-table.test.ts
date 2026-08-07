import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import {
  assertNonDestructive,
  buildSnapshot,
  mergeRow,
  persistSnapshot,
  readJSON,
  type SnapshotDocument,
  type SourceMinimums,
} from "./snapshot-models-table.ts";

const here = dirname(fileURLToPath(import.meta.url));
const fixtureMinimums: SourceMinimums = {
  modelsDev: { anthropic: 1, openai: 1, openrouter: 1 },
  openRouter: 1,
};

async function fixture(name: string): Promise<unknown> {
  return JSON.parse(await readFile(resolve(here, "fixtures", name), "utf8"));
}

async function fixtureSnapshot(): Promise<SnapshotDocument> {
  return buildSnapshot(
    await fixture("models-dev.json"),
    await fixture("openrouter-models.json"),
    await fixture("overrides.json"),
    { generatedAt: "2026-07-10T00:00:00.000Z", minimums: fixtureMinimums },
  );
}

test("maps models.dev object maps and merges only defined values", async () => {
  const snapshot = await fixtureSnapshot();
  assert.deepEqual(
    snapshot.models.map((row) => `${row.provider}/${row.id}`),
    [
      "anthropic/claude-fixture",
      "openai/gpt-fixture",
      "openai/image-zero-limit-fixture",
      "openrouter/openai/gpt-fixture",
      "openrouter/openrouter/auto-fixture",
      "openrouter/openrouter/dynamic-fixture",
    ],
  );
  const row = snapshot.models.find((model) => model.id === "openai/gpt-fixture")!;
  assert.equal(row.context_window, 128000, "missing OpenRouter context must not erase models.dev context");
  assert.equal(row.max_output_tokens, 32768);
  assert.deepEqual(row.pricing, {
    input_per_mtok: 2,
    output_per_mtok: 9,
    cache_read_per_mtok: 0.5,
    cache_write_per_mtok: 2.5,
  });
  assert.equal(row.canonical_id, "openai/gpt-fixture");

  const image = snapshot.models.find((model) => model.id === "image-zero-limit-fixture")!;
  assert.equal(image.context_window, undefined);
  assert.equal(image.max_output_tokens, undefined);
  assert.deepEqual(image.pricing, { input_per_mtok: 5 });

  const auto = snapshot.models.find((model) => model.id === "openrouter/auto-fixture")!;
  assert.equal(auto.context_window, undefined);
  assert.equal(auto.max_output_tokens, undefined);
  assert.equal(auto.pricing, undefined);

  const dynamic = snapshot.models.find((model) => model.id === "openrouter/dynamic-fixture")!;
  assert.deepEqual(dynamic.pricing, { output_per_mtok: 4 });
  for (const model of snapshot.models) {
    for (const price of Object.values(model.pricing ?? {})) assert.ok(price >= 0);
  }
  assert.deepEqual(snapshot, await fixtureSnapshot(), "fixture snapshots must be deterministic");
});

test("mergeRow never clobbers defined values with undefined", () => {
  assert.deepEqual(
    mergeRow(
      { provider: "openai", id: "gpt", context_window: 128000, pricing: { input_per_mtok: 2 } },
      { provider: "openai", id: "gpt", context_window: undefined, pricing: { input_per_mtok: undefined } },
    ),
    { provider: "openai", id: "gpt", context_window: 128000, pricing: { input_per_mtok: 2 } },
  );
});

test("derives ordered efforts and complete context pricing tiers from models.dev", async () => {
  const modelsDev = structuredClone(await fixture("models-dev.json")) as any;
  const model = modelsDev.anthropic.models["claude-fixture"];
  model.reasoning_options = [
    { type: "toggle", values: ["ignored"] },
    { type: "effort", values: ["high", "default", null, "low", "high"] },
    { type: "budget_tokens", min: 1024 },
  ];
  model.cost.tiers = [
    { input: 6, output: 30, cache_read: 0, tier: { type: "context", size: 100_000 } },
    { input: 99, tier: { type: "regional", size: 1 } },
  ];
  const snapshot = buildSnapshot(modelsDev, await fixture("openrouter-models.json"), await fixture("overrides.json"), {
    generatedAt: "2026-08-05T00:00:00Z",
    minimums: fixtureMinimums,
  });
  const row = snapshot.models.find((entry) => entry.provider === "anthropic" && entry.id === "claude-fixture")!;
  assert.deepEqual(row.supported_efforts, ["low", "high"]);
  assert.deepEqual(row.pricing?.tiers, [
    {
      input_tokens_above: 100_000,
      input_per_mtok: 6,
      output_per_mtok: 30,
      cache_read_per_mtok: 0,
      cache_write_per_mtok: 3.75,
    },
  ]);
});

test("effort overrides replace derivation and unknown effort values fail", async () => {
  const modelsDev = structuredClone(await fixture("models-dev.json")) as any;
  modelsDev.openai.models["gpt-fixture"].reasoning_options = [
    { type: "effort", values: ["high", "minimal", "medium", "minimal"] },
  ];
  const overrides = {
    models: [{ provider: "openai", id: "gpt-fixture", supported_efforts: ["none", "max"] }],
  };
  const snapshot = buildSnapshot(modelsDev, await fixture("openrouter-models.json"), overrides, {
    minimums: fixtureMinimums,
  });
  assert.deepEqual(
    snapshot.models.find((entry) => entry.provider === "openai" && entry.id === "gpt-fixture")?.supported_efforts,
    ["none", "max"],
  );

  modelsDev.openai.models["gpt-fixture"].reasoning_options = [{ type: "effort", values: ["turbo"] }];
  assert.throws(
    () => buildSnapshot(modelsDev, {}, overrides, { minimums: fixtureMinimums }),
    /openai\/gpt-fixture has unknown reasoning effort "turbo"/,
  );
});

test("context tiers require complete inherited rates and override tiers replace as a unit", async () => {
  const modelsDev = structuredClone(await fixture("models-dev.json")) as any;
  const openRouter = await fixture("openrouter-models.json");
  const model = modelsDev.anthropic.models["claude-fixture"];
  model.cost.tiers = [{ input: 6, output: 30, tier: { type: "context", size: 100_000 } }];
  delete model.cost.cache_write;

  assert.throws(
    () =>
      buildSnapshot(modelsDev, openRouter, { models: [] }, { generatedAt: "2026-08-05T00:00:00Z", minimums: fixtureMinimums }),
    /anthropic\/claude-fixture pricing tier 100000 has no cache_write_per_mtok/,
  );

  const overrides = {
    models: [
      {
        provider: "anthropic",
        id: "claude-fixture",
        pricing: {
          tiers: [
            {
              input_tokens_above: 200_000,
              input_per_mtok: 7,
              output_per_mtok: 35,
              cache_read_per_mtok: 0.7,
              cache_write_per_mtok: 8.75,
            },
          ],
        },
      },
    ],
  };
  const snapshot = buildSnapshot(modelsDev, openRouter, overrides, {
    minimums: fixtureMinimums,
  });
  assert.deepEqual(
    snapshot.models.find((entry) => entry.provider === "anthropic" && entry.id === "claude-fixture")?.pricing?.tiers,
    overrides.models[0].pricing.tiers,
  );

  const incomplete = structuredClone(overrides) as any;
  delete incomplete.models[0].pricing.tiers[0].cache_write_per_mtok;
  assert.throws(
    () => buildSnapshot(modelsDev, openRouter, incomplete, { minimums: fixtureMinimums }),
    /cache_write_per_mtok is required/,
  );
});

test("rejects malformed sources, missing providers, and implausible row counts", async () => {
  const modelsDev = (await fixture("models-dev.json")) as Record<string, unknown>;
  const openRouter = await fixture("openrouter-models.json");
  const overrides = await fixture("overrides.json");

  const missing = structuredClone(modelsDev);
  delete missing.anthropic;
  assert.throws(
    () => buildSnapshot(missing, openRouter, overrides, { minimums: fixtureMinimums }),
    /models.dev provider anthropic must be an object/,
  );
  assert.throws(
    () => buildSnapshot(modelsDev, { data: {} }, overrides, { minimums: fixtureMinimums }),
    /OpenRouter root.data must be an array/,
  );
  assert.throws(
    () => buildSnapshot(modelsDev, openRouter, overrides),
    /want at least/,
  );
  assert.throws(
    () => buildSnapshot(modelsDev, openRouter, { models: [{ provider: "zai", id: "glm" }] }, { minimums: fixtureMinimums }),
    /not a current provider/,
  );
  assert.throws(
    () =>
      buildSnapshot(
        modelsDev,
        openRouter,
        { models: [{ provider: "openai", id: "gpt-fixture", pricing: { input_per_mtok: -1 } }] },
        { minimums: fixtureMinimums },
      ),
    /non-negative finite number/,
  );
});

test("detects total replacement and substantial ID removal even when counts are stable", () => {
  const previous: SnapshotDocument = {
    generated_at: "before",
    models: [
      { provider: "anthropic", id: "a1" },
      { provider: "anthropic", id: "a2" },
      { provider: "anthropic", id: "a3" },
      { provider: "anthropic", id: "a4" },
      { provider: "anthropic", id: "a5" },
      { provider: "anthropic", id: "a6" },
      { provider: "anthropic", id: "a7" },
      { provider: "anthropic", id: "a8" },
    ],
  };
  const replacement: SnapshotDocument = {
    generated_at: "after",
    models: previous.models.map((row, index) => ({ ...row, id: `b${index + 1}` })),
  };
  assert.throws(() => assertNonDestructive(previous, replacement), /all 8 previous model IDs/);

  const materialChurn: SnapshotDocument = {
    generated_at: "after",
    models: [
      { provider: "anthropic", id: "a1" },
      { provider: "anthropic", id: "a2" },
      { provider: "anthropic", id: "a3" },
      { provider: "anthropic", id: "a4" },
      { provider: "anthropic", id: "a5" },
      { provider: "anthropic", id: "b1" },
      { provider: "anthropic", id: "b2" },
      { provider: "anthropic", id: "b3" },
    ],
  };
  assert.throws(() => assertNonDestructive(previous, materialChurn), /3\/8 model IDs \(37.5%/);
  assert.doesNotThrow(() => assertNonDestructive(previous, replacement, true));
});

test("detects wholesale metadata loss across retained models", () => {
  const previous: SnapshotDocument = {
    generated_at: "before",
    models: Array.from({ length: 4 }, (_, index) => ({
      provider: "anthropic",
      id: `a${index + 1}`,
      display_name: `A ${index + 1}`,
      context_window: 200000,
      max_output_tokens: 8192,
      pricing: { input_per_mtok: 3, output_per_mtok: 15 },
    })),
  };
  const stripped: SnapshotDocument = {
    generated_at: "after",
    models: previous.models.map(({ provider, id }) => ({ provider, id })),
  };
  assert.throws(() => assertNonDestructive(previous, stripped), /lost context_window for 4\/4/);
});

test("allows small ID churn and isolated metadata loss", () => {
  const previous: SnapshotDocument = {
    generated_at: "before",
    models: Array.from({ length: 8 }, (_, index) => ({
      provider: "anthropic",
      id: `a${index + 1}`,
      context_window: 200000,
      pricing: { input_per_mtok: 3 },
    })),
  };
  const next = structuredClone(previous);
  next.generated_at = "after";
  next.models = next.models.filter((row) => row.id !== "a8");
  next.models.push({
    provider: "anthropic",
    id: "b1",
    context_window: 200000,
    pricing: { input_per_mtok: 3 },
  });
  delete next.models[0].context_window;
  assert.doesNotThrow(() => assertNonDestructive(previous, next));
});

test("writes atomically and preserves the destination on rejected deltas", async () => {
  const directory = await mkdtemp(resolve(tmpdir(), "go-llm-models-"));
  const output = resolve(directory, "models.json");
  const snapshot = await fixtureSnapshot();
  await persistSnapshot(output, snapshot);
  assert.deepEqual(JSON.parse(await readFile(output, "utf8")), snapshot);

  const before = await readFile(output, "utf8");
  const stripped = {
    generated_at: "later",
    models: snapshot.models.map(({ provider, id }) => ({ provider, id })),
  };
  await assert.rejects(() => persistSnapshot(output, stripped), /lost display_name/);
  assert.equal(await readFile(output, "utf8"), before);

  const destructive = {
    generated_at: "later",
    models: snapshot.models.filter((row) => row.provider !== "openrouter"),
  };
  await assert.rejects(() => persistSnapshot(output, destructive), /--allow-destructive/);
  assert.equal(await readFile(output, "utf8"), before);

  const circular = structuredClone(snapshot) as SnapshotDocument & { self?: unknown };
  circular.self = circular;
  await assert.rejects(() => persistSnapshot(output, circular), /circular/i);
  assert.equal(await readFile(output, "utf8"), before);
  assert.deepEqual(
    (await readdir(directory)).filter((name) => name.includes(".tmp-")),
    [],
  );
});

test("file-backed destructive checks preserve supported efforts", async (t) => {
  const directory = await mkdtemp(resolve(tmpdir(), "go-llm-efforts-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const output = resolve(directory, "models.json");
  const previous: SnapshotDocument = {
    generated_at: "before",
    models: Array.from({ length: 4 }, (_, index) => ({
      provider: "anthropic",
      id: `claude-${index + 1}`,
      supported_efforts: ["none", "low", "high"],
    })),
  };
  await persistSnapshot(output, previous);

  const stripped: SnapshotDocument = {
    generated_at: "after",
    models: previous.models.map(({ provider, id }) => ({ provider, id })),
  };
  await assert.rejects(() => persistSnapshot(output, stripped), /lost supported_efforts for 4\/4/);
  assert.deepEqual(JSON.parse(await readFile(output, "utf8")), previous);
});

test("file-backed destructive checks preserve pricing tiers", async (t) => {
  const directory = await mkdtemp(resolve(tmpdir(), "go-llm-tiers-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const output = resolve(directory, "models.json");
  const previous: SnapshotDocument = {
    generated_at: "before",
    models: Array.from({ length: 4 }, (_, index) => ({
      provider: "anthropic",
      id: `claude-${index + 1}`,
      pricing: {
        input_per_mtok: 1,
        tiers: [
          {
            input_tokens_above: 100_000,
            input_per_mtok: 2,
            output_per_mtok: 3,
            cache_read_per_mtok: 4,
            cache_write_per_mtok: 5,
          },
        ],
      },
    })),
  };
  await persistSnapshot(output, previous);

  const stripped = structuredClone(previous);
  for (const row of stripped.models) delete row.pricing?.tiers;
  await assert.rejects(() => persistSnapshot(output, stripped), /lost pricing.tiers for 4\/4/);
  assert.deepEqual(JSON.parse(await readFile(output, "utf8")), previous);
});

test("remote JSON reads are bounded, timed out, and parsed after a complete read", async () => {
  const source = "https://models.test/api.json";
  const validFetch = async () => new Response(`{"models":["ok"]}`, { status: 200 });
  assert.deepEqual(await readJSON(source, { fetchImpl: validFetch as typeof fetch, maxBytes: 64, timeoutMs: 100 }), {
    models: ["ok"],
  });

  const declaredOversize = async () =>
    new Response("{}", { status: 200, headers: { "content-length": "65" } });
  await assert.rejects(
    () => readJSON(source, { fetchImpl: declaredOversize as typeof fetch, maxBytes: 64, timeoutMs: 100 }),
    /exceeds 64 byte limit/,
  );

  const streamedOversize = async () => new Response("x".repeat(65), { status: 200 });
  await assert.rejects(
    () => readJSON(source, { fetchImpl: streamedOversize as typeof fetch, maxBytes: 64, timeoutMs: 100 }),
    /exceeds 64 byte limit/,
  );

  const stalledFetch = ((_input: RequestInfo | URL, init?: RequestInit) =>
    new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
    })) as typeof fetch;
  await assert.rejects(
    () => readJSON(source, { fetchImpl: stalledFetch, maxBytes: 64, timeoutMs: 5 }),
    /timed out after 5ms/,
  );
});

test("remote JSON rejects bad responses and cancels unread bodies", async () => {
  const source = "https://models.test/api.json";
  let statusCanceled = false;
  const nonOK = async () =>
    new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode("untrusted error"));
        },
        cancel() {
          statusCanceled = true;
        },
      }),
      { status: 503, statusText: "Unavailable" },
    );
  await assert.rejects(
    () => readJSON(source, { fetchImpl: nonOK as typeof fetch, maxBytes: 64, timeoutMs: 100 }),
    /503 Unavailable/,
  );
  assert.equal(statusCanceled, true);

  const missingBody = async () => new Response(null, { status: 200 });
  await assert.rejects(
    () => readJSON(source, { fetchImpl: missingBody as typeof fetch, maxBytes: 64, timeoutMs: 100 }),
    /response body is missing/,
  );

  const malformed = async () => new Response("{not-json", { status: 200 });
  await assert.rejects(
    () => readJSON(source, { fetchImpl: malformed as typeof fetch, maxBytes: 64, timeoutMs: 100 }),
    /JSON/,
  );
});

test("remote JSON timeout cancels a stalled body reader", async () => {
  const source = "https://models.test/api.json";
  let canceled = false;
  const stalledBody = async () =>
    new Response(
      new ReadableStream({
        pull() {
          return new Promise(() => undefined);
        },
        cancel() {
          canceled = true;
        },
      }),
      { status: 200 },
    );
  await assert.rejects(
    () => readJSON(source, { fetchImpl: stalledBody as typeof fetch, maxBytes: 64, timeoutMs: 5 }),
    /timed out after 5ms/,
  );
  assert.equal(canceled, true);
});

test("local JSON reads remain file-backed", async (t) => {
  const directory = await mkdtemp(resolve(tmpdir(), "go-llm-read-json-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const path = resolve(directory, "source.json");
  await writeFile(path, '{"local":true}\n');
  assert.deepEqual(await readJSON(path), { local: true });
  await writeFile(path, "{broken");
  await assert.rejects(() => readJSON(path), /JSON/);
});
