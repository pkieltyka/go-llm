# Cost completeness and OpenAI image authority — 2026-10-27

Authorized by Agent Console Plan 8 B2, stage 1, against Go LLM `b62ba7b`
(v0.6.1). This amendment supersedes the historical coupling of incomplete
costs to estimated provenance and Plan 4's exclusion of a documented OpenAI
API image map. Historical completed plans remain evidence of their own inputs.
This stage changes Go LLM only; Go Agent and Console integration remain pending.

## Accounting contract

`Usage.CostIncomplete` records missing charges independently of the source of
known dollars. A non-nil `CostUSD` with this flag is a **subtotal**. Native
subtotal provenance stays native; an estimated component or legacy known cost
without provenance makes the known-dollar sum estimated. A nil `CostUSD`
means unknown even when the flag is false. Non-nil zero is known zero.

`Usage.Add(next)` returns independent cost storage and sums all seven token
counters, including the informational one-hour-write and reasoning subsets
without adding them to other counters. It retains explicit incomplete flags
(even with no tokens or cost), and marks incomplete when any token-bearing
input has no cost. Empty usage contributes no charge and no missing-price
signal. Legacy nil-cost token values become explicitly incomplete on addition;
legacy known costs keep their meaning without requiring provider changes.
The aggregate drops `Raw`, which has no single provider-payload meaning.

Addition needs no private accumulator state: grouping and input order preserve
accounting, subject to normal floating-point roundoff for dollar sums. Metadata
survives aggregation of aggregates, intermediate JSON persistence, cloning,
UsageTracker totals/buckets and Session totals. The optional JSON key
`cost_incomplete` is omitted when false; old envelopes still read and legacy
envelopes without the flag retain byte-stable encoding. Older readers ignore
the additive key; they cannot expose the new subtotal qualification. Historical
aggregates that already lost missing-price evidence cannot be repaired from
their known dollars alone.

No missing charge is invented. Price estimation keeps its existing per-call
all-or-nothing behavior and preserves explicit incompleteness. In particular,
one-hour cache writes still have no table rate and leave estimated cost nil.

## Image authority and limits

Official [Models API reference](https://developers.openai.com/api/reference/resources/models)
defines model identity and ownership, with no input modalities. That endpoint
alone cannot prove image support. The nine official model pages below were
fetched on 2026-10-27 (their `.md` equivalents also inspected). Each explicitly
declares image input, Responses endpoint support, and lists these aliases and
snapshots. The implementation uses exact equality only and applies the map to
returned API-key catalog rows; it neither adds unlisted models nor requests
documentation at runtime.

| Official API model page | Exact IDs proved for image input |
| --- | --- |
| [GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1) | `gpt-4.1`, `gpt-4.1-2025-04-14` |
| [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini) | `gpt-4.1-mini`, `gpt-4.1-mini-2025-04-14` |
| [GPT-4.1 nano](https://developers.openai.com/api/docs/models/gpt-4.1-nano) | `gpt-4.1-nano`, `gpt-4.1-nano-2025-04-14` |
| [GPT-4o](https://developers.openai.com/api/docs/models/gpt-4o) | `gpt-4o`, `gpt-4o-2024-05-13`, `gpt-4o-2024-08-06`, `gpt-4o-2024-11-20` |
| [GPT-4o mini](https://developers.openai.com/api/docs/models/gpt-4o-mini) | `gpt-4o-mini`, `gpt-4o-mini-2024-07-18` |
| [GPT-5](https://developers.openai.com/api/docs/models/gpt-5) | `gpt-5`, `gpt-5-2025-08-07` |
| [GPT-5 mini](https://developers.openai.com/api/docs/models/gpt-5-mini) | `gpt-5-mini`, `gpt-5-mini-2025-08-07` |
| [GPT-5 nano](https://developers.openai.com/api/docs/models/gpt-5-nano) | `gpt-5-nano`, `gpt-5-nano-2025-08-07` |
| [GPT-5.1](https://developers.openai.com/api/docs/models/gpt-5.1) | `gpt-5.1`, `gpt-5.1-2025-11-13` |

This is a bounded positive set, not an exhaustive OpenAI capability catalog.
Unmapped IDs, arbitrary suffixes, future snapshots, fine-tunes, differently
cased IDs and provider-prefixed IDs remain unknown. Another provider's rows or
the embedded price table never establish OpenAI image support. Each mapped
result owns its capability slice. Capability claims remain advisory; no
model-name-based request rejection is added to Go LLM.

ChatGPT OAuth rows continue to use only their live `input_modalities`; a
documented API-key ID without that live signal does not inherit the API-key
capability. Catalog membership determines the account's visible models;
documentation does not prove account entitlement, continued availability of
deprecated snapshots, custom endpoint conformance or successful image inference.
Custom endpoints using the OpenAI provider receive the same documented metadata;
callers remain responsible for their endpoint's compatibility.

Later Console display should distinguish positive image input from unknown
and qualify unknown API-key rows: the API catalog does not report modalities,
and the exact row has no documented mapping. Its positive-proof admission rule
remains unchanged. Native, live provider/account/image and hosted acceptance
are not established by offline fixtures or documentation and remain open.
