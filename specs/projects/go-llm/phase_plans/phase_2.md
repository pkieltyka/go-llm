---
status: complete
---

# Phase 2: Serialization + Schema

> Historical, non-normative execution record. `functional_spec.md` and
> `architecture.md` define the current contract.

## Overview

This phase adds persistence-safe canonical JSON for the core `llm` vocabulary
and introduces the stdlib-only `schema` subpackage for tool input schemas and
tool argument validation. The work makes messages/responses round-trip without
losing reasoning payloads, provenance, or forward-compatible unknown parts.

## Steps

1. Add `serialize.go` in package `llm` with canonical JSON for `Message`,
   `Part`, `Response`, and `Usage`; implement part discriminators for
   `text`, `image`, `file`, `tool_call`, `tool_result`, and `reasoning`.
2. Add `UnknownPart` plus `RegisterPartType(name string, decode func(json.RawMessage) (Part, error)) error`
   so provider packages can decode extension parts and older versions can
   preserve newer part types on re-marshal.
3. Add `MarshalMessage`, `UnmarshalMessage`, `MarshalMessages`,
   `UnmarshalMessages`, `MarshalResponse`, and `UnmarshalResponse` helpers;
   message lists use a versioned `{"version":1,"messages":[...]}` envelope.
4. Extend clone support for `UnknownPart` so histories and accessors keep
   defensive-copy behavior for serialized unknown parts.
5. Add `schema/schema.go` with `For[T]`, `MustFor[T]`, `WithModifier`, the
   `JSONSchemaer` hook, and reflection over json/jsonschema tags for structs,
   primitives, slices, `map[string]T`, and nested structs.
6. Add `schema/validate.go` with `ValidateArgs(t llm.Tool, args json.RawMessage) error`
   covering required fields, primitive types, arrays, objects, enums, and
   `additionalProperties` in the supported schema subset.
7. Add serialization tests for envelope/single-message/response
   byte-identity round trips, `ReasoningPart.Raw`, message provenance,
   unknown-part preservation, registered extension decoding, and
   `Response`/`Usage` raw exclusion.
8. Add schema golden tests and validation tables for generated schemas,
   tag handling, unsupported types/constraints, modifiers, self-described
   schemas, and tool argument validation failures.

## Tests

- `TestMarshalMessagesRoundTripByteIdentity`: verifies canonical envelope
  marshal/unmarshal/marshal identity, reasoning raw preservation, and
  assistant provenance.
- `TestMarshalMessagePreservesRawPayloadBytes`: verifies raw-byte
  preservation for single-message canonical serialization.
- `TestUnmarshalMessagesPreservesUnknownPart`: verifies unknown part types
  decode to `UnknownPart` and re-marshal without data loss.
- `TestRegisterPartTypeDecodesExtensionPart`: verifies registered extension
  decoders participate in message decoding.
- `TestResponseJSONExcludesRaw`: verifies `Response.Raw` and `Usage.Raw` are
  not serialized while normalized fields round-trip.
- `TestMarshalResponsePreservesRawPayloadBytes`: verifies raw-byte
  preservation for canonical response serialization.
- `TestForGeneratesGoldenSchema`: verifies struct tag, required/optional,
  enum, nesting, and `additionalProperties:false` output.
- `TestForRejectsUnsupportedTypesAndTags`: verifies unsupported Go types and
  strict-mode-unsupported tag constraints return errors.
- `TestForSupportsModifierAndJSONSchemaer`: verifies the modifier hook and
  self-described schema escape hatch.
- `TestValidateArgs`: verifies valid args, missing required fields, wrong
  types, bad enums, unknown object properties, and invalid JSON.
