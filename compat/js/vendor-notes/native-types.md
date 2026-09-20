# Native SDK: `src/types.ts`

Pinned source: `typesafe-ai/typesafe-sdk-js` at
`66880ccded6cb642dc1809620c2b108c33730214`.

- `JsonValue` is recursive JSON; `EntryType` is string, object, array, or null
  (`src/types.ts:1-11`). State, instructions, and descriptions use this type.
- Noul questions have `type: "noul"`, optional `instructions`, and optional/null
  `criteria` with optional `true` and `false` descriptions (`:20-32`). Choice
  questions require a map of labels to descriptions and have optional
  instructions (`:34-46`). Score questions require an ordered tuple/list with at
  least two entries (`:48-58`).
- Response shapes are: noul `{type,noul}` (`:72-76`); choice
  `{type,choice,confidence,probabilities}` where probabilities are keyed by
  label (`:79-89`); score `{type,score,confidence,legend,probabilities}` with
  stringified score keys (`:101-115`). The response types do not declare native
  `action` or `confidence` on noul.
- A system-one result requires `model`, `answers`, and `usage`; usage requires
  numeric `input_tokens` and `output_tokens` (`:126-142`).
- A request requires `state` and `questions`; `model` is optional and resolved by
  the client (`:160-170`). The type comment says additional request properties
  are forwarded (`:153-158`).
- Retry policy types document two default retries, 500 ms initial / 5000 ms cap,
  statuses 408/429/500-599, and Retry-After support (`:174-196`). Per-call
  options include timeout, retry overrides, signal, and headers (`:198-206`).
- Client configuration names `apiKey`, `baseURL`, `defaultModel`, retry, and
  timeout (`:222-243`). The documented environment names are recorded in
  `native-env.md`.

Source URL for all ranges:
`https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-js/66880ccded6cb642dc1809620c2b108c33730214/src/types.ts`
