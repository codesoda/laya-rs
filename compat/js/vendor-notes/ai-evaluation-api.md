# Vercel provider: `typesafe-ai-evaluation-api.ts`

Pinned source revision:
`20dd00abba618d5a516e0fee40ccd3e18a2bd1fb`.

The success zod schema accepts optional/nullish `model`, required `answers`,
and optional/nullish `usage` (`src/typesafe-ai-evaluation-api.ts:4-32`).
Choice answers require `type`, string `choice`, numeric `probabilities` record,
and optional/nullish numeric `confidence`; score answers require numeric
`score`, numeric probability record, and optional/nullish confidence. Noul
requires only `type` and numeric `noul`. The schema does not require
probabilities to sum to 1, does not require values to be <=1, and does not
limit decimal places. The local fixtures prove that two-decimal, four-decimal,
and non-summing maps are accepted. Zod's normal object parsing also strips
unknown answer fields from the normalized caller result; `legend` and
confidence do not appear in its normalized score/choice answer, while
confidence is separately placed in provider metadata.

The failed-response schema accepts optional/nullish `message`, arbitrary
`detail`, `error` as either a string or an object with optional/nullish message,
and optional/nullish `error_type` (`:34-51`). Its message precedence is
`message`, then `error` string/object message, then stringified `detail`, then
`error_type`, then `TypeSafe request failed` (`:42-50`). It does not map HTTP
statuses to provider-specific subclasses; provider-utils reports `APICallError`
with `statusCode`. Retry-After is not interpreted by this source.

Source URL:
`https://raw.githubusercontent.com/vercel/ai/20dd00abba618d5a516e0fee40ccd3e18a2bd1fb/packages/typesafe-ai/src/typesafe-ai-evaluation-api.ts`
