# TypeSafe wire compatibility draft

This is a compatibility matrix for the L0 spike. It describes request/response
wire behavior, not equal intelligence, calibration, context, or service
features. This project is independent of TypeSafe AI and Convai Innovations.

## Levels

| Level | Contract |
| --- | --- |
| Required | Raw `POST /v1/systemone` and `GET /v1/models` shape consumed by native `@typesafe-ai/sdk@0.6.0`; local mock fixtures in `compat/fixtures/native/`. |
| Required | `@ai-sdk/typesafe-ai@3.0.4` through `createTypeSafeAi`, including its `/v1` base convention and boolean-to-`noul` conversion; fixtures in `compat/fixtures/ai/`. |
| Best effort | Official Python client, after its real package and wire behavior are separately retrieved and pinned. |
| Not promised | Hard-coded TypeSafe URL interception, billing/account services, or Vercel AI Gateway's proprietary transport. Gateway callers must select the direct TypeSafe provider and configure its base URL. |

## Verified transport facts

The native SDK's root is `baseURL` (default `https://api.typesafe.ai`) and
`systemOne` appends `/v1/systemone`; it resolves omitted model to `jev-latest`
(`compat/js/vendor-notes/native-client.md`, source `client.ts:40-42,267-323`).
The native environment names are `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`,
`TYPESAFE_DEFAULT_MODEL`, and `TYPESAFE_LOG_LEVEL` (`src/env.ts:2-11`). It
sends `Authorization: Bearer <key>` plus JSON content type and SDK/runtime user
agent headers (`src/client.ts:349-361`). An explicit empty key is accepted by
this pinned implementation; missing/nullish key is rejected.

The Vercel provider removes trailing slashes from its base (default
`https://api.typesafe.ai/v1`) and appends `/systemone`
(`typesafe-ai-provider.ts:32-50`, `typesafe-ai-evaluation-model.ts:107-122`).
Its key environment variable is distinct: `TYPESAFE_AI_API_KEY`
(`typesafe-ai-provider.ts:23-40`). It sends the same bearer format. Factory
signature: `createTypeSafeAi({ apiKey?, baseURL?, headers?, fetch? })`, then
`provider.evaluationModel(modelId)` (`compat/js/vendor-notes/ai-provider.md`).
The provider exposes no models-list resource.

## Wire and error decisions

Use the native request fields `model`, `state`, and `questions`, with state and
instructions as string/object/array/null; score criteria are ordered arrays and
noul criteria may describe true/false. The native request types make
instructions optional, while the upstream Laya Python runtime directly indexes
that field; missing-instruction behavior therefore remains an adapter decision.
The AI provider accepts boolean questions and converts only their outgoing type
to `noul`.

For a server error, accept the union of both parsers: `message`, `detail`,
`error` as a string or `{message}`, and `error_type`. The AI adapter's precedence
is message, error, detail, error_type; the native parser also accepts text bodies,
`detail.message`, and validation arrays but ignores `error_type`
(`compat/js/vendor-notes/*errors*`). Native status classes are 400 bad request,
401 auth, 403 permission, 404 not found, 422 validation, 429 rate limit, and
500+ internal; 413 is generic `APIError`. Native parses `Retry-After` into
`retryAfterMs` and retries 408/429/5xx by default. The AI adapter throws generic
`APICallError` with `statusCode`; its response headers expose Retry-After but the
provider does not parse or retry it in this source. Laya should return the
structured union and the relevant status without promising the provider's
class names.

The provider declares `{ probabilityDecimals: 2, scoreDecimals: 2 }`; use two
decimal places for Jev-wire adapter display values. Native Laya output keeps
its four-decimal rounding and native `confidence`/`action` semantics. Do not
renormalize rounded distributions. The provider schema accepts four-decimal and
non-summing numeric probabilities: it checks numeric type only, not bounds or
sum (`typesafe-ai-evaluation-api.ts:6-32`).

Laya-only `{confidence, action:{act_probability}}` is defined in
`schemas/laya-native-response.schema.json`, based on `.cache/src/laya/laya/agent.py:312-370`;
it is an extension, not part of the upstream response schema.

## Open questions for Fable

1. Should single-option Choice be a deterministic adapter result or an explicit
   unsupported-capability error? The provider advertises the option count but
   the upstream Laya action head expects two entries.
2. Should the server normalize missing `instructions` to `null`/empty string,
   or reject it to match the current Python runtime's direct indexing?
3. Should native responses be validated at all? The pinned native client casts
   parsed JSON without response-field validation, while the AI provider uses
   the zod schema.
4. Should `jev-latest` map explicitly to the configured Laya profile while the
   response identifies the actual Laya revision? It should not impersonate a
   Jev model name.
5. Should the native adapter include Laya confidence/action by opt-in only, and
   how should its four-decimal native mode be selected?
6. Should 413 and provider `APICallError` class differences be documented only,
   or should the local server expose a common error code extension?
7. Should the native models route list release dates for actual checkpoints or
   omit unknown dates rather than invent metadata?
