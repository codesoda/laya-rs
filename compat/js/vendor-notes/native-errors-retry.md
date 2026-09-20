# Native SDK: errors, retry, env, and models resources

Pinned source revision:
`66880ccded6cb642dc1809620c2b108c33730214`.

## Error parsing and classes

`extractMessage` checks a string body, then `error` string, `error.message`,
`message`, `detail` string, `detail.message`, and validation-error arrays in
that order (`src/errors.ts:15-46`). The final message is `<status> <detail>`,
with a raw JSON/text fallback capped at 200 characters (`:58-65`). `error_type`
is **not** used by the native parser.

Status mapping is exact: 400 `BadRequestError`, 401 `AuthenticationError`,
403 `PermissionDeniedError`, 404 `NotFoundError`, 422
`UnprocessableEntityError`, 429 `RateLimitError`, and 500+ `InternalServerError`;
other statuses are plain `APIError` (`src/errors.ts:68-97`). A 413 is therefore
plain `APIError`. `RateLimitError.retryAfterMs` parses `retry-after-ms` first,
then numeric/date `Retry-After` (`src/retry.ts:33-50`).

## Environment and routes

`TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`, and
`TYPESAFE_LOG_LEVEL` are defined in `src/env.ts:2-11`. `models.list()` sends
`GET /v1/models` and unwraps an object with a `models` array; each model card
has `name`, `description`, and `release_date` (`src/resources/models.ts:14-27`,
src/types.ts:144-148). The AI provider has no models resource; its fixture
records that limitation.

Source URLs:
- `https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-js/66880ccded6cb642dc1809620c2b108c33730214/src/errors.ts`
- `https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-js/66880ccded6cb642dc1809620c2b108c33730214/src/retry.ts`
- `https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-js/66880ccded6cb642dc1809620c2b108c33730214/src/env.ts`
- `https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-js/66880ccded6cb642dc1809620c2b108c33730214/src/resources/models.ts`
