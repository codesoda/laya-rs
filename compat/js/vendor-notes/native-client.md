# Native SDK: `src/client.ts`

Pinned source revision:
`66880ccded6cb642dc1809620c2b108c33730214`.

- Defaults are root `https://api.typesafe.ai` and model `jev-latest`
  (`src/client.ts:40-42`). The constructor resolves an API key from code or
  `TYPESAFE_API_KEY`, a root URL from code or `TYPESAFE_BASE_URL`, and a model
  from code or `TYPESAFE_DEFAULT_MODEL`, stripping trailing URL slashes
  (`:267-278`). Missing/nullish key throws, but an explicit empty string is
  accepted and is sent as `Bearer `; the fixture probe verified this edge.
- `systemOne` forwards the request object and resolves `model` as
  `request.model ?? defaultModel`, then POSTs to `/v1/systemone`
  (`:310-323`). Thus the native base URL is the server root, not a `/v1` URL.
  Every request body field is JSON-stringified as supplied. The exact body
  fields are `model`, `state`, `questions`, plus any extra request properties
  (`src/types.ts:153-170`).
- The request sends `Authorization: Bearer <key>`, `Accept: application/json`,
  `User-Agent: typesafe-sdk/0.6.0`, `X-TypeSafe-SDK`, runtime metadata, and JSON
  content type (`:349-361`). The fixture captures all incoming headers with no
  redaction.
- Default timeout is 10 seconds per attempt. Default retry policy is two
  retries for 408, 429, and 500-599; connection errors and timeouts are also
  retryable (`src/retry.ts:5-24`, `src/client.ts:349-400`). The fixture sets
  `maxRetries: 0` so each canned error is one request. Retry-After is honored
  for retry delay (up to 60 seconds), but the thrown 429 `RateLimitError` also
  exposes `retryAfterMs` (`src/retry.ts:33-67`, `src/errors.ts:91-95`).
- A success response is returned without response-field validation in the native
  client: the parsed JSON is cast to the requested result (`:342-346`).
- Empty/whitespace environment values are ignored by `readEnv`, but explicit
  `apiKey: ""` is not nullish and therefore passes construction; this differs
  from the plan's stronger “nonempty key” expectation (`src/env.ts:13-21`,
  `src/client.ts:270`).

Source URL:
`https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-js/66880ccded6cb642dc1809620c2b108c33730214/src/client.ts`
