# Vercel provider: `typesafe-ai-evaluation-model.ts`

Pinned source: `vercel/ai` at
`20dd00abba618d5a516e0fee40ccd3e18a2bd1fb`; the package declares version
`3.0.4` (`packages/typesafe-ai/package.json:1-4`), matching npm
`@ai-sdk/typesafe-ai@3.0.4`.

- The model accepts `jev-latest` or any string model id
  (`typesafe-ai-evaluation-model.ts:26`). Its `doEvaluate` input is state,
  named questions, headers, abort signal, and provider options (`:62-70`).
  Choice >255 and score >10 are rejected locally (`:71-91`).
- It calls `${baseURL}/systemone` and sends `{model, state, questions}`
  (`:107-122`). Boolean questions are copied with `type: 'noul'` before send
  (`:114-120`); the fixture verifies this conversion. Native noul input is not
  rewritten.
- The default auth path uses `TYPESAFE_AI_API_KEY`, sends
  `Authorization: Bearer <key>`, and adds the provider user agent
  (`:97-105`). In the provider factory, explicitly supplied headers replace
  that generated header set (`typesafe-ai-provider.ts:37-44`).
- Successful answers normalize noul to `{type:'boolean', probability:noul}`;
  choice and score retain choice/probability maps and score. Usage maps to
  `inputTokens`/`outputTokens`; confidence is put in
  `providerMetadata.typesafe.confidence` for non-noul answers
  (`:131-179`). The caller result declares
  `rounding: { probabilityDecimals: 2, scoreDecimals: 2 }` (`:172-175`).
- The provider source has no timeout or retry option. It delegates transport to
  `postJsonToApi` (`:123-128`); in the local mock each error made one request.
  The error fixture shows provider-utils throws `APICallError` with a numeric
  `statusCode`; Retry-After remains visible in response headers but is not
  converted to `retryAfterMs` by this provider.

Source URL:
`https://raw.githubusercontent.com/vercel/ai/20dd00abba618d5a516e0fee40ccd3e18a2bd1fb/packages/typesafe-ai/src/typesafe-ai-evaluation-model.ts`
