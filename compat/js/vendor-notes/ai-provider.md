# Vercel provider: `typesafe-ai-provider.ts`

Pinned source revision:
`20dd00abba618d5a516e0fee40ccd3e18a2bd1fb`.

- `TypeSafeAiProviderSettings` is `{apiKey?, baseURL?, headers?, fetch?}`;
  API key defaults to `TYPESAFE_AI_API_KEY`, and base URL defaults to
  `https://api.typesafe.ai/v1` (`src/typesafe-ai-provider.ts:23-30`).
- `createTypeSafeAi(options = {})` removes trailing slashes from the supplied
  base URL and stores the resulting base (`:32-36`). Its `evaluationModel` is
  the only TypeSafe resource and constructs an evaluation model with provider
  id `typesafe.evaluation` (`:46-54`). There is no models-list method.
- The provider's direct factory signature is therefore
  `createTypeSafeAi({ apiKey?, baseURL?, headers?, fetch? })`, followed by
  `provider.evaluationModel(modelId)` (`:32-50`). `languageModel`,
  `embeddingModel`, and `imageModel` deliberately throw `NoSuchModelError`
  (`:55-63`).
- The source uses the distinct env var `TYPESAFE_AI_API_KEY`; an explicit empty
  `apiKey: ""` is accepted by the installed provider-utils loader in the local
  probe, and is sent as an empty bearer value. No client-side nonempty check is
  present in this provider source.

Npm correspondence: registry `@ai-sdk/typesafe-ai@3.0.4` has the same version
as the pinned package.json. Its peer dependency is only
`zod: ^3.25.76 || ^4.1.8`, not `ai`; this fixture pins `ai@7.0.107` because its
provider/provider-utils dependencies exactly match the provider's published
`@ai-sdk/provider@4.0.17` and `@ai-sdk/provider-utils@5.0.45`.

Source URL:
`https://raw.githubusercontent.com/vercel/ai/20dd00abba618d5a516e0fee40ccd3e18a2bd1fb/packages/typesafe-ai/src/typesafe-ai-provider.ts`
