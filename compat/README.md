# SDK compatibility fixtures

This directory pins and exercises the two JavaScript clients required by the L0
compatibility contract. It is a wire-format test, not a TypeSafe service client.

## Pins

- Native client: `@typesafe-ai/sdk@0.6.0`, from `typesafe-ai/typesafe-sdk-js`
  revision `66880ccded6cb642dc1809620c2b108c33730214`.
- Vercel provider: `@ai-sdk/typesafe-ai@3.0.4`, from `vercel/ai`
  revision `20dd00abba618d5a516e0fee40ccd3e18a2bd1fb`; the package version at
  that source revision is also `3.0.4`.
- AI SDK core: `ai@7.0.107`. The provider has no `ai` peer dependency; this
  exact core release supplies the matching `@ai-sdk/provider@4.0.17` and
  `@ai-sdk/provider-utils@5.0.45` API used by the provider.
- Provider peer: `zod@4.1.8`.

`compat/js/package-lock.json` is the install proof. Node 24.15.0 and npm are
used by this fixture.

## Run

```sh
cd compat/js
npm ci
npm test
```

The test starts only a local `node:http` mock. It never calls `api.typesafe.ai`
or another paid endpoint. It writes deterministic JSON observations below
`compat/fixtures/native/` and `compat/fixtures/ai/`. All incoming request
headers are captured without redaction; only inherently variable loopback host
and HTTP date values are replaced with explicit placeholders so a second run
can be diffed byte-for-byte.

The native SDK uses a root base URL and calls `/v1/systemone`; the Vercel
provider uses a base URL ending in `/v1` and calls `/systemone`. See
`vendor-notes/` for source-line evidence and `schemas/` for the extracted
wire contracts.

This project is independent of TypeSafe AI and Convai Innovations. Compatibility
means wire-format compatibility only; it does not claim equal models,
calibration, quality, service availability, or affiliation.
