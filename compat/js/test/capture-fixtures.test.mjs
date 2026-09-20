import { once } from 'node:events';
import { createServer } from 'node:http';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { TypeSafeClient } from '@typesafe-ai/sdk';
import { createTypeSafeAi } from '@ai-sdk/typesafe-ai';

const FIXTURE_ROOT = join(import.meta.dirname, '../../fixtures');
const DUMMY_KEY = 'dummy-local-key';

const successResponse = {
  model: 'laya-english@fixture',
  answers: {
    department: {
      type: 'choice', choice: 'billing', confidence: 0.53,
      probabilities: { billing: 0.9, support: 0.1 },
    },
    urgency: {
      type: 'score', score: 1.2, confidence: 0.18,
      legend: { '0': 'low', '1': 'medium', '2': 'high' },
      probabilities: { '0': 0.1, '1': 0.6, '2': 0.3 },
    },
    refund: { type: 'noul', noul: 0.95 },
  },
  usage: { input_tokens: 123, output_tokens: 0 },
};

const choiceResponse = {
  model: 'laya-english@fixture',
  answers: {
    route: {
      type: 'choice', choice: 'billing', confidence: 0.75,
      probabilities: { billing: 0.75, support: 0.25 },
    },
  },
  usage: { input_tokens: 10, output_tokens: 0 },
};

const noulResponse = {
  model: 'laya-english@fixture',
  answers: { refund: { type: 'noul', noul: 0.95 } },
  usage: { input_tokens: 8, output_tokens: 0 },
};

function sortedHeaders(headers) {
  return Object.fromEntries(Object.entries(headers).sort(([a], [b]) => a.localeCompare(b)));
}

function stableCapturedHeaders(headers) {
  const normalized = { ...headers };
  for (const key of Object.keys(normalized)) {
    if (key.toLowerCase() === 'host') normalized[key] = '<local-mock>';
    if (key.toLowerCase() === 'date') normalized[key] = '<runtime-date>';
  }
  return sortedHeaders(normalized);
}

function stable(value) {
  return JSON.stringify(value, null, 2) + '\n';
}

async function saveFixture(client, name, value) {
  const directory = join(FIXTURE_ROOT, client);
  await mkdir(directory, { recursive: true });
  await writeFile(join(directory, `${name}.json`), stable(value));
}

class MockServer {
  constructor() {
    this.requests = [];
    this.nextResponse = null;
    this.server = createServer(async (request, response) => {
      const chunks = [];
      for await (const chunk of request) chunks.push(chunk);
      const rawBody = Buffer.concat(chunks).toString('utf8');
      let body = null;
      try { body = rawBody === '' ? null : JSON.parse(rawBody); } catch { body = rawBody; }
      const captured = {
        method: request.method,
        path: request.url,
        headers: stableCapturedHeaders(request.headers),
        rawBody,
        body,
      };
      this.requests.push(captured);
      const planned = this.nextResponse ?? { status: 200, body: successResponse };
      const responseBody = planned.body === undefined ? '' : JSON.stringify(planned.body);
      response.writeHead(planned.status, {
        'content-type': 'application/json',
        ...(planned.headers ?? {}),
      });
      response.end(responseBody);
    });
  }

  async start() {
    this.server.listen(0, '127.0.0.1');
    await once(this.server, 'listening');
    return this.server.address().port;
  }

  async stop() {
    this.server.close();
    await once(this.server, 'close');
  }

  plan(response) {
    this.requests = [];
    this.nextResponse = response;
  }
}

const exampleQuestionsNative = {
  department: {
    type: 'choice', instructions: 'Which team should handle this?',
    criteria: { billing: 'Payments and refunds', support: 'Other support' },
  },
  urgency: {
    type: 'score', instructions: 'How urgent is this?', criteria: ['low', 'medium', 'high'],
  },
  refund: { type: 'noul', instructions: 'Is a refund requested?' },
};

const exampleQuestionsAi = {
  department: exampleQuestionsNative.department,
  urgency: exampleQuestionsNative.urgency,
  refund: { type: 'boolean', instructions: 'Is a refund requested?' },
};

const baseState = { body: 'I was charged twice. Please refund.' };

let mock;
let port;
let native;
let ai;

before(async () => {
  mock = new MockServer();
  port = await mock.start();
  native = new TypeSafeClient({
    apiKey: DUMMY_KEY,
    baseURL: `http://127.0.0.1:${port}`,
    logLevel: 'off',
    retry: { maxRetries: 0 },
  });
  ai = createTypeSafeAi({ apiKey: DUMMY_KEY, baseURL: `http://127.0.0.1:${port}/v1` });
});

after(async () => mock.stop());

async function captureNative(name, operation, planned = { status: 200, body: successResponse }) {
  mock.plan(planned);
  let result;
  let error;
  try { result = await operation(); } catch (caught) { error = caught; }
  const observation = {
    request: mock.requests[0] ?? null,
    response: planned,
    result: result ?? null,
    error: error ? {
      className: error.constructor.name,
      message: error.message,
      status: error.status ?? null,
      retryAfterMs: error.retryAfterMs ?? null,
      retryAfterRead: error.retryAfterMs !== undefined,
    } : null,
  };
  await saveFixture('native', name, observation);
  return { result, error, observation };
}

async function captureAi(name, operation, planned = { status: 200, body: successResponse }) {
  mock.plan(planned);
  let result;
  let error;
  try { result = await operation(); } catch (caught) { error = caught; }
  const responseHeaders = error?.responseHeaders;
  const responseHeaderObject = responseHeaders
    ? (typeof responseHeaders.entries === 'function'
      ? Object.fromEntries(responseHeaders.entries())
      : responseHeaders)
    : null;
  const retryAfter = responseHeaders?.get?.('retry-after') ?? responseHeaderObject?.['retry-after'] ?? null;
  const stableResult = result?.response?.headers
    ? { ...result, response: { ...result.response, headers: stableCapturedHeaders(result.response.headers) } }
    : result;
  const observation = {
    request: mock.requests[0] ?? null,
    response: planned,
    result: stableResult ?? null,
    error: error ? {
      className: error.constructor.name,
      message: error.message,
      status: error.statusCode ?? error.status ?? null,
      retryAfterMs: error.retryAfterMs ?? null,
      retryAfterRead: error.retryAfterMs !== undefined,
      responseHeaders: responseHeaderObject ? stableCapturedHeaders(responseHeaderObject) : null,
      retryAfterHeaderVisible: retryAfter !== null,
    } : null,
  };
  await saveFixture('ai', name, observation);
  return { result, error, observation };
}

function evaluateAi(state, questions) {
  return ai.evaluationModel('jev-latest').doEvaluate({
    state,
    questions,
    headers: undefined,
    abortSignal: undefined,
    providerOptions: undefined,
  });
}

const nativeSuccess = () => native.systemOne({ model: 'jev-latest', state: baseState, questions: exampleQuestionsNative });
const aiSuccess = () => evaluateAi(baseState, exampleQuestionsAi);

for (const [clientName, capture, operation] of [
  ['native', captureNative, nativeSuccess],
  ['ai', captureAi, aiSuccess],
]) {
  test(`${clientName} captures the §8 choice/score/noul example`, async () => {
    const { result, error } = await capture('example-choice-score-noul', operation);
    assert.ifError(error);
    assert.ok(result);
    if (clientName === 'native') {
      assert.equal(result.answers.department.choice, 'billing');
      assert.equal(result.answers.urgency.score, 1.2);
      assert.equal(result.answers.refund.noul, 0.95);
    } else {
      assert.deepEqual(result.answers, {
        department: { type: 'choice', choice: 'billing', probabilities: { billing: 0.9, support: 0.1 } },
        urgency: { type: 'score', score: 1.2, probabilities: { '0': 0.1, '1': 0.6, '2': 0.3 } },
        refund: { type: 'boolean', probability: 0.95 },
      });
      assert.deepEqual(result.rounding, { probabilityDecimals: 2, scoreDecimals: 2 });
    }
  });
}

for (const [clientName, capture] of [['native', captureNative], ['ai', captureAi]]) {
  test(`${clientName} captures choice array criteria without descriptions`, async () => {
    const operation = clientName === 'native'
      ? () => native.systemOne({ state: 'route this', questions: { route: { type: 'choice', criteria: ['billing', 'support'] } } })
      : () => evaluateAi('route this', { route: { type: 'choice', criteria: ['billing', 'support'] } });
    const { result, error } = await capture('choice-array-criteria', operation, { status: 200, body: choiceResponse });
    assert.ifError(error);
    assert.ok(result);
  });

  test(`${clientName} captures noul true/false descriptions`, async () => {
    const question = { type: clientName === 'native' ? 'noul' : 'boolean', instructions: null, criteria: { true: 'yes', false: 'no' } };
    const operation = clientName === 'native'
      ? () => native.systemOne({ state: 'state', questions: { refund: question } })
      : () => evaluateAi('state', { refund: question });
    const { result, error } = await capture('noul-descriptions', operation, { status: 200, body: noulResponse });
    assert.ifError(error);
    assert.ok(result);
  });

  for (const [label, state] of [['state-string', 'plain text'], ['state-object', { nested: true }], ['state-array', ['one', 2]], ['state-null', null]]) {
    test(`${clientName} captures ${label} with omitted instructions`, async () => {
      const question = clientName === 'native'
        ? { type: 'noul' }
        : { type: 'boolean' };
      const operation = clientName === 'native'
        ? () => native.systemOne({ state, questions: { q: question } })
        : () => evaluateAi(state, { q: question });
      const { result, error } = await capture(label, operation, { status: 200, body: noulResponse });
      assert.ifError(error);
      assert.ok(result);
    });
  }
}

test('native captures GET /v1/models and unwraps models', async () => {
  const planned = { status: 200, body: { models: [{ name: 'jev-latest', description: 'fixture', release_date: '2026-01-01' }] } };
  const { result, error } = await captureNative('models-list', () => native.models.list(), planned);
  assert.ifError(error);
  assert.deepEqual(result, planned.body.models);
});

test('AI provider has no models resource; capture that compatibility observation', async () => {
  await saveFixture('ai', 'models-list', {
    request: null,
    response: null,
    result: null,
    error: null,
    observation: 'The pinned @ai-sdk/typesafe-ai provider exposes evaluationModel only; it has no models-list method or route.'
  });
});

const errorCases = [
  ['error-400-message', 400, { message: 'bad request' }, null],
  ['error-401', 401, { error: 'unauthorized' }, null],
  ['error-404-unknown-model', 404, { detail: 'unknown model' }, null],
  ['error-413', 413, { error_type: 'payload_too_large' }, null],
  ['error-429-retry-after', 429, { message: 'slow down' }, { 'retry-after': '7' }],
  ['error-500', 500, { detail: 'server error' }, null],
];

for (const [clientName, capture] of [['native', captureNative], ['ai', captureAi]]) {
  for (const [name, status, body, headers] of errorCases) {
    test(`${clientName} records ${status} error behavior`, async () => {
      const operation = clientName === 'native' ? nativeSuccess : aiSuccess;
      const { result, error } = await capture(name, operation, { status, body, headers: headers ?? undefined });
      assert.equal(result, undefined);
      assert.ok(error);
      if (clientName === 'native') {
        assert.equal(error.status, status);
        if (status === 429) assert.equal(error.retryAfterMs, 7000);
      } else {
        assert.equal(error.statusCode, status);
      }
    });
  }
}

for (const [clientName, capture] of [['native', captureNative], ['ai', captureAi]]) {
  test(`${clientName} accepts two-decimal probabilities`, async () => {
    const response = { model: 'fixture', answers: { route: { type: 'choice', choice: 'a', confidence: 0.5, probabilities: { a: 0.12, b: 0.88 } } }, usage: { input_tokens: 1, output_tokens: 0 } };
    const operation = clientName === 'native'
      ? () => native.systemOne({ state: 'x', questions: { route: { type: 'choice', criteria: { a: null, b: null } } } })
      : () => evaluateAi('x', { route: { type: 'choice', criteria: { a: null, b: null } } });
    const { result, error } = await capture('schema-two-decimal', operation, { status: 200, body: response });
    assert.ifError(error);
    assert.ok(result);
  });

  test(`${clientName} accepts four-decimal and non-summing probabilities`, async () => {
    const response = { model: 'fixture', answers: { route: { type: 'choice', choice: 'a', confidence: 0.5, probabilities: { a: 0.1234, b: 0.1234 } } }, usage: { input_tokens: 1, output_tokens: 0 } };
    const operation = clientName === 'native'
      ? () => native.systemOne({ state: 'x', questions: { route: { type: 'choice', criteria: { a: null, b: null } } } })
      : () => evaluateAi('x', { route: { type: 'choice', criteria: { a: null, b: null } } });
    const { result, error } = await capture('schema-four-decimal-non-summing', operation, { status: 200, body: response });
    assert.ifError(error);
    assert.ok(result);
  });
}
