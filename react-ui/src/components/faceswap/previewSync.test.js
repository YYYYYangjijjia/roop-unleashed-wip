import test from 'node:test';
import assert from 'node:assert/strict';
import { setImmediate } from 'node:timers';
import { createPreviewQueue, createPreviewPairLoader } from './previewSync.js';

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const tick = () => new Promise((resolve) => setImmediate(resolve));

test('HTTP-200 original-frame fallback with error is never committed or cached as a swap', async () => {
  const commits = [], errors = [], busy = [];
  const queue = createPreviewQueue({
    isCurrent: () => true,
    run: async () => ({ image: 'original-frame', error: 'swap failed' }),
    commit: (...args) => commits.push(args),
    onError: (e) => errors.push(e.message), onBusy: (value) => busy.push(value),
  });
  await queue.enqueue({ key: '73/hyperswap' });
  assert.deepEqual(commits, []);
  assert.deepEqual(errors, ['swap failed']);
  assert.deepEqual(busy, [true, false]);
});

test('queued frame uses its own detector payload; old results never commit', async () => {
  const first = deferred();
  let current = '188/scrfd';
  const calls = [], commits = [], busy = [];
  const queue = createPreviewQueue({
    isCurrent: (key) => key === current,
    run: async (task) => { calls.push(task.payload); return calls.length === 1 ? first.promise : 'new-image'; },
    commit: (task, image) => commits.push([task.key, image]),
    onError: assert.fail, onBusy: (value) => busy.push(value),
  });
  const done = queue.enqueue({ key: current, payload: { frame: 188, detector: 'scrfd' } });
  current = '189/scrfd';
  queue.enqueue({ key: current, payload: { frame: 189, detector: 'scrfd' } });
  current = '190/yoloface';
  queue.enqueue({ key: current, payload: { frame: 190, detector: 'yoloface' } });
  first.resolve('old-image');
  await done;
  assert.deepEqual(calls, [{ frame: 188, detector: 'scrfd' }, { frame: 190, detector: 'yoloface' }]);
  assert.deepEqual(commits, [['190/yoloface', 'new-image']]);
  assert.deepEqual(busy, [true, false]);
});

test('changing source/target invalidates an in-flight result without another request', async () => {
  const response = deferred();
  let current = '188/source-a/target-a';
  const commits = [];
  const queue = createPreviewQueue({
    isCurrent: (key) => key === current, run: () => response.promise,
    commit: (...args) => commits.push(args), onError: assert.fail, onBusy: () => {},
  });
  const done = queue.enqueue({ key: current });
  current = '188/source-b/target-b';
  response.resolve('wrong-identity');
  await done;
  assert.equal(commits.length, 0);
});

test('failure clears queued work and busy state; explicit retry still works', async () => {
  const response = deferred();
  let attempts = 0;
  const errors = [], busy = [], commits = [];
  const queue = createPreviewQueue({
    isCurrent: () => true,
    run: () => ++attempts === 1 ? response.promise : Promise.resolve('recovered'),
    commit: (_task, image) => commits.push(image),
    onError: (e) => errors.push(e.message), onBusy: (value) => busy.push(value),
  });
  const done = queue.enqueue({ key: 'a' });
  queue.enqueue({ key: 'b' });
  response.reject(new Error('Bad Gateway'));
  await done;
  assert.equal(attempts, 1);
  assert.deepEqual(errors, ['Bad Gateway']);
  assert.equal(busy.at(-1), false);
  await queue.enqueue({ key: 'c' });
  assert.deepEqual(commits, ['recovered']);
});

test('before and after publish together even when raw decodes much later', async () => {
  const raw = deferred(), swapped = deferred();
  const pairs = [];
  const loader = createPreviewPairLoader((src) => src === 'raw188' ? raw.promise : swapped.promise,
    (pair) => pairs.push(pair));
  const pair = { key: '188', beforeSrc: 'raw188', afterSrc: 'swap188' };
  const done = loader.load(pair);
  swapped.resolve();
  await tick();
  assert.equal(pairs.length, 0);
  raw.resolve();
  await done;
  assert.deepEqual(pairs, [pair]);
});

test('late older pairs never replace a newer frame, including reverse seeks', async () => {
  const loads = new Map();
  const pairs = [];
  const loader = createPreviewPairLoader((src) => {
    if (!loads.has(src)) loads.set(src, deferred());
    return loads.get(src).promise;
  }, (pair) => pairs.push(pair));
  const old = loader.load({ key: '598', beforeSrc: 'raw598', afterSrc: 'swap598' });
  const latest = { key: '188', beforeSrc: 'raw188', afterSrc: 'swap188' };
  const fresh = loader.load(latest);
  loads.get('raw188').resolve(); loads.get('swap188').resolve();
  await fresh;
  loads.get('raw598').resolve(); loads.get('swap598').resolve();
  await old;
  assert.deepEqual(pairs, [latest]);
});

test('failed or cancelled image pair preserves the last complete pair', async () => {
  const blocked = deferred();
  const pairs = [], errors = [];
  const loader = createPreviewPairLoader((src) => src === 'bad' ? Promise.reject(new Error('decode')) : blocked.promise,
    (pair) => pairs.push(pair), (e) => errors.push(e.message));
  await loader.load({ beforeSrc: 'bad', afterSrc: 'bad' });
  const done = loader.load({ beforeSrc: 'new', afterSrc: 'new' });
  loader.cancel(); blocked.resolve(); await done;
  assert.deepEqual(errors, ['decode']);
  assert.equal(pairs.length, 0);
});
