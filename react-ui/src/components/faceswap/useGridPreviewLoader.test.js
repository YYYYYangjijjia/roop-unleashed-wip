import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { setImmediate } from 'node:timers/promises';

// Exercise the loader effect without a DOM; substitute only React's effect
// registration and the network/serialization boundary.
const source = readFileSync(new URL('./useGridPreviewLoader.js', import.meta.url), 'utf8')
  .replace(/^import .*;\r?\n/gm, '')
  .replace('export default function', 'function');

test('an HTTP-200 fallback image with an error is not displayed or cached; other cells continue', async () => {
  let cleanup;
  const calls = [];
  const state = { previews: {}, times: {}, timers: {}, errors: {} };
  const setter = (key) => (update) => { state[key] = update(state[key]); };
  const cache = { current: {} };
  const intervals = { current: {} };
  const post = async (_path, payload) => {
    calls.push(payload.swap_model);
    return payload.swap_model === 'hyperswap'
      ? { image: 'original-frame', error: 'swap failed' }
      : { image: 'swapped-frame', faces: [[1, 2, 3, 4]] };
  };
  const hook = new Function('useEffect', 'postJSON', 'runExclusive',
    `${source}\nreturn useGridPreviewLoader;`)(
    (effect) => { cleanup = effect(); }, post, (run) => run());

  try {
    hook({
      enabled: true, selection: ['hyperswap', 'inswapper'],
      allowed: ['hyperswap', 'inswapper'], paramKey: 'swap_model',
      setPreviews: setter('previews'), setTimes: setter('times'),
      setTimers: setter('timers'), setErrors: setter('errors'),
      intervalsRef: intervals, settings: {}, fakePreview: true,
      selTarget: 0, frame: 191, targetCount: 1,
      buildPreviewPayload: (params) => params,
      previewSignature: (params) => params.swap_model,
      previewCacheRef: cache, cacheSuffix: 'source', reloadKey: '1',
    });
    await setImmediate();
    assert.deepEqual(calls, ['hyperswap', 'inswapper']);
    assert.equal(state.errors.hyperswap, 'swap failed');
    assert.equal(state.previews.hyperswap, undefined);
    assert.equal(cache.current['0_191_hyperswap_source'], undefined);
    assert.equal(state.previews.inswapper, 'swapped-frame');
    assert.equal(cache.current['0_191_inswapper_source'].image, 'swapped-frame');
    assert.equal(state.timers.hyperswap, null);
    assert.deepEqual(intervals.current, {});
  } finally {
    cleanup?.();
  }
});
