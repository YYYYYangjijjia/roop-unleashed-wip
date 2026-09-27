import test from 'node:test';
import assert from 'node:assert/strict';
import { validateSwapModelSupport } from './swapModelSupport.js';

const meta = { capabilities: { alphaface: true }, swap_models: ['inswapper', 'alphaface'], swap_model_status: { alphaface: { installed: true } } };

test('AlphaFace requests reject old backend metadata rather than silently falling back', () => {
  assert.throws(() => validateSwapModelSupport({ swap_model: 'alphaface' }, { swap_models: ['inswapper'] }), /Restart/);
  assert.doesNotThrow(() => validateSwapModelSupport({ swap_model: 'inswapper' }, {}));
  assert.doesNotThrow(() => validateSwapModelSupport({ swap_model: 'alphaface', fake_preview: false }, {}));
});

test('AlphaFace supports average identity but refuses compatibility and missing files without changing params', () => {
  const params = { swap_model: 'alphaface', source_identity_mode: 'average', a_compatibility_mode: false };
  assert.doesNotThrow(() => validateSwapModelSupport(params, meta));
  assert.throws(() => validateSwapModelSupport({ ...params, a_compatibility_mode: true }, meta), /Legacy rendering compatibility/);
  assert.throws(() => validateSwapModelSupport(params, { ...meta, swap_model_status: { alphaface: { installed: false } } }), /missing locally/);
  assert.deepEqual(params, { swap_model: 'alphaface', source_identity_mode: 'average', a_compatibility_mode: false });
});
