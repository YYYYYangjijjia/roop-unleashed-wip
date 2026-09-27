import test from 'node:test';
import assert from 'node:assert/strict';
import { clipRuleKey, frameRuleStorageKey, normalizeFrameRules, readFrameRules, unresolvedFrameRules, frameRulePolicy, queuedTargetIndex, withoutFrameRules } from './frameRules.js';

test('queue target identity wins over name, never silently picks a different file', () => {
  const targets = [{ name: 'video.mp4', clip_id: 'a' }, { name: 'video.mp4', clip_id: 'b' }];
  assert.equal(queuedTargetIndex({ target_name: 'video.mp4', target_clip_id: 'b' }, targets), 1);
  assert.equal(queuedTargetIndex({ target_name: 'video.mp4', target_clip_id: 'gone' }, targets), -1);
  assert.equal(queuedTargetIndex({ target_name: 'video.mp4' }, targets), -1);
  assert.equal(queuedTargetIndex({ target_name: 'video.mp4' }, targets.slice(0, 1)), 0);
});

test('history settings cannot import old clip rules into global snapshots', () => {
  const settings = { swap_model: 'inswapper', frame_rules: [{ mode: 'skip' }], frame_rules_by_target: { a: [] } };
  assert.deepEqual(withoutFrameRules(settings), { swap_model: 'inswapper' });
  assert.equal(settings.frame_rules.length, 1);
  assert.deepEqual(withoutFrameRules(null), {});
});

test('rules use inclusive boundaries; skip wins and only intersections cannot broaden permissions', () => {
  const rules = [
    { start: 10, end: 20, mode: 'only', person_ids: ['a', 'b'] },
    { start: 15, end: 25, mode: 'only', person_ids: ['b', 'c'] },
    { start: 18, end: 19, mode: 'skip', person_ids: [] },
  ];
  assert.deepEqual(frameRulePolicy(rules, 9), { skip: false, person_ids: null });
  assert.deepEqual(frameRulePolicy(rules, 10), { skip: false, person_ids: ['a', 'b'] });
  assert.deepEqual(frameRulePolicy(rules, 20), { skip: false, person_ids: ['b'] });
  assert.equal(frameRulePolicy(rules, 18).skip, true);
  assert.deepEqual(frameRulePolicy(rules, 26), { skip: false, person_ids: null });
  assert.equal(frameRulePolicy([...rules, { start: 15, end: 17, mode: 'only', person_ids: ['z'] }], 16).skip, true);
  assert.equal(frameRulePolicy([{ start: 1, end: 1, mode: 'only', person_ids: [] }], 1).skip, true);
});

test('normalization clamps valid ranges without changing opaque person tokens', () => {
  const rules = normalizeFrameRules([
    { id: 'a', start: 40, end: -1, mode: 'only', person_ids: ['x', 'x', 1, ''] },
    { start: NaN, end: 8, mode: 'skip' }, { start: 1, end: 2, mode: 'unknown' },
  ], 30);
  assert.deepEqual(rules, [{ id: 'a', start: 1, end: 30, mode: 'only', person_ids: ['x'] }]);
  assert.deepEqual(unresolvedFrameRules(rules, ['new-x']), rules);
  assert.deepEqual(unresolvedFrameRules(rules, ['x']), []);
});

test('same-named videos are isolated and reload preserves all intervals', () => {
  const a = { name: 'video.mp4', clip_id: 'file-a', frames: 200, fps: 25 };
  const b = { ...a, clip_id: 'file-b' };
  const saved = [{ id: 'x', mode: 'skip', start: 100, end: 150, person_ids: [] }];
  const storage = new Map([[frameRuleStorageKey(clipRuleKey(a)), JSON.stringify(saved)]]);
  const fakeStorage = { getItem: (k) => storage.get(k) };
  assert.deepEqual(readFrameRules(a, fakeStorage), saved);
  assert.deepEqual(readFrameRules(b, fakeStorage), []);
  assert.deepEqual(readFrameRules(a, { getItem: () => '{bad' }), []);
  assert.notEqual(clipRuleKey({ name: 'x', frames: 50, fps: 25 }), clipRuleKey({ name: 'x', frames: 100, fps: 25 }));
});
