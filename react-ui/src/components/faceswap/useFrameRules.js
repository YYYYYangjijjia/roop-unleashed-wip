import { useState } from 'react';
import { clipRuleKey, frameRuleStorageKey, normalizeFrameRules, readFrameRules } from './frameRules';

export default function useFrameRules(target, targets, notify) {
  // Synchronous per-key selection avoids briefly sending the previous clip's
  // rules while React effects catch up after a target switch.
  const [edits, setEdits] = useState({});
  const key = clipRuleKey(target);
  const getRules = (t) => edits[clipRuleKey(t)] ?? readFrameRules(t);
  const rules = getRules(target);
  const replaceFor = (clip, list) => {
    const clipKey = clipRuleKey(clip);
    if (!clipKey) return;
    const next = normalizeFrameRules(list, clip.frames);
    setEdits((old) => ({ ...old, [clipKey]: next }));
    try { localStorage.setItem(frameRuleStorageKey(clipKey), JSON.stringify(next)); }
    catch { notify?.('Frame rules are temporary in this tab; the browser could not save them.', 'error'); }
  };
  const persist = (list) => replaceFor(target, list);
  return {
    key, rules, getRules, replaceFor,
    byTarget: (clips = targets) => Object.fromEntries(clips.filter((t) => t.clip_id).map((t) => [t.clip_id, getRules(t)])),
    add: (rule) => persist([...rules, { ...rule, id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}` }]),
    update: (id, rule) => persist(rules.map((r) => r.id === id ? { ...rule, id } : r)),
    remove: (id) => persist(rules.filter((r) => r.id !== id)),
    clear: () => persist([]),
  };
}
