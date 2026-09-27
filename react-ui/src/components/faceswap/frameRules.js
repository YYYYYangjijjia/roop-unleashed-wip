// Frame numbers are absolute, one-based and inclusive. Rules are clip data,
// never global settings or snapshot data.
export const clipRuleKey = (target) => target?.clip_id || (target
  ? JSON.stringify([target.path || target.name, target.frames, target.fps]) : '');
export const frameRuleStorageKey = (key) => `roop_frame_rules_v1_${key}`;

// Older jobs can resolve by name only when that name is unambiguous. A new
// job's missing clip must never silently fall back to another same-named file.
export function queuedTargetIndex(job, targets) {
  if (job.target_clip_id) return targets.findIndex((t) => t.clip_id === job.target_clip_id);
  const matches = targets.map((t, i) => t.name === job.target_name ? i : -1).filter((i) => i >= 0);
  return matches.length === 1 ? matches[0] : -1;
}

export function withoutFrameRules(settings = {}) {
  const copy = { ...settings };
  delete copy.frame_rules;
  delete copy.frame_rules_by_target;
  return copy;
}

export function normalizeFrameRules(list, maxFrames) {
  if (!Array.isArray(list) || !Number.isFinite(maxFrames) || maxFrames < 1) return [];
  return list.filter((r) => r && ['skip', 'only'].includes(r.mode)
    && Number.isFinite(r.start) && Number.isFinite(r.end)).map((r, i) => {
    const clamp = (v) => Math.max(1, Math.min(Math.round(v), maxFrames));
    const a = clamp(r.start), b = clamp(r.end);
    return { id: String(r.id || `rule-${i}`), start: Math.min(a, b), end: Math.max(a, b), mode: r.mode,
      person_ids: r.mode === 'only' && Array.isArray(r.person_ids)
        ? [...new Set(r.person_ids.filter((id) => typeof id === 'string' && id))] : [] };
  }).sort((a, b) => a.start - b.start || a.end - b.end);
}

export function readFrameRules(target, storage = globalThis.localStorage) {
  if (!target) return [];
  try { return normalizeFrameRules(JSON.parse(storage.getItem(frameRuleStorageKey(clipRuleKey(target))) || '[]'), target.frames); }
  catch { return []; }
}

export function unresolvedFrameRules(rules, personIds) {
  const known = new Set(personIds);
  return rules.filter((r) => r.mode === 'only' && r.person_ids.some((id) => !known.has(id)));
}

export function frameRulePolicy(rules, frame) {
  const active = rules.filter((r) => r.start <= frame && frame <= r.end);
  if (active.some((r) => r.mode === 'skip')) return { skip: true, person_ids: [] };
  const only = active.filter((r) => r.mode === 'only');
  if (!only.length) return { skip: false, person_ids: null };
  const ids = only[0].person_ids.filter((id) => only.every((r) => r.person_ids.includes(id)));
  return { skip: ids.length === 0, person_ids: ids };
}
