import React, { useState } from 'react';
import { Button, InfoBadge } from '../ui';
import { frameRulePolicy, unresolvedFrameRules } from './frameRules';

export default function FrameRulesBar({ rules, people, frame, maxFrames, enabled, busy, detection, settings = {}, fps,
  onAdd, onUpdate, onRemove, onClear, onJump }) {
  const [start, setStart] = useState(frame), [end, setEnd] = useState(frame);
  const [mode, setMode] = useState('skip'), [selected, setSelected] = useState([]);
  const [editing, setEditing] = useState(null);
  const invalid = unresolvedFrameRules(rules, people.map((p) => p.id));
  const policy = frameRulePolicy(rules, frame);
  const disabled = busy || !enabled;
  const validRange = Number.isInteger(start) && Number.isInteger(end) && start >= 1 && end >= start && end <= maxFrames;
  const validPeople = mode !== 'only' || (selected.length > 0 && selected.every((id) => people.some((p) => p.id === id)));
  const label = (id) => people.find((p) => p.id === id)?.label || 'Missing person (reselect)';
  const conflicts = [];
  if (rules.length) {
    if (settings.upscale_after_swap) conflicts.push('turn off full-video AI upscale');
    if (settings.interp_after_swap && settings.interp_after_swap !== 'off') conflicts.push('turn off Frame interpolation');
    if ([settings.mask_engine, settings.mask_engine_2].includes('Segment Anything 2 (tracked)')) conflicts.push('use a frame-by-frame mask instead of SAM2 tracked');
    if (settings.a_compatibility_mode && rules.some((r) => r.mode === 'only')) conflicts.push('turn off Legacy rendering compatibility for person-specific rules');
    if (detection === 'All input faces' && rules.some((r) => r.mode === 'only')) conflicts.push('change Face selection to Selected face for person-specific rules');
    if (Number(settings.video_fps) > 0 && Math.abs(Number(settings.video_fps) - fps) > 0.01) conflicts.push('set Video FPS to 0 or the source FPS');
  }
  const save = () => {
    const rule = { start, end, mode, person_ids: mode === 'only' ? selected : [] };
    if (editing) onUpdate(editing, rule); else onAdd(rule);
    setEditing(null);
  };
  const inputClass = 'w-20 rounded border border-white/20 bg-black/30 px-2 py-1 text-white';
  return <section className="mt-2 rounded-xl border border-white/15 bg-black/20 px-3 py-3" aria-label="Frame rules">
    <div className="flex items-center gap-1.5 text-xs font-bold mb-2">Frame rules <InfoBadge info={{ en: 'Mark multiple inclusive frame ranges. Keep original skips face detection and swapping entirely. Only selected people limits which known people may be swapped; detection and identity matching still run. Rules apply to the current video only. The full video and export In/Out remain unchanged. Yellow rules take priority over blue rules where they overlap.', zh: '可标记多个包含起止帧的区间。“保留原画面”完全跳过人脸检测和替换；“仅选中人物”仍会检测并匹配身份，只限制可替换的人物。规则仅属于当前视频，不改变完整视频或导出的 In/Out。区间重叠时黄色原画面规则优先。' }} /> <span className="font-normal text-white/50">Full video preserved · Export In/Out unchanged</span></div>
    {!enabled && <p role="status" className="text-xs text-amber-300 mb-2">Frame rules are unavailable until the backend is restarted.</p>}
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <label>Start frame <input aria-label="Rule start frame" type="number" min="1" max={maxFrames} value={start} onChange={(e) => setStart(Number(e.target.value))} className={inputClass} disabled={disabled} /></label>
      <Button size="sm" variant="secondary" onClick={() => setStart(frame)} disabled={disabled}>Use current frame</Button>
      <label>End frame <input aria-label="Rule end frame" type="number" min="1" max={maxFrames} value={end} onChange={(e) => setEnd(Number(e.target.value))} className={inputClass} disabled={disabled} /></label>
      <Button size="sm" variant="secondary" onClick={() => setEnd(frame)} disabled={disabled}>Use current frame</Button>
      <select aria-label="Frame rule action" value={mode} onChange={(e) => setMode(e.target.value)} disabled={disabled} className="rounded border border-white/20 bg-[var(--input-bg)] px-2 py-1">
        <option value="skip">Keep original frames</option>
        <option value="only">Only selected people</option>
      </select>
      <Button size="sm" variant="primary" onClick={save} disabled={disabled || !validRange || !validPeople}>{editing ? 'Save changes' : 'Add rule'}</Button>
      {editing && <Button size="sm" variant="ghost" onClick={() => setEditing(null)}>Cancel</Button>}
    </div>
    {mode === 'only' && <div className="flex flex-wrap gap-3 mt-2 text-xs">
      {people.length === 0 && <span className="text-amber-300">Capture and group target people in Target media first.</span>}
      {people.map((p) => <label key={p.id} className="flex items-center gap-1"><input type="checkbox" checked={selected.includes(p.id)} disabled={disabled} onChange={(e) => setSelected((old) => e.target.checked ? [...old, p.id] : old.filter((id) => id !== p.id))} />{p.label}</label>)}
      <InfoBadge info={{ en: 'This limits who may be swapped. Detection and identity matching still run; it never forces an unknown person to match a selected person.', zh: '这里只限制允许换脸的人物。系统仍会检测并匹配身份，不会强制把陌生人换成所选人物。' }} />
    </div>}
    {invalid.length > 0 && <p role="alert" className="text-xs text-red-300 mt-2">{invalid.length} rule(s) refer to missing people. Preview and render are blocked. Edit or remove these rules.</p>}
    {conflicts.length > 0 && <p role="alert" className="text-xs text-amber-300 mt-2">Before export: {conflicts.join('; ')}. Settings will not change automatically.</p>}
    {rules.some((r) => r.mode === 'only') && detection !== 'Selected face' && detection !== 'All input faces' && <p role="status" className="text-xs text-amber-300 mt-2">Person-specific ranges match captured people; outside them, the current selection mode applies.</p>}
    <div className="flex flex-wrap gap-2 mt-2">
      {rules.map((r) => <div key={r.id} className={`rounded-lg border px-2 py-1 text-xs ${r.mode === 'skip' ? 'border-amber-500/50 bg-amber-500/10' : 'border-sky-500/50 bg-sky-500/10'}`}>
        <button type="button" onClick={() => onJump(r)} title="Jump to range start">{r.start}–{r.end} · {r.mode === 'skip' ? 'Keep original' : r.person_ids.length ? r.person_ids.map(label).join(', ') : 'No eligible people'}</button>
        <button type="button" className="ml-2 underline" disabled={disabled} onClick={() => { setStart(r.start); setEnd(r.end); setMode(r.mode); setSelected(r.person_ids.filter((id) => people.some((p) => p.id === id))); setEditing(r.id); }}>Edit</button>
        <button type="button" className="ml-2 text-red-300" aria-label={`Remove rule ${r.start} to ${r.end}`} disabled={disabled} onClick={() => { onRemove(r.id); if (editing === r.id) setEditing(null); }}>×</button>
      </div>)}
      {rules.length > 0 && <Button size="sm" variant="ghost" disabled={disabled} onClick={() => { onClear(); setEditing(null); }}>Clear rules</Button>}
    </div>
    <p className="text-xs text-white/50 mt-2 mb-0">Current frame: {policy.skip ? 'Keep original' : policy.person_ids ? policy.person_ids.map(label).join(', ') : 'Normal processing'}. <InfoBadge info={{ en: 'Ranges include both endpoints. Yellow = keep original; blue = selected people only. Keep original wins on overlap; overlapping selected-people ranges use their intersection. Kept frames are still decoded and re-encoded on export.', zh: '区间包含起止帧。黄色为保留原画面，蓝色为限定人物。重叠时保留原画面优先；多个限定人物区间取人物交集。导出时这些帧仍会解码并重新编码。' }} /></p>
  </section>;
}
