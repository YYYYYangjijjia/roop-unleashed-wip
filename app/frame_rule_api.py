"""Stable UI identities and strict request validation for timeline rules."""
import hashlib
import os
import threading
import uuid

from roop.frame_rules import normalize_frame_rules


class PersonTokens:
    """Keep a token while a group's reference objects overlap; never reuse it."""
    def __init__(self):
        self._groups = {}
        self._lock = threading.RLock()

    def clear(self):
        with self._lock:
            self._groups.clear()

    def sync(self, groups, faces):
        with self._lock:
            members = {}
            for group, face in zip(groups, faces):
                members.setdefault(group, []).append(face)
            updated = {}
            for group, refs in members.items():
                old = self._groups.get(group)
                # Retain the objects, not merely id(), to avoid address reuse.
                overlap = old and any(a is b for a in refs for b in old[1])
                updated[group] = (old[0] if overlap else uuid.uuid4().hex, refs)
            self._groups = updated
            return [updated[g][0] for g in groups], {v[0]: g for g, v in updated.items()}


def clip_identity(filename):
    path = os.path.normcase(os.path.abspath(filename))
    try:
        stat = os.stat(path)
        stamp = f'{path}|{stat.st_size}|{stat.st_mtime_ns}'
    except OSError:
        stamp = path
    return hashlib.sha256(stamp.encode('utf-8')).hexdigest()[:24]


def resolve_rules(raw, token_groups, total_frames):
    if raw is None:
        raw = []
    if not isinstance(raw, list) or len(raw) > 1000:
        raise ValueError('处理区间必须是列表，且不能超过 1000 条。')
    resolved = []
    for rule in raw:
        if not isinstance(rule, dict):
            raise ValueError('处理区间格式无效。')
        item = dict(rule)
        if item.get('mode') == 'only':
            tokens = item.get('person_ids', [])
            if not isinstance(tokens, list):
                raise ValueError('请重新指定处理区间中的人物。')
            if any(not isinstance(t, str) or t not in token_groups for t in tokens):
                raise ValueError('处理区间引用的人物已失效（可能已删除、重分组或重启）。请在时间轴重新指定人物。')
            item['person_ids'] = [token_groups[t] for t in tokens]
        elif item.get('mode') == 'skip':
            item['person_ids'] = []
        resolved.append(item)
    rules = normalize_frame_rules(resolved)
    if any(r['end'] > total_frames for r in rules):
        raise ValueError('处理区间超出当前视频长度。')
    return rules


def validate_rule_options(rules, payload, cfg=None):
    if not rules:
        return
    def setting(key, default=None):
        return payload.get(key, getattr(cfg, key, default))
    if setting('upscale_after_swap', False) or setting('interp_after_swap', 'off') not in (None, False, 'off'):
        raise ValueError('处理区间需要保持原帧；请关闭整段 AI upscale 和 Frame interpolation 后再运行。')
    if any(setting(key) == 'Segment Anything 2 (tracked)' for key in ('mask_engine', 'mask_engine_2')):
        raise ValueError('处理区间暂不支持 SAM2 tracked 整段预扫描，请选择逐帧遮罩。')
    if setting('a_compatibility_mode', False) and any(r['mode'] == 'only' for r in rules):
        raise ValueError('指定人物区间需要关闭 A 兼容处理；整段保持原样区间仍支持 A 兼容。')
    if any(r['mode'] == 'only' for r in rules) and setting('detection', setting('face_detection_mode')) in ('All input faces', 'all_input'):
        raise ValueError('指定人物区间使用身份匹配，请将 All input faces 改为 Selected face。')
