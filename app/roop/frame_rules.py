"""Manual source-video frame rules; frame numbers are 1-based and inclusive.

None means normal processing. An empty frozenset means preserve the frame,
and a nonempty frozenset restricts matching to captured target group IDs.
Rules never identify a person by their left-to-right position in a frame.
"""
import hashlib
import json


def normalize_frame_rules(raw):
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > 1000:
        raise ValueError('frame_rules must be a list of at most 1000 intervals')
    result = []
    for rule in raw:
        if not isinstance(rule, dict):
            raise ValueError('Each frame rule must be an object')
        start, end = rule.get('start'), rule.get('end')
        if (type(start) is not int or type(end) is not int
                or start < 1 or end < start):
            raise ValueError('Frame rule bounds must be positive integers, start <= end')
        mode = rule.get('mode')
        if mode not in ('skip', 'only'):
            raise ValueError('Frame rule mode must be skip or only')
        ids = rule.get('person_ids', [])
        if not isinstance(ids, list) or any(type(pid) is not int or pid < 0 for pid in ids):
            raise ValueError('Frame rule person_ids must be nonnegative target group IDs')
        result.append({'id': str(rule.get('id', len(result))), 'start': start,
                       'end': end, 'mode': mode,
                       'person_ids': sorted(set(ids)) if mode == 'only' else []})
    return result


def frame_rule_policy(rules, source_frame):
    allowed = None
    for rule in rules or ():
        if rule['start'] <= source_frame <= rule['end']:
            if rule['mode'] == 'skip':
                return frozenset()
            ids = frozenset(rule['person_ids'])
            allowed = ids if allowed is None else allowed.intersection(ids)
    return allowed


def frame_rule_signature(rules, source_frame):
    """Contiguous interval key, so even two normal regions across a skip differ."""
    boundaries = {b for r in rules or () for b in (r['start'], r['end'] + 1)}
    return sum(b <= source_frame for b in boundaries)


def frame_rules_digest(rules):
    encoded = json.dumps(rules or [], sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]
