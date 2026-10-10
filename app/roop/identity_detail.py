"""Persistent source-identity detail extraction and restoration.

FaceSet V2 owns the representation built here.  The representation is a
small, signed luminance residual in a canonical face space plus persistence
confidence; it is deliberately not a source-image patch.  At render time the
residual is warped into the active swap template, scaled to the target's local
contrast, and composited only where the swap/mask pipeline says the generated
face owns the pixel.
"""

from __future__ import annotations

import math
import threading

import cv2
import numpy as np


# A requested restoration that has no representation to work from is reported
# once per distinct cause, never silently skipped.
#
# Bounded like the detector reporter in face_util: one line per distinct cause
# per process, because a per-face warning on a long render is its own problem.
_DETAIL_MISSING_SEEN = set()
_DETAIL_MISSING_LOCK = threading.Lock()


def warn_identity_detail_unavailable(source_index=0, faceset=None,
                                     strength=None):
    """Announce once that identity detail was requested but is unavailable."""
    version = getattr(faceset, 'format_version', None)
    if version is not None and int(version) < 2:
        cause = ("the source FaceSet v%s has no usable reference images and "
                 "landmarks for detail extraction" % version)
    elif faceset is None:
        cause = "no source FaceSet was resolved for this face"
    else:
        cause = ("the FaceSet has no usable high-frequency residual or "
                 "reference image for source index %s" % source_index)
    with _DETAIL_MISSING_LOCK:
        if cause in _DETAIL_MISSING_SEEN:
            return
        _DETAIL_MISSING_SEEN.add(cause)
    detail = "" if strength is None else " (strength %.3g)" % float(strength)
    print("[IdentityDetail] identity_detail_strength is set%s but NOTHING WAS "
          "RESTORED: %s.\n"
          "[IdentityDetail] The render continues unchanged and reports success; "
          "load a faceset with usable reference images or set identity_detail_strength to 0 to "
          "stop asking for it." % (detail, cause), flush=True)


def identity_detail_unavailable_causes():
    """Distinct skip causes seen so far, for harness and test reporting."""
    with _DETAIL_MISSING_LOCK:
        return sorted(_DETAIL_MISSING_SEEN)


def reset_identity_detail_warnings():
    """Clear the once-per-process record. Used by tests."""
    with _DETAIL_MISSING_LOCK:
        _DETAIL_MISSING_SEEN.clear()


DETAIL_SCHEMA = "roop.identity_detail.v1"
DETAIL_SCHEMA_HIGH_RES = "roop.identity_detail.v2"
DETAIL_SIZE = 64
DETAIL_SIZE_HIGH_RES = 256
CANONICAL_SIZE = 128
RESIDUAL_SCALE = 0.25
RESIDUAL_LIMIT = 24.0


def _clamp01(value):
    try:
        return float(max(0.0, min(1.0, float(value))))
    except (TypeError, ValueError):
        return 0.0


def _decode_channel(value, shape, scale=1.0, offset=0.0):
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.size != int(np.prod(shape)):
        return None
    arr = arr.reshape(shape) * float(scale) + float(offset)
    return arr if np.isfinite(arr).all() else None


def _encode_channel(value, scale=1.0, offset=0.0):
    arr = np.asarray(value, dtype=np.float32)
    encoded = np.rint((arr - float(offset)) / float(scale))
    return np.clip(encoded, 0, 255).astype(np.uint8).reshape(-1).tolist()


def decode_detail(detail):
    """Decode a validated-or-untrusted V2 detail object safely."""
    if not isinstance(detail, dict):
        return None
    schema = detail.get("schema")
    if schema == DETAIL_SCHEMA:
        size = DETAIL_SIZE
    elif schema == DETAIL_SCHEMA_HIGH_RES:
        size = DETAIL_SIZE_HIGH_RES
    else:
        return None
    shape = detail.get("shape") or [size, size]
    try:
        shape = tuple(int(v) for v in shape)
    except (TypeError, ValueError):
        return None
    if shape != (size, size):
        return None
    residual = _decode_channel(detail.get("residual_q"), shape,
                               RESIDUAL_SCALE, -128.0 * RESIDUAL_SCALE)
    confidence = _decode_channel(detail.get("confidence_q"), shape,
                                  1.0 / 255.0)
    mask = _decode_channel(detail.get("mask_q"), shape, 1.0 / 255.0)
    if residual is None or confidence is None or mask is None:
        return None
    return {
        "residual": residual.astype(np.float32),
        "confidence": np.clip(confidence, 0.0, 1.0).astype(np.float32),
        "mask": np.clip(mask, 0.0, 1.0).astype(np.float32),
        "support": int(detail.get("support", detail.get("source_count", 1)) or 1),
        "source_count": int(detail.get("source_count", 1) or 1),
        "confidence_scalar": _clamp01(detail.get("confidence", 0.0)),
    }


def build_detail_representation(crop, quality_confidence=0.0,
                                detail_size=DETAIL_SIZE):
    """Build one source observation in canonical face detail space.

    The broad Gaussian is the source illumination/skin field and is discarded.
    Only the signed residual remains.  A robust local noise estimate lowers the
    confidence of camera/JPEG-scale fluctuations; the later cross-reference
    aggregation is what turns this observation into persistent identity detail.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return {}
    if detail_size not in (DETAIL_SIZE, DETAIL_SIZE_HIGH_RES):
        return {}
    try:
        small = cv2.resize(crop, (detail_size, detail_size),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        fine_base = cv2.GaussianBlur(gray, (0, 0),
                                     1.35 if detail_size == DETAIL_SIZE else 1.5)
        residual = gray - fine_base
        if detail_size == DETAIL_SIZE_HIGH_RES:
            # Freckles span several pixels at 256px. A fine high-pass keeps
            # their edges but loses the dark centre, so retain the negative
            # middle-scale component separately from broad lighting/shading.
            middle = gray - cv2.GaussianBlur(gray, (0, 0), 5.0)
            residual += 0.8 * np.minimum(middle, 0.0)
        centre = residual[5:-5, 5:-5]
        median = float(np.median(centre)) if centre.size else 0.0
        mad = float(np.median(np.abs(centre - median))) if centre.size else 0.0
        noise_floor = max(0.35, 1.4826 * mad)

        # Soft confidence: low-amplitude independent noise is not emitted as a
        # detail point, while stronger marks and stable wrinkle strokes remain.
        signal = np.clip((np.abs(residual) - max(1.25, 1.5 * noise_floor)) / 10.0,
                         0.0, 1.0)
        quality = _clamp01(quality_confidence)
        confidence = quality * (0.12 + 0.88 * signal)
        mask = np.clip(signal * (0.35 + 0.65 * quality), 0.0, 1.0)
        residual = np.clip(residual, -RESIDUAL_LIMIT, RESIDUAL_LIMIT)
        return {
            "schema": (DETAIL_SCHEMA if detail_size == DETAIL_SIZE
                       else DETAIL_SCHEMA_HIGH_RES),
            "space": f"arcface_{detail_size}",
            "shape": [detail_size, detail_size],
            "residual_q": _encode_channel(residual, RESIDUAL_SCALE,
                                            -128.0 * RESIDUAL_SCALE),
            "confidence_q": _encode_channel(confidence, 1.0 / 255.0),
            "mask_q": _encode_channel(mask, 1.0 / 255.0),
            "support": 1,
            "source_count": 1,
            "confidence": round(quality, 6),
            "noise_floor": round(noise_floor, 5),
        }
    except (cv2.error, TypeError, ValueError):
        return {}


def aggregate_detail_representations(details, weights=None):
    """Keep only detail agreeing across the FaceSet V2 reference gallery."""
    decoded = [decode_detail(item) for item in (details or [])]
    decoded = [item for item in decoded if item is not None]
    if not decoded:
        return {}
    stack = np.stack([item["residual"] for item in decoded], axis=0)
    median = np.median(stack, axis=0).astype(np.float32)
    tolerance = np.maximum(1.25, 0.35 * np.abs(median) + 0.75)
    support = np.mean(np.abs(stack - median[None, ...]) <= tolerance[None, ...], axis=0)
    confidence_stack = np.stack([item["confidence"] for item in decoded], axis=0)
    confidence = np.mean(confidence_stack, axis=0) * support
    # A median near zero is exactly what independent sensor/JPEG noise produces.
    signal = np.clip((np.abs(median) - 1.0) / 9.0, 0.0, 1.0)
    confidence *= 0.20 + 0.80 * signal
    mask = np.clip(confidence * (0.35 + 0.65 * signal), 0.0, 1.0)
    source_weights = np.asarray(weights if weights is not None else
                                [1.0] * len(decoded), dtype=np.float32)
    source_weights = source_weights[:len(decoded)]
    scalar = float(np.average([item["confidence_scalar"] for item in decoded],
                              weights=source_weights)) if source_weights.size else 0.0
    scalar *= min(1.0, float(len(decoded)) / 2.0)
    return {
        "schema": DETAIL_SCHEMA,
        "space": "arcface_128",
        "shape": [DETAIL_SIZE, DETAIL_SIZE],
        "residual_q": _encode_channel(np.clip(median, -RESIDUAL_LIMIT, RESIDUAL_LIMIT),
                                        RESIDUAL_SCALE, -128.0 * RESIDUAL_SCALE),
        "confidence_q": _encode_channel(confidence, 1.0 / 255.0),
        "mask_q": _encode_channel(mask, 1.0 / 255.0),
        "support": len(decoded),
        "source_count": len(decoded),
        "confidence": round(_clamp01(scalar), 6),
    }


def _landmarks_crop(target_face, matrix, output_shape, matrix_shape=None):
    value = None
    # Only 68-point layouts have the eye/nose/lip indices used below. The
    # InsightFace 106-point layout has a different ordering and must not be
    # interpreted as its first 68 points.
    for name in ("landmark_2d_68", "landmark_3d_68", "landmarks_2d"):
        try:
            value = getattr(target_face, name, None)
        except Exception:
            value = None
        if value is None and isinstance(target_face, dict):
            value = target_face.get(name)
        if value is not None:
            break
    try:
        lm = np.asarray(value, dtype=np.float32)
        if lm.ndim != 2 or lm.shape[0] != 68 or lm.shape[1] < 2:
            value = (target_face.get('kps') if isinstance(target_face, dict)
                     else getattr(target_face, 'kps', None))
            lm = np.asarray(value, dtype=np.float32)
            if lm.ndim != 2 or lm.shape[0] != 5 or lm.shape[1] < 2:
                return None
        lm = lm[:, :2]
        M = np.asarray(matrix, dtype=np.float32)
        crop = np.hstack([lm, np.ones((lm.shape[0], 1), np.float32)]) @ M.T
        source_shape = matrix_shape or output_shape
        if source_shape != output_shape:
            crop[:, 0] *= float(output_shape[1]) / max(1.0, source_shape[1])
            crop[:, 1] *= float(output_shape[0]) / max(1.0, source_shape[0])
        return crop if np.isfinite(crop).all() else None
    except (TypeError, ValueError):
        return None


def _protected_feature_mask(shape, landmarks):
    h, w = shape[:2]
    central = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(central, (w // 2, h // 2), (int(w * .39), int(h * .45)),
                0, 0, 360, 1.0, -1)
    central = cv2.GaussianBlur(central, (0, 0), max(1.0, min(h, w) / 80.0))
    if landmarks is None:
        return central
    protected = np.zeros((h, w), dtype=np.uint8)
    try:
        if len(landmarks) == 5:
            eye_dist = max(4.0, float(np.linalg.norm(landmarks[1] - landmarks[0])))
            mouth_dist = max(4.0, float(np.linalg.norm(landmarks[4] - landmarks[3])))
            features = (
                (landmarks[0], (.20 * eye_dist, .13 * eye_dist)),
                (landmarks[1], (.20 * eye_dist, .13 * eye_dist)),
                (landmarks[2], (.16 * eye_dist, .19 * eye_dist)),
                ((landmarks[3] + landmarks[4]) / 2,
                 (.82 * mouth_dist, .17 * eye_dist)),
            )
            for centre, axes in features:
                cv2.ellipse(protected, tuple(np.rint(centre).astype(int)),
                            tuple(max(2, int(v)) for v in axes),
                            0, 0, 360, 255, -1)
        else:
            eye_dist = max(4.0, float(np.linalg.norm(
                landmarks[45].astype(np.float32) - landmarks[36].astype(np.float32))))
            # Protect eye/lid, nostril and lip structures while leaving skin.
            for points, radius in ((range(36, 48), 0.12 * eye_dist),
                                   (range(31, 36), 0.09 * eye_dist),
                                   (range(48, 68), 0.25 * eye_dist)):
                pts = landmarks[list(points)].astype(np.float32)
                centre = np.mean(pts, axis=0)
                axes = (max(2, int(np.ptp(pts[:, 0]) * 0.65 + radius)),
                        max(2, int(np.ptp(pts[:, 1]) * 0.65 + radius)))
                cv2.ellipse(protected, tuple(np.rint(centre).astype(int)), axes,
                            0, 0, 360, 255, -1)
        # The crop boundary/hairline is never a safe place to invent source
        # detail. A soft central surface floor handles missing landmarks too.
        allowed = 1.0 - (protected.astype(np.float32) / 255.0)
        return central * cv2.GaussianBlur(allowed, (0, 0), max(0.8, min(h, w) / 180.0))
    except (TypeError, ValueError, cv2.error):
        return central


def _template_warp(channel, output_shape, template="arcface", target_landmarks=None):
    """Map the stored arcface canonical map into the active swap template."""
    h, w = output_shape[:2]
    base = np.asarray(channel, dtype=np.float32)
    canonical_size = base.shape[1]
    if str(template or "arcface").lower() == "arcface":
        return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    try:
        from roop.face_util import swap_template_points
        src = swap_template_points(canonical_size, "arcface").astype(np.float32)
        dst = swap_template_points(w, template, landmarks=target_landmarks).astype(np.float32)
        if h != w:
            dst[:, 1] *= float(h) / max(1.0, w)
        forward, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if forward is None:
            return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
        # warpAffine without WARP_INVERSE_MAP expects canonical -> target.
        return cv2.warpAffine(base, forward, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    except (ImportError, TypeError, ValueError, cv2.error):
        return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


def restore_identity_detail(face_img, detail, target_face=None, matrix=None,
                            strength=0.0, visibility_mask=None,
                            target_template="arcface", temporal_manager=None,
                            track_id=None, motion=0.0, source_index=None,
                            transition_alpha=1.0, matrix_shape=None,
                            return_metrics=False):
    """Restore a confidence-weighted source detail map onto a swapped crop."""
    original = face_img
    s = _clamp01(strength)
    decoded = decode_detail(detail)
    if s <= 0.0 or decoded is None or face_img is None or getattr(face_img, "size", 0) == 0:
        return (original, {"applied_fraction": 0.0, "energy": 0.0}) if return_metrics else original
    try:
        h, w = face_img.shape[:2]
        target_landmarks = getattr(target_face, 'kps', None)
        residual = _template_warp(decoded["residual"], (h, w), target_template, target_landmarks)
        confidence = _template_warp(decoded["confidence"], (h, w), target_template, target_landmarks)
        detail_mask = _template_warp(decoded["mask"], (h, w), target_template, target_landmarks)
        landmarks = (_landmarks_crop(target_face, matrix, (h, w),
                                     matrix_shape) if target_face is not None and matrix is not None else None)
        safe = _protected_feature_mask((h, w), landmarks)
        if visibility_mask is not None:
            visible = np.asarray(visibility_mask, dtype=np.float32)
            if visible.ndim == 3:
                visible = visible[..., 0]
            if visible.shape[:2] != (h, w):
                visible = cv2.resize(visible, (w, h), interpolation=cv2.INTER_LINEAR)
            safe *= np.clip(visible, 0.0, 1.0)

        # Adapt amplitude to the target's own exposure/texture field. This
        # preserves the target lighting direction and reduces detail in dark,
        # blurred, or very small faces rather than punching bright source marks.
        face = face_img.astype(np.float32)
        gray = cv2.cvtColor(np.clip(face, 0, 255).astype(np.uint8),
                            cv2.COLOR_BGR2GRAY).astype(np.float32)
        target_residual = gray - cv2.GaussianBlur(gray, (0, 0), 1.35)
        source_scale = max(1.0, float(np.percentile(np.abs(residual), 75)))
        target_scale = float(np.percentile(np.abs(target_residual), 75))
        # A smooth swap is exactly where source marks need restoration. Using
        # its near-zero texture energy to halve the source signal made 0 and 1
        # look the same on those frames.
        lighting_gain = np.clip(target_scale / source_scale, 1.0, 1.15)
        luma_gain = np.clip((cv2.GaussianBlur(gray, (0, 0), 3.0) / 128.0),
                            0.45, 1.15)
        # Smooth/cap the residual so an isolated noisy pixel cannot become an
        # artificial sharp point after upscaling or a later codec pass.
        residual = cv2.GaussianBlur(residual, (0, 0), max(0.45, w / 900.0))
        residual = RESIDUAL_LIMIT * np.tanh(residual / RESIDUAL_LIMIT)
        # Confidence and mask both encode the same source signal; multiplying
        # both and the scalar made real marks disappear after uint8 rounding.
        applied = (s * np.maximum(confidence, detail_mask)
                   * safe * lighting_gain * luma_gain)
        if temporal_manager is not None and track_id is not None:
            try:
                residual = temporal_manager.blend_detail(
                    track_id, residual, confidence=decoded["confidence_scalar"],
                    motion=motion, source_index=source_index,
                    transition_alpha=transition_alpha)
            except Exception:
                pass
        delta = residual * applied
        out = np.clip(face + delta[..., None], 0.0, 255.0).astype(np.uint8)
        metrics = {
            "applied_fraction": float(np.mean(applied > 0.03)),
            "energy": float(np.mean(np.abs(delta))),
            "confidence": float(decoded["confidence_scalar"]),
            "lighting_gain": float(lighting_gain),
        }
        return (out, metrics) if return_metrics else out
    except (TypeError, ValueError, cv2.error):
        return (original, {"applied_fraction": 0.0, "energy": 0.0}) if return_metrics else original


__all__ = [
    "DETAIL_SCHEMA", "DETAIL_SIZE", "build_detail_representation",
    "aggregate_detail_representations", "decode_detail",
    "restore_identity_detail",
]
