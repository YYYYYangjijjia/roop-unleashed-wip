"""Thin, opt-in runtime adapter around the frozen A (1b311e8) frame kernel.

This module owns its source copies, detector, inference sessions and lock. It
never patches roop.globals or routes a frame through C's rendering mixins.
The surrounding C manager continues to own video I/O, cancellation and UI.
"""
from copy import copy, deepcopy
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import math

from .frame_engine import AFrameEngine, eNoFaceAction


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _setting(options, cfg, name, default=None):
    value = _get(options, name, None)
    return _get(cfg, name, default) if value is None else value


def _zero(value):
    try:
        return math.isfinite(float(value)) and float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _clone_face(face):
    # InsightFace Face returns None for unknown attributes, including pickle
    # hooks inspected by deepcopy. Copy its plain mapping, then reconstruct it.
    if isinstance(face, dict):
        return type(face)(deepcopy(dict(face)))
    return deepcopy(face)


def _clone_faceset(faceset):
    # Reference images and metadata are read-only to the frozen frame engine;
    # only its faces receive mask offsets, matrices or other mutable fields.
    result = copy(faceset)
    result.faces = [_clone_face(face) for face in faceset.faces]
    return result


def validate_a_configuration(options, cfg):
    """Validate before allocating sessions; unsupported options fail explicitly."""
    errors = []
    if not _setting(options, cfg, "a_compatibility_mode", False):
        errors.append("a_compatibility_mode must be enabled")
    if _get(options, "swap_model", "inswapper") != "inswapper":
        errors.append("swap model must be inswapper")
    if _get(options, "swap_mode", "first") != "first":
        errors.append("face selection must be First found")
    if _setting(options, cfg, "source_identity_mode", "pose") != "average":
        errors.append("source_identity_mode must be average")
    if str(_get(cfg, "selected_enhancer", "None")).lower() != "none":
        errors.append("enhancer must be None")
    if str(_get(cfg, "enhancer_type", "none")).lower() != "none":
        errors.append("enhancer_type must be none")
    if _get(cfg, "mask_engine", "DFL XSeg") != "DFL XSeg":
        errors.append("mask engine must be DFL XSeg")
    if str(_get(cfg, "mask_engine_2", "None")).lower() != "none":
        errors.append("second mask engine must be None")
    processor_keys = set(_get(options, "processors", {}) or {})
    if processor_keys != {"faceswap", "mask_xseg"}:
        errors.append("processors must contain only faceswap and mask_xseg")
    if _get(options, "num_swap_steps", 1) != 1:
        errors.append("swap steps must be 1")
    if _get(options, "subsample_size", 256) not in (128, 256, 512):
        errors.append("subsample size must be 128, 256 or 512")
    if _get(options, "imagemask", None) is not None:
        errors.append("manual masks are not supported in A compatibility mode")
    if _get(options, "show_face_masking", False) or _get(options, "show_face_area_overlay", False):
        errors.append("mask previews/overlays must be disabled")

    disabled = (
        "restore_original_mouth", "restore_original_eyes", "use_3d_recon",
        "use_source_bank", "use_frontalization", "stabilize_face",
        "stabilize_enhancer", "temporal_detection", "temporal_roi_hint",
        "track_identities", "rescue_small_faces", "refine_landmarks",
        "jaw_reshape", "enhancer_align", "color_match_after_enhance",
        "lipsync_enabled", "upscale_after_swap", "vr_mode",
    )
    for name in disabled:
        if _setting(options, cfg, name, False):
            errors.append(f"{name} must be disabled")
    zero_fields = (
        "temporal_smooth_strength", "expression_restore_strength",
        "detail_transfer_strength", "swap_model_mask_strength", "output_face_scale",
        "merger_hist_match", "merger_sharpen", "merger_motion_blur",
        "merger_grain_match", "merger_degrade",
    )
    for name in zero_fields:
        if not _zero(_setting(options, cfg, name, 0)):
            errors.append(f"{name} must be 0")
    if str(_get(cfg, "color_transfer_mode", "none")).lower() != "none":
        errors.append("color transfer must be none")
    if _get(cfg, "interp_after_swap", "off") != "off":
        errors.append("frame interpolation must be off")
    if _get(cfg, "detector_engine", "scrfd") != "scrfd":
        errors.append("detector must be scrfd")
    if _get(cfg, "no_face_action", "Use untouched original frame") != "Use untouched original frame":
        errors.append("no-face action must be Use untouched original frame")
    if _get(cfg, "provider", "cuda") not in ("cuda", "cpu"):
        errors.append("provider must be cuda or cpu (no TensorRT)")

    for name, default, low, high in (("a_mask_erosion", 1, 1, 3), ("a_mask_blur", 15, 10, 50)):
        value = _setting(options, cfg, name, default)
        try:
            number = float(value)
            valid = math.isfinite(number) and number.is_integer() and low <= number <= high
        except (TypeError, ValueError):
            valid = False
        if not valid:
            errors.append(f"{name} must be an integer from {low} to {high}")
    for name in ("mask_top", "mask_bottom", "mask_left", "mask_right"):
        try:
            value = float(_get(cfg, name, 0))
            valid = math.isfinite(value) and 0 <= value < 1
        except (TypeError, ValueError):
            valid = False
        if not valid:
            errors.append(f"{name} must be between 0 (inclusive) and 1")
    if not errors:
        if float(_get(cfg, "mask_top", 0)) + float(_get(cfg, "mask_bottom", 0)) >= 1:
            errors.append("top and bottom mask offsets must leave a nonempty mask")
        if float(_get(cfg, "mask_left", 0)) + float(_get(cfg, "mask_right", 0)) >= 1:
            errors.append("left and right mask offsets must leave a nonempty mask")
    if errors:
        raise ValueError("Unsupported A compatibility settings: " + "; ".join(errors))


class ACompatRenderer:
    def __init__(self, input_faces, target_faces, options, cfg):
        validate_a_configuration(options, cfg)
        if not input_faces:
            raise ValueError("A compatibility requires a source loaded in A mode; reload the source .fsz.")
        for faceset in input_faces:
            if not _get(faceset, "a_compat_ready", False):
                raise ValueError("Source was not parsed with A semantics. Reload the source .fsz in A compatibility mode.")
            if not _get(faceset, "faces", None):
                raise ValueError("A compatibility source contains no faces; reload the source.")
        index = _get(options, "selected_index", 0)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(input_faces):
            raise ValueError("A compatibility selected source index is out of range.")

        self._lock = RLock()
        self._released = False
        self._processors = []
        self.detector = None
        self.engine = None
        self.last_num_swapped = 0
        self.total_swaps = 0
        self._cfg = deepcopy(cfg)
        sources = [_clone_faceset(faceset) for faceset in input_faces]
        targets = [_clone_face(face) for face in target_faces]
        offsets = [float(_get(cfg, name, 0)) for name in
                   ("mask_top", "mask_bottom", "mask_left", "mask_right")]
        offsets.extend((int(_setting(options, cfg, "a_mask_erosion", 1)),
                        int(_setting(options, cfg, "a_mask_blur", 15))))
        for faceset in sources:
            for face in faceset.faces:
                face.mask_offsets = tuple(offsets)

        legacy_options = SimpleNamespace(
            swap_modelname="InSwapper 128", swap_output_size=128,
            face_distance_threshold=float(_get(options, "face_distance_threshold", 0.65)),
            blend_ratio=float(_get(options, "blend_ratio", 0.65)), swap_mode="first",
            selected_index=index, masking_text=_get(options, "masking_text", ""),
            imagemask=None, num_swap_steps=1,
            subsample_size=int(_get(options, "subsample_size", 256)),
            show_face_area_overlay=False, show_face_masking=False,
            restore_original_mouth=False, max_num_reuse_frame=15,
        )
        runtime = SimpleNamespace(
            no_face_action=eNoFaceAction.USE_ORIGINAL_FRAME,
            autorotate_faces=bool(_get(cfg, "autorotate_faces", True)), vr_mode=False,
        )
        device = _get(cfg, "provider", "cuda")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda"
                     else ["CPUExecutionProvider"])
        from roop.runtime_paths import models_directory
        models_dir = models_directory()
        # All files must already exist. Neither processor contains a downloader.
        from .detector import AFaceDetector
        from .processors import AInSwapper, AXSeg
        try:
            self.detector = AFaceDetector(self._cfg, providers)
            self._processors.append(AInSwapper(models_dir, providers, device))
            self._processors.append(AXSeg(models_dir, providers, device))
            self.engine = AFrameEngine(sources, targets, legacy_options, runtime,
                                       self.detector, self._processors)
        except Exception:
            self.release()
            raise

    def process_frame(self, frame):
        with self._lock:
            if self._released:
                raise RuntimeError("A compatibility renderer has been released.")
            self.engine.last_num_swapped = 0
            result = self.engine.process_frame(frame)
            self.last_num_swapped = self.engine.last_num_swapped
            self.total_swaps += self.last_num_swapped
            return result

    def release(self):
        with self._lock:
            for processor in self._processors:
                processor.Release()
            self._processors.clear()
            if self.detector is not None:
                release = getattr(self.detector, "release", None)
                if release is not None:
                    release()
            self.detector = None
            self.engine = None
            self._released = True
