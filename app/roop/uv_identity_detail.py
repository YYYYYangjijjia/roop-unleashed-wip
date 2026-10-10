"""Optional face-surface transfer of source freckles and fixed skin marks.

The swapper still receives the ordinary averaged identity. This module only
builds a skin-only UV residual from reference photographs and projects it onto
the target's *original* face geometry after swapping. It intentionally selects
one sharp observation per UV region instead of averaging image pixels: a mark
visible in one good reference must not disappear because other photos are soft.
"""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from roop.runtime_paths import models_directory, runtime_config


_ASSETS = {
    'face_landmarker.task': (
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/'
        'face_landmarker/float16/1/face_landmarker.task',
        '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff'),
    'canonical_face_model.obj': (
        'https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/'
        'mediapipe/modules/face_geometry/data/canonical_face_model.obj',
        '8bac80443397e113f41a8b565ea72c59390bc031d9defab289dba7bc0c54e618'),
    'resnet18.onnx': (
        'https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx',
        '0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f'),
}
_SIZE = 512
_TRIAL_LIMIT = 16  # parser work is bounded even for a very large faceset
_INSTANCE = None
_INSTANCE_LOCK = threading.Lock()


def _assets():
    root = models_directory()
    offline = (runtime_config().get('offline') or
               os.environ.get('ROOP_OFFLINE', '').lower() in ('1', 'true', 'yes', 'on'))
    for name, (url, digest) in _ASSETS.items():
        path = root / name
        if not path.is_file():
            if offline:
                raise FileNotFoundError(f'UV detail requires {path}; offline mode prevents download')
            tmp = path.with_suffix(path.suffix + '.download')
            try:
                with urllib.request.urlopen(url, timeout=45) as source, tmp.open('wb') as dest:
                    while True:
                        part = source.read(1024 * 1024)
                        if not part:
                            break
                        dest.write(part)
                if hashlib.sha256(tmp.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'UV detail model checksum mismatch: {name}')
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'UV detail model checksum mismatch: {path}')
    return root


def _barycentric_grid(points, bounds):
    x0, y0, x1, y1 = bounds
    xx, yy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    a, b, c = points
    det = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
    if abs(det) < 1e-6:
        return None, None
    w0 = ((b[1]-c[1])*(xx-c[0]) + (c[0]-b[0])*(yy-c[1])) / det
    w1 = ((c[1]-a[1])*(xx-c[0]) + (a[0]-c[0])*(yy-c[1])) / det
    weights = np.stack((w0, w1, 1-w0-w1), axis=-1).astype(np.float32)
    return weights, np.all(weights >= -1e-4, axis=-1)


def _compact_dark_marks(gray, skin):
    """Keep small dark skin blobs; reject shadows, hair strokes and wrinkles.

    A broad high-pass treats the entire nose shadow as a mole. Connected blob
    constraints make the accepted feature class explicit and bounded. The
    result is a soft opacity, not a copied RGB patch or source illumination.
    """
    smooth = cv2.GaussianBlur(gray, (0, 0), .65)
    dark = np.clip(cv2.GaussianBlur(smooth, (0, 0), 5)-smooth, 0, 24)
    binary = ((dark > 5) & (skin > 0)).astype(np.uint8)
    count, ids, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    accepted = np.zeros(binary.shape, np.uint8)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if not (2 <= area <= 90 and 2 <= w <= 15 and 2 <= h <= 15):
            continue
        if max(w, h) / max(1, min(w, h)) > 2.5:
            continue
        patch = (ids[y:y+h, x:x+w] == index)
        yy, xx = np.where(patch)
        if len(xx) >= 3:
            axes = np.linalg.eigvalsh(np.cov(np.stack((xx, yy))))
            if axes[-1] / max(.5, axes[0]) > 6:
                continue
        accepted[y:y+h, x:x+w][patch] = 1
    return dark * cv2.GaussianBlur(accepted.astype(np.float32), (0, 0), .7)


def _coverage_gate(support, coverage):
    """Select source mark confidence independently from final blend strength."""
    # With sixteen references, a genuine mark seen in just one sharp photo
    # naturally has support around 1/16. A linear .6 threshold would erase
    # nearly everything in an edited/mixed gallery. The quadratic schedule
    # makes 0 selective (~5 references), .5 moderate (~2), 1 permissive (~1).
    amount = np.clip(float(coverage), 0, 1)
    threshold = .30*(1-amount)**2 + .01
    width = .025 + .025*(1-amount)
    return np.clip((support-threshold)/width, 0, 1)


def _parse_geometry(path: Path):
    vertices, coords, faces = [], [], []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('v '):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith('vt '):
            coords.append([float(v) for v in line.split()[1:3]])
        elif line.startswith('f '):
            faces.append([[int(v)-1 for v in item.split('/')[:2]]
                          for item in line.split()[1:4]])
    vertices = np.asarray(vertices, np.float32)
    coords = np.asarray(coords, np.float32)
    faces = np.asarray(faces, np.int32)
    if len(vertices) != 468 or len(faces) < 800:
        raise ValueError('Unexpected MediaPipe canonical mesh')
    triangles = faces[:, :, 0]
    uv = np.zeros((len(vertices), 2), np.float32)
    for vertex, texcoord in faces.reshape(-1, 2):
        uv[vertex] = coords[texcoord]
    uv[:, 1] = 1 - uv[:, 1]
    uv *= _SIZE-1
    normals = np.cross(vertices[triangles[:, 1]]-vertices[triangles[:, 0]],
                       vertices[triangles[:, 2]]-vertices[triangles[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)
    ids = np.full((_SIZE, _SIZE), -1, np.int16)
    bary = np.zeros((_SIZE, _SIZE, 3), np.float32)
    for index, tri in enumerate(triangles):
        points = uv[tri]
        x0 = max(0, int(np.floor(points[:, 0].min())))
        x1 = min(_SIZE, int(np.ceil(points[:, 0].max()))+1)
        y0 = max(0, int(np.floor(points[:, 1].min())))
        y1 = min(_SIZE, int(np.ceil(points[:, 1].max()))+1)
        if x1 <= x0 or y1 <= y0:
            continue
        weights, inside = _barycentric_grid(points, (x0, y0, x1, y1))
        if weights is None:
            continue
        ids[y0:y1, x0:x1][inside] = index
        bary[y0:y1, x0:x1][inside] = weights[inside]
    return triangles, uv, normals, ids, bary


class UVIdentityDetail:
    def __init__(self):
        import mediapipe as mp
        import onnxruntime as ort
        root = _assets()
        self.mp = mp
        self.tri, self.uv, self.normals, self.ids, self.bary = _parse_geometry(
            root / 'canonical_face_model.obj')
        self.safe_uv = np.ones((_SIZE, _SIZE), np.uint8)
        # These structures have dark edges that look like "marks" after a
        # simple local-contrast extraction. Keep the initial feature scope to
        # cheek/forehead/chin skin rather than inventing nostril/lip strokes.
        for vertices, axes in (((33, 133), (62, 28)),
                               ((362, 263), (62, 28)),
                               ((1, 2), (42, 50)),
                               ((61, 291), (82, 38))):
            centre = np.mean(self.uv[list(vertices)], axis=0).astype(int)
            cv2.ellipse(self.safe_uv, tuple(centre), axes, 0, 0, 360, 0, -1)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(root / 'face_landmarker.task')),
            num_faces=1, output_facial_transformation_matrixes=True)
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        parser_path = root / 'resnet18.onnx'
        self.parser = ort.InferenceSession(str(parser_path), providers=['CPUExecutionProvider'])
        self.parser_input = self.parser.get_inputs()[0].name
        self.lock = threading.Lock()  # MediaPipe Task and ONNX session share workers

    def _detect(self, image):
        mp = self.mp
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=np.ascontiguousarray(rgb)))
        if not result.face_landmarks:
            return None
        h, w = image.shape[:2]
        landmarks = np.array([[p.x*w, p.y*h] for p in result.face_landmarks[0][:468]],
                             dtype=np.float32)
        rotation = np.asarray(result.facial_transformation_matrixes[0],
                              dtype=np.float32)[:3, :3]
        return landmarks, rotation

    def _skin(self, image):
        resized = cv2.resize(image, (_SIZE, _SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array([.485, .456, .406], np.float32)) / np.array(
            [.229, .224, .225], np.float32)
        blob = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)
        labels = self.parser.run(None, {self.parser_input: blob})[0][0].argmax(0)
        return cv2.resize((labels == 1).astype(np.uint8),
                          (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    def _uv_maps(self, landmarks, rotation):
        valid = self.ids >= 0
        ids = self.ids[valid]
        front = (self.normals @ rotation.T)[:, 2]
        visible = np.zeros((_SIZE, _SIZE), np.uint8)
        visible[valid] = (front[ids] > .3).astype(np.uint8)
        points = np.sum(landmarks[self.tri[ids]] * self.bary[valid, :, None], axis=1)
        mapx = np.full((_SIZE, _SIZE), -1, np.float32)
        mapy = np.full((_SIZE, _SIZE), -1, np.float32)
        mapx[valid], mapy[valid] = points[:, 0], points[:, 1]
        facing = np.zeros((_SIZE, _SIZE), np.float32)
        facing[valid] = np.clip(front[ids], 0, 1)
        return mapx, mapy, visible, facing

    def build(self, reference_images):
        """Build a selected-observation skin atlas; never average raw pixels."""
        candidates = []
        with self.lock:
            for index, image in enumerate(reference_images):
                if image is None or getattr(image, 'size', 0) == 0:
                    continue
                observation = self._detect(image)
                if observation is None:
                    continue
                landmarks, rotation = observation
                extent = landmarks.max(axis=0) - landmarks.min(axis=0)
                x0, y0 = np.maximum(np.floor(landmarks.min(axis=0)).astype(int), 0)
                x1, y1 = np.minimum(np.ceil(landmarks.max(axis=0)).astype(int),
                                    np.array([image.shape[1], image.shape[0]]))
                if x1 <= x0 or y1 <= y0:
                    continue
                gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                sharp = float(cv2.Laplacian(gray, cv2.CV_32F).var())
                size = float(min(extent))
                score = np.log1p(sharp) + .2*np.log1p(size)
                yaw = float(np.degrees(np.arctan2(rotation[0, 2], rotation[2, 2])))
                candidates.append((score, index, yaw, image, landmarks, rotation))
            # Preserve angle coverage before filling remaining slots by quality.
            ranked = sorted(candidates, reverse=True, key=lambda row: row[0])
            # A bin can be overfull; retain at most three per bin and then global best.
            chosen = []
            for low, high in ((-90, -25), (-25, -10), (-10, 10), (10, 25), (25, 90)):
                chosen.extend([row for row in ranked if low <= row[2] < high][:3])
            seen = {row[1] for row in chosen}
            chosen.extend(row for row in ranked if row[1] not in seen)
            chosen = chosen[:_TRIAL_LIMIT]
            if not chosen:
                raise ValueError('UV detail needs at least one readable face reference')
            best_score = np.full((_SIZE, _SIZE), -np.inf, np.float32)
            residual = np.zeros((_SIZE, _SIZE), np.float32)
            source_ids = np.full((_SIZE, _SIZE), -1, np.int16)
            observed = np.zeros((_SIZE, _SIZE), np.float32)
            marked = np.zeros((_SIZE, _SIZE), np.float32)
            for quality, index, _, image, landmarks, rotation in chosen:
                mapx, mapy, visible, facing = self._uv_maps(landmarks, rotation)
                source_skin = cv2.erode(self._skin(image),
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
                skin = cv2.remap(source_skin, mapx, mapy, cv2.INTER_NEAREST)
                skin = (cv2.erode(skin, np.ones((3, 3), np.uint8)) *
                        visible * self.safe_uv)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
                dark = _compact_dark_marks(gray, source_skin)
                mark = cv2.remap(dark, mapx, mapy, cv2.INTER_LINEAR)
                local = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)),
                                         (0, 0), 3)
                sharp = cv2.remap(local, mapx, mapy, cv2.INTER_LINEAR)
                # Smooth selection scores so a different photo does not win on
                # each freckle pixel. This chooses a good observation per area.
                score = cv2.GaussianBlur(sharp, (0, 0), 9) * facing
                score = np.clip(score / 12.0, .15, 2.0)
                score *= min(1.0, max(.25, quality / 8.0)) * skin
                # Compare mark presence only among reliably visible skin.
                # A blurred photo has a smaller local-sharpness weight, so its
                # missing freckle is weak negative evidence. A sharp conflicting
                # photo counts strongly. A 2px tolerance absorbs UV jitter.
                present = cv2.GaussianBlur((mark > 4.0).astype(np.float32),
                                           (0, 0), 1.4)
                observed += score
                marked += score * present
                good = (skin > 0) & (mark > 3.0) & (score > best_score)
                residual[good], best_score[good], source_ids[good] = (
                    mark[good], score[good], index)
            support = np.divide(marked, np.maximum(observed, 1e-6))
            support[observed <= 1e-6] = 0
            alpha = cv2.GaussianBlur((source_ids >= 0).astype(np.float32),
                                     (0, 0), 1.0)
            return {'residual': residual, 'alpha': alpha,
                    'support': np.clip(support, 0, 1),
                    'source_ids': source_ids, 'source_count': len(chosen),
                    'selected_indices': [row[1] for row in chosen],
                    'coverage': float(np.mean(source_ids >= 0))}

    def apply(self, swapped, target_original, atlas, strength, coverage=.5,
              visibility_mask=None):
        """Project the source UV atlas through target landmarks onto a crop."""
        if atlas is None or strength <= 0:
            return swapped
        h, w = swapped.shape[:2]
        original = cv2.resize(target_original, (w, h), interpolation=cv2.INTER_LINEAR)
        # Small swap crops are scaled for more reliable landmark detection.
        scale = max(1.0, 384.0/min(h, w))
        detection_image = (cv2.resize(original, (round(w*scale), round(h*scale)))
                           if scale > 1 else original)
        with self.lock:
            observation = self._detect(detection_image)
            if observation is None:
                return swapped
            landmarks, rotation = observation
            landmarks /= scale
            skin = self._skin(original)
        mapx = np.full((h, w), -1, np.float32)
        mapy = np.full((h, w), -1, np.float32)
        weight = np.zeros((h, w), np.float32)
        facing = (self.normals @ rotation.T)[:, 2]
        for index, tri in enumerate(self.tri):
            if facing[index] <= .3:
                continue
            points = landmarks[tri]
            x0 = max(0, int(np.floor(points[:, 0].min())))
            x1 = min(w, int(np.ceil(points[:, 0].max()))+1)
            y0 = max(0, int(np.floor(points[:, 1].min())))
            y1 = min(h, int(np.ceil(points[:, 1].max()))+1)
            if x1 <= x0 or y1 <= y0:
                continue
            bary, inside = _barycentric_grid(points, (x0, y0, x1, y1))
            if bary is None:
                continue
            coords = np.sum(self.uv[tri] * bary[:, :, :, None], axis=2)
            mapx[y0:y1, x0:x1][inside] = coords[:, :, 0][inside]
            mapy[y0:y1, x0:x1][inside] = coords[:, :, 1][inside]
            weight[y0:y1, x0:x1][inside] = float(min(1, facing[index]))
        mark = cv2.remap(atlas['residual'], mapx, mapy, cv2.INTER_LINEAR)
        alpha = cv2.remap(atlas['alpha'], mapx, mapy, cv2.INTER_LINEAR) * weight
        support = cv2.remap(atlas['support'], mapx, mapy, cv2.INTER_LINEAR)
        # Coverage controls *which* marks are admitted. The existing strength
        # parameter separately controls the alpha of those selected marks.
        alpha *= _coverage_gate(support, coverage)
        alpha *= cv2.GaussianBlur(cv2.erode(skin, np.ones((5, 5), np.uint8))
                                  .astype(np.float32), (0, 0), 1.0)
        if visibility_mask is not None:
            visible = np.asarray(visibility_mask, np.float32)
            if visible.ndim == 3:
                visible = visible[:, :, 0]
            if visible.shape != (h, w):
                visible = cv2.resize(visible, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha *= np.clip(visible, 0, 1)
        # A 512px UV freckle commonly occupies only 1-2 output pixels on a
        # 256px face crop. Without contrast compensation its sub-3-level delta
        # vanishes in JPEG/video quantisation. Cap the resulting darkening so
        # a single mark cannot become a black paste-on dot.
        delta = np.minimum(mark * 2.2, 40) * np.clip(alpha, 0, 1) * np.clip(strength, 0, 1)
        return np.clip(swapped.astype(np.float32)-delta[:, :, None], 0, 255).astype(np.uint8)


def uv_detail_engine():
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = UVIdentityDetail()
    return _INSTANCE
