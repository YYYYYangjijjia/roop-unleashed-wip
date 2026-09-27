"""A source intake: local PNG detections, raw mean, fixed first-face identity."""
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

import cv2
import numpy as np

from roop.FaceSet import FaceSet
from .detector import AFaceDetector


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _extract_archive(path, destination):
    """Extract into our own temporary directory; reject unsafe archive names."""
    root = Path(destination).resolve()
    with zipfile.ZipFile(path) as archive:
        seen = set()
        members = []
        for member in archive.infolist():
            name = member.filename.replace('\\', '/')
            parts = name.split('/')
            if name.startswith('/') or any(p in ('.', '..') or ':' in p for p in parts):
                raise ValueError(f'Unsafe faceset archive member: {member.filename}')
            target = (root / name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f'Faceset archive member escapes temporary directory: {member.filename}')
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f'Faceset archive symlink is unsupported: {member.filename}')
            key = os.path.normcase(str(target))
            if key in seen:
                raise ValueError(f'Duplicate faceset archive member: {member.filename}')
            seen.add(key)
            members.append((member, target))
        for member, target in members:
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)


def _collect(paths, original_path, cfg, average):
    detector = AFaceDetector(cfg)
    faceset = FaceSet()
    thumbnail = None
    entries = []
    try:
        for path in paths:
            data = detector.extract_face_images(str(path), (False, 0))
            entries.append({'name': Path(path).name, 'sha256': _sha256(path), 'faces': len(data)})
            for face, crop in data:
                face.mask_offsets = (0, 0, 0, 0, 1, 20)
                faceset.faces.append(face)
                if thumbnail is None:
                    thumbnail = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                faceset.ref_images.append(image)
        if not faceset.faces:
            raise ValueError('A compatibility detected no source faces.')
        if average and len(faceset.faces) > 1:
            # Exactly A: raw vectors, arithmetic mean, only overwrite faces[0].
            faceset.embeddings_backup = faceset.faces[0]['embedding']
            embeddings = [face.embedding for face in faceset.faces]
            faceset.faces[0]['embedding'] = np.mean(embeddings, axis=0)
        faceset._source_path = str(Path(original_path).resolve())
        faceset.a_compat_ready = True
        faceset.a_compat_manifest = {
            'mode': 'a_compat', 'reference_commit': '1b311e8',
            'source_sha256': _sha256(original_path),
            'files_in_detection_order': entries,
            'face_count': len(faceset.faces),
            'aggregation': 'raw_arithmetic_mean_first_face' if average else 'first_detected_face',
            'source_embedding_sha256': hashlib.sha256(
                np.asarray(faceset.faces[0].embedding).tobytes()).hexdigest(),
            'detector': 'buffalo_l', 'det_size': list(detector.det_size),
            'det_threshold': 0.5,
        }
        return faceset, thumbnail
    finally:
        detector.release()


def load_faceset(path, cfg):
    with tempfile.TemporaryDirectory(prefix='roop_a_faceset_') as temporary:
        _extract_archive(path, temporary)
        # Preserve A's filesystem enumeration and case-sensitive PNG filter.
        files = [Path(temporary) / name for name in os.listdir(temporary)
                 if name.endswith('.png') and (Path(temporary) / name).is_file()]
        return _collect(files, path, cfg, average=True)


def load_source_image(path, cfg):
    return _collect([Path(path)], path, cfg, average=False)
