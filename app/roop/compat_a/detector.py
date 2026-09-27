"""Independent, local-only buffalo_l analyser following A's detection path."""
from pathlib import Path
import threading

import cv2
import numpy as np

from .geometry import clamp_cut_values, resize_image_keep_content


class AFaceDetector:
    def __init__(self, cfg, providers=None):
        self.cfg = cfg
        if providers is None:
            providers = getattr(cfg, 'execution_providers', None)
        if providers is None and getattr(cfg, 'provider', None) in ('cuda', 'cpu'):
            providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                         if cfg.provider == 'cuda' else ['CPUExecutionProvider'])
        if providers is None:
            import roop.globals
            providers = roop.globals.execution_providers
        self.providers = list(providers)
        if getattr(cfg, 'force_cpu', False):
            self.providers = ['CPUExecutionProvider']
        from roop.runtime_paths import models_directory
        self.models_dir = Path(getattr(cfg, 'models_dir', None) or models_directory())
        self.det_size = (640, 640) if getattr(cfg, 'default_det_size', True) else (320, 320)
        self.allowed_modules = ['landmark_3d_68', 'landmark_2d_106', 'detection', 'recognition']
        if getattr(cfg, 'swap_mode', '') in ('all_female', 'all_male'):
            self.allowed_modules.append('genderage')
        self._analyser = None
        self._lock = threading.RLock()

    def get_face_analyser(self):
        with self._lock:
            if self._analyser is not None:
                return self._analyser
            # FaceAnalysis.__init__ calls ensure_available, which may download.
            # Build its documented model fields from explicit local ONNX files
            # instead, then use the original prepare/get implementation.
            names = ['1k3d68.onnx', '2d106det.onnx', 'det_10g.onnx', 'w600k_r50.onnx']
            if 'genderage' in self.allowed_modules:
                names.append('genderage.onnx')
            folder = self.models_dir / 'buffalo_l'
            for name in names:
                path = folder / name
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f'A compatibility requires local model: {path}')
            from insightface.app import FaceAnalysis
            from insightface import model_zoo
            fa = FaceAnalysis.__new__(FaceAnalysis)
            fa.models = {}
            fa.model_dir = str(folder)
            for name in sorted(names):
                model = model_zoo.get_model(str(folder / name), providers=self.providers)
                if model is None:
                    raise RuntimeError(f'Unrecognized A compatibility model: {folder / name}')
                requested_cuda = any((p[0] if isinstance(p, (tuple, list)) else p)
                                     == 'CUDAExecutionProvider' for p in self.providers)
                if requested_cuda and 'CUDAExecutionProvider' not in model.session.get_providers():
                    raise RuntimeError(f'A compatibility detector fell back from CUDA: {name}')
                fa.models[model.taskname] = model
            missing = set(self.allowed_modules) - set(fa.models)
            if missing:
                raise RuntimeError(f'A compatibility model tasks missing: {sorted(missing)}')
            fa.det_model = fa.models['detection']
            fa.prepare(ctx_id=0, det_size=self.det_size)
            self._analyser = fa
            return fa

    def get_all_faces(self, frame):
        with self._lock:
            return sorted(self.get_face_analyser().get(frame), key=lambda f: f.bbox[0])

    def get_first_face(self, frame):
        with self._lock:
            faces = self.get_face_analyser().get(frame)
            return min(faces, key=lambda f: f.bbox[0]) if faces else None

    def release(self):
        with self._lock:
            self._analyser = None


    def extract_face_images(self, source_filename, video_info=(False, 0), extra_padding=-1.0):
        face_data = []
        source_image = None

        if video_info[0]:
            from roop.capturer import get_video_frame
            frame = get_video_frame(source_filename, video_info[1])
            if frame is not None:
                source_image = frame
            else:
                return face_data
        else:
            source_image = cv2.imdecode(np.fromfile(source_filename, dtype=np.uint8), cv2.IMREAD_COLOR)

        faces = self.get_all_faces(source_image)
        if faces is None:
            return face_data

        i = 0
        for face in faces:
            (startX, startY, endX, endY) = face["bbox"].astype("int")
            startX, endX, startY, endY = clamp_cut_values(startX, endX, startY, endY, source_image)
            if extra_padding > 0.0:
                if source_image.shape[:2] == (512, 512):
                    i += 1
                    face_data.append([face, source_image])
                    continue

                found = False
                for i in range(1, 3):
                    (startX, startY, endX, endY) = face["bbox"].astype("int")
                    startX, endX, startY, endY = clamp_cut_values(startX, endX, startY, endY, source_image)
                    cutout_padding = extra_padding
                    # top needs extra room for detection
                    padding = int((endY - startY) * cutout_padding)
                    oldY = startY
                    startY -= padding

                    factor = 0.25 if i == 1 else 0.5
                    cutout_padding = factor
                    padding = int((endY - oldY) * cutout_padding)
                    endY += padding
                    padding = int((endX - startX) * cutout_padding)
                    startX -= padding
                    endX += padding
                    startX, endX, startY, endY = clamp_cut_values(
                        startX, endX, startY, endY, source_image
                    )
                    face_temp = source_image[startY:endY, startX:endX]
                    face_temp = resize_image_keep_content(face_temp)
                    testfaces = self.get_all_faces(face_temp)
                    if testfaces is not None and len(testfaces) > 0:
                        i += 1
                        face_data.append([testfaces[0], face_temp])
                        found = True
                        break

                if not found:
                    print("No face found after resizing, this shouldn't happen!")
                continue

            face_temp = source_image[startY:endY, startX:endX]
            if face_temp.size < 1:
                continue

            i += 1
            face_data.append([face, face_temp])
        return face_data
