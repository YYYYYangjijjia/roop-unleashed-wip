"""Isolated AlphaFace face-crop trial; never opens videos or changes app state.

Use --inspect for asset/ONNX verification without inference. Use --run explicitly
with a face-only BGR crop and its local five landmarks (or --detect-landmarks).
Identity always averages every row in the supplied W600K embedding archive.
Community reference: VisoMasterFusion/VisoMaster-Fusion at
d86cc97f499c39d275b7ef5a0d0ba03ea1b4b2a3; see model directory provenance.json.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import onnxruntime as ort
from skimage.transform import SimilarityTransform

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from roop.runtime_paths import models_directory
ASSETS = models_directory() / "alphaface_trial"
MODEL_HASH = "5514d967ab6cc27e1b0edc092e05ee97d235adccb4da68574a9b1a1e221a4c6a"
EMP_HASH = "cee626bc81721d71c5d6cb1f76f830b9ae46f595514b0884dd8ae34785576764"

# Five target yaw templates from the pinned community implementation. This
# selects target alignment only; it never selects a source reference by pose.
YAW_TEMPLATES = np.array([
    [[51.642, 50.115], [57.617, 49.990], [35.740, 69.007], [51.157, 89.050], [57.025, 89.702]],
    [[45.031, 50.118], [65.568, 50.872], [39.677, 68.111], [45.177, 86.190], [64.246, 86.758]],
    [[39.730, 51.138], [72.270, 51.138], [56.000, 68.493], [42.463, 87.010], [69.537, 87.010]],
    [[46.845, 50.872], [67.382, 50.118], [72.737, 68.111], [48.167, 86.758], [67.236, 86.190]],
    [[54.796, 49.990], [60.771, 50.115], [76.673, 69.007], [55.388, 89.702], [61.257, 89.050]],
], dtype=np.float32)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def project_average(embeddings, emp, expected_count):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape != (expected_count, 512) or not np.isfinite(embeddings).all():
        raise ValueError(f"Expected all {expected_count} finite W600K references, got {embeddings.shape}")
    if np.any(np.linalg.norm(embeddings, axis=1) <= 1e-12):
        raise ValueError("Empty identity reference")
    if emp.shape != (512, 512) or not np.isfinite(emp).all():
        raise ValueError("Invalid AlphaFace identity projection")
    # Average raw embeddings, project, normalize ONCE as required by AlphaFace.
    latent = embeddings.mean(axis=0, keepdims=True) @ emp
    norm = np.linalg.norm(latent)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Degenerate projected average identity")
    return np.ascontiguousarray(latent / norm, dtype=np.float32)


def align_face(crop, landmarks):
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        raise ValueError("Landmarks must be finite local crop coordinates, shape 5x2")
    # Equivalent to community 512px template * 112/128 + x32, then /2.
    templates = YAW_TEMPLATES * 2
    templates[:, :, 0] += 16
    candidates = []
    for index, template in enumerate(templates):
        transform = SimilarityTransform()
        if transform.estimate(landmarks, template) and np.isfinite(transform.params).all():
            error = np.linalg.norm(transform(landmarks) - template, axis=1).sum()
            candidates.append((float(error), index, transform.params[:2]))
    if not candidates:
        raise ValueError("Could not align face landmarks")
    error, index, matrix = min(candidates, key=lambda candidate: candidate[0])
    return cv2.warpAffine(crop, matrix, (256, 256), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT), matrix, index, error


def detect_crop_landmarks(crop):
    from insightface.model_zoo import get_model
    path = APP / "models" / "buffalo_l" / "det_10g.onnx"
    detector = get_model(str(path), providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.5)
    boxes, landmarks = detector.detect(crop, max_num=0, metric="default")
    if len(boxes) != 1:
        raise ValueError(f"Expected one face in face-only crop, detected {len(boxes)}; supply explicit landmarks")
    return landmarks[0]


def create_session(model, provider):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    if provider == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA provider unavailable; refusing an unreported CPU fallback")
        providers.insert(0, ("CUDAExecutionProvider", {
            "gpu_mem_limit": 3 * 1024**3, "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": "0",
        }))
    session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    if provider == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("Requested CUDA session did not initialize")
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--inspect", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--face-crop", type=Path, help="Face-only uint8 BGR .npy or image; never a video/full frame")
    parser.add_argument("--landmarks", type=Path, help="JSON 5x2 coordinates local to the crop")
    parser.add_argument("--detect-landmarks", action="store_true", help="Run existing CPU SCRFD on the face crop only")
    parser.add_argument("--embeddings", type=Path, default=APP.parent / "logs/hyperswap-current-fsz-mean.npz")
    parser.add_argument("--expected-count", type=int, default=65)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--production-adapter", action="store_true",
                        help="Validate the registered app processor and alignment, using the same face-only fixture")
    parser.add_argument("--output", type=Path, default=ASSETS / "trial-output")
    args = parser.parse_args()
    model = ASSETS / "alphaface_swapper_fused_norm.onnx"
    emp_path = ASSETS / "emp.npy"
    if sha256(model) != MODEL_HASH or sha256(emp_path) != EMP_HASH:
        raise ValueError("Model or identity projection SHA256 mismatch")
    report = {"model_sha256": MODEL_HASH, "emp_sha256": EMP_HASH,
              "onnxruntime": ort.__version__, "media_inference": False}
    if args.inspect:
        import onnx
        graph = onnx.load(str(model))
        onnx.checker.check_model(graph)
        report["onnx_check"] = "passed"
        report["inputs"] = {v.name: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
                            for v in graph.graph.input}
        report["outputs"] = {v.name: [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
                             for v in graph.graph.output}
        report["emp_shape"] = list(np.load(emp_path, allow_pickle=False).shape)
        print(json.dumps(report, indent=2))
        return
    if not args.face_crop or bool(args.landmarks) == args.detect_landmarks:
        parser.error("--run needs --face-crop and exactly one of --landmarks / --detect-landmarks")
    crop = (np.load(args.face_crop, allow_pickle=False) if args.face_crop.suffix.lower() == ".npy"
            else cv2.imread(str(args.face_crop)))
    if crop is None or crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError("Crop must be uint8 HWC BGR")
    landmarks = (json.loads(args.landmarks.read_text(encoding="utf-8")) if args.landmarks
                 else detect_crop_landmarks(crop))
    aligned, matrix, template, fit_error = align_face(crop, landmarks)
    with np.load(args.embeddings, allow_pickle=False) as archive:
        embeddings = archive["embeddings"]
    latent = project_average(embeddings, np.load(emp_path, allow_pickle=False), args.expected_count)
    blob = np.ascontiguousarray(aligned[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    processor = None
    if args.production_adapter:
        if args.provider == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sys.path.insert(0, str(APP))
        import roop.globals as globals_
        from roop.FaceSet import FaceSet
        from roop.face_util import align_crop
        from roop.processors.FaceSwapInsightFace import FaceSwapInsightFace
        from insightface.app.common import Face
        from roop.offline import mark_offline
        mark_offline('isolated AlphaFace adapter verification')
        globals_.execution_providers = ['CPUExecutionProvider']
        if args.provider == 'cuda':
            # Importing torch loads the CUDA DLLs used by the existing app.
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable for production adapter verification')
            globals_.execution_providers.insert(0, ('CUDAExecutionProvider', {
                'gpu_mem_limit': 3 * 1024**3, 'arena_extend_strategy': 'kSameAsRequested',
                'cudnn_conv_algo_search': 'HEURISTIC', 'cudnn_conv_use_max_workspace': '0',
            }))
        native_crop, native_matrix = align_crop(crop, np.asarray(landmarks, np.float32), 256, mode='alphaface')
        np.testing.assert_allclose(native_matrix, matrix, atol=1e-5)
        np.testing.assert_array_equal(native_crop, aligned)
        faceset = FaceSet()
        faceset.faces = [Face(embedding=np.asarray(e, np.float32)) for e in embeddings]
        source_face = faceset.average_identity_face()
        processor = FaceSwapInsightFace()
        processor.Initialize({'devicename': args.provider, 'swap_model': 'alphaface'})
        np.testing.assert_allclose(processor._compute_latent(source_face), latent, atol=1e-6)
        session = processor.model_swap_insightface
        if processor.loaded_model_key != 'alphaface' or processor.pool is not None:
            raise RuntimeError('Wrong swapper or unexpected multi-instance pool')
    else:
        session = create_session(model, args.provider)
    start = time.perf_counter()
    output = (processor.Run(source_face, None, blob)[None] if processor else
              session.run(["output"], {"target": blob, "source_embedding": latent})[0])
    seconds = time.perf_counter() - start
    if output.shape != (1, 3, 256, 256) or not np.isfinite(output).all() or np.abs(output).mean() < 1e-4:
        raise RuntimeError("AlphaFace returned invalid output; no unchanged-face fallback is permitted")
    swapped = np.rint(np.clip(output[0].transpose(1, 2, 0)[:, :, ::-1], 0, 1) * 255).astype(np.uint8)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, image in [("target-aligned-face.png", aligned), ("alphaface-result.png", swapped),
                        ("comparison-face-only.png", np.concatenate([aligned, swapped], axis=1))]:
        if not cv2.imwrite(str(args.output / name), image):
            raise OSError(f"Could not write {name}")
    report.update(media_inference=True, production_adapter=args.production_adapter,
                  source_identity="all-reference raw mean -> emp -> L2",
                  source_count=len(embeddings), embeddings_path=str(args.embeddings.resolve()),
                  embeddings_sha256=sha256(args.embeddings), face_crop=str(args.face_crop.resolve()),
                  face_crop_sha256=sha256(args.face_crop), landmarks=np.asarray(landmarks).tolist(),
                  providers=session.get_providers(), first_inference_seconds=seconds,
                  target_yaw_template=template, template_fit_error=fit_error, affine=matrix.tolist(),
                  raw_output_range=[float(output.min()), float(output.max())],
                  limitations="face crop only; no masking, enhancement, temporal or full-video quality validation")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if processor:
        processor.Release()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
