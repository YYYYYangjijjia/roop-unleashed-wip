"""Local-only A processors, frozen from A commit 1b311e8.

Numerical Run bodies follow roop/processors/FaceSwapInsightFace.py:39-50
and Mask_XSeg.py:35-52. Only construction, model path and provider injection
change; these classes do not use C's pools, graph rewriting or downloads.
"""
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime


def _local_model(models_dir, name):
    path = Path(models_dir) / name
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"A compatibility requires local model: {path}")
    return str(path)


def _check_provider(session, device):
    if device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("A compatibility requested CUDA, but its session fell back to CPU.")


class AInSwapper:
    processorname = "faceswap"
    type = "swap"

    def __init__(self, models_dir, providers, device):
        path = _local_model(models_dir, "inswapper_128.onnx")
        graph = onnx.load(path).graph
        self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
        self.devicename = device
        opts = onnxruntime.SessionOptions()
        opts.enable_cpu_mem_arena = False
        self.model_swap_insightface = onnxruntime.InferenceSession(path, opts, providers=providers)
        _check_provider(self.model_swap_insightface, device)

    def Run(self, source_face, target_face, temp_frame):
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        io_binding = self.model_swap_insightface.io_binding()
        io_binding.bind_cpu_input("target", temp_frame)
        io_binding.bind_cpu_input("source", latent)
        io_binding.bind_output("output", self.devicename)
        self.model_swap_insightface.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()[0]
        return ort_outs[0]

    def Release(self):
        self.model_swap_insightface = None
        self.emap = None


class AXSeg:
    processorname = "mask_xseg"
    type = "mask"

    def __init__(self, models_dir, providers, device):
        path = _local_model(models_dir, "xseg.onnx")
        self.devicename = device
        self.model_xseg = onnxruntime.InferenceSession(path, None, providers=providers)
        _check_provider(self.model_xseg, device)
        self.model_inputs = self.model_xseg.get_inputs()
        self.model_outputs = self.model_xseg.get_outputs()

    def Run(self, img1, keywords):
        # Preserve A's positional resize call (the third argument is dst).
        temp_frame = cv2.resize(img1, (256, 256), cv2.INTER_CUBIC)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = temp_frame[None, ...]
        io_binding = self.model_xseg.io_binding()
        io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame)
        io_binding.bind_output(self.model_outputs[0].name, self.devicename)
        self.model_xseg.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        result = np.clip(result, 0, 1.0)
        result[result < 0.1] = 0
        result = 1.0 - result
        return result

    def Release(self):
        self.model_xseg = None
