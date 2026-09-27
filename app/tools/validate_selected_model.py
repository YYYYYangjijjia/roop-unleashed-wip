"""Read-only structural and CPU ONNX Runtime load check for one model."""
import json
import sys
import onnx
import onnxruntime as ort

path = sys.argv[1]
onnx.checker.check_model(path)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.log_severity_level = 3
session = ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])
def describe(values):
    return [{'name': v.name, 'shape': v.shape, 'type': v.type} for v in values]
print(json.dumps({'onnx_check': 'passed', 'cpu_session_load': 'passed',
                  'onnxruntime': ort.__version__, 'inputs': describe(session.get_inputs()),
                  'outputs': describe(session.get_outputs()), 'gpu_inference_tested': False}))
