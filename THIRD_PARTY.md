# Source and model attribution

This repository is a fork of
[rishabh4496/roop-unleashed-wip](https://github.com/rishabh4496/roop-unleashed-wip).
Its Git history is retained. The application license is
[AGPL-3.0](LICENSE) (also preserved at `app/LICENSE`).

The optional AlphaFace adapter follows the published AlphaFace/VisoMaster
alignment and identity projection contract. The reference implementation and
`emp.npy` projection come from
[VisoMasterFusion/VisoMaster-Fusion](https://github.com/VisoMasterFusion/VisoMaster-Fusion/tree/d86cc97f499c39d275b7ef5a0d0ba03ea1b4b2a3),
whose AlphaFace software license is reproduced at
[`licenses/AlphaFace-MIT.txt`](licenses/AlphaFace-MIT.txt).
The optional ONNX export is fetched from the pinned
[community release](https://github.com/kodek4/VisoMaster-Fusion/releases/tag/alphaface-model-v1).
Both file hashes are verified by the optional installer. No model weights are
included in this Git repository. Other model URLs are declared in the
application's model loaders and registry; their individual terms remain with
their respective sources.

Optional multi-view skin detail uses the
[MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
task and [canonical UV face mesh](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj),
plus the existing [FaceParser ResNet18](https://github.com/yakhyo/face-parsing).
The Python installer and first-use model loader verify fixed versions/hashes;
none of these model files are included in Git. Review each upstream model's
license and use terms before redistributing downloaded weights.
