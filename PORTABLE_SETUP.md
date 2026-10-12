# Portable setup

This fork does not include model weights, Python environments, facesets, input
media, outputs, or machine-local configuration. Keep the original AGPL-3.0
license and check the licenses/usage terms of separately downloaded weights.
See [source and model attribution](THIRD_PARTY.md).

## Windows local launcher

Requirements: Python 3.10, Node.js/npm, FFmpeg, and a GPU driver compatible
with the selected inference runtime. The first `install_local.bat` or
`start_local.bat` run creates `runtime.local.json` from `runtime.example.json`
when it is missing. Edit the local file for your machine before installation
if you need custom paths. It is ignored by Git and is never overwritten by the
launcher. The launcher also creates `app/config.yaml` from the checked-in
`app/config.default.yaml` when settings are missing; existing settings are
never overwritten. Paths can be absolute or relative to this repo.

```json
{
  "models_dir": "app/models",
  "environment": { "mode": "create", "path": "app/env" },
  "ports": { "api": 8802, "gradio": 8862, "ui": 5174 },
  "ffmpeg_dir": "",
  "offline": false,
  "model_profile": "local_default",
  "optional_models": []
}
```

- `models_dir`: one existing model directory. The default `app/models` is
  created by the launcher. To reuse a library, point at its directory and
  keep the model filenames/subdirectories this application expects. An
  custom missing directory is an error, so a typo cannot
  trigger a second download. With `offline: false`, missing weights download
  into this selected directory; use `offline: true` to leave a shared library
  untouched and receive a missing-model error instead. Some weights download
  during startup, while others download when first selected. URLs and expected
  hashes are defined by the model registry/code, not by this config file.
- `environment.mode: "existing"`: use the selected Python environment as is.
  `install_local.ps1` checks core imports and never installs into it. The
  `path` can point to a virtual environment root or an environment directory
  containing `python.exe`. It must already have the application dependencies.
- `environment.mode: "create"`: make a Python 3.10 environment at `path` if
  needed and install dependencies. The installer reuses matching PyTorch
  CUDA/CPU wheels already in that environment; it installs them when missing
  or when the selected build differs. A new installation requires network
  access. The Windows installer supports NVIDIA CUDA and CPU; AMD users should
  use Pinokio's existing installation path or provide a prepared environment.
- `ffmpeg_dir`: folder containing `ffmpeg.exe`; leave blank to use `PATH`.
- `ports`: ports for the local API, Gradio server, and browser UI. Defaults are
  8802, 8862, and 5174. Change these if Windows reserves a port or another
  application uses it; stop the local services before changing the values.
  The launcher checks that each port can bind before starting the backend.
- `offline`: prevent model downloads at runtime. It does not install missing
  packages. A missing or invalid model is reported rather than silently
  switching models.
- `model_profile: "local_default"`: during `install_local.bat`, verify or
  download the weights used by the checked-in default settings: buffalo_l,
  RetinaFace R50, InSwapper, DFL XSeg, LivePortrait expression restoration,
  and UV identity detail. Existing files with the expected SHA-256 are reused.
  A missing model is downloaded to `models_dir` and checked before installation
  succeeds. Set `"none"` to skip this profile. Offline mode verifies local
  files and fails clearly if any are missing. This profile does not download
  every model shown in the UI. Other models may download when selected, depending
  on their loader; install them separately if the loader does not offer a download.
- `optional_models`: add `"alphaface"` to install its ONNX model and identity
  projection during `install_local.bat`. These are optional because the model
  alone is about 529 MiB. Existing files are checked by SHA-256 and reused;
  bad files are preserved if downloading or verification fails. You can also
  run `app/env/python.exe app/tools/install_alphaface.py` later. AlphaFace
  never downloads inside a preview or render.

Run `install_local.bat` once, then `start_local.bat`. The launcher checks
the configured ports and reuses its own already-running services. Use
`stop_local.bat` to stop only the recorded services. The browser UI is on
`http://127.0.0.1:5174/` with the default UI port. Logs and service records
live under ignored `logs/`. The Windows local launcher skips the old bulk
pre-download of unrelated models at startup; install the default profile first.
On an already configured machine, the installer preserves `app/config.yaml`.
To adopt the tracked profile, back up that file, remove it, and rerun
`install_local.bat`; captured reference faces and media paths must still be
loaded separately.
If the Command Prompt title begins with `Select` (or `选择`), press Esc to leave
text selection mode and let an apparently stalled installation continue.
If you change an environment, model path, or port while services are running,
stop them first and start again; running services cannot adopt the new values.

## Pinokio and other systems

The existing Pinokio `install.js` / `start_react.js` path continues to create
and use its own `app/env` environment. The Python backend reads
`runtime.local.json` for the model directory, including when started through
Pinokio. To reuse an arbitrary Python environment, use the Windows local
launcher above. On other systems, prepare the environment using the existing
manual/Pinokio instructions and set `ROOP_MODELS_DIR` to a model library path
before starting the backend; this environment variable overrides the JSON
model path.

The full roop-unleashed-main rendering compatibility is opt-in. Average source
identity is the factory default and may be combined with the standard renderer.
The top-bar EN / 中文 switch affects help popovers; it does not alter project
settings, labels, or processing results.

## Optional multi-view UV detail

For **Source detail method → Multi-view UV (skin)**, first install MediaPipe in
the **same Python environment used by the backend**:

```powershell
app\env\python.exe app\install_uv_detail.py
```

This installer avoids installing a second OpenCV wheel over the app's headless
OpenCV. It installs the optional CPU face-landmark dependency; no CUDA memory
is reserved by this feature. On first use, the app fetches the official
`face_landmarker.task`, its matching canonical face mesh, and the existing
FaceParser `resnet18.onnx` into the configured `models_dir`. URLs and SHA-256
hashes are pinned in `app/roop/uv_identity_detail.py`. With `offline: true`,
copy these three verified files into that directory first; no download occurs.
The default blend is 0, so the feature costs nothing until enabled. Building
an atlas for a faceset takes several seconds once; frame processing is slower
because face geometry and skin segmentation run on each selected face.
