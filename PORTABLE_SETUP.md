# Portable setup

This fork does not include model weights, Python environments, facesets, input
media, outputs, or machine-local configuration. Keep the original AGPL-3.0
license and check the licenses/usage terms of separately downloaded weights.
See [source and model attribution](THIRD_PARTY.md).

## Windows local launcher

Requirements: Python 3.10, Node.js/npm, FFmpeg, and a GPU driver compatible
with the selected inference runtime. Copy `runtime.example.json` to
`runtime.local.json` in the repository root and edit the paths for your machine.
The latter is ignored by Git. Paths can be absolute or relative to this repo.

```json
{
  "models_dir": "app/models",
  "environment": { "mode": "create", "path": "app/env" },
  "ffmpeg_dir": "",
  "offline": false,
  "optional_models": []
}
```

- `models_dir`: one existing model directory. The default `app/models` is
  created by the launcher. To reuse a library, point at its directory and
  keep the model filenames/subdirectories this application expects. An
  explicitly configured missing directory is an error, so a typo cannot
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
  needed and install dependencies. This requires network access. The Windows
  installer supports NVIDIA CUDA and CPU; AMD users should use Pinokio's
  existing installation path or provide a prepared environment.
- `ffmpeg_dir`: folder containing `ffmpeg.exe`; leave blank to use `PATH`.
- `offline`: prevent model downloads at runtime. It does not install missing
  packages. A missing or invalid model is reported rather than silently
  switching models.
- `optional_models`: add `"alphaface"` to install its ONNX model and identity
  projection during `install_local.bat`. These are optional because the model
  alone is about 529 MiB. Existing files are checked by SHA-256 and reused;
  bad files are preserved if downloading or verification fails. You can also
  run `app/env/python.exe app/tools/install_alphaface.py` later. AlphaFace
  never downloads inside a preview or render.

Run `install_local.bat` once, then `start_local.bat`. The launcher checks
ports 8002, 7862 and 5174 and reuses its own already-running services. Use
`stop_local.bat` to stop only the recorded services. The browser UI is on
`http://127.0.0.1:5174/`. Logs and service records live under ignored `logs/`.
If you change an environment or model path while services are running, stop
them first and start again; running Python cannot adopt the new path.

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
