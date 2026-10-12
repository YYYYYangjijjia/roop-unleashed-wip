"""Install the optional CPU UV-detail dependency without replacing headless cv2.

Run with the same Python interpreter used by the app. Models are fetched on
first use from pinned official URLs unless runtime.local.json sets offline.
"""

import subprocess
import sys


def main():
    # MediaPipe's wheel metadata asks for opencv-contrib-python, but this app
    # already owns a compatible headless cv2. Installing two OpenCV wheels in
    # one environment overwrites whichever was installed first.
    import cv2
    if tuple(map(int, cv2.__version__.split('.')[:2])) < (4, 9):
        raise RuntimeError('Install app/requirements.txt before UV detail')
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    'absl-py==2.3.1', 'certifi', 'flatbuffers>=25.9,<26',
                    'matplotlib==3.9.1', 'sounddevice==0.5.2'], check=True)
    # MediaPipe declares opencv-contrib-python, which would overwrite the
    # application's headless cv2. The Tasks API used here works with that cv2.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                    'mediapipe==1.0.1'], check=True)
    import mediapipe
    if not hasattr(mediapipe.tasks.vision, 'FaceLandmarker'):
        raise RuntimeError('MediaPipe Tasks FaceLandmarker is unavailable')
    print('UV detail dependency ready:', mediapipe.__version__)
    print('Face mesh and skin parser models download on first use, or can be '
          'copied into the configured models directory for offline use.')


if __name__ == '__main__':
    main()
