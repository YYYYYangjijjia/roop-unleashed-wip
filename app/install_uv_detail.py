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
    import flatbuffers
    import numpy
    if tuple(map(int, cv2.__version__.split('.')[:2])) < (4, 9):
        raise RuntimeError('Install app/requirements.txt before UV detail')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                    'mediapipe==1.0.1', 'absl-py==2.3.1',
                    'sounddevice==0.5.2'], check=True)
    import mediapipe
    print('UV detail dependency ready:', mediapipe.__version__)
    print('Face mesh and skin parser models download on first use, or can be '
          'copied into the configured models directory for offline use.')


if __name__ == '__main__':
    main()
