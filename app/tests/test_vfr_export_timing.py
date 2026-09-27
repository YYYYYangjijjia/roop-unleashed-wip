"""VFR exports must keep source picture timing alongside copied audio."""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import av

from roop.util_ffmpeg import is_variable_frame_rate, restore_audio


class TestVFRExportTiming(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is required')
    def test_source_frame_times_and_audio_packets_survive_export(self):
        with tempfile.TemporaryDirectory() as workdir:
            source = os.path.join(workdir, 'source.mp4')
            rendered = os.path.join(workdir, 'rendered.mp4')
            result = os.path.join(workdir, 'result.mp4')
            subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'lavfi', '-i', 'testsrc=duration=2:size=64x64:rate=10',
                '-f', 'lavfi', '-i', 'sine=frequency=1000:duration=2.3',
                '-vf', r'setpts=PTS+gte(N\,10)*0.4/TB', '-fps_mode', 'vfr',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', source,
            ], check=True, capture_output=True)

            with av.open(source) as media:
                video = media.streams.video[0]
                source_times = [frame.pts * video.time_base for frame in media.decode(video)]
                render_rate = str(video.average_rate)
            self.assertTrue(is_variable_frame_rate(source))
            self.assertGreater(max(b - a for a, b in zip(source_times, source_times[1:])),
                               min(b - a for a, b in zip(source_times, source_times[1:])))

            subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'lavfi', '-i', f'testsrc=duration=3:size=64x64:rate={render_rate}',
                '-frames:v', str(len(source_times)), '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', rendered,
            ], check=True, capture_output=True)

            self.assertTrue(restore_audio(rendered, source, 0, len(source_times), result,
                                          timing_video_path=source,
                                          preserve_vfr_timing=True))
            with av.open(result) as media:
                video = media.streams.video[0]
                result_times = [frame.pts * video.time_base for frame in media.decode(video)]
            self.assertEqual(source_times, result_times)

            def audio_hash(path):
                digest = hashlib.sha256()
                with av.open(path) as media:
                    for packet in media.demux(media.streams.audio[0]):
                        if packet.pts is not None:
                            digest.update(bytes(packet))
                return digest.hexdigest()

            self.assertEqual(audio_hash(source), audio_hash(result))

            # Timeline trims also use source frame indices, including VFR gaps.
            trimmed_render = os.path.join(workdir, 'trimmed_render.mp4')
            trimmed_result = os.path.join(workdir, 'trimmed_result.mp4')
            subprocess.run([
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-f', 'lavfi', '-i', f'testsrc=duration=3:size=64x64:rate={render_rate}',
                '-frames:v', '10', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                trimmed_render,
            ], check=True, capture_output=True)
            self.assertTrue(restore_audio(trimmed_render, source, 5, 15, trimmed_result,
                                          timing_video_path=source,
                                          preserve_vfr_timing=True))
            with av.open(trimmed_result) as media:
                video = media.streams.video[0]
                trimmed_times = [frame.pts * video.time_base for frame in media.decode(video)]
            self.assertEqual(trimmed_times,
                             [time - source_times[5] for time in source_times[5:15]])


if __name__ == '__main__':
    unittest.main()
