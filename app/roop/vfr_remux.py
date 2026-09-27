"""Restore a source video's presentation times after a frame-by-frame swap.

The swapper writes raw frames to a constant-rate encoder, which has no way to
carry the source packet timestamps. Remuxing the encoded packets is lossless;
only their timing changes. This path is for the in-memory, one-output-frame-per-
source-frame reader, not the extract-frames path that already resamples frames.
"""

from fractions import Fraction
import os
import subprocess
import uuid


def _frame_pts(video_path):
    from roop.capturer import _ffprobe_binary, _popen_kwargs

    timeout = max(120, min(3600, os.path.getsize(video_path) / (5 * 1024 * 1024)))
    proc = subprocess.run(
        [_ffprobe_binary(), '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'frame=best_effort_timestamp', '-of', 'csv=p=0',
         video_path],
        capture_output=True, timeout=timeout, **_popen_kwargs(),
    )
    if proc.returncode:
        raise RuntimeError('Could not read source frame timestamps: ' +
                           (proc.stderr or b'').decode('utf-8', 'replace')[:300])
    pts = [int(line.strip().split(b',')[0]) for line in proc.stdout.splitlines()
           if line.strip()]
    if len(pts) < 2 or any(a >= b for a, b in zip(pts, pts[1:])):
        raise ValueError('Source frame timestamps are missing or not increasing')
    return pts


def remux_vfr_video(rendered_path, timing_video_path, audio_path, final_path,
                    start_frame=0, end_frame=None):
    """Copy encoded video and original audio with source-frame presentation times.

    A failure leaves the rendered intermediate intact and removes only this
    function's incomplete output. No face pixels are decoded or re-encoded.
    """
    try:
        import av
    except ImportError as exc:
        raise RuntimeError('VFR export requires the av package (PyAV)') from exc

    timestamps = _frame_pts(timing_video_path)
    start = int(start_frame or 0)
    end = len(timestamps) if end_frame is None else min(int(end_frame), len(timestamps))
    if start < 0 or end <= start:
        raise ValueError('Invalid VFR frame range')
    if not audio_path:
        audio_path = timing_video_path
    temp_path = os.path.join(
        os.path.dirname(os.path.abspath(final_path)),
        f'.{os.path.basename(final_path)}.{uuid.uuid4().hex[:8]}.vfr.mp4',
    )
    try:
        with av.open(rendered_path) as rendered, \
             av.open(timing_video_path) as timing, \
             av.open(audio_path) as original_audio, \
             av.open(temp_path, 'w') as output:
            rendered_video = rendered.streams.video[0]
            timing_video = timing.streams.video[0]
            tick = timing_video.time_base
            base_tick = timestamps[start]
            base_time = Fraction(base_tick) * tick
            if end < len(timestamps):
                end_tick = timestamps[end]
            elif timing_video.duration is not None:
                end_tick = max(timing_video.duration, timestamps[-1] + 1)
            else:
                end_tick = timestamps[-1] + (timestamps[-1] - timestamps[-2])
            last_time = Fraction(end_tick) * tick
            output.metadata.update(original_audio.metadata)
            output_video = output.add_stream_from_template(rendered_video)
            output_video.time_base = tick
            output_video.metadata.update(rendered_video.metadata)

            copied_streams = [stream for stream in original_audio.streams
                              if stream.type in ('audio', 'subtitle')]
            stream_map = {}
            for stream in copied_streams:
                target = output.add_stream_from_template(stream)
                target.time_base = stream.time_base
                target.metadata.update(stream.metadata)
                stream_map[stream.index] = target

            frame_step = Fraction(1, 1) / rendered_video.average_rate / rendered_video.time_base
            first_pts = rendered_video.start_time or 0
            first_delta = timestamps[start + 1] - timestamps[start]
            video_count = 0
            audio_count = 0

            def source_tick(index):
                absolute = start + index
                if absolute < start:
                    return index * first_delta
                if absolute >= end:
                    raise ValueError(f'Rendered video frame {absolute} exceeds source range ending at {end}')
                return timestamps[absolute] - base_tick

            def mapped_index(pts):
                value = Fraction(pts - first_pts) / frame_step
                index = round(value)
                if abs(value - index) > Fraction(1, 4):
                    raise ValueError('Rendered video is not on its expected constant-rate grid')
                return index

            def next_video(iterator):
                nonlocal video_count
                for packet in iterator:
                    if packet.pts is None or packet.dts is None:
                        continue
                    frame_index = mapped_index(packet.pts)
                    if frame_index < 0:
                        raise ValueError('Unexpected negative video presentation index')
                    packet.pts = source_tick(frame_index)
                    packet.dts = source_tick(mapped_index(packet.dts))
                    if start + frame_index + 1 < end:
                        packet.duration = timestamps[start + frame_index + 1] - timestamps[start + frame_index]
                    else:
                        packet.duration = round((last_time - base_time) / tick) - packet.pts
                    packet.time_base = tick
                    packet.stream = output_video
                    video_count += 1
                    return Fraction(packet.dts) * tick, packet
                return None

            def next_media(iterator):
                nonlocal audio_count
                for packet in iterator:
                    if packet.pts is None or packet.dts is None:
                        continue
                    stream = packet.stream
                    packet_time = Fraction(packet.dts) * stream.time_base
                    packet_end = packet_time + Fraction(packet.duration or 0) * stream.time_base
                    if (start and packet_end <= base_time) or packet_time >= last_time:
                        continue
                    if start:
                        offset = round(base_time / stream.time_base)
                        packet.pts -= offset
                        packet.dts -= offset
                    packet.stream = stream_map[stream.index]
                    if stream.type == 'audio':
                        audio_count += 1
                    return Fraction(packet.dts) * stream.time_base, packet
                return None

            video_iter = iter(rendered.demux(rendered_video))
            media_iter = iter(original_audio.demux(*copied_streams)) if copied_streams else iter(())
            video_head = next_video(video_iter)
            media_head = next_media(media_iter)
            while video_head is not None or media_head is not None:
                if media_head is None or (video_head is not None and video_head[0] <= media_head[0]):
                    output.mux(video_head[1])
                    video_head = next_video(video_iter)
                else:
                    output.mux(media_head[1])
                    media_head = next_media(media_iter)
            if video_count == 0:
                raise ValueError('Rendered video contains no frames')
        os.replace(temp_path, final_path)
        return video_count, audio_count
    finally:
        if os.path.isfile(temp_path):
            os.remove(temp_path)
