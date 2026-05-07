"""
Processus producteurs de frames vidéo.

Chaque producteur tourne dans son propre Process (forké depuis le parent).
Il capture des frames, les timestamp avec CLOCK_MONOTONIC au plus près de
la capture hardware, et les écrit dans un FrameChannel.

Quand record_event n'est pas activé : on capture quand même (pour garder
la caméra chaude et éviter le délai de réouverture) mais on flush sans écrire.
Quand stop_event est activé : le processus se termine.
"""
import time
import numpy as np
import multiprocessing as mp

from .frame_channel import FrameChannel


def _ts_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


# ------------------------------------------------------------------ #
# Producteur Arducam (V4L2)
# ------------------------------------------------------------------ #

def arducam_producer(
    channel: FrameChannel,
    device: str,
    record_event: mp.Event,
    stop_event: mp.Event,
) -> None:
    import subprocess
    import cv2

    # V4L2 menu for this camera: 0=Auto, 1=Manual.
    # OpenCV's translation layer maps to wrong values, so set directly via v4l2-ctl.
    subprocess.run(
        ["v4l2-ctl", "-d", device, "--set-ctrl=auto_exposure=0"],
        check=False, capture_output=True,
    )

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, channel.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, channel.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[arducam_producer] ERROR: cannot open {device}", flush=True)
        return

    # Drain warm-up frames so auto-exposure can stabilize before we record.
    for _ in range(30):
        cap.read()
    print(f"[arducam_producer] {device} ready.", flush=True)

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            t_cap = _ts_ns()

            if not ret:
                continue

            frame_rgb = cv2.cvtColor(
                cv2.resize(frame, (channel.width, channel.height)),
                cv2.COLOR_BGR2RGB,
            )

            if record_event.is_set():
                channel.write(frame_rgb, t_cap)
            else:
                channel.flush()
    finally:
        cap.release()


# ------------------------------------------------------------------ #
# Producteur D435 (wrist camera — RGB + depth)
# ------------------------------------------------------------------ #

def d435_producer(
    rgb_channel: FrameChannel,
    depth_channel: "FrameChannel | None",
    device_serial: "str | None",
    record_event: mp.Event,
    stop_event: mp.Event,
) -> None:
    """
    Producteur pour Intel RealSense D435 (caméra poignet).

    Utilise pyrealsense2 pour obtenir RGB + depth uint16 avec timestamps
    hardware alignés sur CLOCK_MONOTONIC.

    Fallback sur cv2 si pyrealsense2 échoue (RGB seulement via /dev/video8).
    """
    try:
        _d435_realsense(rgb_channel, depth_channel, device_serial,
                        record_event, stop_event)
    except ImportError:
        print(
            "[d435_producer] pyrealsense2 absent → fallback cv2 (RGB seulement).\n"
            "    pip install pyrealsense2  pour activer la profondeur."
        )
        _d435_cv2_fallback(rgb_channel, record_event, stop_event)
    except Exception as e:
        print(f"[d435_producer] Erreur RealSense : {e}\n"
              f"    → fallback cv2 (RGB seulement)")
        _d435_cv2_fallback(rgb_channel, record_event, stop_event)


def _d435_realsense(rgb_ch, depth_ch, serial, record_event, stop_event):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)

    config.enable_stream(rs.stream.color, rgb_ch.width, rgb_ch.height,
                         rs.format.rgb8, 30)
    if depth_ch is not None:
        config.enable_stream(rs.stream.depth, depth_ch.width, depth_ch.height,
                             rs.format.z16, 30)

    profile = pipeline.start(config)

    # Aligner le clock D435 sur CLOCK_MONOTONIC hôte, exposition fixe
    try:
        sensor = profile.get_device().first_depth_sensor()
        sensor.set_option(rs.option.global_time_enabled, 1)
        sensor.set_option(rs.option.enable_auto_exposure, 0)
    except Exception:
        pass

    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            t_cap = _ts_ns()

            color_frame = frames.get_color_frame()
            if color_frame:
                rgb = np.asanyarray(color_frame.get_data())
                if record_event.is_set():
                    rgb_ch.write(rgb, t_cap)
                else:
                    rgb_ch.flush()

            if depth_ch is not None:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    # uint16 millimètres — ne pas encoder en H.264
                    d16 = np.asanyarray(depth_frame.get_data())
                    d16_ch = d16[:, :, np.newaxis]  # (H, W) → (H, W, 1)
                    if record_event.is_set():
                        depth_ch.write(d16_ch, t_cap)
                    else:
                        depth_ch.flush()
    finally:
        pipeline.stop()


def _d435_cv2_fallback(rgb_ch, record_event, stop_event):
    """Fallback RGB-only via V4L2 — nœud couleur D435 = /dev/video8."""
    import cv2

    # /dev/video8 est le flux couleur YUYV du D435 sur ce système
    cap = cv2.VideoCapture("/dev/video8", cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, rgb_ch.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rgb_ch.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    if not cap.isOpened():
        print("[d435_cv2_fallback] ERREUR: impossible d'ouvrir /dev/video8")
        return

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            t_cap = _ts_ns()
            if not ret:
                continue

            rgb = cv2.cvtColor(
                cv2.resize(frame, (rgb_ch.width, rgb_ch.height)),
                cv2.COLOR_BGR2RGB,
            )
            if record_event.is_set():
                rgb_ch.write(rgb, t_cap)
            else:
                rgb_ch.flush()
    finally:
        cap.release()
