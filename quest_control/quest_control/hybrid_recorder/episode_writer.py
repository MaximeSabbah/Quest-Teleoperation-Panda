"""
Processus écrivain d'épisodes — format LeRobot v2.

Commandes via cmd_q :
  ('start', metadata_dict)   → démarre la session
  ('stop',  success: bool)   → finalise et sauvegarde
  ('discard',)               → supprime les fichiers temporaires
  ('shutdown',)              → arrêt propre

Sortie (format LeRobot v2) :
  {base_dir}/
    meta/
      info.json              description du dataset (mise à jour par épisode)
      episodes.jsonl         une ligne JSON par épisode
    data/
      chunk-000/
        episode_{n:06d}.parquet   état + action alignés sur les frames vidéo
    videos/
      chunk-000/
        observation.images.{cam}_episode_{n:06d}.mp4   H.264 RGB
    depth/
      episode_{n:06d}/
        frame_{i:06d}.png         uint16 PNG (depth D435)

observation.state  (25 dims) = joint_pos(7) + joint_vel(7) + ee_pos(3)
                               + ee_quat(4) + gripper_pos(2) + gripper_vel(2)
action             (15 dims) = action_ee_pos(3) + action_ee_quat(4)
                               + action_gripper_cmd(1) + joint_torques_ff(7)

Pour H.264 propre : pip install av
"""
import json
import os
import queue as _queue
import shutil
import threading
import time
import numpy as np
import multiprocessing as mp

from .frame_channel import FrameChannel

_STATE_DIM  = 25
_ACTION_DIM = 15   # 3 + 4 + 1 + 7

_STATE_NAMES = [
    "joint_pos_1", "joint_pos_2", "joint_pos_3", "joint_pos_4",
    "joint_pos_5", "joint_pos_6", "joint_pos_7",
    "joint_vel_1", "joint_vel_2", "joint_vel_3", "joint_vel_4",
    "joint_vel_5", "joint_vel_6", "joint_vel_7",
    "ee_pos_x", "ee_pos_y", "ee_pos_z",
    "ee_quat_x", "ee_quat_y", "ee_quat_z", "ee_quat_w",
    "gripper_pos_l", "gripper_pos_r",
    "gripper_vel_l", "gripper_vel_r",
]
_ACTION_NAMES = [
    "ee_pos_x", "ee_pos_y", "ee_pos_z",
    "ee_quat_x", "ee_quat_y", "ee_quat_z", "ee_quat_w",
    "gripper_cmd",
    "torque_j1", "torque_j2", "torque_j3", "torque_j4",
    "torque_j5", "torque_j6", "torque_j7",
]


# ------------------------------------------------------------------ #
# Encodeur vidéo (PyAV si dispo, sinon cv2 VideoWriter)
# ------------------------------------------------------------------ #

def _try_import_av():
    try:
        import av
        return av
    except ImportError:
        return None


class _VideoEncoder:
    def __init__(self, path: str, width: int, height: int, fps: int):
        self._idx = 0
        av = _try_import_av()

        if av:
            self._av = av
            self._container = av.open(path, mode="w")
            self._stream = self._container.add_stream("h264", rate=fps)
            self._stream.width = width
            self._stream.height = height
            self._stream.pix_fmt = "yuv420p"
            self._stream.options = {"crf": "18", "preset": "fast"}
            self._cv_writer = None
        else:
            import cv2
            self._av = None
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._cv_writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, rgb_hwc: np.ndarray) -> None:
        if self._av:
            av_frame = self._av.VideoFrame.from_ndarray(rgb_hwc, format="rgb24")
            av_frame.pts = self._idx
            for pkt in self._stream.encode(av_frame):
                self._container.mux(pkt)
        else:
            import cv2
            self._cv_writer.write(cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR))
        self._idx += 1

    def close(self) -> None:
        if self._av:
            for pkt in self._stream.encode():
                self._container.mux(pkt)
            self._container.close()
        else:
            self._cv_writer.release()


def _trim_video_cv2(src: str, dst: str, n_keep: int, cv2) -> bool:
    """Re-encode the first n_keep frames of src into dst using OpenCV."""
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    written = 0
    while written < n_keep:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        written += 1
    cap.release()
    out.release()
    return written == n_keep


# ------------------------------------------------------------------ #
# Session d'enregistrement
# ------------------------------------------------------------------ #

class _RecordingSession:
    """Accumule toutes les données d'un épisode en cours."""

    def __init__(
        self,
        tmp_dir: str,
        ep_idx: int,
        channel_names: list[str],
        widths: list[int],
        heights: list[int],
        fps: int,
        metadata: dict,
        global_frame_offset: int,
    ):
        self.tmp_dir = tmp_dir
        self.ep_idx = ep_idx
        self.metadata = metadata
        self.fps = fps
        self._global_frame_offset = global_frame_offset
        self.t_start_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

        os.makedirs(tmp_dir, exist_ok=True)

        # Un encodeur vidéo par canal RGB (écrits dans tmp_dir)
        self.encoders: dict[str, _VideoEncoder] = {}
        self._cam_dims: dict[str, tuple[int, int]] = {}
        for name, w, h in zip(channel_names, widths, heights):
            path = os.path.join(tmp_dir, f"observation.images.{name}.mp4")
            self.encoders[name] = _VideoEncoder(path, w, h, fps)
            self._cam_dims[name] = (w, h)

        # Timestamps des frames vidéo (pour aligner l'état sur les frames)
        self.frame_timestamps: dict[str, list[int]] = {n: [] for n in channel_names}

        # Profondeur uint16 en RAM — PNGs écrits à la fin
        self.depth_frames: list[tuple[int, np.ndarray]] = []

        # États / actions enregistrés à ~100 Hz
        self.state_rows: list[dict] = []

        self.frame_counts: dict[str, int] = {n: 0 for n in channel_names}

    def push_frame(self, name: str, t_ns: int, rgb: np.ndarray) -> None:
        self.encoders[name].write(rgb)
        self.frame_counts[name] += 1
        self.frame_timestamps[name].append(t_ns)

    def push_depth(self, t_ns: int, depth_2d: np.ndarray) -> None:
        self.depth_frames.append((t_ns, depth_2d.copy()))

    def push_state(self, state: dict) -> None:
        self.state_rows.append(state)

    # ------------------------------------------------------------------ #
    # Construction de la ligne observation.state / action
    # ------------------------------------------------------------------ #

    def _build_state_action(self, s: dict):
        """
        Construire (obs_state: list[float], action: list[float]) depuis un dict état.
        Retourne None en cas de clé manquante ou de dimensions incorrectes.

        observation.state (25) :
          joint_pos(7) + joint_vel(7) + ee_pos(3) + ee_quat(4)
          + gripper_pos(2) + gripper_vel(2)

        action (15) :
          action_ee_pos(3) + action_ee_quat(4) + action_gripper_cmd(1)
          + action_joint_torques(7)  ← feedforward Crocoddyl
        """
        try:
            obs = np.concatenate([
                np.asarray(s["joint_pos"]).ravel(),          # 7
                np.asarray(s["joint_vel"]).ravel(),          # 7
                np.asarray(s["ee_pos"]).ravel(),             # 3
                np.asarray(s["ee_quat"]).ravel(),            # 4
                np.asarray(s["gripper_pos"]).ravel(),        # 2
                np.asarray(s["gripper_vel"]).ravel(),        # 2
            ]).astype(np.float32)

            # Torques feedforward Crocoddyl (zeros si non disponibles)
            torques_raw = s.get("action_joint_torques", None)
            if torques_raw is not None:
                torques = np.asarray(torques_raw).ravel()[:7].astype(np.float32)
                if len(torques) < 7:
                    torques = np.pad(torques, (0, 7 - len(torques)))
            else:
                torques = np.zeros(7, dtype=np.float32)

            act = np.concatenate([
                np.asarray(s["action_ee_pos"]).ravel(),      # 3
                np.asarray(s["action_ee_quat"]).ravel(),     # 4
                [float(s["action_gripper_cmd"])],             # 1
                torques,                                      # 7
            ]).astype(np.float32)

        except (KeyError, ValueError, TypeError):
            return None

        return obs.tolist(), act.tolist()

    # ------------------------------------------------------------------ #
    # Alignement état → frames vidéo
    # ------------------------------------------------------------------ #

    def _build_aligned_rows(self) -> list[dict]:
        """
        Aligner l'état (100 Hz) sur les timestamps des frames vidéo (30 fps).
        Pour chaque frame vidéo, cherche l'état le plus proche en temps.
        Retourne une liste de dicts — un par frame vidéo.
        """
        ep_idx = self.ep_idx

        ref_name = next(
            (n for n in self.frame_timestamps if self.frame_timestamps[n]),
            None,
        )
        if ref_name is None or not self.state_rows:
            return []

        video_ts = np.array(self.frame_timestamps[ref_name], dtype=np.int64)
        state_ts = np.array([r["timestamp_ns"] for r in self.state_rows], dtype=np.int64)
        t0_ns = video_ts[0]
        n = len(video_ts)

        rows = []
        for frame_i, t_vid in enumerate(video_ts):
            closest = int(np.argmin(np.abs(state_ts - t_vid)))
            result = self._build_state_action(self.state_rows[closest])
            if result is None:
                continue
            obs_state, action = result

            rows.append({
                "timestamp":     float(t_vid - t0_ns) * 1e-9,
                "frame_index":   frame_i,
                "episode_index": ep_idx,
                "index":         self._global_frame_offset + frame_i,
                "task_index":    0,
                "next.done":     frame_i == n - 1,
                "observation.state": obs_state,
                "action":            action,
            })

        return rows

    # ------------------------------------------------------------------ #
    # Finalisation
    # ------------------------------------------------------------------ #

    def finalize(self, base_dir: str, success: bool) -> None:
        import cv2

        ep_idx = self.ep_idx

        # 1. Close all video encoders.
        for enc in self.encoders.values():
            enc.close()

        # 2. Synchronise frame counts across cameras.
        #
        #    Pass A — trim to the common timestamp window so boundary frames
        #    captured by one camera but not another are removed.
        #    Pass B — hard-cap every camera to min(counts) so a dropped
        #    internal frame (which leaves start/end timestamps unchanged)
        #    never causes the parquet to reference a video frame that doesn't
        #    exist.
        active_cams = [n for n in self.frame_timestamps if self.frame_timestamps[n]]
        if active_cams:
            # Pass A: timestamp-window intersection
            t_start = max(self.frame_timestamps[n][0]  for n in active_cams)
            t_end   = min(self.frame_timestamps[n][-1] for n in active_cams)
            for name in active_cams:
                ts = self.frame_timestamps[name]
                i0 = next((i for i, t in enumerate(ts) if t >= t_start), 0)
                i1 = next(
                    (len(ts) - 1 - i for i, t in enumerate(reversed(ts)) if t <= t_end),
                    len(ts) - 1,
                )
                self.frame_timestamps[name] = ts[i0 : i1 + 1]

            # Pass B: hard-cap to the shortest camera
            final_count = min(len(self.frame_timestamps[n]) for n in active_cams)
            for name in active_cams:
                self.frame_timestamps[name] = self.frame_timestamps[name][:final_count]

            # Trim video files for any camera that ended up with more frames
            for name in active_cams:
                n_keep = final_count
                if n_keep < self.frame_counts[name]:
                    src = os.path.join(
                        self.tmp_dir, f"observation.images.{name}.mp4"
                    )
                    tmp = src + ".trim.mp4"
                    ok = _trim_video_cv2(src, tmp, n_keep, cv2)
                    if ok and os.path.exists(tmp):
                        os.replace(tmp, src)
                    else:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                        print(
                            f"[writer] WARNING: trim failed for {name} "
                            f"(kept {self.frame_counts[name]}, wanted {n_keep})",
                            flush=True,
                        )
                self.frame_counts[name] = n_keep

        # 3. Move synchronised videos to the LeRobot tree.
        video_dir = os.path.join(base_dir, "videos", "chunk-000")
        os.makedirs(video_dir, exist_ok=True)
        for name in self.encoders:
            src = os.path.join(self.tmp_dir, f"observation.images.{name}.mp4")
            dst = os.path.join(
                video_dir,
                f"observation.images.{name}_episode_{ep_idx:06d}.mp4",
            )
            if os.path.exists(src):
                shutil.move(src, dst)

        # 4. Write depth frames (uint16 PNG, outside LeRobot standard).
        if self.depth_frames:
            depth_dir = os.path.join(base_dir, "depth", f"episode_{ep_idx:06d}")
            os.makedirs(depth_dir, exist_ok=True)
            for i, (_, d) in enumerate(self.depth_frames):
                cv2.imwrite(os.path.join(depth_dir, f"frame_{i:06d}.png"), d)

        # 4. Aligner l'état sur les frames vidéo et écrire le parquet
        n_states = len(self.state_rows)
        print(f"[writer] Finalizing ep {ep_idx}: state_rows={n_states}, video_frames={self.frame_counts}", flush=True)
        rows = self._build_aligned_rows()
        n_frames = len(rows)
        if rows:
            data_dir = os.path.join(base_dir, "data", "chunk-000")
            os.makedirs(data_dir, exist_ok=True)
            _write_parquet(
                os.path.join(data_dir, f"episode_{ep_idx:06d}.parquet"),
                rows,
            )
        else:
            print(f"[writer] WARNING: 0 aligned rows — no parquet written. "
                  f"states={n_states}, video_frames={self.frame_counts}", flush=True)

        # 5. Mettre à jour les métadonnées du dataset
        task = self.metadata.get("language_instruction", "")
        _update_meta(base_dir, ep_idx, task, n_frames, success,
                     self._cam_dims, self.fps)

        # 6. Nettoyage du répertoire temporaire
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

        print(f"[writer] Episode {ep_idx} saved → {base_dir}", flush=True)
        print(f"         Frames  : {self.frame_counts}", flush=True)
        print(f"         Parquet : {n_frames} rows at {self.fps} fps", flush=True)
        print(f"         Depth   : {len(self.depth_frames)} PNG frames", flush=True)

    def discard(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        print(f"[writer] Episode {self.ep_idx} discarded.", flush=True)


# ------------------------------------------------------------------ #
# Écriture parquet (LeRobot v2)
# ------------------------------------------------------------------ #

def _write_parquet(path: str, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        return

    table = pa.table({
        "timestamp":
            pa.array([r["timestamp"] for r in rows], type=pa.float32()),
        "frame_index":
            pa.array([r["frame_index"] for r in rows], type=pa.int64()),
        "episode_index":
            pa.array([r["episode_index"] for r in rows], type=pa.int64()),
        "index":
            pa.array([r["index"] for r in rows], type=pa.int64()),
        "task_index":
            pa.array([r["task_index"] for r in rows], type=pa.int64()),
        "next.done":
            pa.array([r["next.done"] for r in rows], type=pa.bool_()),
        "observation.state":
            pa.array([r["observation.state"] for r in rows],
                     type=pa.list_(pa.float32())),
        "action":
            pa.array([r["action"] for r in rows],
                     type=pa.list_(pa.float32())),
    })
    pq.write_table(table, path, compression="snappy")


# ------------------------------------------------------------------ #
# Métadonnées dataset (info.json + episodes.jsonl)
# ------------------------------------------------------------------ #

def _default_info(cam_dims: dict, fps: int) -> dict:
    features = {
        "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
        "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
        "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
        "index":         {"dtype": "int64",   "shape": [1], "names": None},
        "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        "next.done":     {"dtype": "bool",    "shape": [1], "names": None},
        "observation.state": {
            "dtype": "float32",
            "shape": [_STATE_DIM],
            "names": _STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": [_ACTION_DIM],
            "names": _ACTION_NAMES,
        },
    }
    for name, (w, h) in cam_dims.items():
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": fps,
                "video.height": h,
                "video.width": w,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return {
        "codebase_version": "v2.0",
        "robot_type": "panda",
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:0"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/"
            "{video_key}_episode_{episode_index:06d}.mp4"
        ),
        "tasks": [],
        "features": features,
    }


def _update_meta(
    base_dir: str,
    ep_idx: int,
    task: str,
    n_frames: int,
    success: bool,
    cam_dims: dict,
    fps: int,
) -> None:
    meta_dir = os.path.join(base_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    info_path = os.path.join(meta_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = _default_info(cam_dims, fps)

    info["total_episodes"] = ep_idx + 1
    info["total_frames"] = info.get("total_frames", 0) + n_frames
    info["splits"] = {"train": f"0:{ep_idx + 1}"}

    tasks = info.setdefault("tasks", [])
    if not any(t["task"] == task for t in tasks):
        tasks.append({"task_index": 0, "task": task})

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    episodes_path = os.path.join(meta_dir, "episodes.jsonl")
    with open(episodes_path, "a") as f:
        f.write(json.dumps({
            "episode_index": ep_idx,
            "tasks": [task],
            "length": n_frames,
            "success": success,
        }) + "\n")


# ------------------------------------------------------------------ #
# Compteurs pour reprendre un dataset existant
# ------------------------------------------------------------------ #

def _count_existing_episodes(base_dir: str) -> int:
    info_path = os.path.join(base_dir, "meta", "info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path) as f:
                return json.load(f)["total_episodes"]
        except Exception:
            pass
    data_dir = os.path.join(base_dir, "data", "chunk-000")
    if not os.path.exists(data_dir):
        return 0
    return sum(1 for fn in os.listdir(data_dir) if fn.endswith(".parquet"))


def _count_existing_frames(base_dir: str) -> int:
    info_path = os.path.join(base_dir, "meta", "info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path) as f:
                return json.load(f)["total_frames"]
        except Exception:
            pass
    return 0


# ------------------------------------------------------------------ #
# Boucle principale du processus écrivain
# ------------------------------------------------------------------ #

def writer_process(
    rgb_channels: dict,   # {name: FrameChannel}
    depth_channel,        # FrameChannel | None
    state_q: mp.Queue,
    cmd_q: mp.Queue,
    base_dir: str,
    fps: int = 30,
) -> None:
    """
    Fonction cible du processus écrivain. Tourne en boucle jusqu'à 'shutdown'.
    Polling à ~120 Hz (non-bloquant) pour drainer les canaux caméra.
    """
    session: "_RecordingSession | None" = None
    ep_counter   = _count_existing_episodes(base_dir)
    total_frames = _count_existing_frames(base_dir)

    ch_names = list(rgb_channels.keys())
    widths   = [c.width  for c in rgb_channels.values()]
    heights  = [c.height for c in rgb_channels.values()]

    while True:

        # -- Traiter les commandes --
        try:
            cmd = cmd_q.get_nowait()
            action = cmd[0]

            if action == "start":
                meta = cmd[1] if len(cmd) > 1 else {}
                ep_idx = ep_counter
                ep_counter += 1
                tmp_dir = os.path.join(base_dir, f".tmp_ep_{ep_idx:06d}")
                session = _RecordingSession(
                    tmp_dir, ep_idx, ch_names, widths, heights,
                    fps, meta, global_frame_offset=total_frames,
                )
                print(f"[writer] Episode {ep_idx} started.", flush=True)

            elif action == "stop":
                if session is not None:
                    success = cmd[1] if len(cmd) > 1 else True
                    _drain_all(session, rgb_channels, depth_channel, state_q)
                    # Mettre à jour l'offset avant de passer au thread
                    ref_ch = ch_names[0] if ch_names else None
                    total_frames += (
                        session.frame_counts.get(ref_ch, 0) if ref_ch else 0
                    )
                    snap = session
                    threading.Thread(
                        target=snap.finalize,
                        args=(base_dir, success),
                        daemon=True,
                    ).start()
                    session = None

            elif action == "discard":
                if session is not None:
                    session.discard()
                    session = None
                _flush_all(rgb_channels, depth_channel, state_q)

            elif action == "shutdown":
                if session is not None:
                    session.discard()
                break

        except _queue.Empty:
            pass

        # -- Drainer les canaux --
        if session is not None:
            _drain_all(session, rgb_channels, depth_channel, state_q)
        else:
            _flush_all(rgb_channels, depth_channel, state_q)

        time.sleep(1.0 / 120)


def _drain_all(session, rgb_channels, depth_channel, state_q):
    for name, ch in rgb_channels.items():
        for t_ns, frame in ch.drain():
            session.push_frame(name, t_ns, frame)

    if depth_channel is not None:
        for t_ns, depth in depth_channel.drain():
            session.push_depth(t_ns, depth[:, :, 0])

    while True:
        try:
            session.push_state(state_q.get_nowait())
        except _queue.Empty:
            break


def _flush_all(rgb_channels, depth_channel, state_q):
    for ch in rgb_channels.values():
        ch.flush()
    if depth_channel is not None:
        depth_channel.flush()
    while True:
        try:
            state_q.get_nowait()
        except _queue.Empty:
            break
