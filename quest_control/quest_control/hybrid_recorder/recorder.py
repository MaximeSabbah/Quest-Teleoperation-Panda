"""
HybridRecorder : orchestrateur du système d'enregistrement.

2 Arducams RGB (exterior_1, exterior_2) + 1 Intel RealSense D435 (wrist RGB +
depth uint16) tournent dans des processus séparés. Les données de contrôle
arrivent via Queue depuis le nœud ROS2. Le processus écrivain produit des
épisodes au format LeRobot v2 (parquet + MP4).

IMPORTANT : instancier AVANT tout fork() dans le nœud ROS2 (i.e., dans __init__).
"""
import multiprocessing as mp
import os
import time
import numpy as np

from .frame_channel import FrameChannel
from .producers import arducam_producer, d435_producer
from .episode_writer import writer_process

# ------------------------------------------------------------------ #
# Configuration matérielle
# Adapter /dev/video* si la disposition USB change (v4l2-ctl --list-devices)
# ------------------------------------------------------------------ #

ARDUCAM_CONFIGS = {
    "exterior_1": {"device": "/dev/video0", "width": 640, "height": 480},
    "exterior_2": {"device": "/dev/video2", "width": 640, "height": 480},
}

# Résolution D435 (couleur et profondeur)
WRIST_RGB_W,   WRIST_RGB_H   = 640, 480
WRIST_DEPTH_W, WRIST_DEPTH_H = 640, 480

# None = premier D435 trouvé ; mettre le numéro de série pour fixer un device
WRIST_D435_SERIAL: str | None = None


class HybridRecorder:
    """
    Hybrid recorder : 2 Arducams + D435 (RGB + depth) + robot.

    Paramètres
    ----------
    base_dir : str
        Répertoire racine (ex: "~/demos").
    task_name : str
        Sous-répertoire de la tâche (ex: "cube_grasp").
    arducam_configs : dict | None
        Configs Arducam. None → ARDUCAM_CONFIGS.
    wrist_serial : str | None
        Numéro de série D435. None → premier device trouvé.
    fps : int
        FPS des vidéos (30 par défaut).
    """

    def __init__(
        self,
        base_dir: str,
        task_name: str,
        arducam_configs: dict | None = None,
        wrist_serial: str | None = None,
        fps: int = 30,
    ):
        self._base_dir = os.path.expanduser(os.path.join(base_dir, task_name))
        self._fps = fps
        self._recording = False
        os.makedirs(self._base_dir, exist_ok=True)

        arducam_cfg = arducam_configs or ARDUCAM_CONFIGS

        # --- FrameChannels RGB (Arducams) ---
        rgb_channels: dict[str, FrameChannel] = {
            name: FrameChannel(
                name=f"hrecorder_{name}",
                height=c["height"],
                width=c["width"],
                channels=3,
                dtype=np.uint8,
            )
            for name, c in arducam_cfg.items()
        }

        # --- FrameChannel RGB wrist (D435 couleur) ---
        wrist_rgb_ch = FrameChannel(
            name="hrecorder_wrist",
            height=WRIST_RGB_H,
            width=WRIST_RGB_W,
            channels=3,
            dtype=np.uint8,
        )
        rgb_channels["wrist"] = wrist_rgb_ch

        # --- FrameChannel depth (D435 profondeur uint16) ---
        depth_ch = FrameChannel(
            name="hrecorder_wrist_depth",
            height=WRIST_DEPTH_H,
            width=WRIST_DEPTH_W,
            channels=1,
            dtype=np.uint16,
        )

        self._channels = rgb_channels
        self._depth_ch = depth_ch

        # --- Queues et events partagés ---
        self._state_q: mp.Queue = mp.Queue(maxsize=5000)
        self._cmd_q: mp.Queue = mp.Queue(maxsize=50)
        self._record_event = mp.Event()
        self._stop_event = mp.Event()

        self._procs: list[mp.Process] = []

        # --- Producteurs Arducam ---
        for name, c in arducam_cfg.items():
            p = mp.Process(
                target=arducam_producer,
                args=(rgb_channels[name], c["device"],
                      self._record_event, self._stop_event),
                daemon=True,
                name=f"producer_{name}",
            )
            p.start()
            self._procs.append(p)

        # --- Producteur D435 (RGB + depth) ---
        serial = wrist_serial or WRIST_D435_SERIAL
        wrist_proc = mp.Process(
            target=d435_producer,
            args=(wrist_rgb_ch, depth_ch, serial,
                  self._record_event, self._stop_event),
            daemon=True,
            name="producer_wrist_d435",
        )
        wrist_proc.start()
        self._procs.append(wrist_proc)

        # --- Processus écrivain ---
        self._writer_proc = mp.Process(
            target=writer_process,
            args=(self._channels, self._depth_ch, self._state_q,
                  self._cmd_q, self._base_dir, self._fps),
            daemon=True,
            name="episode_writer",
        )
        self._writer_proc.start()

        print(
            f"[HybridRecorder] Started. Base: {self._base_dir}\n"
            f"  RGB cameras : {list(self._channels.keys())}\n"
            f"  Depth D435  : {WRIST_DEPTH_W}x{WRIST_DEPTH_H}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # API publique
    # ------------------------------------------------------------------ #

    def start_episode(
        self,
        language_instruction: str = "",
        demonstrator_id: str = "",
    ) -> None:
        """Démarrer un nouvel épisode."""
        if self._recording:
            print("[HybridRecorder] WARN: épisode déjà en cours.")
            return
        meta = {
            "language_instruction": language_instruction,
            "demonstrator_id": demonstrator_id,
            "t_wall_start": time.time(),
        }
        self._cmd_q.put(("start", meta))
        self._record_event.set()
        self._recording = True

    def stop_episode(self, success: bool = True) -> None:
        """Arrêter et sauvegarder l'épisode courant."""
        if not self._recording:
            return
        self._record_event.clear()
        self._cmd_q.put(("stop", success))
        self._recording = False

    def discard_episode(self) -> None:
        """Annuler l'épisode courant sans sauvegarder."""
        if not self._recording:
            return
        self._record_event.clear()
        self._cmd_q.put(("discard",))
        self._recording = False

    def record_state(self, state: dict) -> None:
        """
        Enregistrer un pas de contrôle. Appeler à chaque itération du contrôleur.

        Clés attendues (numpy arrays ou scalaires) :
            timestamp_ns       : int    — CLOCK_MONOTONIC en ns
            joint_pos          : (7,)   — positions articulaires [rad]
            joint_vel          : (7,)   — vitesses articulaires [rad/s]
            ee_pos             : (3,)   — position EE dans fer_link0 [m]
            ee_quat            : (4,)   — orientation EE xyzw
            gripper_pos        : (2,)   — positions doigts [m]
            gripper_vel        : (2,)   — vitesses doigts [m/s]
            action_ee_pos      : (3,)   — cible EE MPC [m]
            action_ee_quat     : (4,)   — orientation cible xyzw
            action_gripper_cmd : float  — 0.0=fermé, 1.0=ouvert
        """
        if not self._recording:
            return
        try:
            self._state_q.put_nowait(state)
        except Exception:
            pass

    @property
    def is_recording(self) -> bool:
        return self._recording

    def shutdown(self) -> None:
        """Arrêter tous les processus et libérer la SharedMemory."""
        self._stop_event.set()
        if self._recording:
            self.discard_episode()
        self._cmd_q.put(("shutdown",))

        self._writer_proc.join(timeout=15)
        for p in self._procs:
            p.join(timeout=3)

        for ch in self._channels.values():
            ch.close()
            ch.unlink()

        self._depth_ch.close()
        self._depth_ch.unlink()

        print("[HybridRecorder] Arrêt complet.")
