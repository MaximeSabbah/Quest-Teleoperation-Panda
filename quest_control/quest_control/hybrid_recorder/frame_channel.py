"""
FrameChannel: canal inter-processus zéro-copie pour frames vidéo.

Utilise SharedMemory pour les pixels (anneau de slots) et une mp.Queue
pour les métadonnées (seq, timestamp, index de slot).

Fonctionne avec la méthode start 'fork' (défaut Linux) : les file
descriptors de la Queue et le mapping SHM sont hérités après fork().

Producteur : écrit dans write(frame, t_ns).
Consommateur : récupère via drain() → [(t_ns, frame), ...].
"""
import queue as _queue
import multiprocessing as mp
import numpy as np
from multiprocessing import shared_memory

RING_SLOTS = 16  # fenêtre de buffering (~0.5 s à 30 fps)


class FrameChannel:
    def __init__(self, name: str, height: int, width: int,
                 channels: int = 3, dtype=np.uint8):
        self.name = name
        self.height = height
        self.width = width
        self.channels = channels
        self.dtype = np.dtype(dtype)

        self._nbytes = int(self.dtype.itemsize) * height * width * channels
        self._n = RING_SLOTS

        self._shm = shared_memory.SharedMemory(
            name=name, create=True, size=self._nbytes * RING_SLOTS
        )
        # maxsize = RING_SLOTS : invariant que la queue ne dépasse jamais l'anneau SHM.
        # Quand la queue est pleine, le drop de la plus ancienne notification correspond
        # exactement au slot SHM sur le point d'être écrasé (modulo RING_SLOTS).
        self._q: mp.Queue = mp.Queue(maxsize=RING_SLOTS)
        self._dropped = mp.Value("i", 0)
        self._seq = 0  # compteur local au producteur, non partagé

    # ------------------------------------------------------------------ #
    # API Producteur
    # ------------------------------------------------------------------ #

    def write(self, frame: np.ndarray, t_cap_ns: int) -> None:
        """Écrire une frame dans l'anneau. Supprime la plus ancienne si plein."""
        slot = self._seq % self._n
        shape = (self.height, self.width, self.channels)
        dst = np.ndarray(shape, dtype=self.dtype,
                         buffer=self._shm.buf, offset=slot * self._nbytes)
        np.copyto(dst, frame)

        try:
            self._q.put_nowait((self._seq, t_cap_ns, slot))
        except _queue.Full:
            try:
                self._q.get_nowait()  # supprime la plus ancienne
            except _queue.Empty:
                pass
            with self._dropped.get_lock():
                self._dropped.value += 1
            self._q.put_nowait((self._seq, t_cap_ns, slot))

        self._seq += 1

    # ------------------------------------------------------------------ #
    # API Consommateur
    # ------------------------------------------------------------------ #

    def drain(self) -> list[tuple[int, np.ndarray]]:
        """Retourner toutes les frames disponibles, triées par seq."""
        pending = []
        while True:
            try:
                pending.append(self._q.get_nowait())
            except _queue.Empty:
                break

        pending.sort()  # tri par seq (premier élément du tuple)

        out = []
        for _seq, t_cap_ns, slot in pending:
            shape = (self.height, self.width, self.channels)
            arr = np.ndarray(shape, dtype=self.dtype,
                             buffer=self._shm.buf,
                             offset=slot * self._nbytes).copy()
            out.append((t_cap_ns, arr))
        return out

    def flush(self) -> None:
        """Vider la queue sans traiter (appeler quand on n'enregistre pas)."""
        while True:
            try:
                self._q.get_nowait()
            except _queue.Empty:
                break

    # ------------------------------------------------------------------ #
    # Cycle de vie
    # ------------------------------------------------------------------ #

    @property
    def dropped_frames(self) -> int:
        return self._dropped.value

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass
