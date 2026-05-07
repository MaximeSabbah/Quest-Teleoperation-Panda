gepetto@miyanoura:~/ros2_ws$ 
ros2 launch quest_control quest_control.launch.py  use_rviz:=true robot_ip:=172.17.1.3 aux_computer_ip:=panda2 aux_computer_user:=mkulcsar 

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0

1. Variables d'environnement (à mettre dans ton .bashrc ou avant de lancer)

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0
2. Lance tout

ros2 launch quest_control quest_control.launch.py \
  use_rviz:=true \
  robot_ip:=172.17.1.2 \
  aux_computer_ip:=panda1 \
  aux_computer_user:=mkulcsar
robot_ip:=172.17.1.2 — IP du bras Franka
aux_computer_ip:=panda1 — hostname du PC auxiliaire (celui connecté directement au robot)
aux_computer_user:=mkulcsar — utilisateur SSH sur ce PC
OPENCV_VIDEOIO_PRIORITY_GSTREAMER=0 — important pour que cv2 utilise V4L2 et non GStreamer pour les caméras

Résumé de ce qui a été fait
5 nouveaux fichiers créés dans hybrid_recorder/
frame_channel.py — Canal inter-processus zéro-copie

Anneau SHM (SharedMemory) pour les pixels des frames
mp.Queue pour les métadonnées (seq, timestamp, slot)
maxsize=RING_SLOTS : invariant garanti que les drops correspondent aux slots sur le point d'être écrasés
Testé : écriture/lecture, overflow, flush
producers.py — Processus caméras

arducam_producer() : V4L2 via cv2, exposition fixe pour réduire la gigue
d435_producer() : pyrealsense2 (timestamps hardware + profondeur uint16) avec fallback cv2 si absent
Timestamps CLOCK_MONOTONIC au plus près de la capture
episode_writer.py — Processus écrivain

Loop non-bloquante à ~120 Hz qui draine les canaux
H.264 en streaming (PyAV si dispo, sinon cv2 VideoWriter)
Profondeur uint16 → PNG via cv2
État/action → HDF5 (clé par clé, même format que l'existant)
Finalisation dans un thread → la boucle continue à flusher pendant l'encodage
Déplacement atomique tmp/ → episode_XXXXXX/
Commandes : start/stop/discard/shutdown
recorder.py — Orchestrateur

Crée les canaux SHM, spawne les processus producteurs + écrivain
API simple : start_episode() / stop_episode() / discard_episode() / record_state() / shutdown()
Modifications minimales de quest_streamer.py
Import + instanciation du HybridRecorder avec un flag USE_HYBRID_RECORDER = True
Bouton A → start/stop, bouton B → discard (nouvelle feature)
_build_state_dict() : construit le dict état/action depuis le robot et le plan MPC
record_state() appelé à chaque iteration de stream_quest_to_robot()
hybrid_recorder.shutdown() dans le finally de main()
Format de sortie

~/demos/red_cube_hybrid/
  episode_000000/
    observation.images.exterior_1.mp4   ← H.264 RGB (Arducam gauche)
    observation.images.exterior_2.mp4   ← H.264 RGB (Arducam droite)
    observation.images.wrist.mp4        ← H.264 RGB (D435)
    wrist_depth/frame_000000.png ...    ← uint16 PNG (D435 depth)
    data.hdf5                           ← joint_pos, ee_pos, action, timestamps...
    metadata.json                       ← language, success, n_frames, dropped...
Pour installer les dépendances optionnelles

pip install av           # H.264 propre (vs cv2 VideoWriter)
pip install pyrealsense2 # profondeur D435 + timestamps hardware
donc concretement comment je lance tout ?
Voici exactement ce qu'il faut faire dans l'ordre.

1. Build le workspace ROS2

cd ~/your_ros2_ws   # le workspace qui contient quest_control
colcon build --packages-select quest_control
source install/setup.bash
2. Lancer le système complet (un seul terminal)

ros2 launch quest_control quest_control.launch.py
Ce launch file démarre tout dans l'ordre :

Le bras Franka
L'attente que les joints soient non-nuls
agimus_controller
quest_streamer — qui crée automatiquement le HybridRecorder et spawne les processus caméras en arrière-plan
3. Contrôles pendant la téléopération
Action	Bouton Quest
Démarrer un épisode	A (appui court)
Arrêter et sauvegarder	A (deuxième appui)
Annuler l'épisode en cours	B
Fermer le gripper	GripButton (gâchette)
4. Vérifier que ça enregistre

ls ~/demos/red_cube_hybrid/
# → episode_000000/  episode_000001/  ...

ls ~/demos/red_cube_hybrid/episode_000000/
# → observation.images.exterior_1.mp4
#   observation.images.exterior_2.mp4
#   observation.images.wrist.mp4
#   wrist_depth/
#   data.hdf5
#   metadata.json
5. Si ça ne démarre pas
Problème caméra : Les chemins V4L2 sont hardcodés dans recorder.py:28-38. Vérifie les tiens :


ls /dev/v4l/by-path/
# copier-coller les bons chemins dans DEFAULT_ARDUCAM_CONFIGS et DEFAULT_D435_CONFIG
Désactiver le nouveau recorder si besoin de débugger (retour à l'ancien comportement) :


# dans quest_streamer.py ligne ~124
self.USE_HYBRID_RECORDER = False
Pour la profondeur D435 :


pip install pyrealsense2
Pour H.264 propre (recommandé) :


pip install av
6. Inspecter un épisode enregistré

import h5py, json, numpy as np

# Données robot
with h5py.File("~/demos/red_cube_hybrid/episode_000000/data.hdf5") as f:
    print(list(f.keys()))        # joint_pos, ee_pos, action_ee_pos...
    print(f["joint_pos"].shape)  # (N, 7)

# Métadonnées
with open("~/demos/red_cube_hybrid/episode_000000/metadata.json") as f:
    print(json.load(f))  # n_state_steps, frame_counts, success...

ros2 launch quest_control quest_control.launch.py use_rviz:=true robot_ip:=172.17.1.2 task_name:=grab_the_red_cube trajectory_buffer:=standard
