"""
Visualise a recorded LeRobot episode with Rerun.

Usage:
    ~/rerun_venv/bin/python visualize_episode.py <base_dir> [episode_index]

Example:
    ~/rerun_venv/bin/python visualize_episode.py ~/demos/red_cube_hybrid 0

Displays:
    - All camera streams (exterior_1, exterior_2, wrist) time-aligned
    - Robot state: joint positions, EE position, gripper
    - Action: EE target and gripper command
    - Depth frames (if available)
"""
import sys
import os
import numpy as np
import pyarrow.parquet as pq
import cv2
import rerun as rr
import rerun.blueprint as rrb


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
    "action_ee_pos_x", "action_ee_pos_y", "action_ee_pos_z",
    "action_ee_quat_x", "action_ee_quat_y", "action_ee_quat_z", "action_ee_quat_w",
    "action_gripper_cmd",
    "torque_j1", "torque_j2", "torque_j3", "torque_j4",
    "torque_j5", "torque_j6", "torque_j7",
]


def load_episode(base_dir: str, ep_idx: int) -> dict:
    path = os.path.join(base_dir, "data", "chunk-000", f"episode_{ep_idx:06d}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No parquet at {path}")
    return pq.read_table(path).to_pydict()


def video_frames(video_path: str):
    """Yield (frame_index, bgr_frame) from a video file."""
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield i, frame
        i += 1
    cap.release()


def depth_frames(depth_dir: str):
    """Yield (frame_index, depth_uint16) from a depth PNG directory."""
    if not os.path.isdir(depth_dir):
        return
    files = sorted(f for f in os.listdir(depth_dir) if f.endswith(".png"))
    for i, fname in enumerate(files):
        d = cv2.imread(os.path.join(depth_dir, fname), cv2.IMREAD_UNCHANGED)
        if d is not None:
            yield i, d.astype(np.uint16)


def main():
    if len(sys.argv) < 2:
        print("Usage: visualize_episode.py <base_dir> [episode_index]")
        sys.exit(1)

    base_dir = os.path.expanduser(sys.argv[1])
    ep_idx   = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print(f"Loading episode {ep_idx} from {base_dir} …")
    data = load_episode(base_dir, ep_idx)

    timestamps  = np.array(data["timestamp"], dtype=np.float64)   # s from ep start
    obs_state   = np.array(data["observation.state"])              # (N, 25)
    actions     = np.array(data["action"])                         # (N, 15)
    n_rows      = len(timestamps)

    # --- Blueprint: cameras on top row, scalar plots on bottom ---
    video_dir = os.path.join(base_dir, "videos", "chunk-000")
    cam_names = sorted(
        f.split("_episode")[0].replace("observation.images.", "")
        for f in os.listdir(video_dir)
        if f.endswith(f"_episode_{ep_idx:06d}.mp4")
    )

    camera_views = [rrb.Spatial2DView(name=c, origin=f"/cameras/{c}") for c in cam_names]
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*camera_views, row_shares=[1] * len(camera_views)),
            rrb.Horizontal(
                rrb.TimeSeriesView(name="EE position",  origin="/robot/ee_pos"),
                rrb.TimeSeriesView(name="Gripper",      origin="/robot/gripper"),
                rrb.TimeSeriesView(name="Joint pos",    origin="/robot/joint_pos"),
            ),
            row_shares=[3, 2],
        ),
        collapse_panels=True,
    )

    rr.init("quest_episode_viewer", spawn=True)
    rr.send_blueprint(blueprint)

    # --- Log scalar robot state at each timestep ---
    print("Logging robot state …")
    for i in range(n_rows):
        t = float(timestamps[i])
        rr.set_time_seconds("time", t)

        state  = obs_state[i]
        action = actions[i]

        # EE position (world frame)
        rr.log("robot/ee_pos/x", rr.Scalar(float(state[14])))
        rr.log("robot/ee_pos/y", rr.Scalar(float(state[15])))
        rr.log("robot/ee_pos/z", rr.Scalar(float(state[16])))

        # EE target
        rr.log("robot/ee_target/x", rr.Scalar(float(action[0])))
        rr.log("robot/ee_target/y", rr.Scalar(float(action[1])))
        rr.log("robot/ee_target/z", rr.Scalar(float(action[2])))

        # Gripper
        rr.log("robot/gripper/pos_l",   rr.Scalar(float(state[22])))
        rr.log("robot/gripper/pos_r",   rr.Scalar(float(state[23])))
        rr.log("robot/gripper/command", rr.Scalar(float(action[7])))

        # Joint positions
        for j in range(7):
            rr.log(f"robot/joint_pos/j{j+1}", rr.Scalar(float(state[j])))

    # --- Log camera frames ---
    for cam in cam_names:
        vpath = os.path.join(
            video_dir, f"observation.images.{cam}_episode_{ep_idx:06d}.mp4"
        )
        if not os.path.exists(vpath):
            continue
        print(f"Logging camera {cam} …")
        for frame_i, bgr in video_frames(vpath):
            if frame_i >= n_rows:
                break
            rr.set_time_seconds("time", float(timestamps[frame_i]))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rr.log(f"cameras/{cam}", rr.Image(rgb))

    # --- Log depth frames (if present) ---
    depth_dir = os.path.join(base_dir, "depth", f"episode_{ep_idx:06d}")
    depth_gen = list(depth_frames(depth_dir))
    if depth_gen:
        print("Logging depth frames …")
        for frame_i, d16 in depth_gen:
            if frame_i >= n_rows:
                break
            rr.set_time_seconds("time", float(timestamps[frame_i]))
            rr.log("cameras/wrist_depth", rr.DepthImage(d16, meter=1000.0))

    print(f"Done. Episode {ep_idx}: {n_rows} rows, {len(cam_names)} cameras.")


if __name__ == "__main__":
    main()
