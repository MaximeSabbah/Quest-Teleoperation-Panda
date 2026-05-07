"""
Synchronization assessment between video streams and robot state data.

Protocol:
  1. Record an episode where you flash a light at all cameras
     AND press the grip button at the same moment.
  2. Run: python assess_sync.py ~/demos/red_cube_hybrid 0
     (replace 0 with the episode index you want to analyse)

What it measures:
  - Video: frame brightness peak (the flash)
  - Parquet: gripper_pos step (grip close event)
  - Both are in CLOCK_MONOTONIC time — offset should be < 33 ms (one frame)
"""
import sys
import os
import numpy as np
import pyarrow.parquet as pq
import cv2
import matplotlib.pyplot as plt


def load_episode(base_dir: str, ep_idx: int):
    parquet_path = os.path.join(
        base_dir, "data", "chunk-000", f"episode_{ep_idx:06d}.parquet"
    )
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    table = pq.read_table(parquet_path)
    return table.to_pydict()


def brightness_per_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    values = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        values.append(float(frame.mean()))
    cap.release()
    return np.array(values)


def find_flash_frame(brightness: np.ndarray) -> int:
    """Frame index of the sharpest brightness jump (the flash)."""
    diff = np.diff(brightness)
    return int(np.argmax(diff)) + 1


def find_grip_row(gripper_pos_col: list) -> int:
    """Row index where gripper first closes (pos drops below half its initial value)."""
    pos = np.array([p[0] if isinstance(p, (list, tuple)) else p
                    for p in gripper_pos_col])
    # Gripper open ≈ 0.039 m, closed ≈ 0 m
    threshold = pos[:5].mean() * 0.5
    candidates = np.where(pos < threshold)[0]
    return int(candidates[0]) if len(candidates) else -1


def main():
    if len(sys.argv) < 2:
        print("Usage: python assess_sync.py <base_dir> [episode_index]")
        sys.exit(1)

    base_dir = os.path.expanduser(sys.argv[1])
    ep_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print(f"Loading episode {ep_idx} from {base_dir}")
    data = load_episode(base_dir, ep_idx)

    timestamps = np.array(data["timestamp"])        # seconds from episode start
    frame_indices = np.array(data["frame_index"])

    # Extract gripper_pos_l (first gripper finger, index 22 in observation.state)
    obs = np.array(data["observation.state"])       # shape (N, 25)
    gripper_l = obs[:, 22]                          # gripper_pos_l

    grip_row = find_grip_row(gripper_l)
    if grip_row == -1:
        print("WARNING: no clear grip event found in parquet — did you close the gripper?")
    else:
        grip_time = timestamps[grip_row]
        print(f"\nGripper close event: row={grip_row}, t={grip_time:.3f}s, "
              f"gripper_pos={gripper_l[grip_row]:.4f}m")

    # Check available video streams
    video_dir = os.path.join(base_dir, "videos", "chunk-000")
    available = [f for f in os.listdir(video_dir)
                 if f.endswith(f"_episode_{ep_idx:06d}.mp4")]
    if not available:
        print(f"No video files found for episode {ep_idx} in {video_dir}")
        sys.exit(1)

    print(f"\nVideo streams found: {[f.split('_episode')[0] for f in available]}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    ax_video = axes[0]
    ax_robot = axes[1]

    flash_times = {}
    for fname in sorted(available):
        cam_name = fname.replace(f"_episode_{ep_idx:06d}.mp4", "")
        vpath = os.path.join(video_dir, fname)
        brightness = brightness_per_frame(vpath)
        fps = 30.0
        video_times = np.arange(len(brightness)) / fps

        flash_frame = find_flash_frame(brightness)
        flash_t = video_times[flash_frame]
        flash_times[cam_name] = flash_t
        print(f"  {cam_name}: flash at frame={flash_frame}, t={flash_t:.3f}s")

        ax_video.plot(video_times, brightness, label=cam_name, alpha=0.8)
        ax_video.axvline(flash_t, linestyle="--", alpha=0.5)

    ax_video.set_title("Camera brightness over time (flash = vertical dashed line)")
    ax_video.set_xlabel("Time (s)")
    ax_video.set_ylabel("Mean brightness")
    ax_video.legend()

    # Robot state plot
    ax_robot.plot(timestamps, gripper_l, label="gripper_pos_l (m)", color="tab:blue")
    if grip_row != -1:
        ax_robot.axvline(grip_time, color="tab:blue", linestyle="--",
                         label=f"grip event t={grip_time:.3f}s")

    ax_robot.set_title("Robot gripper position over time")
    ax_robot.set_xlabel("Time from episode start (s)  [parquet timestamp]")
    ax_robot.set_ylabel("Gripper pos (m)")
    ax_robot.legend()

    plt.tight_layout()

    # Summary
    print("\n=== Sync summary ===")
    if flash_times:
        flash_mean = np.mean(list(flash_times.values()))
        print(f"Camera flash times : { {k: f'{v:.3f}s' for k, v in flash_times.items()} }")
        if len(flash_times) > 1:
            cam_spread = max(flash_times.values()) - min(flash_times.values())
            print(f"Camera-to-camera spread : {cam_spread*1000:.1f} ms  "
                  f"({'OK' if cam_spread < 0.033 else 'WARNING: > 1 frame'})")
        if grip_row != -1:
            robot_video_offset = grip_time - flash_mean
            print(f"Robot vs video offset   : {robot_video_offset*1000:+.1f} ms  "
                  f"(robot state leads if negative)  "
                  f"({'OK' if abs(robot_video_offset) < 0.033 else 'WARNING: > 1 frame'})")
    print()

    out_path = os.path.join(base_dir, f"sync_assessment_ep{ep_idx}.png")
    plt.savefig(out_path, dpi=120)
    print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
