"""
Dataset management tool — delete episodes and compact the index.

Usage:
    python manage_dataset.py <dataset_dir> delete <ep1> [ep2 ep3 ...]
    python manage_dataset.py <dataset_dir> delete 3 7 11-14   # ranges ok
    python manage_dataset.py <dataset_dir> status

The 'delete' command:
  1. Removes parquet + all video files for the listed episodes.
  2. Renumbers the surviving episodes 0, 1, 2, … N-1 (compacts gaps).
  3. Patches episode_index and global index columns inside each parquet.
  4. Rewrites meta/episodes.jsonl and meta/info.json.

Run with:
    ~/rerun_venv/bin/python scripts/manage_dataset.py <dataset_dir> ...
"""
import json
import os
import sys
import shutil
import glob
import pyarrow as pa
import pyarrow.parquet as pq


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _data_dir(base: str) -> str:
    return os.path.join(base, "data", "chunk-000")

def _video_dir(base: str) -> str:
    return os.path.join(base, "videos", "chunk-000")

def _depth_dir(base: str, ep: int) -> str:
    return os.path.join(base, "depth", f"episode_{ep:06d}")

def _parquet_path(base: str, ep: int) -> str:
    return os.path.join(_data_dir(base), f"episode_{ep:06d}.parquet")

def _video_paths(base: str, ep: int) -> list[str]:
    pattern = os.path.join(_video_dir(base), f"*_episode_{ep:06d}.mp4")
    return glob.glob(pattern)

def _existing_episodes(base: str) -> list[int]:
    """Return sorted list of episode indices that have a parquet file."""
    d = _data_dir(base)
    if not os.path.isdir(d):
        return []
    idxs = []
    for fn in os.listdir(d):
        if fn.endswith(".parquet"):
            try:
                idxs.append(int(fn[len("episode_"):-len(".parquet")]))
            except ValueError:
                pass
    return sorted(idxs)

def _parse_indices(tokens: list[str]) -> set[int]:
    """Parse ['3', '7', '11-14'] → {3, 7, 11, 12, 13, 14}."""
    result = set()
    for tok in tokens:
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(tok))
    return result


# ------------------------------------------------------------------ #
# Status
# ------------------------------------------------------------------ #

def cmd_status(base: str) -> None:
    existing = _existing_episodes(base)
    info_path = os.path.join(base, "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    print(f"Dataset : {base}")
    print(f"  info.json  total_episodes : {info.get('total_episodes')}")
    print(f"  info.json  total_frames   : {info.get('total_frames')}")
    print(f"  Parquet files on disk     : {len(existing)}")
    if existing:
        print(f"  Episode indices          : {existing[0]}..{existing[-1]}")
        gaps = [i for i in range(existing[0], existing[-1] + 1) if i not in existing]
        if gaps:
            print(f"  GAPS (missing)           : {gaps}")
        else:
            print("  No gaps — index is contiguous.")


# ------------------------------------------------------------------ #
# Delete + compact
# ------------------------------------------------------------------ #

def cmd_delete(base: str, to_delete: set[int]) -> None:
    existing = _existing_episodes(base)
    if not existing:
        print("No episodes found.")
        return

    # Validate
    invalid = to_delete - set(existing)
    if invalid:
        print(f"WARNING: episodes not found on disk (skipping): {sorted(invalid)}")
    to_delete &= set(existing)

    survivors = [ep for ep in existing if ep not in to_delete]

    if not to_delete and existing == list(range(len(existing))):
        print("Nothing to delete and index is already contiguous.")
        return
    print(f"Deleting episodes : {sorted(to_delete)}")
    print(f"Keeping  episodes : {survivors}")
    print(f"Will renumber to  : 0 … {len(survivors) - 1}")
    answer = input("Proceed? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    # --- Step 1: delete files for removed episodes ---
    for ep in sorted(to_delete):
        p = _parquet_path(base, ep)
        if os.path.exists(p):
            os.remove(p)
            print(f"  Removed {os.path.basename(p)}")
        for v in _video_paths(base, ep):
            os.remove(v)
            print(f"  Removed {os.path.basename(v)}")
        d = _depth_dir(base, ep)
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"  Removed depth dir episode_{ep:06d}")

    # --- Step 2: renumber survivors ---
    # Load episodes.jsonl (may have stale entries — rebuild from parquets)
    episode_meta: dict[int, dict] = {}
    jsonl_path = os.path.join(base, "meta", "episodes.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    episode_meta[e["episode_index"]] = e

    global_frame_offset = 0
    new_episodes_meta = []

    for new_idx, old_idx in enumerate(survivors):
        if new_idx == old_idx:
            # No rename needed, but still need to patch parquet if offset changed
            pass

        # -- patch parquet --
        parquet_src = _parquet_path(base, old_idx)
        table = pq.read_table(parquet_src)
        n_frames = len(table)

        schema = table.schema
        col_names = schema.names

        new_cols = {}
        if "episode_index" in col_names:
            new_cols["episode_index"] = pa.array(
                [new_idx] * n_frames,
                type=table.schema.field("episode_index").type,
            )
        if "index" in col_names:
            new_cols["index"] = pa.array(
                list(range(global_frame_offset, global_frame_offset + n_frames)),
                type=table.schema.field("index").type,
            )
        if "frame_index" in col_names:
            # frame_index is relative to episode — keep as-is
            pass

        for col, arr in new_cols.items():
            idx_col = col_names.index(col)
            table = table.set_column(idx_col, col, arr)

        # Write to temp then rename
        tmp_path = parquet_src + ".tmp"
        pq.write_table(table, tmp_path)

        # Rename video files first (before overwriting parquet in case old==new)
        new_parquet = _parquet_path(base, new_idx)
        for v_old in _video_paths(base, old_idx):
            cam_key = os.path.basename(v_old).split(f"_episode_{old_idx:06d}")[0]
            v_new = os.path.join(_video_dir(base), f"{cam_key}_episode_{new_idx:06d}.mp4")
            if v_old != v_new:
                os.rename(v_old, v_new)

        # Rename depth dir
        d_old = _depth_dir(base, old_idx)
        d_new = _depth_dir(base, new_idx)
        if os.path.isdir(d_old) and d_old != d_new:
            os.rename(d_old, d_new)

        # Move parquet (tmp → new path, handles old_idx==new_idx cleanly)
        os.replace(tmp_path, new_parquet)
        if old_idx != new_idx and parquet_src != new_parquet:
            # parquet_src was already overwritten by os.replace if same path,
            # but if different, the old file still exists — remove it
            if os.path.exists(parquet_src):
                os.remove(parquet_src)

        # -- update episode meta --
        meta = episode_meta.get(old_idx, {
            "tasks": [""],
            "length": n_frames,
            "success": True,
        })
        meta["episode_index"] = new_idx
        meta["length"] = n_frames
        new_episodes_meta.append(meta)

        global_frame_offset += n_frames
        print(f"  ep {old_idx:>4} → {new_idx:>4}  ({n_frames} frames)")

    # --- Step 3: rewrite episodes.jsonl ---
    with open(jsonl_path, "w") as f:
        for m in new_episodes_meta:
            f.write(json.dumps(m) + "\n")

    # --- Step 4: rewrite info.json ---
    info_path = os.path.join(base, "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)
    info["total_episodes"] = len(survivors)
    info["total_frames"] = global_frame_offset
    info["splits"] = {"train": f"0:{len(survivors)}"}
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone. {len(survivors)} episodes, {global_frame_offset} frames total.")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    base = os.path.expanduser(sys.argv[1])
    command = sys.argv[2]

    if command == "status":
        cmd_status(base)
    elif command == "compact":
        # Renumber existing episodes to fill gaps without deleting anything
        cmd_delete(base, set())
    elif command == "delete":
        if len(sys.argv) < 4:
            print("Usage: manage_dataset.py <dir> delete <ep> [ep ...]")
            sys.exit(1)
        indices = _parse_indices(sys.argv[3:])
        cmd_delete(base, indices)
    else:
        print(f"Unknown command: {command}")
        print("Commands: status, compact, delete")
        sys.exit(1)


if __name__ == "__main__":
    main()
