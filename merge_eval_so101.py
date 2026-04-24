import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parent
DATASET_PREFIX = "eval_so101"
MERGED_DATASET = "eval_so101_merged"


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def is_lerobot_dataset_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "data").is_dir()
        and (path / "meta").is_dir()
        and (path / "meta" / "info.json").is_file()
    )


def discover_eval_datasets(base_dir: Path, prefix: str = DATASET_PREFIX) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(prefix)}(?:_\d+)?$")
    return [
        child.resolve()
        for child in sorted(base_dir.iterdir(), key=natural_sort_key)
        if pattern.match(child.name) and is_lerobot_dataset_dir(child)
    ]


def resolve_input(input_token: str, base_dir: Path) -> list[Path]:
    path = Path(input_token)
    resolved = path if path.is_absolute() else base_dir / path

    if resolved.exists():
        if is_lerobot_dataset_dir(resolved):
            return [resolved.resolve()]

        if resolved.is_dir():
            return [
                child.resolve()
                for child in sorted(resolved.iterdir(), key=natural_sort_key)
                if is_lerobot_dataset_dir(child)
            ]

    pattern = re.compile(rf"^{re.escape(input_token)}(?:_\d+)?$")
    return [
        child.resolve()
        for child in sorted(base_dir.iterdir(), key=natural_sort_key)
        if pattern.match(child.name) and is_lerobot_dataset_dir(child)
    ]


def offset_stats_dict(stats: dict, key: str, offset: int) -> None:
    if key in stats:
        for metric in ["min", "max", "mean"]:
            if metric in stats[key]:
                stats[key][metric] = [val + offset for val in stats[key][metric]]


def load_tasks(tasks_file: Path) -> dict[str, int]:
    tasks_vocab: dict[str, int] = {}
    if not tasks_file.exists():
        return tasks_vocab

    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            task = json.loads(line)
            tasks_vocab[task["task"]] = task["task_index"]
    return tasks_vocab


def build_task_mapping(current_tasks_file: Path, tasks_vocab: dict[str, int]) -> dict[int, int]:
    task_mapping: dict[int, int] = {}
    if not current_tasks_file.exists():
        return task_mapping

    with open(current_tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            task = json.loads(line)
            old_idx = task["task_index"]
            task_str = task["task"]

            if task_str not in tasks_vocab:
                tasks_vocab[task_str] = len(tasks_vocab)
            task_mapping[old_idx] = tasks_vocab[task_str]

    return task_mapping


def merge_parquet_files(
    current_dir: Path,
    merged_dir: Path,
    ep_offset: int,
    frame_offset: int,
    chunk_size: int,
    task_mapping: dict[int, int],
) -> int:
    pq_count = 0
    data_dir = current_dir / "data"
    if not data_dir.exists():
        return pq_count

    for pq_file in sorted(data_dir.rglob("episode_*.parquet"), key=natural_sort_key):
        try:
            ep_idx = int(pq_file.stem.split("_")[1])
        except (ValueError, IndexError):
            continue

        new_ep_idx = ep_idx + ep_offset
        new_chunk_idx = new_ep_idx // chunk_size
        target_chunk = merged_dir / "data" / f"chunk-{new_chunk_idx:03d}"
        target_chunk.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(pq_file)
        schema = table.schema

        if "episode_index" in table.column_names:
            idx = schema.get_field_index("episode_index")
            new_col = pc.add(
                table.column(idx),
                pa.scalar(ep_offset, type=schema.field(idx).type),
            )
            table = table.set_column(idx, "episode_index", new_col)

        if "index" in table.column_names:
            idx = schema.get_field_index("index")
            new_col = pc.add(
                table.column(idx),
                pa.scalar(frame_offset, type=schema.field(idx).type),
            )
            table = table.set_column(idx, "index", new_col)

        if "task_index" in table.column_names and task_mapping:
            idx = schema.get_field_index("task_index")
            task_arr = table.column(idx).to_pylist()
            new_task_arr = [task_mapping.get(t, t) for t in task_arr]
            new_col = pa.array(new_task_arr, type=schema.field(idx).type)
            table = table.set_column(idx, "task_index", new_col)

        pq.write_table(table, target_chunk / f"episode_{new_ep_idx:06d}.parquet")
        pq_count += 1

    return pq_count


def merge_video_files(
    current_dir: Path,
    merged_dir: Path,
    ep_offset: int,
    chunk_size: int,
) -> int:
    vid_count = 0
    video_dir = current_dir / "videos"
    if not video_dir.exists():
        return vid_count

    for chunk_dir in sorted(video_dir.glob("chunk-*"), key=natural_sort_key):
        for camera_dir in sorted(chunk_dir.iterdir(), key=natural_sort_key):
            if not camera_dir.is_dir():
                continue

            for vid_file in sorted(camera_dir.glob("episode_*.mp4"), key=natural_sort_key):
                try:
                    ep_idx = int(vid_file.stem.split("_")[1])
                except (ValueError, IndexError):
                    continue

                new_ep_idx = ep_idx + ep_offset
                new_chunk_idx = new_ep_idx // chunk_size
                target_camera_dir = (
                    merged_dir
                    / "videos"
                    / f"chunk-{new_chunk_idx:03d}"
                    / camera_dir.name
                )
                target_camera_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(vid_file, target_camera_dir / f"episode_{new_ep_idx:06d}.mp4")
                vid_count += 1

    return vid_count


def append_episodes(current_dir: Path, merged_dir: Path, ep_offset: int) -> int:
    ep_count = 0
    with open(current_dir / "meta" / "episodes.jsonl", "r", encoding="utf-8") as f_in:
        with open(merged_dir / "meta" / "episodes.jsonl", "a", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                ep_data = json.loads(line)
                ep_data["episode_index"] += ep_offset
                f_out.write(json.dumps(ep_data, ensure_ascii=False) + "\n")
                ep_count += 1
    return ep_count


def append_episode_stats(
    current_dir: Path,
    merged_dir: Path,
    ep_offset: int,
    frame_offset: int,
    task_mapping: dict[int, int],
) -> int:
    stats_file = current_dir / "meta" / "episodes_stats.jsonl"
    if not stats_file.exists():
        return 0

    stats_count = 0
    with open(stats_file, "r", encoding="utf-8") as f_in:
        with open(merged_dir / "meta" / "episodes_stats.jsonl", "a", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                stat_data = json.loads(line)
                stat_data["episode_index"] += ep_offset

                if "stats" in stat_data:
                    offset_stats_dict(stat_data["stats"], "episode_index", ep_offset)
                    offset_stats_dict(stat_data["stats"], "index", frame_offset)

                    if "task_index" in stat_data["stats"] and task_mapping:
                        old_task_idx = int(stat_data["stats"]["task_index"]["min"][0])
                        new_task_idx = task_mapping.get(old_task_idx, old_task_idx)
                        stat_data["stats"]["task_index"]["min"] = [new_task_idx]
                        stat_data["stats"]["task_index"]["max"] = [new_task_idx]
                        stat_data["stats"]["task_index"]["mean"] = [float(new_task_idx)]

                f_out.write(json.dumps(stat_data, ensure_ascii=False) + "\n")
                stats_count += 1

    return stats_count


def write_tasks(merged_dir: Path, tasks_vocab: dict[str, int]) -> None:
    with open(merged_dir / "meta" / "tasks.jsonl", "w", encoding="utf-8") as f:
        for task_str, idx in sorted(tasks_vocab.items(), key=lambda item: item[1]):
            f.write(
                json.dumps(
                    {"task_index": idx, "task": task_str},
                    ensure_ascii=False,
                )
                + "\n"
            )


def run_safety_audit(merged_dir: Path, tasks_vocab: dict[str, int]) -> bool:
    problems: list[str] = []

    info_path = merged_dir / "meta" / "info.json"
    episodes_path = merged_dir / "meta" / "episodes.jsonl"
    stats_path = merged_dir / "meta" / "episodes_stats.jsonl"

    if not info_path.exists():
        print("[fail] Missing meta/info.json")
        return False

    with open(info_path, "r", encoding="utf-8") as f:
        final_info = json.load(f)

    print("\n" + "=" * 60)
    print("  Post-merge audit")
    print("=" * 60)
    print(f"  total_episodes: {final_info.get('total_episodes', 0)}")
    print(f"  total_frames  : {final_info.get('total_frames', 0)}")
    print(f"  total_videos  : {final_info.get('total_videos', 0)}")
    print(f"  total_tasks   : {final_info.get('total_tasks', 0)}")

    expected_eps = final_info.get("total_episodes", 0)

    if episodes_path.exists():
        with open(episodes_path, "r", encoding="utf-8") as f:
            ep_lines = [line for line in f if line.strip()]
        if len(ep_lines) != expected_eps:
            problems.append(f"episodes.jsonl lines: got {len(ep_lines)}, expected {expected_eps}")
        else:
            print(f"[ok] episodes.jsonl lines: {len(ep_lines)}")
    else:
        problems.append("Missing meta/episodes.jsonl")

    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stat_lines = [line for line in f if line.strip()]
        if len(stat_lines) != expected_eps:
            problems.append(
                f"episodes_stats.jsonl lines: got {len(stat_lines)}, expected {expected_eps}"
            )
        else:
            print(f"[ok] episodes_stats.jsonl lines: {len(stat_lines)}")

        if stat_lines:
            first_stats = json.loads(stat_lines[0])
            last_stats = json.loads(stat_lines[-1])
            if first_stats.get("episode_index") != 0:
                problems.append(f"First episode_index is {first_stats.get('episode_index')}, not 0")
            if last_stats.get("episode_index") != expected_eps - 1:
                problems.append(
                    f"Last episode_index is {last_stats.get('episode_index')}, "
                    f"expected {expected_eps - 1}"
                )
            else:
                print(f"[ok] episode_index range: 0..{last_stats.get('episode_index')}")

            frame_index_min = last_stats.get("stats", {}).get("frame_index", {}).get("min", [0])[0]
            if frame_index_min != 0:
                problems.append(f"Last frame_index min is {frame_index_min}, not 0")
            else:
                print("[ok] frame_index still starts at 0 per episode")
    else:
        problems.append("Missing meta/episodes_stats.jsonl")

    pq_files = list((merged_dir / "data").rglob("episode_*.parquet"))
    if len(pq_files) != expected_eps:
        problems.append(f"Parquet files: got {len(pq_files)}, expected {expected_eps}")
    else:
        print(f"[ok] parquet files: {len(pq_files)}")

    expected_videos = final_info.get("total_videos", 0)
    vid_files = list((merged_dir / "videos").rglob("episode_*.mp4"))
    if expected_videos and len(vid_files) != expected_videos:
        problems.append(f"Video files: got {len(vid_files)}, expected {expected_videos}")
    else:
        print(f"[ok] video files: {len(vid_files)}")

    print("\n  Task vocab:")
    for task_str, idx in sorted(tasks_vocab.items(), key=lambda item: item[1]):
        print(f"    [{idx}] {task_str}")

    if problems:
        print("\n[fail] Problems found:")
        for problem in problems:
            print(f"  - {problem}")
        return False

    print("\n[ok] Audit passed.")
    return True


def merge_eval_so101(dataset_dirs: list[Path], output_dir: Path) -> bool:
    if output_dir in dataset_dirs:
        print(f"[error] Output dir cannot be one of the input dirs: {output_dir}")
        return False

    print("=" * 70)
    print("[merge] eval_so101 LeRobot datasets")
    print(f"[merge] input datasets: {len(dataset_dirs)}")
    print("=" * 70)

    for dataset_dir in dataset_dirs:
        if not is_lerobot_dataset_dir(dataset_dir):
            print(f"[error] Invalid dataset dir: {dataset_dir}")
            return False
        print(f"  [input] {dataset_dir}")

    if output_dir.exists():
        print(f"\n[merge] Removing existing output dir: {output_dir}")
        shutil.rmtree(output_dir)

    print(f"\n[merge] Copying base dataset: {dataset_dirs[0].name}")
    shutil.copytree(dataset_dirs[0], output_dir)
    print("[merge] Base copy complete")

    tasks_vocab = load_tasks(output_dir / "meta" / "tasks.jsonl")

    for ds_idx, current_dir in enumerate(dataset_dirs[1:], start=2):
        print("\n" + "=" * 60)
        print(f"[{ds_idx}/{len(dataset_dirs)}] Merge dataset: {current_dir.name}")
        print("=" * 60)

        task_mapping = build_task_mapping(current_dir / "meta" / "tasks.jsonl", tasks_vocab)

        with open(output_dir / "meta" / "info.json", "r", encoding="utf-8") as f:
            merged_info = json.load(f)
        with open(current_dir / "meta" / "info.json", "r", encoding="utf-8") as f:
            current_info = json.load(f)

        ep_offset = merged_info["total_episodes"]
        frame_offset = merged_info["total_frames"]
        chunk_size = merged_info.get("chunks_size", 1000)

        print(f"[offset] episode +{ep_offset}, frame +{frame_offset}, chunk_size={chunk_size}")

        pq_count = merge_parquet_files(
            current_dir=current_dir,
            merged_dir=output_dir,
            ep_offset=ep_offset,
            frame_offset=frame_offset,
            chunk_size=chunk_size,
            task_mapping=task_mapping,
        )
        print(f"[done] parquet files: {pq_count}")

        vid_count = merge_video_files(
            current_dir=current_dir,
            merged_dir=output_dir,
            ep_offset=ep_offset,
            chunk_size=chunk_size,
        )
        print(f"[done] video files: {vid_count}")

        ep_count = append_episodes(current_dir, output_dir, ep_offset)
        print(f"[done] episodes.jsonl rows: {ep_count}")

        stats_count = append_episode_stats(
            current_dir=current_dir,
            merged_dir=output_dir,
            ep_offset=ep_offset,
            frame_offset=frame_offset,
            task_mapping=task_mapping,
        )
        print(f"[done] episodes_stats.jsonl rows: {stats_count}")

        merged_info["total_episodes"] += current_info["total_episodes"]
        merged_info["total_frames"] += current_info["total_frames"]
        merged_info["total_videos"] = merged_info.get("total_videos", 0) + current_info.get(
            "total_videos", 0
        )
        merged_info["total_tasks"] = len(tasks_vocab)
        merged_info["total_chunks"] = (merged_info["total_episodes"] - 1) // chunk_size + 1

        if "splits" in merged_info and "train" in merged_info["splits"]:
            merged_info["splits"]["train"] = f"0:{merged_info['total_episodes']}"

        with open(output_dir / "meta" / "info.json", "w", encoding="utf-8") as f:
            json.dump(merged_info, f, indent=4, ensure_ascii=False)

    write_tasks(output_dir, tasks_vocab)
    ok = run_safety_audit(output_dir, tasks_vocab)

    print("\n" + "=" * 70)
    print("[merge] Finished")
    print(f"[output] {output_dir}")
    print("=" * 70)
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge eval_so101 LeRobot datasets with the same offsets as merge_multi.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python merge_eval_so101.py\n"
            "  python merge_eval_so101.py -o eval_so101_merged\n"
            "  python merge_eval_so101.py eval_so101_1 eval_so101_2 -o eval_so101_pair\n"
            "  python merge_eval_so101.py eval_so101 -b D:/HuggingFaceCache/lerobot/put_paper_cup_in_red_box\n"
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "Input dataset dirs, a parent dir, or a lazy prefix such as eval_so101. "
            "If omitted, the script scans eval_so101_*. "
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=MERGED_DATASET,
        help=f"Output dataset dir. Default: {MERGED_DATASET}",
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        default=str(BASE_DIR),
        help="Base dir for resolving relative inputs and output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()

    if not base_dir.exists():
        print(f"[error] base-dir does not exist: {base_dir}")
        sys.exit(1)

    if args.inputs:
        dataset_dirs: list[Path] = []
        unresolved_inputs: list[str] = []
        for token in args.inputs:
            discovered = resolve_input(token, base_dir)
            if not discovered:
                unresolved_inputs.append(token)
                continue
            for dataset_dir in discovered:
                if dataset_dir not in dataset_dirs:
                    dataset_dirs.append(dataset_dir)

        if unresolved_inputs:
            print("[error] Could not find datasets for:")
            for token in unresolved_inputs:
                print(f"  - {token}")
            sys.exit(1)
    else:
        dataset_dirs = discover_eval_datasets(base_dir)

    if len(dataset_dirs) < 2:
        print(f"[error] Need at least 2 datasets, found {len(dataset_dirs)}")
        for dataset_dir in dataset_dirs:
            print(f"  - {dataset_dir}")
        sys.exit(1)

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir = output_dir.resolve()

    print(f"\n[config] base-dir: {base_dir}")
    print(f"[config] output : {output_dir}")
    print(f"[config] datasets: {len(dataset_dirs)}")

    for idx, dataset_dir in enumerate(dataset_dirs, start=1):
        with open(dataset_dir / "meta" / "info.json", "r", encoding="utf-8") as f:
            info = json.load(f)
        print(
            f"  [{idx}] {dataset_dir.name}: "
            f"{info.get('total_episodes', 0)} eps / "
            f"{info.get('total_frames', 0)} frames / "
            f"{info.get('total_videos', 0)} videos"
        )

    ok = merge_eval_so101(dataset_dirs, output_dir)
    if not ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
