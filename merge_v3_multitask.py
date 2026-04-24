import os
import shutil
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

# ==========================================
# ！！！配置你的绝对路径！！！
# ==========================================
BASE_DIR = Path(r"C:\Users\86158\.cache\huggingface\lerobot")

DATASET_1 = "task2_right_random_20"
DATASET_2 = "task2_left_random_20"
MERGED_DATASET = "task2_random_40"


def offset_stats_dict(stats, key, offset):
    """专门用来平移 episodes_stats.jsonl 内部的 min/max/mean 嵌套数组"""
    if key in stats:
        for metric in ["min", "max", "mean"]:
            if metric in stats[key]:
                stats[key][metric] = [val + offset for val in stats[key][metric]]


def perfect_multitask_merge():
    dir1 = BASE_DIR / DATASET_1 / "demo"
    dir2 = BASE_DIR / DATASET_2 / "demo"
    merged_dir = BASE_DIR / MERGED_DATASET / "demo"

    print("=" * 60)
    print("🛡️ 正在启动 v2.1 【多任务动态映射】完美合并引擎...")
    if not dir1.exists() or not dir2.exists():
        print(f"❌ 找不到源文件夹！请检查 BASE_DIR: {BASE_DIR}")
        return

    # 1. 复制 Dataset_1 作为基础
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    shutil.copytree(dir1, merged_dir)
    print("✅ 基础数据 (Dataset 1) 拷贝完成。")

    # ========================================================
    # 【新增核心逻辑】：合并 tasks.jsonl 并建立映射表
    # ========================================================
    print("\n🧠 正在分析并合并多模态任务词典 (Tasks Vocabulary)...")
    tasks_vocab = {}

    # 读入数据集1的任务
    with open(dir1 / "meta" / "tasks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            tasks_vocab[t["task"]] = t["task_index"]

    # 读入数据集2的任务并建立映射
    task2_mapping = {}  # 格式: { 旧index : 新index }
    with open(dir2 / "meta" / "tasks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            old_idx = t["task_index"]
            task_str = t["task"]

            if task_str in tasks_vocab:
                # 任务一样，复用原来的 index
                task2_mapping[old_idx] = tasks_vocab[task_str]
                print(f"  👉[复用任务]: '{task_str}' (Index 保持为 {tasks_vocab[task_str]})")
            else:
                # 发现新任务，分配新的 index
                new_idx = len(tasks_vocab)
                tasks_vocab[task_str] = new_idx
                task2_mapping[old_idx] = new_idx
                print(f"  ✨ [发现新任务]: '{task_str}' (旧Index: {old_idx} -> 新Index: {new_idx})")

    # 将合并后的新任务表写入 merged_dir
    with open(merged_dir / "meta" / "tasks.jsonl", "w", encoding="utf-8") as f:
        # 按照 index 排序写入
        sorted_tasks = sorted(tasks_vocab.items(), key=lambda x: x[1])
        for task_str, idx in sorted_tasks:
            f.write(json.dumps({"task_index": idx, "task": task_str}) + "\n")
    # ========================================================

    # 读取统计信息，计算偏移量
    with open(merged_dir / "meta" / "info.json", "r", encoding="utf-8") as f:
        info1 = json.load(f)
    with open(dir2 / "meta" / "info.json", "r", encoding="utf-8") as f:
        info2 = json.load(f)

    ep_offset = info1["total_episodes"]
    frame_offset = info1["total_frames"]
    chunk_size = info1.get("chunks_size", 1000)

    print(f"\n📊 物理帧偏移量计算完成：从第 {ep_offset} 集、第 {frame_offset} 帧开始追加...")

    # 3. 处理 Parquet (无损追加 + Task Index 动态替换)
    data_dir2 = dir2 / "data"
    if data_dir2.exists():
        for chunk_dir in data_dir2.glob("chunk-*"):
            for pq_file in chunk_dir.glob("episode_*.parquet"):
                try:
                    ep_idx = int(pq_file.stem.split("_")[1])
                except ValueError:
                    continue

                new_ep_idx = ep_idx + ep_offset
                new_chunk_idx = new_ep_idx // chunk_size
                target_chunk = merged_dir / "data" / f"chunk-{new_chunk_idx:03d}"
                target_chunk.mkdir(parents=True, exist_ok=True)
                new_pq_name = f"episode_{new_ep_idx:06d}.parquet"

                table = pq.read_table(pq_file)
                schema = table.schema

                # 平移 episode_index
                if "episode_index" in table.column_names:
                    idx = schema.get_field_index("episode_index")
                    new_col = pc.add(table.column(idx), pa.scalar(ep_offset, type=schema.field(idx).type))
                    table = table.set_column(idx, "episode_index", new_col)

                # 平移 index (全局帧)
                if "index" in table.column_names:
                    idx = schema.get_field_index("index")
                    new_col = pc.add(table.column(idx), pa.scalar(frame_offset, type=schema.field(idx).type))
                    table = table.set_column(idx, "index", new_col)

                # 【核心修复】：动态映射 task_index
                if "task_index" in table.column_names:
                    idx = schema.get_field_index("task_index")
                    # 将列提取出来遍历映射，再安全打包回去，绝不破坏底层数据类型
                    task_arr = table.column(idx).to_pylist()
                    new_task_arr = [task2_mapping.get(t, t) for t in task_arr]
                    new_col = pa.array(new_task_arr, type=schema.field(idx).type)
                    table = table.set_column(idx, "task_index", new_col)

                pq.write_table(table, target_chunk / new_pq_name)

    # 4. 处理 Videos (严格路由到对应 Chunk 和摄像头)
    video_dir2 = dir2 / "videos"
    if video_dir2.exists():
        for chunk_dir in video_dir2.glob("chunk-*"):
            for cam_dir in chunk_dir.iterdir():
                if not cam_dir.is_dir(): continue
                for vid_file in cam_dir.glob("episode_*.mp4"):
                    ep_idx = int(vid_file.stem.split("_")[1])
                    new_ep_idx = ep_idx + ep_offset
                    new_chunk_idx = new_ep_idx // chunk_size

                    target_cam_dir = merged_dir / "videos" / f"chunk-{new_chunk_idx:03d}" / cam_dir.name
                    target_cam_dir.mkdir(parents=True, exist_ok=True)
                    new_vid_name = f"episode_{new_ep_idx:06d}.mp4"
                    shutil.copy2(vid_file, target_cam_dir / new_vid_name)

    # 5. 更新 episodes.jsonl (由于里面的 task 是字符串，只需平移剧集编号)
    with open(dir2 / "meta" / "episodes.jsonl", "r", encoding="utf-8") as f_in, \
            open(merged_dir / "meta" / "episodes.jsonl", "a", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip(): continue
            ep_data = json.loads(line)
            ep_data["episode_index"] += ep_offset
            f_out.write(json.dumps(ep_data) + "\n")

    # 6. 更新 episodes_stats.jsonl 的多重嵌套索引
    stats_file = dir2 / "meta" / "episodes_stats.jsonl"
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f_in, \
                open(merged_dir / "meta" / "episodes_stats.jsonl", "a", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip(): continue
                stat_data = json.loads(line)

                stat_data["episode_index"] += ep_offset
                if "stats" in stat_data:
                    offset_stats_dict(stat_data["stats"], "episode_index", ep_offset)
                    offset_stats_dict(stat_data["stats"], "index", frame_offset)

                    # 【核心修复】：替换统计表里的 task_index 映射
                    if "task_index" in stat_data["stats"]:
                        old_task_idx = int(stat_data["stats"]["task_index"]["min"][0])
                        new_task_idx = task2_mapping.get(old_task_idx, old_task_idx)
                        stat_data["stats"]["task_index"]["min"] = [new_task_idx]
                        stat_data["stats"]["task_index"]["max"] = [new_task_idx]
                        stat_data["stats"]["task_index"]["mean"] = [float(new_task_idx)]

                f_out.write(json.dumps(stat_data) + "\n")

    # 7. 重写 info.json 累加器和全局 Split 分段
    info1["total_episodes"] += info2["total_episodes"]
    info1["total_frames"] += info2["total_frames"]
    info1["total_videos"] = info1.get("total_videos", 0) + info2.get("total_videos", 0)

    # 【核心修复】：更新总任务数量
    info1["total_tasks"] = len(tasks_vocab)
    info1["total_chunks"] = (info1["total_episodes"] - 1) // chunk_size + 1

    if "splits" in info1 and "train" in info1["splits"]:
        info1["splits"]["train"] = f"0:{info1['total_episodes']}"

    with open(merged_dir / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(info1, f, indent=4)

    print("\n=====================================================")
    print(f"🎉 包含 Multi-Task 映射的绝对安全合并完成！")
    print(f"📈 最终总任务数：{info1['total_tasks']} 个")
    print(f"🎬 最终总集数：{info1['total_episodes']} 集")
    print(f"🎞️ 最终总帧数：{info1['total_frames']} 帧")
    print("\n✅ 现在请使用以下参数开始多任务联合训练：")
    print("--dataset.repo_id=merged_multitask_data/demo")
    print("=====================================================")


if __name__ == "__main__":
    perfect_multitask_merge()