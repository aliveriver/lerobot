import os
import shutil
import json
from pathlib import Path
import pyarrow as pa  # 【修复点1】引入主 pyarrow 模块
import pyarrow.parquet as pq
import pyarrow.compute as pc

# ==========================================
# ！！！配置你的绝对路径！！！
# 根据你的终端输出，用户名是 vipuser，请核对：
# ==========================================
BASE_DIR = Path(r"C:\Users\86158\.cache\huggingface\lerobot")

DATASET_1 = "task1_fixed"
DATASET_2 = "task1_random"
MERGED_DATASET = "task1_data"


def offset_stats_dict(stats, key, offset):
    """专门用来平移 episodes_stats.jsonl 内部的 min/max/mean 嵌套数组"""
    if key in stats:
        for metric in ["min", "max", "mean"]:
            if metric in stats[key]:
                stats[key][metric] = [val + offset for val in stats[key][metric]]


def perfect_merge():
    dir1 = BASE_DIR / DATASET_1 / "demo"
    dir2 = BASE_DIR / DATASET_2 / "demo"
    merged_dir = BASE_DIR / MERGED_DATASET / "demo"

    print("🛡️ 正在启动 v2.1 完美合并引擎...")
    if not dir1.exists() or not dir2.exists():
        print(f"❌ 找不到源文件夹！请检查 BASE_DIR: {BASE_DIR}")
        return

    # 1. 复制 Test4 作为基础
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    shutil.copytree(dir1, merged_dir)
    print("✅ 基础数据 (Test4) 拷贝完成。")

    # 2. 读取统计信息，计算偏移量
    with open(merged_dir / "meta" / "info.json", "r", encoding="utf-8") as f:
        info1 = json.load(f)
    with open(dir2 / "meta" / "info.json", "r", encoding="utf-8") as f:
        info2 = json.load(f)

    ep_offset = info1["total_episodes"]
    frame_offset = info1["total_frames"]
    chunk_size = info1.get("chunks_size", 1000)

    print(f"📊 偏移量计算完成：从第 {ep_offset} 集、第 {frame_offset} 帧开始追加...")

    # 3. 处理 Parquet (无损追加)
    data_dir2 = dir2 / "data"
    if data_dir2.exists():
        for chunk_dir in data_dir2.glob("chunk-*"):
            for pq_file in chunk_dir.glob("episode_*.parquet"):
                try:
                    ep_idx = int(pq_file.stem.split("_")[1])
                except ValueError:
                    continue

                new_ep_idx = ep_idx + ep_offset
                new_chunk_idx = new_ep_idx // chunk_size  # 严格遵循 1000集分块原则
                target_chunk = merged_dir / "data" / f"chunk-{new_chunk_idx:03d}"
                target_chunk.mkdir(parents=True, exist_ok=True)
                new_pq_name = f"episode_{new_ep_idx:06d}.parquet"

                table = pq.read_table(pq_file)
                schema = table.schema

                # 【修复点2】使用 pa.scalar 替换 pc.scalar
                if "episode_index" in table.column_names:
                    idx = schema.get_field_index("episode_index")
                    new_col = pc.add(table.column(idx), pa.scalar(ep_offset, type=schema.field(idx).type))
                    table = table.set_column(idx, "episode_index", new_col)

                if "index" in table.column_names:
                    idx = schema.get_field_index("index")
                    new_col = pc.add(table.column(idx), pa.scalar(frame_offset, type=schema.field(idx).type))
                    table = table.set_column(idx, "index", new_col)

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

    # 5. 更新 episodes.jsonl (平移外层剧集索引)
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

                f_out.write(json.dumps(stat_data) + "\n")

    # 7. 重写 info.json 累加器和全局 Split 分段
    info1["total_episodes"] += info2["total_episodes"]
    info1["total_frames"] += info2["total_frames"]
    info1["total_videos"] = info1.get("total_videos", 0) + info2.get("total_videos", 0)

    # 【修复点3】动态更新总分块数量 (total_chunks)，让框架知道是否产生了新的 chunk
    info1["total_chunks"] = (info1["total_episodes"] - 1) // chunk_size + 1

    # 动态修复拆分，保证模型获取整个完整数据段
    if "splits" in info1 and "train" in info1["splits"]:
        info1["splits"]["train"] = f"0:{info1['total_episodes']}"

    with open(merged_dir / "meta" / "info.json", "w", encoding="utf-8") as f:
        json.dump(info1, f, indent=4)

    print("=====================================================")
    print(f"🎉 绝对安全合并完成！\n总集数：{info1['total_episodes']}\n总帧数：{info1['total_frames']}")
    print(f"更新后的分块数量(total_chunks)：{info1['total_chunks']}")
    print("你的 test4 和 test5 源文件已完好无损地保留。")
    print("\n✅ 现在请使用以下参数开始炼丹：")
    print("--dataset.repo_id=merged_data/demo")
    print("=====================================================")


if __name__ == "__main__":
    perfect_merge()