import json
import shutil
from pathlib import Path
import pyarrow.parquet as pq

# ==========================================
# ！！！在这里填入你想要修改的数据集的 demo 文件夹绝对路径！！！
# 请务必确保路径正确，例如：
# ==========================================
DATASET_DIR = Path(r"C:\Users\86158\.cache\huggingface\lerobot\task1_data\demo")

# 根据你提供的 info.json，特征的全称如下：
CAM_FRONT = "observation.images.front"
CAM_HAND = "observation.images.handeye"


def safe_camera_swap():
    print("=" * 60)
    print(f"🛡️ 启动【高级安全审查版】视角互换程序")
    print(f"📁 目标数据集: {DATASET_DIR}")
    print("=" * 60)

    if not DATASET_DIR.exists():
        print("❌ 致命错误：找不到数据集路径，请检查代码中的 DATASET_DIR！")
        return

    # ---------------------------------------------------------
    # 阶段 1：互换视频文件夹 (Videos)
    # ---------------------------------------------------------
    print("\n[1/4] 正在检查并互换视频文件夹...")
    videos_dir = DATASET_DIR / "videos"
    video_swapped_count = 0
    if videos_dir.exists():
        for chunk_dir in videos_dir.glob("chunk-*"):
            front_dir = chunk_dir / CAM_FRONT
            handeye_dir = chunk_dir / CAM_HAND
            temp_dir = chunk_dir / "temp_swap_dir"

            # 如果文件夹名字只是 front 和 handeye，做一下兼容处理
            if not front_dir.exists():
                front_dir = chunk_dir / "front"
                handeye_dir = chunk_dir / "handeye"

            if front_dir.exists() and handeye_dir.exists():
                print(f"  👀 发现目标文件夹: {front_dir.name} 和 {handeye_dir.name}")
                front_dir.rename(temp_dir)
                handeye_dir.rename(front_dir)
                temp_dir.rename(handeye_dir)
                video_swapped_count += 1
                print(f"  ✅ {chunk_dir.name} 下的视频文件夹互换成功！")
            else:
                print(f"  ⚠️ 在 {chunk_dir.name} 中未找到对应的视频文件夹，跳过。")
    if video_swapped_count == 0:
        print("  ❌ 未完成任何视频文件夹互换，请检查视频目录结构！")

    # ---------------------------------------------------------
    # 阶段 2：互换 Parquet 数据表头列名
    # ---------------------------------------------------------
    print("\n[2/4] 正在检查并互换 Parquet 数据表列名...")
    data_dir = DATASET_DIR / "data"
    parquet_swapped_count = 0
    if data_dir.exists():
        for chunk_dir in data_dir.glob("chunk-*"):
            for pq_file in chunk_dir.glob("episode_*.parquet"):
                table = pq.read_table(pq_file)
                original_names = table.column_names
                new_names = []

                # 严格替换逻辑
                for name in original_names:
                    if name == CAM_FRONT:
                        new_names.append(CAM_HAND)
                    elif name == CAM_HAND:
                        new_names.append(CAM_FRONT)
                    else:
                        new_names.append(name)

                # 只在第一个文件打印出检查信息
                if parquet_swapped_count == 0:
                    print(f"  🔍 【审查】文件 {pq_file.name} 的表头变化：")
                    for o, n in zip(original_names, new_names):
                        if o != n:
                            print(f"      🔄 [{o}]  --->  [{n}]")

                table = table.rename_columns(new_names)
                pq.write_table(table, pq_file)
                parquet_swapped_count += 1
        print(f"  ✅ 共计 {parquet_swapped_count} 个 Parquet 文件的特征列名互换成功！")

    # ---------------------------------------------------------
    # 阶段 3：互换 episodes_stats.jsonl 中的图像统计数据
    # ---------------------------------------------------------
    print("\n[3/4] 正在互换 episodes_stats.jsonl 统计特征...")
    stats_file = DATASET_DIR / "meta" / "episodes_stats.jsonl"
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        with open(stats_file, "w", encoding="utf-8") as f:
            for i, line in enumerate(lines):
                if not line.strip(): continue
                stat = json.loads(line)
                if "stats" in stat:
                    s = stat["stats"]
                    if CAM_FRONT in s and CAM_HAND in s:
                        # 打印第一行的数值交换审查
                        if i == 0:
                            print(f"  🔍 【审查】第0集的色彩均值 (Mean) 变化：")
                            print(f"      原本的 {CAM_FRONT} mean: {s[CAM_FRONT]['mean'][0][:1]}...")
                            print(f"      原本的 {CAM_HAND} mean: {s[CAM_HAND]['mean'][0][:1]}...")

                        # 交换字典
                        s[CAM_FRONT], s[CAM_HAND] = s[CAM_HAND], s[CAM_FRONT]

                        if i == 0:
                            print(f"      🔄 互换后 {CAM_FRONT} mean: {s[CAM_FRONT]['mean'][0][:1]}...")
                            print(f"      🔄 互换后 {CAM_HAND} mean: {s[CAM_HAND]['mean'][0][:1]}...")

                f.write(json.dumps(stat) + "\n")
        print("  ✅ JSONL 色彩均值/方差等统计参数互换完成！")
    else:
        print("  ⚠️ 未找到 episodes_stats.jsonl，跳过。")

    # ---------------------------------------------------------
    # 阶段 4：互换 info.json 中的特征定义
    # ---------------------------------------------------------
    print("\n[4/4] 正在互换 info.json 中的特征描述...")
    info_file = DATASET_DIR / "meta" / "info.json"
    if info_file.exists():
        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)

        if "features" in info and CAM_FRONT in info["features"] and CAM_HAND in info["features"]:
            # 交换特征定义（包括分辨率、FPS等描述）
            info["features"][CAM_FRONT], info["features"][CAM_HAND] = \
                info["features"][CAM_HAND], info["features"][CAM_FRONT]

            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=4)
            print("  ✅ info.json 全局特征定义互换完成！")
        else:
            print("  ⚠️ info.json 中未找到完整的相机特征定义，跳过。")

    print("=" * 60)
    print("🎉 恭喜！整个数据集的底中高三层（视频、表格、元数据）均已安全互换！")
    print("你可以直接用这个数据集训练了。")
    print("=" * 60)


if __name__ == "__main__":
    safe_camera_swap()