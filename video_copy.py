import os
import re
import shutil
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    VIDEO_SOURCE_DIR,
    VIDEO_COPY_DIR,
    VIDEO_EXTS,
    MAX_COPY_WORKERS
)


def sanitize_filename(name):
    """清理文件名中的非法字符"""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)


def make_target_name(src_path, src_root):
    """
    根据源视频的相对路径生成稳定的新文件名。
    """
    src_path = Path(src_path)
    src_root = Path(src_root)

    relative_path = src_path.relative_to(src_root)
    relative_str = str(relative_path).replace("\\", "/")

    hash_id = hashlib.md5(relative_str.encode("utf-8")).hexdigest()[:8]

    stem = sanitize_filename(src_path.stem)
    ext = src_path.suffix.lower()

    return f"{stem}_{hash_id}{ext}"


def copy_file_task(src_path, src_root, dst_dir):
    """
    单个视频复制任务。

    逻辑：
    1. 目标文件不存在：复制
    2. 目标文件已存在：直接跳过
    """
    src_path = Path(src_path)
    src_root = Path(src_root)
    dst_dir = Path(dst_dir)

    target_name = make_target_name(src_path, src_root)
    dst_path = dst_dir / target_name

    if dst_path.exists():
        return {
            "status": "skipped",
            "source": str(src_path),
            "target": str(dst_path),
            "message": "目标文件已存在，跳过"
        }

    try:
        shutil.copy2(src_path, dst_path)
        return {
            "status": "copied",
            "source": str(src_path),
            "target": str(dst_path),
            "message": "复制成功"
        }
    except Exception as e:
        return {
            "status": "failed",
            "source": str(src_path),
            "target": str(dst_path),
            "message": f"复制失败: {e}"
        }


def find_video_files(src_dir):
    src_dir = Path(src_dir)

    video_files = []

    for root, _, files in os.walk(src_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in VIDEO_EXTS:
                video_files.append(Path(root) / file)

    return sorted(video_files)


def main():
    src_dir = Path(VIDEO_SOURCE_DIR)
    dst_dir = Path(VIDEO_COPY_DIR)

    dst_dir.mkdir(parents=True, exist_ok=True)

    print("正在扫描视频文件...")
    print(f"源文件夹: {src_dir}")
    print(f"目标文件夹: {dst_dir}")

    if not src_dir.exists():
        print(f"源文件夹不存在: {src_dir}")
        return

    video_files = find_video_files(src_dir)
    total = len(video_files)

    if total == 0:
        print(f"没有找到视频文件，请检查路径: {src_dir}")
        return

    print(f"找到 {total} 个视频，开始复制...")

    copied = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_COPY_WORKERS) as executor:
        futures = [
            executor.submit(copy_file_task, src_path, src_dir, dst_dir)
            for src_path in video_files
        ]

        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            status = result["status"]

            if status == "copied":
                copied += 1
            elif status == "skipped":
                skipped += 1
            elif status == "failed":
                failed += 1
                print(f"失败: {result['source']} | {result['message']}")

            if i % 10 == 0 or i == total:
                print(f"进度: {i}/{total}")

    print("\n视频复制完成")
    print(f"新增复制: {copied}")
    print(f"跳过已存在: {skipped}")
    print(f"失败: {failed}")


if __name__ == "__main__":
    main()
