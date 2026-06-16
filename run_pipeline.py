import os
import sys
import subprocess
import time
from pathlib import Path

from config import (
    VIDEO_SOURCE_DIR,
    VIDEO_COPY_DIR,
    CUT_IMAGE_DIR,
    GATHER_IMAGE_DIR,
    VIDEO_EXTS,
    IMAGE_EXTS
)


BASE_DIR = Path(__file__).parent

SCRIPTS = [
    ("复制视频", "vedioCopy_v2.py"),
    ("视频抽帧", "vedio_cut_v2.py"),
    ("汇总图片", "pic_gather_v2.py"),
    ("相似图片去重", "SimilarPic_v2.py"),
]


def count_files(folder, exts):
    folder = Path(folder)

    if not folder.exists():
        return 0

    count = 0

    for root, _, files in os.walk(folder):
        for file in files:
            if Path(file).suffix.lower() in exts:
                count += 1

    return count


def print_status():
    print("\n当前数据状态:")
    print(f"原始视频数量: {count_files(VIDEO_SOURCE_DIR, VIDEO_EXTS)}")
    print(f"工作视频数量: {count_files(VIDEO_COPY_DIR, VIDEO_EXTS)}")
    print(f"抽帧图片数量: {count_files(CUT_IMAGE_DIR, IMAGE_EXTS)}")
    print(f"汇总图片数量: {count_files(GATHER_IMAGE_DIR, IMAGE_EXTS)}")


def run_script(step_name, script_name):
    script_path = BASE_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"找不到脚本: {script_path}")

    print("\n" + "=" * 60)
    print(f"开始执行: {step_name}")
    print("=" * 60)

    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR)
    )

    end = time.time()

    if result.returncode != 0:
        raise RuntimeError(f"{step_name} 执行失败，请检查 {script_name}")

    print(f"完成: {step_name}，耗时 {end - start:.1f} 秒")


def main():
    print("数据处理流水线启动")

    print_status()

    for step_name, script_name in SCRIPTS:
        run_script(step_name, script_name)
        print_status()

    print("\n" + "=" * 60)
    print("全部流程完成")
    print("=" * 60)

    print_status()


if __name__ == "__main__":
    main()