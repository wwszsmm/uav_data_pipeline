import os
import sys
import subprocess
import time
from pathlib import Path

from pipeline_utils import atomic_json, digest, read_json, MANIFEST
from pipeline_config import (
    OUTPUT_ROOT,
    DEDUP_IMAGE_DIR,
    config_path,
    FRAME_INTERVAL,
    MIN_SIMILARITY_THRESHOLD,
    VIDEO_SOURCE_DIR,
    VIDEO_COPY_DIR,
    CUT_IMAGE_DIR,
    GATHER_IMAGE_DIR,
    VIDEO_EXTS,
    IMAGE_EXTS
)


BASE_DIR = Path(__file__).parent

SCRIPTS = [
    ("复制视频", "video_copy.py"),
    ("视频抽帧", "frame_extraction.py"),
    ("汇总图片", "image_gather.py"),
    ("相似图片去重", "image_deduplication.py"),
    ("检查图片", "check_images.py"),
    ("生成报告", "generate_report.py"),
]


def count_files(folder, exts):
    folder = Path(folder)

    if not folder.exists():
        return 0

    record = read_json(folder / MANIFEST)
    if record is not None:
        return sum(Path(n).suffix.lower() in exts for n in record["files"])

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
        cwd=str(BASE_DIR),
        env={**os.environ, "UAV_CONFIG": str(config_path)}
    )

    end = time.time()

    if result.returncode != 0:
        raise RuntimeError(f"{step_name} 执行失败，请检查 {script_name}")

    print(f"完成: {step_name}，耗时 {end - start:.1f} 秒")


def main():
    print("数据处理流水线启动")
    status_path = Path(OUTPUT_ROOT) / "pipeline_status.json"
    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "running", "stages": [],
              "configuration": {"sha256": digest(config_path), "frame_interval": FRAME_INTERVAL,
                                "similarity_threshold": MIN_SIMILARITY_THRESHOLD}}
    atomic_json(status_path, status)
    print_status()
    for step_name, script_name in SCRIPTS[:-1]:
        entry = {"stage": script_name, "status": "running"}
        status["stages"].append(entry)
        atomic_json(status_path, status)
        try:
            if digest(config_path) != status["configuration"]["sha256"]:
                raise RuntimeError("Configuration changed during this run")
            run_script(step_name, script_name)
        except Exception as error:
            entry.update(status="failed", error=str(error))
            status["status"] = "failed"
            atomic_json(status_path, status)
            # Preserve a failure report even when later processing cannot proceed.
            run_script(*SCRIPTS[-1])
            raise
        entry["status"] = "passed"
        atomic_json(status_path, status)
        print_status()
    final_step = {"stage": "verify_outputs_and_report", "status": "running"}
    status["stages"].append(final_step)
    atomic_json(status_path, status)
    try:
        from generate_report import state_fingerprint
        status["state_sha256"] = state_fingerprint()
        rejected = read_json(Path(DEDUP_IMAGE_DIR) / "rejected_images.json", [])
        status["status"] = "complete_with_exclusions" if rejected else "complete"
        final_step["status"] = "passed"
        atomic_json(status_path, status)
        run_script(*SCRIPTS[-1])
    except Exception as error:
        status["status"] = "failed"
        final_step.update(status="failed", error=str(error))
        atomic_json(status_path, status)
        try:
            run_script(*SCRIPTS[-1])
        except Exception as report_error:
            print(f"Failure report could not be regenerated: {report_error}")
        raise

    print("\n" + "=" * 60)
    print(f"全部流程完成: {status['status']}")
    print("=" * 60)

    print_status()


if __name__ == "__main__":
    main()
