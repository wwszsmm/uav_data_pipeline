from datetime import datetime
from pathlib import Path

from config import (
    OUTPUT_ROOT,
    VIDEO_SOURCE_DIR,
    VIDEO_COPY_DIR,
    CUT_IMAGE_DIR,
    GATHER_IMAGE_DIR,
    DEDUP_IMAGE_DIR,
    VIDEO_EXTS,
    IMAGE_EXTS,
    FRAME_INTERVAL,
    MIN_SIMILARITY_THRESHOLD,
)


def count_files(folder, extensions):
    folder = Path(folder)

    if not folder.exists():
        return 0

    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def count_duplicate_images():
    csv_path = Path(DEDUP_IMAGE_DIR) / "duplicate_pairs.csv"

    if not csv_path.exists():
        return 0

    with open(csv_path, "r", encoding="utf-8-sig") as file:
        return max(sum(1 for _ in file) - 1, 0)


def main():
    original_videos = count_files(VIDEO_SOURCE_DIR, VIDEO_EXTS)
    copied_videos = count_files(VIDEO_COPY_DIR, VIDEO_EXTS)
    cut_images = count_files(CUT_IMAGE_DIR, IMAGE_EXTS)
    gathered_images = count_files(GATHER_IMAGE_DIR, IMAGE_EXTS)
    deduped_images = count_files(DEDUP_IMAGE_DIR, IMAGE_EXTS)
    duplicate_images = count_duplicate_images()

    duplicate_rate = (
        duplicate_images / gathered_images * 100
        if gathered_images > 0
        else 0
    )

    report = f"""# UAV Data Pipeline Report

Generated: {datetime.now():%Y-%m-%d %H:%M:%S}

## Configuration

- Frame interval: {FRAME_INTERVAL}
- Similarity threshold: {MIN_SIMILARITY_THRESHOLD}

## Results

| Item | Count |
|---|---:|
| Original videos | {original_videos} |
| Copied videos | {copied_videos} |
| Extracted images | {cut_images} |
| Gathered images | {gathered_images} |
| Duplicate images | {duplicate_images} |
| Deduplicated images | {deduped_images} |
| Duplicate rate | {duplicate_rate:.2f}% |
"""

    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    report_path = output_root / "pipeline_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("\n统计报告生成完成")
    print(f"原始视频: {original_videos}")
    print(f"抽帧图片: {cut_images}")
    print(f"汇总图片: {gathered_images}")
    print(f"重复图片: {duplicate_images}")
    print(f"去重后图片: {deduped_images}")
    print(f"重复比例: {duplicate_rate:.2f}%")
    print(f"报告路径: {report_path}")


if __name__ == "__main__":
    main()