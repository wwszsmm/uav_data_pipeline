import os
import csv
import shutil
import hashlib
from pathlib import Path

from config import CUT_IMAGE_DIR, GATHER_IMAGE_DIR, GATHER_PREFIX, IMAGE_EXTS


def make_target_name(src_path, source_folder, base_name):
    """
    根据图片相对路径生成稳定短文件名。
    """
    src_path = Path(src_path)
    source_folder = Path(source_folder)

    relative_path = src_path.relative_to(source_folder)
    relative_str = str(relative_path).replace("\\", "/")

    hash_id = hashlib.md5(relative_str.encode("utf-8")).hexdigest()[:8]
    ext = src_path.suffix.lower()

    return f"{base_name}_{hash_id}{ext}", relative_str


def find_image_files(source_folder):
    source_folder = Path(source_folder)

    image_paths = []

    for root, _, files in os.walk(source_folder):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in IMAGE_EXTS:
                image_paths.append(Path(root) / file)

    return sorted(image_paths)


def write_mapping_csv(mapping_rows, mapping_path):
    """写入图片映射表，方便根据新文件名追溯原始图片"""
    with open(mapping_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "new_filename",
            "source_relative_path",
            "source_absolute_path"
        ])

        writer.writerows(mapping_rows)


def copy_and_rename_images(source_folder, target_folder, base_name):
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)

    target_folder.mkdir(parents=True, exist_ok=True)

    if not source_folder.exists():
        print(f"源图片文件夹不存在: {source_folder}")
        return

    image_paths = find_image_files(source_folder)
    total = len(image_paths)

    if total == 0:
        print(f"没有找到图片，请检查路径: {source_folder}")
        return

    print(f"找到 {total} 张图片")
    print(f"源文件夹: {source_folder}")
    print(f"目标文件夹: {target_folder}")

    copied = 0
    skipped = 0
    updated = 0
    failed = 0

    mapping_rows = []

    for i, src_path in enumerate(image_paths, start=1):
        target_name, source_relative_path = make_target_name(
            src_path,
            source_folder,
            base_name
        )

        dst_path = target_folder / target_name

        mapping_rows.append([
            target_name,
            source_relative_path,
            str(src_path)
        ])

        try:
            src_size = src_path.stat().st_size
        except Exception as e:
            failed += 1
            print(f"读取源图片失败: {src_path} | {e}")
            continue

        if dst_path.exists():
            try:
                dst_size = dst_path.stat().st_size
            except Exception as e:
                failed += 1
                print(f"读取目标图片失败: {dst_path} | {e}")
                continue

            if src_size == dst_size:
                skipped += 1
                continue

            try:
                shutil.copy2(src_path, dst_path)
                updated += 1
            except Exception as e:
                failed += 1
                print(f"覆盖图片失败: {src_path} -> {dst_path} | {e}")
                continue

        else:
            try:
                shutil.copy2(src_path, dst_path)
                copied += 1
            except Exception as e:
                failed += 1
                print(f"复制图片失败: {src_path} -> {dst_path} | {e}")
                continue

        if i % 100 == 0 or i == total:
            print(f"进度: {i}/{total}")

    mapping_path = target_folder / "image_mapping.csv"
    write_mapping_csv(mapping_rows, mapping_path)

    print("\n图片汇总完成")
    print(f"新增复制: {copied}")
    print(f"跳过已存在: {skipped}")
    print(f"覆盖更新: {updated}")
    print(f"失败: {failed}")
    print(f"目标文件夹: {target_folder}")
    print(f"映射表已保存: {mapping_path}")


def main():
    copy_and_rename_images(
        source_folder=CUT_IMAGE_DIR,
        target_folder=GATHER_IMAGE_DIR,
        base_name=GATHER_PREFIX
    )

if __name__ == "__main__":
    main()
