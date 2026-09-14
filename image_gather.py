import csv
import hashlib
from pathlib import Path

from pipeline_config import CUT_IMAGE_DIR, GATHER_IMAGE_DIR, GATHER_PREFIX, IMAGE_EXTS
from pipeline_utils import managed_files, sync_files, unique_mapping


def make_target_name(src_path, source_folder, base_name):
    """
    根据图片相对路径生成稳定短文件名。
    """
    src_path = Path(src_path)
    source_folder = Path(source_folder)

    relative_path = src_path.relative_to(source_folder)
    relative_str = str(relative_path).replace("\\", "/")

    hash_id = hashlib.sha256(relative_str.encode("utf-8")).hexdigest()
    ext = src_path.suffix.lower()

    return f"{base_name}_{hash_id}{ext}", relative_str


def find_image_files(source_folder):
    return managed_files(source_folder, IMAGE_EXTS)


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
    source_folder, target_folder = Path(source_folder).resolve(), Path(target_folder).resolve()
    image_paths = find_image_files(source_folder)
    if not image_paths:
        raise RuntimeError(f"No current frames found: {source_folder}")
    rows = []
    pairs = []
    for source in image_paths:
        name, relative = make_target_name(source, source_folder, base_name)
        pairs.append((name, source))
        rows.append([name, relative, str(source)])
    targets = unique_mapping(pairs)
    copied, skipped = sync_files(targets, target_folder)
    write_mapping_csv(rows, target_folder / "image_mapping.csv")
    print(f"Image gathering complete: copied/updated={copied}, unchanged={skipped}")


def main():
    copy_and_rename_images(
        source_folder=CUT_IMAGE_DIR,
        target_folder=GATHER_IMAGE_DIR,
        base_name=GATHER_PREFIX
    )

if __name__ == "__main__":
    main()
