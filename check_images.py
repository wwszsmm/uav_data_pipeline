import csv
from pathlib import Path

import cv2

from config import DEDUP_IMAGE_DIR, OUTPUT_ROOT, IMAGE_EXTS


def check_images(folder):
    folder = Path(folder)
    invalid_images = []
    total = 0

    if not folder.exists():
        raise FileNotFoundError(f"图片目录不存在: {folder}")

    for image_path in folder.rglob("*"):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        total += 1

        try:
            if image_path.stat().st_size == 0:
                invalid_images.append((str(image_path), "empty_file"))
                continue

            image = cv2.imread(str(image_path))

            if image is None or image.size == 0:
                invalid_images.append((str(image_path), "unreadable_image"))

        except OSError as error:
            invalid_images.append((str(image_path), str(error)))

    return total, invalid_images


def save_invalid_images(invalid_images, output_path):
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "problem"])
        writer.writerows(invalid_images)


def main():
    total, invalid_images = check_images(DEDUP_IMAGE_DIR)

    report_path = Path(OUTPUT_ROOT) / "invalid_images.csv"
    save_invalid_images(invalid_images, report_path)

    print("\n图片质量检查完成")
    print(f"检查图片: {total}")
    print(f"异常图片: {len(invalid_images)}")
    print(f"正常图片: {total - len(invalid_images)}")
    print(f"异常记录: {report_path}")


if __name__ == "__main__":
    main()