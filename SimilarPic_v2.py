import csv
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
import numpy as np
import torch
from imagededup.methods.cnn import CNN
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from tqdm import tqdm

from config import (
    GATHER_IMAGE_DIR,
    DEDUP_IMAGE_DIR,
    FEATURE_FILE,
    MIN_SIMILARITY_THRESHOLD,
    BATCH_SIZE,
    MAX_THREADS,
    IMAGE_EXTS,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def find_images(image_dir):
    image_dir = Path(image_dir)

    if not image_dir.exists():
        return []

    return sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def load_image(img_path):
    try:
        with Image.open(img_path) as img:
            tensor = preprocess(img.convert("RGB"))
        return img_path.name, tensor
    except (UnidentifiedImageError, OSError):
        return None


def load_feature_cache(feature_file):
    feature_file = Path(feature_file)

    if not feature_file.exists():
        return None

    with np.load(feature_file) as data:
        return {name: data[name].copy() for name in data.files}


def save_feature_cache(feature_file, encodings):
    feature_file = Path(feature_file)
    feature_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_file, **encodings)


def build_features(image_files, feature_file):
    """
    有缓存就加载；缓存和当前图片名不一致就重建。
    """
    current_names = {p.name for p in image_files}

    cached = load_feature_cache(feature_file)
    if cached is not None and set(cached.keys()) == current_names:
        print(f"加载已缓存特征，共 {len(cached)} 张图片")
        return cached

    if cached is not None:
        print("特征缓存和当前图片不一致，重新生成特征...")

    print("开始生成图片特征...")

    encoder = CNN()
    encoder.model.to(device)
    encoder.model.eval()

    encodings = {}

    for start in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Encoding"):
        batch_files = image_files[start:start + BATCH_SIZE]

        tensors = []
        names = []

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = [executor.submit(load_image, p) for p in batch_files]

            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue

                name, tensor = result
                names.append(name)
                tensors.append(tensor)

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            features = encoder.model(batch_tensor).cpu().numpy()

        for name, feature in zip(names, features):
            encodings[name] = feature

    save_feature_cache(feature_file, encodings)
    print(f"特征生成完成，共 {len(encodings)} 张图片")
    print(f"特征缓存已保存: {feature_file}")

    return encodings


def build_index(encodings):
    names = sorted(encodings.keys())

    features = np.vstack([encodings[name] for name in names]).astype("float32")

    faiss.normalize_L2(features)

    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)

    return index, features, names


def find_duplicates(index, features, names, threshold):
    if len(names) < 2:
        return [], set()

    k = min(20, len(names))

    scores, indices = index.search(features, k)

    duplicate_pairs = []
    duplicate_names = set()

    for idx, name in enumerate(names):
        if name in duplicate_names:
            continue

        for score, neighbor_idx in zip(scores[idx][1:], indices[idx][1:]):
            if neighbor_idx < 0:
                continue

            if score < threshold:
                break

            duplicate_name = names[neighbor_idx]

            if duplicate_name == name or duplicate_name in duplicate_names:
                continue

            duplicate_pairs.append((
                name,
                duplicate_name,
                round(float(score), 4)
            ))

            duplicate_names.add(duplicate_name)

    return duplicate_pairs, duplicate_names


def copy_unique_images(image_files, output_dir, duplicate_names):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_files = [
        p for p in image_files
        if p.name not in duplicate_names
    ]

    unique_names = {p.name for p in unique_files}

    removed = 0
    copied = 0
    skipped = 0
    failed = 0

    # 删除 dedup_images 里已经不属于当前唯一结果的旧图片
    for old_file in output_dir.iterdir():
        if old_file.is_file() and old_file.suffix.lower() in IMAGE_EXTS:
            if old_file.name not in unique_names:
                old_file.unlink()
                removed += 1

    for src_path in tqdm(unique_files, desc="Copy unique"):
        dst_path = output_dir / src_path.name

        if dst_path.exists():
            skipped += 1
            continue

        try:
            shutil.copy2(src_path, dst_path)
            copied += 1
        except Exception as e:
            failed += 1
            print(f"复制失败: {src_path} -> {dst_path} | {e}")

    return copied, skipped, removed, failed


def save_duplicate_csv(output_dir, duplicate_pairs):
    csv_path = Path(output_dir) / "duplicate_pairs.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["original", "duplicate", "similarity"])
        writer.writerows(duplicate_pairs)

    return csv_path


def main():
    image_dir = Path(GATHER_IMAGE_DIR)
    output_dir = Path(DEDUP_IMAGE_DIR)
    feature_file = Path(FEATURE_FILE)

    print("开始相似图片去重")
    print(f"输入图片文件夹: {image_dir}")
    print(f"输出图片文件夹: {output_dir}")
    print(f"特征缓存文件: {feature_file}")
    print(f"相似度阈值: {MIN_SIMILARITY_THRESHOLD}")
    print(f"运行设备: {device}")

    image_files = find_images(image_dir)

    if not image_files:
        print(f"没有找到图片，请检查输入路径: {image_dir}")
        return

    print(f"找到 {len(image_files)} 张图片")

    encodings = build_features(image_files, feature_file)

    if not encodings:
        print("没有成功提取到图片特征，去重终止")
        return

    index, features, names = build_index(encodings)

    duplicate_pairs, duplicate_names = find_duplicates(
        index=index,
        features=features,
        names=names,
        threshold=MIN_SIMILARITY_THRESHOLD
    )

    copied, skipped, removed, failed = copy_unique_images(
        image_files=image_files,
        output_dir=output_dir,
        duplicate_names=duplicate_names
    )

    csv_path = save_duplicate_csv(output_dir, duplicate_pairs)

    print("\n相似图片去重完成")
    print(f"原始图片数量: {len(image_files)}")
    print(f"重复图片数量: {len(duplicate_names)}")
    print(f"唯一图片数量: {len(image_files) - len(duplicate_names)}")
    print(f"本次新增复制: {copied}")
    print(f"已存在跳过: {skipped}")
    print(f"删除旧结果: {removed}")
    print(f"复制失败: {failed}")
    print(f"唯一图片输出文件夹: {output_dir}")
    print(f"重复图片 CSV: {csv_path}")


if __name__ == "__main__":
    main()