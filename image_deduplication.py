import csv
import json
import os
import tempfile
import zipfile
import zlib
from importlib.metadata import version
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import faiss
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from pipeline_config import (GATHER_IMAGE_DIR, DEDUP_IMAGE_DIR, FEATURE_FILE,
                             MIN_SIMILARITY_THRESHOLD, BATCH_SIZE, MAX_THREADS, IMAGE_EXTS)
from pipeline_utils import atomic_json, digest, managed_files, manifest, sync_files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_images(image_dir):
    return managed_files(image_dir, IMAGE_EXTS)


def load_image(img_path, preprocess):
    try:
        with Image.open(img_path) as img:
            tensor = preprocess(img.convert("RGB"))
        return img_path.name, tensor
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def feature_identity(image_files):
    return {
        "schema": 2,
        "images": {p.name: digest(p) for p in image_files},
        "model": "mobilenet_v3_small/IMAGENET1K_V1",
        "preprocess": "encoder.transform",
        "versions": {p: version(p) for p in ("imagededup", "torch", "torchvision", "Pillow")},
    }


def load_feature_cache(feature_file, identity):
    try:
        with np.load(feature_file, allow_pickle=False) as data:
            if json.loads(str(data["metadata"].item())) != identity:
                return None
            names, features = data["names"].tolist(), data["features"]
            if (not isinstance(names, list) or len(names) != len(features) or features.ndim != 2
                    or not all(isinstance(n, str) for n in names)
                    or len(set(names)) != len(names) or not set(names).issubset(identity["images"])
                    or not np.isfinite(features).all()
                    or (len(features) and (np.linalg.norm(features, axis=1) == 0).any())):
                return None
            return dict(zip(names, features.copy()))
    except (OSError, ValueError, KeyError, EOFError, TypeError, zipfile.BadZipFile, zlib.error):
        return None


def save_feature_cache(feature_file, encodings, identity):
    feature_file = Path(feature_file)
    feature_file.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=feature_file.parent, suffix=".npz")
    names = sorted(encodings)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(identity, sort_keys=True),
                                names=np.asarray(names, dtype=str),
                                features=np.vstack([encodings[n] for n in names]) if names else np.empty((0, 0)))
        os.replace(temporary, feature_file)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_features(image_files, feature_file):
    identity = feature_identity(image_files)
    cached = load_feature_cache(feature_file, identity)
    if cached is not None:
        print(f"Loaded verified feature cache: {len(cached)} images")
        return cached
    # Construct only after a cache miss. First construction may download public ImageNet weights.
    from imagededup.methods.cnn import CNN
    encoder = CNN()
    encoder.model.to(device)
    encoder.model.eval()
    encodings = {}
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for start in range(0, len(image_files), BATCH_SIZE):
            results = executor.map(lambda p: load_image(p, encoder.transform),
                                   image_files[start:start + BATCH_SIZE])
            results = [r for r in results if r is not None]
            if not results:
                continue
            names, tensors = zip(*results)
            with torch.inference_mode():
                features = encoder.model(torch.stack(tensors).to(device)).cpu().numpy()
            for name, feature in zip(names, features):
                if not np.isfinite(feature).all() or np.linalg.norm(feature) == 0:
                    raise RuntimeError(f"Invalid model feature: {name}")
                encodings[name] = feature
    if feature_identity(image_files) != identity:
        raise RuntimeError("Images changed while encoding; retry with stable input")
    save_feature_cache(feature_file, encodings, identity)
    return encodings


def build_index(encodings):
    names = sorted(encodings)
    features = np.vstack([encodings[name] for name in names]).astype("float32")
    if not np.isfinite(features).all() or (np.linalg.norm(features, axis=1) == 0).any():
        raise ValueError("Features must be finite and non-zero")
    faiss.normalize_L2(features)
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    return index, features, names


def find_duplicates(index, features, names, threshold):
    """Greedy representatives in filename order; every match points to a retained image."""
    pairs, duplicates = [], set()
    for idx, name in enumerate(names):
        if name in duplicates:
            continue
        # Expand top-k only while the tail may hide further qualifying neighbors.
        k = min(20, len(names))
        while k:
            scores, neighbors = index.search(features[idx:idx + 1], k)
            if k == len(names) or scores[0][-1] < threshold:
                break
            k = min(k * 2, len(names))
        if not k:
            continue
        for score, neighbor in sorted(zip(scores[0], neighbors[0]), key=lambda item: int(item[1])):
            if neighbor <= idx or score < threshold:
                continue
            duplicate = names[neighbor]
            if duplicate not in duplicates:
                duplicates.add(duplicate)
                pairs.append((name, duplicate, round(float(score), 6)))
    return pairs, duplicates


def copy_unique_images(image_files, output_dir, duplicate_names):
    unique = {p.name: p for p in image_files if p.name not in duplicate_names}
    previous = manifest(output_dir)
    removed = len(set(previous["files"]) - set(unique))
    copied, skipped = sync_files(unique, output_dir)
    return copied, skipped, removed, 0


def save_duplicate_csv(output_dir, duplicate_pairs):
    csv_path = Path(output_dir) / "duplicate_pairs.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["original", "duplicate", "similarity"])
        writer.writerows(duplicate_pairs)
    return csv_path


def main():
    image_files = find_images(GATHER_IMAGE_DIR)
    if not image_files:
        raise RuntimeError("No current gathered images; run image_gather.py first")
    manifest(DEDUP_IMAGE_DIR)
    encodings = build_features(image_files, FEATURE_FILE)
    rejected = sorted(p.name for p in image_files if p.name not in encodings)
    atomic_json(Path(DEDUP_IMAGE_DIR) / "rejected_images.json", rejected)
    if not encodings:
        sync_files({}, DEDUP_IMAGE_DIR)
        raise RuntimeError("No readable images could be encoded")
    index, features, names = build_index(encodings)
    pairs, duplicates = find_duplicates(index, features, names, MIN_SIMILARITY_THRESHOLD)
    # Unreadable images never enter the cleaned output.
    valid_files = [p for p in image_files if p.name in encodings]
    copied, skipped, removed, failed = copy_unique_images(valid_files, DEDUP_IMAGE_DIR, duplicates)
    save_duplicate_csv(DEDUP_IMAGE_DIR, pairs)
    stats = {"input": len(image_files), "valid_input": len(encodings), "invalid_input": len(rejected),
             "duplicates": len(duplicates), "valid_output": len(encodings) - len(duplicates),
             "copy_failures": failed, "threshold": MIN_SIMILARITY_THRESHOLD}
    atomic_json(Path(DEDUP_IMAGE_DIR) / "dedup_summary.json", stats)
    print(f"Deduplication: {stats}; copied/updated={copied}, unchanged={skipped}, removed_owned={removed}")


if __name__ == "__main__":
    main()
