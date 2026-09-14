import hashlib
import json
import os
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import cv2
from tqdm import tqdm

from pipeline_config import VIDEO_COPY_DIR, CUT_IMAGE_DIR, FRAME_INTERVAL, MAX_CUT_PROCESSES, USE_GPU, VIDEO_EXTS
from pipeline_utils import atomic_json, digest, finish_manifest, managed_files, manifest, owned_path, read_json


def extract_frames(video_path, output_folder, frame_interval, use_gpu):
    """Reuse only a complete snapshot of the same video, interval and decoder mode."""
    if type(frame_interval) is not int or frame_interval <= 0:
        raise ValueError("frame_interval must be a positive integer")
    video_path, output_folder = Path(video_path), Path(output_folder)
    identity = {"video": video_path.name, "sha256": digest(video_path),
                "interval": frame_interval, "opencv": cv2.__version__, "gpu_requested": use_gpu}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    snapshot = owned_path(output_folder, key)
    snapshot.mkdir(parents=True, exist_ok=True)
    marker = snapshot / "_done.json"
    try:
        done = read_json(marker)
    except (ValueError, OSError):
        done = None
    if done and done.get("identity") == identity and done.get("files"):
        if all(owned_path(snapshot, n).is_file() and digest(owned_path(snapshot, n)) == h
               for n, h in done["files"].items()):
            return {"video": video_path.name, "status": "skipped_done", "saved": 0,
                    "files": {f"{key}/{n}": h for n, h in done["files"].items()}}
    marker.unlink(missing_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gpu_mode = False
    if use_gpu:
        try:
            gpu_cap = cv2.cudacodec.createVideoReader(str(video_path))
            cap.release()
            cap, gpu_mode = gpu_cap, True
        except (AttributeError, cv2.error) as error:
            print(f"GPU decoding unavailable; using CPU: {error}")
    files = {}
    frame_idx = 0
    try:
        while True:
            if gpu_mode:
                ret, gpu_frame = cap.nextFrame()
                frame = gpu_frame.download() if ret else None
            else:
                ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                name = f"frame{frame_idx}.jpg"
                target = owned_path(snapshot, name)
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    raise RuntimeError(f"Cannot encode image: {target}")
                temporary = target.with_suffix(".jpg.tmp")
                try:
                    temporary.write_bytes(encoded.tobytes())
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
                files[name] = digest(target)
            frame_idx += 1
    finally:
        if hasattr(cap, "release"):
            cap.release()
    if not files or (expected_frames > 0 and frame_idx < expected_frames):
        raise RuntimeError(f"Incomplete video decoding: {video_path} ({frame_idx}/{expected_frames} frames)")
    if digest(video_path) != identity["sha256"]:
        raise RuntimeError(f"Video changed during extraction: {video_path}")
    atomic_json(marker, {"identity": identity, "files": files, "decoded_frames": frame_idx})
    return {"video": video_path.name, "status": "done", "saved": len(files),
            "files": {f"{key}/{n}": h for n, h in files.items()}}


def main():
    video_files = managed_files(VIDEO_COPY_DIR, VIDEO_EXTS)
    if not video_files:
        raise RuntimeError("No current videos; run video_copy.py first")
    previous = manifest(CUT_IMAGE_DIR)
    worker = partial(extract_frames, output_folder=CUT_IMAGE_DIR,
                     frame_interval=FRAME_INTERVAL, use_gpu=USE_GPU)
    files = {}
    with Pool(processes=MAX_CUT_PROCESSES) as pool:
        for result in tqdm(pool.imap(worker, video_files), total=len(video_files)):
            if files.keys() & result["files"].keys():
                raise RuntimeError("Frame snapshot name collision")
            files.update(result["files"])
            print(f"{result['video']}: {result['status']}, saved={result['saved']}")
    # Historical snapshots stay on disk; downstream stages read only this active list.
    finish_manifest(CUT_IMAGE_DIR, files, previous, prune=False)
    print(f"Current extracted images: {len(files)}")


if __name__ == "__main__":
    main()
