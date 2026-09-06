import os
import re
import cv2
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from config import (
    VIDEO_COPY_DIR,
    CUT_IMAGE_DIR,
    FRAME_INTERVAL,
    MAX_CUT_PROCESSES,
    USE_GPU,
    VIDEO_EXTS
)


def sanitize_filename(name):
    """清理文件名中的非法字符"""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)


def extract_frames(video_path, output_folder, frame_interval, use_gpu):
    """
    从单个视频中抽帧。
    """
    video_path = Path(video_path)
    output_folder = Path(output_folder)

    video_basename = video_path.name
    video_name = sanitize_filename(video_path.stem)

    video_output_folder = output_folder / video_name
    video_output_folder.mkdir(parents=True, exist_ok=True)

    done_marker = video_output_folder / f"_DONE_interval_{frame_interval}.txt"

    if done_marker.exists():
        return {
            "video": video_basename,
            "status": "skipped_done",
            "saved": 0,
            "skipped": 0,
            "message": "已完成，跳过"
        }

    cap = None
    gpu_mode = False

    if use_gpu:
        try:
            cap = cv2.cudacodec.createVideoReader(str(video_path))
            gpu_mode = True
        except Exception as e:
            print(f"GPU 解码失败（{video_basename}），回退到 CPU: {e}")
            cap = None

    if cap is None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                "video": video_basename,
                "status": "failed",
                "saved": 0,
                "skipped": 0,
                "message": "无法打开视频"
            }

    frame_idx = 0
    saved_count = 0
    skipped_count = 0

    while True:
        if gpu_mode:
            ret, frame_gpu = cap.nextFrame()
            if not ret:
                break
            frame = frame_gpu.download()
        else:
            ret, frame = cap.read()
            if not ret:
                break

        if frame_idx % frame_interval == 0:
            img_name = f"{video_name}_frame{frame_idx}.jpg"
            img_path = video_output_folder / img_name

            if img_path.exists():
                skipped_count += 1
            else:
                ok = cv2.imwrite(str(img_path), frame)
                if ok:
                    saved_count += 1
                else:
                    print(f"保存图片失败: {img_path}")

        frame_idx += 1

    if not gpu_mode:
        cap.release()

    with open(done_marker, "w", encoding="utf-8") as f:
        f.write(f"video: {video_basename}\n")
        f.write(f"frame_interval: {frame_interval}\n")
        f.write(f"saved_count: {saved_count}\n")
        f.write(f"skipped_count: {skipped_count}\n")
        f.write("status: done\n")

    return {
        "video": video_basename,
        "status": "done",
        "saved": saved_count,
        "skipped": skipped_count,
        "message": "抽帧完成"
    }


def main():
    input_folder = Path(VIDEO_COPY_DIR)
    output_folder = Path(CUT_IMAGE_DIR)

    output_folder.mkdir(parents=True, exist_ok=True)

    if not input_folder.exists():
        print(f"输入视频文件夹不存在: {input_folder}")
        return

    video_files = [
        file for file in input_folder.iterdir()
        if file.is_file() and file.suffix.lower() in VIDEO_EXTS
    ]

    video_files = sorted(video_files)
    total_videos = len(video_files)

    if total_videos == 0:
        print(f"没有找到视频文件，请检查输入路径: {input_folder}")
        return

    print(f"找到 {total_videos} 个视频")
    print(f"开始抽帧，每隔 {FRAME_INTERVAL} 帧截取一张图片")
    print(f"输入目录: {input_folder}")
    print(f"输出目录: {output_folder}")

    worker_func = partial(
        extract_frames,
        output_folder=output_folder,
        frame_interval=FRAME_INTERVAL,
        use_gpu=USE_GPU
    )

    done = 0
    skipped_done = 0
    failed = 0
    total_saved = 0
    total_skipped = 0

    with Pool(processes=MAX_CUT_PROCESSES) as pool:
        for result in tqdm(pool.imap(worker_func, video_files), total=total_videos):
            status = result["status"]

            if status == "done":
                done += 1
            elif status == "skipped_done":
                skipped_done += 1
            elif status == "failed":
                failed += 1

            total_saved += result["saved"]
            total_skipped += result["skipped"]

            print(
                f"{result['video']} | {result['message']} | "
                f"新增 {result['saved']} 张 | 跳过 {result['skipped']} 张"
            )

    print("\n视频抽帧完成")
    print(f"本次完成视频: {done}")
    print(f"跳过已完成视频: {skipped_done}")
    print(f"失败视频: {failed}")
    print(f"新增图片: {total_saved}")
    print(f"跳过已存在图片: {total_skipped}")


if __name__ == "__main__":
    main()
