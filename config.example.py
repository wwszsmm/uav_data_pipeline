from pathlib import Path

# Input directory containing the original UAV videos.
INPUT_PATH = Path(r"replace_with_your_video_directory")

# All generated outputs are stored next to the input directory.
OUTPUT_ROOT = INPUT_PATH.parent / f"{INPUT_PATH.name}_output"

GATHER_FOLDER_NAME = "gather_images"
GATHER_PREFIX = "test_result"

FRAME_INTERVAL = 75

MAX_COPY_WORKERS = 8
MAX_CUT_PROCESSES = 10

MIN_SIMILARITY_THRESHOLD = 0.90
BATCH_SIZE = 32
MAX_THREADS = 12

# OpenCV CUDA video decoding is optional.
USE_GPU = False

VIDEO_SOURCE_DIR = INPUT_PATH
VIDEO_COPY_DIR = OUTPUT_ROOT / "copied_videos"
CUT_IMAGE_DIR = OUTPUT_ROOT / "cut_images"
GATHER_IMAGE_DIR = OUTPUT_ROOT / GATHER_FOLDER_NAME
DEDUP_IMAGE_DIR = OUTPUT_ROOT / "deduped_images"

FEATURE_DIR = OUTPUT_ROOT / "features"
FEATURE_FILE = FEATURE_DIR / f"{GATHER_FOLDER_NAME}_features.npz"

REPORT_DIR = OUTPUT_ROOT / "reports"

VIDEO_EXTS = {
    ".mp4", ".avi", ".mkv", ".flv", ".wmv", ".ts", ".m4v", ".mov"
}

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"
}
