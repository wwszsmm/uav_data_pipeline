"""Load one configuration, resolving paths relative to the config file."""
import importlib.util
import math
import os
from pathlib import Path

config_path = Path(os.environ.get("UAV_CONFIG", Path(__file__).with_name("config.py"))).resolve()
if not config_path.is_file():
    raise RuntimeError("Create config.py from config.example.py, or set UAV_CONFIG to a configuration file.")
spec = importlib.util.spec_from_file_location("uav_local_config", config_path)
config = importlib.util.module_from_spec(spec)
exec(compile(config_path.read_text(encoding="utf-8-sig"), str(config_path), "exec"), config.__dict__)

for name in dir(config):
    if name.isupper():
        value = getattr(config, name)
        if name.endswith(("_DIR", "_PATH", "_ROOT", "_FILE")):
            value = Path(value).expanduser()
            value = (config_path.parent / value).resolve() if not value.is_absolute() else value.resolve()
        globals()[name] = value


def validate():
    for name in ("FRAME_INTERVAL", "MAX_COPY_WORKERS", "MAX_CUT_PROCESSES", "BATCH_SIZE", "MAX_THREADS"):
        value = globals()[name]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(USE_GPU, bool):
        raise ValueError("USE_GPU must be a boolean (video decoding only)")
    if not math.isfinite(MIN_SIMILARITY_THRESHOLD) or not -1 <= MIN_SIMILARITY_THRESHOLD <= 1:
        raise ValueError("MIN_SIMILARITY_THRESHOLD must be finite and between -1 and 1")
    if not VIDEO_SOURCE_DIR.is_dir():
        raise ValueError(f"Video input directory does not exist: {VIDEO_SOURCE_DIR}")
    if OUTPUT_ROOT == VIDEO_SOURCE_DIR or OUTPUT_ROOT.is_relative_to(VIDEO_SOURCE_DIR) or VIDEO_SOURCE_DIR.is_relative_to(OUTPUT_ROOT):
        raise ValueError("Input and OUTPUT_ROOT must be separate, non-nested directories")
    folders = [VIDEO_COPY_DIR, CUT_IMAGE_DIR, GATHER_IMAGE_DIR, DEDUP_IMAGE_DIR, FEATURE_DIR]
    for i, folder in enumerate(folders):
        if folder == OUTPUT_ROOT or not folder.is_relative_to(OUTPUT_ROOT):
            raise ValueError(f"Stage directory must be inside OUTPUT_ROOT: {folder}")
        for other in folders[:i]:
            if folder == other or folder.is_relative_to(other) or other.is_relative_to(folder):
                raise ValueError("Stage directories must not overlap")
    if not FEATURE_FILE.is_relative_to(FEATURE_DIR):
        raise ValueError("FEATURE_FILE must be inside FEATURE_DIR")
    if not GATHER_PREFIX or any(c in GATHER_PREFIX for c in '<>:"/\\|?*') or GATHER_PREFIX.endswith((" ", ".")):
        raise ValueError("GATHER_PREFIX must be a safe filename prefix")


validate()
