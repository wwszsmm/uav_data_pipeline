from datetime import datetime
import hashlib
import json
from pathlib import Path

from pipeline_config import (OUTPUT_ROOT, VIDEO_SOURCE_DIR, VIDEO_COPY_DIR, CUT_IMAGE_DIR,
                             GATHER_IMAGE_DIR, DEDUP_IMAGE_DIR, VIDEO_EXTS, IMAGE_EXTS,
                             config_path)
from pipeline_utils import digest, managed_files, read_json, MANIFEST


def state_fingerprint():
    """Bind a successful report to the inputs and active outputs that were checked."""
    state = {}
    folders = [(VIDEO_SOURCE_DIR, VIDEO_EXTS), (VIDEO_COPY_DIR, VIDEO_EXTS),
               (CUT_IMAGE_DIR, IMAGE_EXTS), (GATHER_IMAGE_DIR, IMAGE_EXTS), (DEDUP_IMAGE_DIR, IMAGE_EXTS)]
    for index, (folder, extensions) in enumerate(folders):
        paths = (sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
                 if index == 0 else managed_files(folder, extensions))
        state[str(index)] = {p.relative_to(folder).as_posix(): digest(p) for p in paths}
    for path in (DEDUP_IMAGE_DIR / "dedup_summary.json", OUTPUT_ROOT / "image_check.json"):
        state[path.name] = digest(path)
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def count_files(folder, extensions):
    folder = Path(folder)
    if not folder.exists():
        return 0
    record = read_json(folder / MANIFEST)
    if record is not None:
        return sum(Path(n).suffix.lower() in extensions for n in record["files"])
    return sum(p.is_file() and p.suffix.lower() in extensions for p in folder.rglob("*"))


def main():
    status = read_json(OUTPUT_ROOT / "pipeline_status.json",
                       {"status": "not_verified", "stages": []})
    dedup = read_json(DEDUP_IMAGE_DIR / "dedup_summary.json", {})
    check = read_json(OUTPUT_ROOT / "image_check.json", {})
    successful = status["status"] in ("complete", "complete_with_exclusions")
    configuration = status.get("configuration", {})
    if successful and configuration.get("sha256") != digest(config_path):
        status["status"] = "configuration_changed_since_run"
        successful = False
    if successful:
        try:
            successful = status.get("state_sha256") == state_fingerprint()
        except (OSError, ValueError, RuntimeError):
            successful = False
        if not successful:
            status["status"] = "data_changed_or_unverified"
    # Failed runs must not label previous-run summaries as current verified results.
    counts = [
        ("Original videos", count_files(VIDEO_SOURCE_DIR, VIDEO_EXTS)),
        ("Current copied videos", count_files(VIDEO_COPY_DIR, VIDEO_EXTS)),
        ("Current extracted images", count_files(CUT_IMAGE_DIR, IMAGE_EXTS)),
        ("Current gathered images", count_files(GATHER_IMAGE_DIR, IMAGE_EXTS)),
        ("Invalid input images excluded", dedup.get("invalid_input", "Not verified") if successful else "Not verified"),
        ("Duplicate images", dedup.get("duplicates", "Not verified") if successful else "Not verified"),
        ("Verified valid output images", check.get("valid", "Not verified") if successful else "Not verified"),
        ("Output integrity failures", check.get("invalid", "Not verified") if successful else "Not verified"),
        ("Failed stages", sum(s["status"] == "failed" for s in status["stages"])),
    ]
    table = "\n".join(f"| {name} | {value} |" for name, value in counts)
    stages = "\n".join(f"- {s['stage']}: {s['status']}" for s in status["stages"]) or "- No orchestrated run recorded"
    rate = (f"{dedup['duplicates'] / dedup['valid_input']:.2%}"
            if successful and dedup.get("valid_input") else "Not verified")
    report = f"""# UAV Data Pipeline Report

Generated: {datetime.now():%Y-%m-%d %H:%M:%S}
Run started: {status.get('started', 'Not recorded')}
Run status: **{status['status']}**

## Configuration
- Recorded frame interval: {configuration.get('frame_interval', 'Not recorded')}
- Recorded similarity threshold: {configuration.get('similarity_threshold', 'Not recorded')}

## Results
| Item | Count |
|---|---:|
{table}

Duplicate rate among valid input images: {rate}

Counts describe active output manifests. After a failed run, some manifests can
still describe an earlier stage/run; they are not proof of successful processing.
Historical frame snapshots are excluded from these counts.
An independently invoked report checks recorded configuration/content fingerprints;
it does not rerun video decoding, CNN inference or the image integrity stage.

## Stages
{stages}
"""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "pipeline_report.md"
    path.write_text(report, encoding="utf-8")
    print(f"Report: {path}; status={status['status']}")


if __name__ == "__main__":
    main()
