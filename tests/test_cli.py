import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def make_config(tmp_path, **overrides):
    source = tmp_path / "input"
    source.mkdir(exist_ok=True)
    text = (ROOT / "config.example.py").read_text(encoding="utf-8")
    text = text.replace('Path(r"replace_with_your_video_directory")', 'Path("input")')
    text += "\nFRAME_INTERVAL = 3\nMAX_COPY_WORKERS = 2\nMAX_CUT_PROCESSES = 2\nMAX_THREADS = 2\n"
    for key, value in overrides.items():
        text += f"{key} = {value!r}\n"
    config = tmp_path / "run_config.py"
    config.write_text(text, encoding="utf-8")
    return config, source, tmp_path / "input_output"


def invoke(config, script="run_pipeline.py", cwd=None):
    result = subprocess.run([sys.executable, str(ROOT / script)], cwd=cwd or ROOT,
                            env={**os.environ, "UAV_CONFIG": str(config), "PYTHONUTF8": "1"},
                            capture_output=True, text=True, encoding="utf-8", timeout=180)
    return result


def test_empty_input_stops_and_reports_failure(tmp_path):
    config, _, output = make_config(tmp_path)
    result = invoke(config)
    assert result.returncode != 0
    status = json.loads((output / "pipeline_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert len(status["stages"]) == 1
    assert "Not verified" in (output / "pipeline_report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("override", [{"FRAME_INTERVAL": 0}, {"MAX_THREADS": -1},
                                      {"MIN_SIMILARITY_THRESHOLD": 1.5}])
def test_invalid_configuration_stops_before_outputs(tmp_path, override):
    config, _, output = make_config(tmp_path, **override)
    result = invoke(config)
    assert result.returncode != 0
    assert not output.exists()


def test_overlapping_outputs_are_rejected(tmp_path):
    config, _, output = make_config(tmp_path)
    with config.open("a") as stream:
        stream.write("\nDEDUP_IMAGE_DIR = GATHER_IMAGE_DIR\n")
    assert invoke(config).returncode != 0
    assert not output.exists()


def test_report_does_not_relabel_old_results_with_changed_configuration(tmp_path):
    import hashlib
    config, _, output = make_config(tmp_path)
    output.mkdir()
    status = {"status": "complete", "stages": [], "configuration": {
        "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "frame_interval": 3, "similarity_threshold": 0.90}}
    (output / "pipeline_status.json").write_text(json.dumps(status), encoding="utf-8")
    with config.open("a", encoding="utf-8") as stream:
        stream.write("\nFRAME_INTERVAL = 75\nMIN_SIMILARITY_THRESHOLD = 0.99\n")
    result = invoke(config, "generate_report.py")
    assert result.returncode == 0, result.stderr
    report = (output / "pipeline_report.md").read_text(encoding="utf-8")
    assert "configuration_changed_since_run" in report
    assert "Recorded frame interval: 3" in report
    assert "Verified valid output images | Not verified" in report


def write_static_video(path, seed):
    frame = np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    assert writer.isOpened()
    for _ in range(12):
        writer.write(frame)
    writer.release()


@pytest.mark.skipif(os.environ.get("UAV_RUN_MODEL_TEST") != "1",
                    reason="Opt-in: requires cached public MobileNet weights")
def test_real_cpu_pipeline_rerun_change_and_removal(tmp_path):
    config, source, output = make_config(tmp_path)
    write_static_video(source / "a.avi", 1)
    def run():
        result = invoke(config, cwd=tmp_path)  # Different cwd from both scripts and config.
        assert result.returncode == 0, result.stdout + result.stderr
        status = json.loads((output / "pipeline_status.json").read_text(encoding="utf-8"))
        assert status["status"] == "complete"
        return json.loads((output / "deduped_images/dedup_summary.json").read_text(encoding="utf-8"))
    first = run()
    assert first["input"] == 4 and first["valid_output"] == 1
    cache = output / "features/gather_images_features.npz"
    before = cache.stat().st_mtime_ns
    assert run() == first
    assert cache.stat().st_mtime_ns == before
    write_static_video(source / "a.avi", 2)
    run()
    assert cache.stat().st_mtime_ns != before
    write_static_video(source / "b.avi", 3)
    assert run()["input"] == 8
    (source / "a.avi").unlink()
    assert run()["input"] == 4
    assert len(list((output / "copied_videos").glob("*.avi"))) == 1
    assert len(list((output / "gather_images").glob("*.jpg"))) == 4
    # A standalone report cannot claim the previous verified result after data changes.
    write_static_video(source / "b.avi", 4)
    report = invoke(config, "generate_report.py")
    assert report.returncode == 0, report.stderr
    assert "data_changed_or_unverified" in (output / "pipeline_report.md").read_text(encoding="utf-8")
