import csv
import json
from pathlib import Path
import subprocess
import sys
import struct
import zipfile

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

import frame_extraction as frames
import image_deduplication as dedup
import image_gather as gather
import pipeline_utils as utils
import video_copy as videos


def video(path, count=9):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for i in range(count):
        writer.write(np.full((48, 64, 3), i * 15, dtype=np.uint8))
    writer.release()
    return path


def test_existing_partial_copy_is_repaired(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.write_bytes(b"complete source")
    dest.write_bytes(b"partial")
    assert utils.atomic_copy(source, dest) == "copied"
    assert dest.read_bytes() == source.read_bytes()
    assert utils.atomic_copy(source, dest) == "skipped"


def test_failed_copy_does_not_replace_existing_file(tmp_path, monkeypatch):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.write_bytes(b"new")
    dest.write_bytes(b"old")
    def broken(src, dst):
        Path(dst).write_bytes(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(utils.shutil, "copy2", broken)
    with pytest.raises(OSError):
        utils.atomic_copy(source, dest)
    assert dest.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_source_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(videos, "VIDEO_SOURCE_DIR", tmp_path / "absent")
    monkeypatch.setattr(videos, "VIDEO_COPY_DIR", tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        videos.main()


def test_same_size_update_and_owned_removal(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    source.write_bytes(b"old")
    utils.sync_files({"image.jpg": source}, out)
    source.write_bytes(b"new")
    utils.sync_files({"image.jpg": source}, out)
    assert (out / "image.jpg").read_bytes() == b"new"
    (out / "personal.txt").write_text("keep")
    utils.sync_files({}, out)
    assert not (out / "image.jpg").exists()
    assert (out / "personal.txt").read_text() == "keep"


def test_unmanaged_directory_is_preserved(tmp_path):
    (tmp_path / "old.jpg").write_bytes(b"private")
    with pytest.raises(RuntimeError, match="Unmanaged"):
        utils.manifest(tmp_path)
    assert (tmp_path / "old.jpg").read_bytes() == b"private"


def test_modified_stale_output_is_not_deleted(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    source.write_bytes(b"original")
    utils.sync_files({"image.jpg": source}, out)
    (out / "image.jpg").write_bytes(b"manual edit")
    with pytest.raises(RuntimeError, match="externally modified"):
        utils.sync_files({}, out)
    assert (out / "image.jpg").read_bytes() == b"manual edit"


@pytest.mark.parametrize("name", ["../escape", "..", "."])
def test_manifest_path_boundary(tmp_path, name):
    with pytest.raises(ValueError):
        utils.owned_path(tmp_path, name)


def test_original_hash_collision_is_resolved(tmp_path):
    a = gather.make_target_name(tmp_path / "video/frame55504.jpg", tmp_path, "sample")[0]
    b = gather.make_target_name(tmp_path / "video/frame91380.jpg", tmp_path, "sample")[0]
    assert a != b
    with pytest.raises(RuntimeError, match="collision"):
        utils.unique_mapping([("same.jpg", "a"), ("same.jpg", "b")])


def test_frames_restart_and_parameter_change(tmp_path):
    source = video(tmp_path / "video.avi")
    out = tmp_path / "frames"
    first = frames.extract_frames(source, out, 3, False)
    assert first["saved"] == 3
    assert frames.extract_frames(source, out, 3, False)["status"] == "skipped_done"
    changed = frames.extract_frames(source, out, 4, False)
    assert changed["saved"] == 3
    assert not (first["files"].keys() & changed["files"].keys())
    # Corrupt a saved image: the done marker must not cause it to be reused.
    (out / next(iter(first["files"]))).write_bytes(b"bad")
    assert frames.extract_frames(source, out, 3, False)["status"] == "done"


def test_failed_frame_write_leaves_no_done_marker(tmp_path, monkeypatch):
    source = video(tmp_path / "video.avi")
    monkeypatch.setattr(frames.cv2, "imencode", lambda *a: (False, None))
    with pytest.raises(RuntimeError, match="encode"):
        frames.extract_frames(source, tmp_path / "frames", 3, False)
    assert not list((tmp_path / "frames").rglob("_done.json"))


def test_incomplete_decode_leaves_no_done_marker(tmp_path, monkeypatch):
    source = tmp_path / "video.avi"
    source.write_bytes(b"video")
    class Capture:
        def isOpened(self): return True
        def get(self, key): return 9
        def read(self): return False, None
        def release(self): pass
    monkeypatch.setattr(frames.cv2, "VideoCapture", lambda _: Capture())
    with pytest.raises(RuntimeError, match="Incomplete"):
        frames.extract_frames(source, tmp_path / "frames", 3, False)
    assert not list((tmp_path / "frames").rglob("_done.json"))


def test_cache_invalidates_same_name_content_change(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"old")
    identity = dedup.feature_identity([image])
    cache = tmp_path / "cache.npz"
    dedup.save_feature_cache(cache, {image.name: np.array([1, 2], dtype=np.float32)}, identity)
    assert dedup.load_feature_cache(cache, identity) is not None
    image.write_bytes(b"new")
    assert dedup.load_feature_cache(cache, dedup.feature_identity([image])) is None
    cache.write_bytes(b"interrupted cache")
    assert dedup.load_feature_cache(cache, identity) is None


def test_corrupted_compressed_payload_is_a_cache_miss(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    identity = dedup.feature_identity([image])
    cache = tmp_path / "cache.npz"
    dedup.save_feature_cache(cache, {image.name: np.arange(1, 100, dtype=np.float32)}, identity)
    with zipfile.ZipFile(cache) as archive:
        info = archive.getinfo("features.npy")
    data = bytearray(cache.read_bytes())
    offset = info.header_offset
    name_length, extra_length = struct.unpack_from("<HH", data, offset + 26)
    payload_start = offset + 30 + name_length + extra_length
    data[payload_start + info.compress_size // 2] ^= 0xFF
    cache.write_bytes(data)
    assert dedup.load_feature_cache(cache, identity) is None


def test_faiss_more_than_twenty_identical_images():
    index, features, names = dedup.build_index({f"image{i:03}.jpg": np.array([1, 0]) for i in range(65)})
    pairs, duplicates = dedup.find_duplicates(index, features, names, .90)
    assert len(duplicates) == 64
    assert set(names) - duplicates == {names[0]}
    assert all(original == names[0] for original, _, _ in pairs)


def test_nontransitive_neighbors_keep_valid_representatives():
    vectors = {f"{i}.jpg": np.array([np.cos(a), np.sin(a)]) for i, a in enumerate([0, .3, .6])}
    index, features, names = dedup.build_index(vectors)
    pairs, duplicates = dedup.find_duplicates(index, features, names, .95)
    assert duplicates == {"1.jpg"}
    assert all(original not in duplicates for original, _, _ in pairs)


def test_encoder_uses_its_transform_and_rejects_bad_images(tmp_path, monkeypatch):
    import imagededup.methods.cnn as cnn
    called = []
    class Encoder:
        model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        def transform(self, image):
            called.append(image.size)
            return torch.ones(3, 4, 4)
    monkeypatch.setattr(cnn, "CNN", Encoder)
    good, bad = tmp_path / "good.jpg", tmp_path / "bad.jpg"
    Image.new("RGB", (8, 8), "red").save(good)
    bad.write_bytes(b"not an image")
    cache = tmp_path / "cache.npz"
    result = dedup.build_features([good, bad], cache)
    assert set(result) == {good.name}
    assert called == [(8, 8)]
    gathered = tmp_path / "gathered"
    utils.sync_files({good.name: good, bad.name: bad}, gathered)
    out = tmp_path / "out"
    monkeypatch.setattr(dedup, "GATHER_IMAGE_DIR", gathered)
    monkeypatch.setattr(dedup, "DEDUP_IMAGE_DIR", out)
    monkeypatch.setattr(dedup, "FEATURE_FILE", cache)
    dedup.main()
    assert (out / good.name).exists()
    assert not (out / bad.name).exists()
    assert utils.read_json(out / "rejected_images.json") == [bad.name]
    assert utils.read_json(out / "dedup_summary.json")["invalid_input"] == 1


def test_gather_uses_current_manifest_not_historical_frames(tmp_path):
    source, out = tmp_path / "frames", tmp_path / "gather"
    previous = utils.manifest(source)
    (source / "current.jpg").write_bytes(b"current")
    (source / "historical.jpg").write_bytes(b"old")
    utils.finish_manifest(source, {"current.jpg": utils.digest(source / "current.jpg")}, previous, prune=False)
    gather.copy_and_rename_images(source, out, "sample")
    assert len(utils.managed_files(out, {".jpg"})) == 1
    with (out / "image_mapping.csv").open(encoding="utf-8-sig") as stream:
        assert list(csv.DictReader(stream))[0]["source_relative_path"] == "current.jpg"


def test_final_verification_failure_records_failed_status(tmp_path, monkeypatch):
    import generate_report
    import run_pipeline
    monkeypatch.setattr(run_pipeline, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(run_pipeline, "print_status", lambda: None)
    calls = []
    monkeypatch.setattr(run_pipeline, "run_script", lambda name, script: calls.append(script))
    def broken_verification():
        raise RuntimeError("Output changed after validation")
    monkeypatch.setattr(generate_report, "state_fingerprint", broken_verification)
    with pytest.raises(RuntimeError, match="Output changed"):
        run_pipeline.main()
    status = utils.read_json(tmp_path / "pipeline_status.json")
    assert status["status"] == "failed"
    assert status["stages"][-1]["status"] == "failed"
    assert calls[-1] == "generate_report.py"
