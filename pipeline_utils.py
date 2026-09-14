"""Shared file integrity and output ownership rules for the existing stages."""
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

MANIFEST = "_pipeline_manifest.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def owned_path(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    target = root / relative
    if relative.is_absolute() or ".." in relative.parts or target.resolve() == root:
        raise ValueError(f"Invalid output name: {name}")
    if not target.resolve().is_relative_to(root) or target.is_symlink():
        raise ValueError(f"Output escapes its directory: {name}")
    return target


def manifest(folder):
    folder = Path(folder)
    record = read_json(folder / MANIFEST)
    if record is None:
        if folder.exists() and any(folder.iterdir()):
            raise RuntimeError(f"Unmanaged output directory: {folder}. Choose a new OUTPUT_ROOT; old outputs are preserved.")
        folder.mkdir(parents=True, exist_ok=True)
        record = {"version": 1, "files": {}}
        atomic_json(folder / MANIFEST, record)
    if record.get("version") != 1 or not isinstance(record.get("files"), dict):
        raise ValueError(f"Invalid output manifest: {folder}")
    for name in record["files"]:
        owned_path(folder, name)
    return record


def managed_files(folder, extensions):
    record = manifest(folder)
    paths = [owned_path(folder, n) for n in sorted(record["files"]) if Path(n).suffix.lower() in extensions]
    for path in paths:
        relative = path.relative_to(Path(folder).resolve()).as_posix()
        if not path.is_file() or digest(path) != record["files"][relative]:
            raise RuntimeError(f"Output changed or missing; rerun the preceding stage: {path}")
    return paths


def atomic_copy(source, target):
    source, target = Path(source), Path(target)
    expected = digest(source)
    if target.is_file() and digest(target) == expected:
        return "skipped"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        if digest(temporary) != expected or digest(source) != expected:
            raise RuntimeError(f"Source changed during copy: {source}")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return "copied"


def finish_manifest(folder, files, previous, *, prune=True):
    """Only remove files previously recorded by this stage; never arbitrary files."""
    folder = Path(folder)
    for name, expected in previous["files"].items():
        if prune and name not in files:
            path = owned_path(folder, name)
            if path.exists():
                if digest(path) != expected:
                    raise RuntimeError(f"Refusing to remove externally modified output: {path}")
                path.unlink()
    atomic_json(folder / MANIFEST, {"version": 1, "files": files})


def sync_files(sources, folder):
    """sources maps unique relative destination names to source paths."""
    previous = manifest(folder)
    copied = skipped = 0
    files = {}
    for name, source in sorted(sources.items()):
        target = owned_path(folder, name)
        # A prior interrupted copy may have installed this exact content already.
        if target.exists() and name not in previous["files"] and digest(target) != digest(source):
            raise RuntimeError(f"Untracked output conflicts with source: {target}")
        status = atomic_copy(source, target)
        copied += status == "copied"
        skipped += status == "skipped"
        files[name] = digest(target)
    finish_manifest(folder, files, previous)
    return copied, skipped


def unique_mapping(pairs):
    mapping = {}
    for name, path in pairs:
        if name in mapping and Path(mapping[name]) != Path(path):
            raise RuntimeError(f"Filename collision: {name}")
        mapping[name] = path
    return mapping
