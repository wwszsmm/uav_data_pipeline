import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_workspace = tempfile.TemporaryDirectory(prefix="uav-tests-")
_root = Path(_workspace.name)
(_root / "input").mkdir()
_config = _root / "config.py"
_config.write_text((ROOT / "config.example.py").read_text(encoding="utf-8").replace(
    'Path(r"replace_with_your_video_directory")', repr(_root / "input").replace("WindowsPath", "Path").replace("PosixPath", "Path")), encoding="utf-8")
os.environ["UAV_CONFIG"] = str(_config)


def pytest_sessionfinish(session, exitstatus):
    _workspace.cleanup()
