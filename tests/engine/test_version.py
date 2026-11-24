"""The engine reports the real release version, also when frozen by PyInstaller (no dist metadata)."""
import json
from pathlib import Path

import bsmaker
from bsmaker.engine import server

ROOT = Path(__file__).resolve().parents[2]


def test_engine_version_comes_from_the_package_not_dist_metadata():
    assert server._get_version() == bsmaker.__version__


def test_package_version_matches_the_app_release():
    app = json.loads((ROOT / "app" / "package.json").read_text(encoding="utf-8"))
    assert bsmaker.__version__ == app["version"]
