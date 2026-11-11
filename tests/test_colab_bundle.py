import importlib.util
import json
import re
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "make_colab_bundle", ROOT / "scripts" / "make_colab_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    (r / "src" / "pkg").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "src" / "pkg" / "a.py").write_text("x = 1\n")
    (r / "tests" / "t.py").write_text("y = 2\n")
    for name in ("pyproject.toml", "dataset_stats.json", "playability.toml", "other.txt"):
        (r / name).write_text("data\n")
    _git(r, "init", "-q")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    return r


def test_bundle_contents_and_commit(repo: Path, tmp_path: Path) -> None:
    mod = _load_script()
    out = tmp_path / "out" / "bundle.zip"
    commit = mod.build_bundle(repo, out)
    assert commit == _git(repo, "rev-parse", "HEAD")
    with zipfile.ZipFile(out) as zf:
        names = {n for n in zf.namelist() if not n.endswith("/")}
        assert names == {"src/pkg/a.py", "pyproject.toml", "dataset_stats.json",
                         "playability.toml", "BUNDLE_COMMIT.txt"}
        assert zf.read("BUNDLE_COMMIT.txt").decode().strip() == commit


def test_dirty_src_is_refused(repo: Path, tmp_path: Path) -> None:
    mod = _load_script()
    (repo / "src" / "pkg" / "a.py").write_text("x = 2\n")
    out = tmp_path / "bundle.zip"
    with pytest.raises(RuntimeError, match="a.py"):
        mod.build_bundle(repo, out)
    assert not out.exists()


def _notebook() -> dict:
    return json.loads((ROOT / "notebooks" / "colab_train.ipynb").read_text("utf-8"))


def test_notebook_valid_and_clean() -> None:
    nb = _notebook()
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells
    for c in code_cells:
        src = "".join(c["source"])
        lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))]
        compile("\n".join(lines), "<cell>", "exec")
        assert "F:\\" not in src
        assert not re.search(r"\S+@\S+", src)
    assert not re.search(r"\S+@\S+\.\S+", json.dumps(nb))
