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


def test_dirty_other_bundled_path_is_refused(repo: Path, tmp_path: Path) -> None:
    mod = _load_script()
    (repo / "playability.toml").write_text("changed")
    out = tmp_path / "bundle.zip"
    with pytest.raises(RuntimeError, match="playability.toml"):
        mod.build_bundle(repo, out)
    assert not out.exists()


def test_untracked_bundled_path_is_refused(repo: Path, tmp_path: Path) -> None:
    mod = _load_script()
    _git(repo, "rm", "-q", "dataset_stats.json")
    _git(repo, "commit", "-q", "-m", "rm")
    out = tmp_path / "bundle.zip"
    with pytest.raises(RuntimeError, match="not tracked"):
        mod.build_bundle(repo, out)
    assert not out.exists() and not out.with_suffix(".tmp").exists()


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


def test_notebook_streams_output_and_marks_unpack() -> None:
    srcs = ["".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == "code"]
    run_cell = next(s for s in srcs if "def run(" in s)
    assert "Popen" in run_cell and "stderr=subprocess.STDOUT" in run_cell
    assert any(".done" in s for s in srcs)
    assert any("flush_and_unmount" in s for s in srcs)


# ---- delta bundle ----

def _make_data(root: Path, keys: list[str], stats: bool = True) -> None:
    (root / "events").mkdir(parents=True)
    (root / "features").mkdir()
    (root / "events" / "index.json").write_text(json.dumps([{"map_key": k} for k in keys]))
    (root / "events" / "events.npz").write_bytes(b"npz")
    for k in keys:
        (root / "features" / f"{k}.npy").write_bytes(b"n")
        (root / "features" / f"{k}.json").write_text("{}")
    if stats:
        (root / "dataset_stats.json").write_text("{}")


def _base_index(path: Path, keys: list[str]) -> Path:
    path.write_text(json.dumps([{"map_key": k} for k in keys]))
    return path


def test_delta_holds_new_keys_and_events(tmp_path: Path) -> None:
    import tarfile
    mod = _load_script()
    _make_data(tmp_path / "d", ["a", "b", "c"])
    base = _base_index(tmp_path / "base.json", ["a"])
    out, n = mod.build_delta(base, tmp_path / "d", tmp_path / "out")
    assert n == 2 and out.name == "bsmaker-train-delta.tar"
    with tarfile.open(out) as tf:
        names = {m.name for m in tf.getmembers() if m.isfile()}
    assert names == {"events/index.json", "events/events.npz", "dataset_stats.json",
                     "features/b.npy", "features/b.json", "features/c.npy", "features/c.json"}


def test_delta_without_stats_and_refuses_overwrite(tmp_path: Path) -> None:
    import tarfile
    mod = _load_script()
    _make_data(tmp_path / "d", ["a", "b"], stats=False)
    base = _base_index(tmp_path / "base.json", ["a"])
    out, _ = mod.build_delta(base, tmp_path / "d", tmp_path / "out")
    with tarfile.open(out) as tf:
        assert "dataset_stats.json" not in tf.getnames()
    with pytest.raises(RuntimeError, match="overwrite"):
        mod.build_delta(base, tmp_path / "d", tmp_path / "out")


def test_delta_missing_feature_names_key_and_leaves_no_tar(tmp_path: Path) -> None:
    mod = _load_script()
    _make_data(tmp_path / "d", ["a", "bkey"])
    (tmp_path / "d" / "features" / "bkey.npy").unlink()
    base = _base_index(tmp_path / "base.json", ["a"])
    with pytest.raises(RuntimeError, match="bkey"):
        mod.build_delta(base, tmp_path / "d", tmp_path / "out")
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


# ---- VM scripts ----

def _load_vm(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "colab" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tar(path: Path, files: dict[str, str]) -> None:
    import tarfile
    src = path.parent / (path.name + ".src")
    for name, text in files.items():
        f = src / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    with tarfile.open(path, "w") as tf:
        for top in sorted({n.split("/")[0] for n in files}):
            tf.add(src / top, arcname=top)


@pytest.fixture
def drive(tmp_path: Path) -> Path:
    d = tmp_path / "drive"
    d.mkdir()
    _tar(d / "bsmaker-train-data.tar", {"events/index.json": "[1, 2]", "features/a.npy": "x"})
    _tar(d / "bsmaker-train-delta.tar", {"events/index.json": "[1, 2, 3]", "features/b.npy": "x",
                                         "dataset_stats.json": '{"new": 1}'})
    with zipfile.ZipFile(d / "bsmaker-code.zip", "w") as zf:
        zf.writestr("src/bsmaker/__init__.py", "")
        zf.writestr("dataset_stats.json", '{"old": 1}')
        zf.writestr("BUNDLE_COMMIT.txt", "abc123\n")
    return d


def test_vm_setup_layout_and_idempotence(drive: Path, tmp_path: Path, capsys) -> None:
    mod = _load_vm("vm_setup")
    work = tmp_path / "work"
    assert mod.main(["--drive-dir", str(drive), "--work", str(work), "--no-pip"]) == 0
    out = capsys.readouterr().out
    assert "feature files (.npy): 2" in out and "index entries: 3" in out and "abc123" in out
    assert (work / "data" / ".base.done").is_file() and (work / "data" / ".delta.done").is_file()
    assert not (work / "data.partial").exists()
    assert json.loads((work / "bsmaker" / "dataset_stats.json").read_text()) == {"new": 1}
    toml = (work / "bsmaker" / "bsmaker.toml").read_text()
    assert (work / "data").as_posix() in toml and "custom_levels" in toml
    (work / "data" / "features" / "marker.txt").write_text("keep")
    assert mod.main(["--drive-dir", str(drive), "--work", str(work), "--no-pip"]) == 0
    assert "skipping" in capsys.readouterr().out
    assert (work / "data" / "features" / "marker.txt").exists()


def test_vm_setup_recovers_from_partial(drive: Path, tmp_path: Path) -> None:
    mod = _load_vm("vm_setup")
    work = tmp_path / "work"
    (work / "data.partial" / "features").mkdir(parents=True)
    (work / "data.partial" / "features" / "junk.npy").write_text("j")
    assert mod.main(["--drive-dir", str(drive), "--work", str(work), "--no-pip"]) == 0
    assert not (work / "data.partial").exists()
    assert not (work / "data" / "features" / "junk.npy").exists()


def test_vm_setup_missing_delta(drive: Path, tmp_path: Path) -> None:
    mod = _load_vm("vm_setup")
    (drive / "bsmaker-train-delta.tar").unlink()
    work = tmp_path / "work"
    with pytest.raises(SystemExit, match="bsmaker-train-delta.tar"):
        mod.main(["--drive-dir", str(drive), "--work", str(work), "--no-pip"])
    assert mod.main(["--drive-dir", str(drive), "--work", str(work), "--no-pip", "--no-delta"]) == 0
    assert not (work / "data" / ".delta.done").exists()


def test_vm_setup_refuses_unsafe_tar(tmp_path: Path) -> None:
    import io
    import tarfile
    mod = _load_vm("vm_setup")
    bad = tmp_path / "bad.tar"
    with tarfile.open(bad, "w") as tf:
        info = tarfile.TarInfo("../evil.txt")
        info.size = 1
        tf.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(SystemExit, match="unsafe"):
        mod.safe_extract(bad, tmp_path / "dest")
    assert not (tmp_path / "evil.txt").exists()


def test_vm_train_dry_run_and_validation(tmp_path: Path, capsys) -> None:
    import sys as _sys
    mod = _load_vm("vm_train")
    drv = tmp_path / "drv"
    assert mod.main(["--model", "timing", "--name", "r1", "--drive-dir", str(drv),
                     "--work", str(tmp_path / "w"), "--dry-run", "--", "--max-steps", "5"]) == 0
    out = capsys.readouterr().out.strip()
    expected = (f"$ {_sys.executable} -u -m bsmaker train timing --out "
                f"{drv / 'checkpoints' / 'gen2' / 'r1'} --max-steps 5")
    assert out == expected and not drv.exists()
    with pytest.raises(SystemExit):
        mod.main(["--model", "timing", "--name", "bad name", "--dry-run"])
    with pytest.raises(SystemExit):
        mod.main(["--model", "timing", "--name", "r", "--dry-run", "--", "--out", "x"])


def test_vm_train_runs_logs_and_propagates_exit(tmp_path: Path, capsys) -> None:
    mod = _load_vm("vm_train")
    work, drv = tmp_path / "w", tmp_path / "drv"
    pkg = work / "bsmaker" / "src" / "bsmaker"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text("import sys\nprint('hello', sys.argv[1:])\nraise SystemExit(3)\n")
    code = mod.main(["--model", "pattern", "--name", "r2", "--drive-dir", str(drv),
                     "--work", str(work), "--", "--max-steps", "1"])
    assert code == 3
    log = (drv / "checkpoints" / "gen2" / "r2" / "console.log").read_text()
    assert log.startswith("$ ") and "hello ['train', 'pattern'" in log
    assert log.strip().endswith("vm_train: exit code 3")
    assert capsys.readouterr().out.strip().endswith("vm_train: exit code 3")
