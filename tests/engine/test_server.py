import io
import json
import queue
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from test_generate import LAYER_STATS, STATS, make_models, write_audio  # noqa: E402
from bsmaker.data.layer_stats import write_layer_stats
from bsmaker.engine.server import EngineConfig, EngineServer
from bsmaker.models.common import save_checkpoint


def _setup_engine_dir(tmp_path: Path) -> EngineConfig:
    ckpts = tmp_path / "checkpoints"
    t, p = make_models()
    for name, loaded in (("timing", t), ("pattern", p)):
        save_checkpoint(ckpts / name / "best.pt", {
            "model": loaded.model.state_dict(), "model_config": asdict(loaded.config),
            "stats": STATS, "step": 1, "best": {},
        })
    play = tmp_path / "playability.toml"
    play.write_text("[mask]\n[data]\n", encoding="utf-8")
    lstats = tmp_path / "layer_stats.json"
    write_layer_stats(LAYER_STATS, lstats)
    return EngineConfig(
        checkpoints_dir=ckpts, playability_path=play, layer_stats_path=lstats,
        device="cpu", log_path=tmp_path / "engine.log",
    )


def _run_cmds(cfg: EngineConfig, cmds: list[dict]) -> list[dict]:
    stdin = io.StringIO("\n".join(json.dumps(c) for c in cmds) + "\n")
    stdout = io.StringIO()
    stderr = io.StringIO()
    srv = EngineServer(cfg, stdin=stdin, stdout=stdout, stderr=stderr)
    assert srv.serve() == 0
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def test_engine_info_and_shutdown(tmp_path: Path) -> None:
    cfg = _setup_engine_dir(tmp_path)
    msgs = _run_cmds(cfg, [{"id": "1", "cmd": "info"}, {"id": "2", "cmd": "shutdown"}])
    info = next(m for m in msgs if m["id"] == "1")
    assert info["type"] == "result"
    assert info["data"]["has_layer_models"] is False
    assert "DefaultEnvironment" in info["data"]["environments"]
    assert "tech" in info["data"]["styles"]


def test_engine_generate_progress_result_and_vram_release(tmp_path: Path, monkeypatch) -> None:
    cfg = _setup_engine_dir(tmp_path)
    audio = write_audio(tmp_path / "song.wav")
    out_dir = tmp_path / "CustomLevels"
    out_dir.mkdir()
    freed = []
    import bsmaker.engine.server as S
    orig_cpu = S.pipeline_to_cpu
    monkeypatch.setattr(S, "pipeline_to_cpu", lambda p: (freed.append(True), orig_cpu(p))[1])

    msgs = _run_cmds(cfg, [
        {"id": "g1", "cmd": "generate", "audio_path": str(audio), "out_dir": str(out_dir),
         "nps": 3.0, "styles": ["tech"], "seed": 1, "title": "Song", "artist": "Artist",
         "lights": 2, "environment": "DefaultEnvironment"},
        {"id": "s1", "cmd": "shutdown"},
    ])
    progress = [m for m in msgs if m["id"] == "g1" and m["type"] == "progress"]
    res = next(m for m in msgs if m["id"] == "g1" and m["type"] == "result")
    assert progress and progress[-1]["pct"] == 100
    assert Path(res["data"]["folder"]).is_dir()
    assert res["data"]["used_rule_lights"] is True
    assert freed == [True]


def test_engine_internal_exception_never_leaks_traceback(tmp_path: Path, monkeypatch) -> None:
    cfg = _setup_engine_dir(tmp_path)
    import bsmaker.engine.server as S
    monkeypatch.setattr(S, "search_youtube", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("secret stack detail")))

    res = _run_cmds(cfg, [
        {"id": "q1", "cmd": "search", "query": "hello"},
        {"id": "s1", "cmd": "shutdown"},
    ])
    err = next(m for m in res if m["id"] == "q1")
    assert err["type"] == "error" and err["code"] == "internal"
    assert "Traceback" not in json.dumps(err)
    assert "secret stack detail" not in err["message_pl"]
    assert "secret stack detail" in (tmp_path / "engine.log").read_text(encoding="utf-8")


def test_engine_cancel_mid_generate_returns_cancelled_and_releases_vram(tmp_path: Path, monkeypatch) -> None:
    cfg = _setup_engine_dir(tmp_path)
    audio = write_audio(tmp_path / "song.wav")
    out_dir = tmp_path / "CustomLevels"
    out_dir.mkdir()

    import bsmaker.engine.server as S
    started = threading.Event()
    freed = []
    orig_cpu = S.pipeline_to_cpu
    monkeypatch.setattr(S, "pipeline_to_cpu", lambda p: (freed.append(True), orig_cpu(p))[1])

    def slow_generate(*a, is_cancelled=None, **kw):
        started.set()
        for _ in range(100):
            if is_cancelled and is_cancelled():
                from bsmaker.errors import GenerateCancelled
                raise GenerateCancelled("cancelled")
            time.sleep(0.01)
        raise AssertionError("cancel flag never observed")

    monkeypatch.setattr(S, "run_generate", slow_generate)

    lines: queue.Queue[str | None] = queue.Queue()

    class QueueReader:
        def readline(self):
            item = lines.get()
            return "" if item is None else item

    stdout = io.StringIO()
    srv = EngineServer(cfg, stdin=QueueReader(), stdout=stdout, stderr=io.StringIO())
    t = threading.Thread(target=srv.serve, daemon=True)
    t.start()
    lines.put(json.dumps({
        "id": "g1", "cmd": "generate", "audio_path": str(audio), "out_dir": str(out_dir),
        "nps": 3.0, "styles": ["tech"], "seed": 1, "title": "S", "artist": "A",
    }) + "\n")
    assert started.wait(timeout=5.0)
    lines.put(json.dumps({"id": "c1", "cmd": "cancel"}) + "\n")
    lines.put(json.dumps({"id": "s1", "cmd": "shutdown"}) + "\n")
    lines.put(None)
    t.join(timeout=5.0)
    msgs = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    g1_err = next(m for m in msgs if m["id"] == "g1" and m["type"] == "error")
    assert g1_err["code"] == "cancelled"
    assert freed == [True]
    assert list(out_dir.iterdir()) == []


def test_engine_pipelined_jobs_and_unicode_non_cp1250_stdio(tmp_path: Path, monkeypatch) -> None:
    cfg = _setup_engine_dir(tmp_path)
    audio = write_audio(tmp_path / "song.wav")
    out_dir = tmp_path / "CustomLevels"
    out_dir.mkdir()
    unicode_title = "Żółć – 日本語 🎵"
    msgs = _run_cmds(cfg, [
        {"id": "i1", "cmd": "info"},
        {"id": "g1", "cmd": "generate", "audio_path": str(audio), "out_dir": str(out_dir),
         "nps": 3.0, "styles": ["tech"], "seed": 1, "title": unicode_title, "artist": "Artysta – 🎤"},
        {"id": "s1", "cmd": "shutdown"},
    ])
    assert [m["id"] for m in msgs if m["type"] == "result"] == ["i1", "g1", "s1"]
    gen_res = next(m for m in msgs if m["id"] == "g1" and m["type"] == "result")
    assert gen_res["data"]["title"] == unicode_title
    assert gen_res["data"]["artist"] == "Artysta – 🎤"

    # Also verify CLI / __main__ UTF-8 reconfigure helper does not raise on TextIOWrapper
    raw_in = io.TextIOWrapper(io.BytesIO(b'{"id":"1","cmd":"shutdown"}\n'), encoding="cp1250")
    raw_out = io.TextIOWrapper(io.BytesIO(), encoding="cp1250")
    raw_in.reconfigure(encoding="utf-8", errors="replace")
    raw_out.reconfigure(encoding="utf-8", errors="replace")
    srv = EngineServer(cfg, stdin=raw_in, stdout=raw_out, stderr=io.StringIO())
    assert srv.serve() == 0
    raw_out.flush()
    assert json.loads(raw_out.buffer.getvalue().decode("utf-8").splitlines()[0])["id"] == "1"
