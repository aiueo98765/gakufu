"""ブラウザ版エンジン（Pyodide の中で動く Python）の窓口。

AI の推論（音源分離・採譜・拍）はブラウザの JS（onnxruntime-web / WebGPU）が行い、
それ以外（音符への変換・拍の精密化・編曲・五線譜・MIDI・タブ譜）は PC 版と同じ
Python のコードをここから呼ぶ。データは Pyodide の仮想ファイルシステム（IndexedDB に保存）の
/data/jobs/<id>/ に、PC 版と同じ形で置く。

JS からの受け渡しは JSON 文字列と Float32Array。
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
from urllib.parse import parse_qs

import numpy as np

from . import minilibrosa

# librosa は numba が要るので Pyodide では動かない。同じ計算をする numpy 版に差し替える。
sys.modules.setdefault("librosa", minilibrosa)

from . import pipeline, separation, service  # noqa: E402
from .jobs import STEPS, job_dir, manager  # noqa: E402
from .service import ServiceError  # noqa: E402
from .transcription import basicpitch, bd_post, bp_notes, bytedance, pitchtrack  # noqa: E402

# 推論は JS 側で行うので、採譜 AI は「使える」扱いにする（choose_engine の判定に使われる）
bytedance.available = lambda: True
bytedance.checkpoint_ready = lambda: True
basicpitch.available = lambda: True

SERVER_STOPPED = "サーバーの再起動で処理が中断されました。もう一度実行してください。"
PAGE_CLOSED = "処理中にページが閉じられた（または再読み込みされた）ため中断しました。もう一度実行してください。"

# 起動時、「処理中」のまま残っていた曲は中断扱いになる（jobs.JobManager._load）。
# ただし別のタブがまだ処理している曲（GAKUFU_LIVE_JOBS）は、そのまま処理中として扱う。
_LIVE = {x for x in os.environ.get("GAKUFU_LIVE_JOBS", "").split(",") if x}
BOOT_FIXED: list[str] = []
for _job in list(manager.jobs.values()):
    if _job.data.get("error") == SERVER_STOPPED:
        if _job.id in _LIVE:
            _job.data["status"] = "running"
            _job.data["error"] = None
            continue
        _job.data["error"] = PAGE_CLOSED
        _job.data["message"] = "中断"
        manager._save(_job, force=True)
        BOOT_FIXED.append(_job.id)


def boot_fixed() -> str:
    """起動時に中断扱いへ書き換えた曲（JS がこの分を保存し直す）"""
    return _j(BOOT_FIXED)


def refresh(jid: str) -> None:
    """別のタブが書き換えた曲を、ファイルから読み直す。"""
    from .jobs import Job

    p = job_dir(jid) / "job.json"
    if not p.exists():
        manager.jobs.pop(jid, None)
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return
    job = manager.jobs.get(jid)
    if job:
        job.data = data
    else:
        manager.jobs[jid] = Job(data)


def refresh_all(ids_json: str, keep_json: str = "[]") -> None:
    """別のタブの変更を一覧に反映する。keep（このタブが処理中・作成直後・保存待ちの曲）には触らない。"""
    ids = set(json.loads(ids_json))
    keep = set(json.loads(keep_json or "[]"))
    for jid in list(manager.jobs):
        if jid not in ids and jid not in keep:
            manager.jobs.pop(jid, None)
    for jid in ids - keep:
        refresh(jid)


def _job(jid: str):
    """処理中の曲。見つからなければ（別のタブで消されたなど）はっきりエラーにする。"""
    job = manager.get(jid)
    if not job:
        raise RuntimeError("この曲のデータが見つからなくなりました（別のタブで削除された可能性があります）")
    return job


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def _f32(a) -> np.ndarray:
    if hasattr(a, "to_py"):
        a = a.to_py()
    return np.asarray(a, dtype=np.float32)


# --------------------------------------------------------------------------- API（main.py と同じ受け答え）
def handle(method: str, path: str, query: str, body: str) -> str:
    """戻り値（JSON 文字列）: {"status", "json"} / {"status", "text", "type"} / {"status", "b64", "filename", "type"}"""
    q = {k: v[-1] for k, v in parse_qs(query or "").items()}
    try:
        data = json.loads(body) if body else {}
    except ValueError:
        return _j({"status": 400, "json": {"detail": "本文が JSON ではありません"}})
    try:
        return _j(_route(method.upper(), path, q, data))
    except ServiceError as e:
        return _j({"status": e.status, "json": {"detail": e.detail}})
    except Exception as e:  # 想定外のエラーも画面に出す
        import traceback

        traceback.print_exc()
        return _j({"status": 500, "json": {"detail": f"{type(e).__name__}: {e}"}})


def _file(t: tuple[bytes, str, str]) -> dict:
    data, filename, media = t
    return {"status": 200, "b64": base64.b64encode(data).decode("ascii"), "filename": filename, "type": media}


def _route(method: str, path: str, q: dict, data: dict) -> dict:
    mode = q.get("mode", "standard")
    inst = q.get("instrument", "piano")
    if path == "/api/static" and method == "GET":
        return {"status": 200, "json": service.static_info()}
    if path == "/api/jobs" and method == "GET":
        return {"status": 200, "json": manager.list()}
    if path == "/api/notation/musicxml" and method == "POST":
        return {"status": 200, "text": service.musicxml(data.get("score"), data.get("options")), "type": "application/vnd.recordare.musicxml+xml"}
    if path == "/api/notation/fingering" and method == "POST":
        return {"status": 200, "json": service.fingering(data.get("score"))}
    if path == "/api/export/tab" and method == "POST":
        return _file(service.export_tab(data.get("score")))
    if path == "/api/export/midi" and method == "POST":
        return _file(service.export_midi(data.get("score"), data.get("tempoScale", 1.0)))
    if path == "/api/export/musicxml" and method == "POST":
        return _file(service.export_musicxml(data.get("score"), data.get("options")))
    m = re.fullmatch(r"/api/jobs/([0-9a-f]{12})(/[\w/-]*)?", path)
    if m:
        jid, sub = m.group(1), (m.group(2) or "")
        if sub == "" and method == "GET":
            return {"status": 200, "json": service.job_or_404(jid).public()}
        if sub == "" and method == "DELETE":
            service.job_or_404(jid)
            return {"status": 200, "json": {"ok": manager.delete(jid)}}
        if sub == "/score" and method == "GET":
            return {"status": 200, "json": service.get_score(jid, mode, inst)}
        if sub == "/score" and method == "PUT":
            return {"status": 200, "json": service.put_score(jid, mode, inst, data.get("score"))}
        if sub == "/arrange" and method == "POST":
            return {
                "status": 200,
                "json": service.rearrange(jid, data.get("mode", "standard"), data.get("options") or {}, data.get("instrument", "piano"), data.get("guitar") or {}),
            }
        if sub == "/reset" and method == "POST":
            return {"status": 200, "json": service.reset(jid, mode, inst)}
        if sub == "/export/raw-midi" and method == "GET":
            return _file(service.raw_midi(jid))
    raise ServiceError(404, f"{method} {path} はありません")


# --------------------------------------------------------------------------- ジョブの状態（JS の処理役から呼ぶ）
def job_create(req_json: str, meta_json: str) -> str:
    job = manager.create(json.loads(req_json), json.loads(meta_json))
    manager.update(job, status="running", force=True)
    return _j(job.public())


def job_get(jid: str) -> str:
    job = manager.get(jid)
    return _j(job.public() if job else None)


def job_update(jid: str, patch_json: str) -> None:
    manager.update(_job(jid), force=True, **json.loads(patch_json))


def job_meta(jid: str, meta_json: str) -> None:
    job = _job(jid)
    manager.update(job, meta={**job.data.get("meta", {}), **json.loads(meta_json)}, force=True)


def job_step(jid: str, key: str, state: str, detail: str = "") -> None:
    manager.step(_job(jid), key, state, detail)


def job_progress(jid: str, key: str, frac: float, msg: str = "") -> None:
    job = _job(jid)
    with manager.lock:
        manager._recalc(job, key, frac)
        if msg:
            job.data["message"] = msg
            job.data["steps"][key]["detail"] = msg
    manager._save(job)


def job_error(jid: str, msg: str) -> None:
    job = manager.get(jid)
    if not job:
        return
    for k, _ in STEPS:
        if job.data["steps"][k]["state"] == "running":
            job.data["steps"][k]["state"] = "error"
    manager.log(job, msg)
    manager.update(job, status="error", error=msg, message="エラー", force=True)


def job_canceled(jid: str) -> None:
    job = manager.get(jid)
    if job:
        manager.update(job, status="canceled", message="キャンセルしました", force=True)


def job_retranscribe_start(jid: str, engine: str) -> str:
    """やり直しの受付（main.py の /retranscribe と同じ検査と状態の更新）。"""
    job = service.job_or_404(jid)
    if job.data["status"] in ("queued", "running"):
        raise_status = {"status": 409, "detail": "処理中です"}
        return _j({"error": raise_status})
    if engine not in ("auto", "bytedance", "basicpitch"):
        return _j({"error": {"status": 400, "detail": "engine は auto / bytedance / basicpitch"}})
    if not (job_dir(jid) / "raw.json").exists():
        return _j({"error": {"status": 409, "detail": "まだ採譜が終わっていないジョブです"}})
    req = dict(job.data["request"], engine=engine)
    steps = job.data["steps"]
    for k in ("transcribe", "analyze", "arrange"):
        steps[k]["state"] = "pending"
        steps[k]["detail"] = ""
    manager.update(job, request=req, status="running", message="採譜をやり直します", error=None, steps=steps, force=True)
    raw = pipeline.load_json(job_dir(jid) / "raw.json")
    analysis = pipeline.load_json(job_dir(jid) / "analysis.json")
    return _j({"job": job.public(), "kind": raw.get("sourceKind", "piano"), "ratios": analysis.get("ratios") or {}, "target": raw.get("target")})


# --------------------------------------------------------------------------- 採譜の前後（PC 版の pipeline と同じ判定）
def plan(energies_json: str, source_req: str, engine_req: str) -> str:
    """分離した音のエネルギー → 曲の種類・採譜にかける音・使う採譜 AI。"""
    energies = json.loads(energies_json) if energies_json else None
    if energies:
        cls = separation.classify_source(energies)
        names = list(energies)
    else:
        cls = {"kind": "piano", "ratios": {}, "drums": 0}
        names = []
    kind = pipeline.source_kind(cls["kind"], names, source_req)
    use, label = pipeline.target_plan(kind, names, cls["ratios"])
    engine_id, profile = pipeline.choose_engine(kind, cls["ratios"], engine_req)
    song = kind == "song" and "vocals" in names
    ratio_txt = " / ".join(f"{k}:{v * 100:.0f}%" for k, v in cls["ratios"].items())
    return _j(
        {
            "kind": kind,
            "kindLabel": pipeline.KIND_LABEL.get(kind, kind),
            "ratios": cls["ratios"],
            "drums": cls.get("drums", 0),
            "ratioText": ratio_txt,
            "use": use,
            "label": label,
            "engineId": engine_id,
            "profile": profile,
            "song": song,
            "bass": song and "bass" in names and cls["ratios"].get("bass", 0) > 0.01,
        }
    )


def retranscribe_plan(kind: str, ratios_json: str, stem_names_json: str, engine_req: str) -> str:
    ratios = json.loads(ratios_json)
    names = json.loads(stem_names_json)
    engine_id, profile = pipeline.choose_engine(kind, ratios, engine_req)
    song = kind == "song" and "vocals" in names
    return _j({"engineId": engine_id, "profile": profile, "song": song, "bass": song and "bass" in names and ratios.get("bass", 0) > 0.01})


def bytedance_notes(T: int, onset, offset, frame, velocity, ped_onset, ped_offset, ped_frame) -> str:
    out = {
        "reg_onset_output": _f32(onset).reshape(T, 88),
        "reg_offset_output": _f32(offset).reshape(T, 88),
        "frame_output": _f32(frame).reshape(T, 88),
        "velocity_output": _f32(velocity).reshape(T, 88),
        "reg_pedal_onset_output": _f32(ped_onset).reshape(T),
        "reg_pedal_offset_output": _f32(ped_offset).reshape(T),
        "pedal_frame_output": _f32(ped_frame).reshape(T),
    }
    note_events, pedal_events = bd_post.to_events(out)
    notes, pedals = bytedance.events_to_notes(note_events, pedal_events)
    return _j({"notes": notes, "pedals": pedals})


def basicpitch_notes(n_windows: int, audio_len: int, note, onset, profile: str) -> str:
    p = basicpitch.PROFILES.get(profile, basicpitch.PROFILES["ensemble"])
    nt = bp_notes.unwrap_output(_f32(note).reshape(n_windows, bp_notes.ANNOT_N_FRAMES, 88), audio_len)
    on = bp_notes.unwrap_output(_f32(onset).reshape(n_windows, bp_notes.ANNOT_N_FRAMES, 88), audio_len)
    events = bp_notes.model_output_to_events(nt, on, nt.shape[0], p["onset"], p["frame"], p["min_ms"], p["fmin"], p["fmax"])
    return _j(basicpitch.events_to_notes(events))


def track_part(y16k, kind: str) -> str:
    return _j(pitchtrack.track(_f32(y16k), pitchtrack.SR, kind))


def assemble_notes(engine_id: str, kind: str, main_json: str, melody_json: str, bass_json: str) -> str:
    """transcribe_target の後半と同じ：役割（piano / harmony / melody / bass）と表示名を付ける。"""
    notes = json.loads(main_json)
    engine = pipeline.ENGINE_LABEL[engine_id]
    role = "harmony" if kind == "song" else "piano"
    for n in notes:
        n["role"] = role
    melody = json.loads(melody_json) if melody_json else None
    if melody is not None:
        for n in melody:
            n["role"] = "melody"
        notes += melody
        bass = json.loads(bass_json) if bass_json else []
        for n in bass:
            n["role"] = "bass"
        notes += bass
        engine += " ＋ pYIN（歌メロ・ベース）"
    return _j({"notes": notes, "engine": engine})


def save_raw(jid: str, notes_json: str, pedals_json: str, engine: str, engine_id: str, kind: str, target_label: str) -> str:
    raw = pipeline.save_raw(job_dir(jid), json.loads(notes_json), json.loads(pedals_json), engine, engine_id, kind, target_label)
    return pipeline.transcribe_detail(raw)


def analyze(jid: str, y22050, kind: str, ratios_json: str, dnn_json: str) -> str:
    d = job_dir(jid)
    raw = pipeline.load_json(d / "raw.json")
    analysis = pipeline.analyze(d, _f32(y22050), 22050, raw["notes"], kind, json.loads(ratios_json), json.loads(dnn_json) if dnn_json else None)
    return _j({"detail": pipeline.analyze_detail(analysis)})


def arrange_first(jid: str) -> str:
    job = _job(jid)
    d = job_dir(jid)
    req = job.data["request"]
    mode = req.get("mode", "standard")
    raw = pipeline.load_json(d / "raw.json")
    analysis = pipeline.load_json(d / "analysis.json")
    score = pipeline.first_arrange(d, raw, analysis, mode, req.get("options") or {}, job.data["meta"])
    return _j({"detail": pipeline.arrange_detail(score), "analysis": {"key": analysis["key"], "audioBpm": analysis.get("audioBpm"), "bpm": analysis["bpm"]}})


def job_finish(jid: str, sep_json: str, kind: str, engine: str, engine_id: str, target_label: str) -> None:
    job = _job(jid)
    d = job_dir(jid)
    analysis = pipeline.load_json(d / "analysis.json")
    mode = job.data["request"].get("mode", "standard")
    result = pipeline.job_result(mode, kind, engine, engine_id, json.loads(sep_json), target_label, analysis)
    manager.update(job, status="done", progress=1.0, message="完成", result=result, force=True)


def finish_retranscribe(jid: str, notes_json: str, pedals_json: str, engine: str, engine_id: str, kind: str, target_label: str) -> None:
    job = _job(jid)
    d = job_dir(jid)
    raw = pipeline.save_raw(d, json.loads(notes_json), json.loads(pedals_json), engine, engine_id, kind, target_label)
    manager.step(job, "transcribe", "done", pipeline.transcribe_detail(raw))
    analysis = pipeline.load_json(d / "analysis.json")
    pipeline.finish_retranscribe(job, manager, raw, analysis)
