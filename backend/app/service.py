"""楽譜まわりの受け答え（PC 版の FastAPI とブラウザ版エンジンで共通）。

FastAPI に依存しない形で、入力の検査と、楽譜の取得・保存・編曲し直し・書き出しを行う。
エラーは ServiceError（HTTP の状態コード付き）で返す。
"""
from __future__ import annotations

import re

from . import pipeline
from .arrange import DEFAULT_OPTIONS, LEFT_HAND_LABEL, MODE_LABEL, MODES, rearrange_from_score
from .guitar import TUNINGS, ascii_tab, refret
from .jobs import job_dir, manager
from .midi_export import raw_to_midi, score_to_midi
from .notation import score_to_musicxml
from .separation import QUALITY

SOURCE_TYPES = ("auto", "piano", "song", "instrumental")
SEPARATIONS = ("auto", "off", *QUALITY.keys())


class ServiceError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


# --------------------------------------------------------------------------- 入力の検査
def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ServiceError(400, f"mode は {', '.join(MODES)} のいずれか")
    return mode


def check_instrument(inst: str) -> str:
    if inst not in ("piano", "guitar"):
        raise ServiceError(400, "instrument は piano か guitar")
    return inst


def clean_options(opts: dict) -> dict:
    out = {}
    for k, v in (opts or {}).items():
        if k not in DEFAULT_OPTIONS:
            continue
        if k == "tempoScale" and v not in ("auto", 1, 2, 0.5, "1", "2", "0.5"):
            continue
        if k == "bpm" and v is not None:
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if not 30 <= v <= 300:
                continue
        if k == "timeSig" and v not in ("auto", "4/4", "3/4", "2/4", "6/8", "12/8"):
            continue
        if k == "transpose":
            v = int(max(-12, min(12, int(v or 0))))
        if k == "leftHand" and v not in LEFT_HAND_LABEL:
            continue
        out[k] = v
    return out


def clean_guitar(opts: dict) -> dict:
    out = {}
    opts = opts or {}
    if opts.get("tuning") in TUNINGS:
        out["tuning"] = opts["tuning"]
    cap = opts.get("capo")
    if cap in ("auto", None, ""):
        out["capo"] = "auto"
    else:
        try:
            out["capo"] = int(max(0, min(9, int(cap))))
        except (TypeError, ValueError):
            out["capo"] = "auto"
    return out


def validate_score(score: dict) -> dict:
    if not isinstance(score, dict):
        raise ServiceError(400, "score が不正です")
    notes = score.get("notes")
    if not isinstance(notes, list) or len(notes) > 60000:
        raise ServiceError(400, "notes が不正です")
    try:
        ts = [int(x) for x in score["timeSignature"]]
        assert (len(ts) == 1 or ts[1] == 4) and ts[0] in (2, 3, 4) or (len(ts) == 2 and ts[1] == 8 and ts[0] in (6, 12))
        int(score["key"]["fifths"])
        for n in notes:
            p = int(n["pitch"])
            assert 0 <= p <= 127
            assert float(n["dur"]) > 0 and float(n["start"]) >= 0
            assert n.get("hand", "R") in ("R", "L")
        b = score.get("bars")
        if b is not None:
            assert isinstance(b, list) and len(b) >= 2 and len(b) < 20000
            assert all(float(y) - float(x) >= 0.5 for x, y in zip(b, b[1:]))
    except (KeyError, TypeError, ValueError, AssertionError) as e:
        raise ServiceError(400, f"score の形式が不正です: {e}")
    return score


def safe_filename(title: str, ext: str) -> str:
    base = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", title or "score").strip()[:80] or "score"
    return f"{base}{ext}"


def job_or_404(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{12}", job_id or ""):
        raise ServiceError(404, "ジョブが見つかりません")
    job = manager.get(job_id)
    if not job:
        raise ServiceError(404, "ジョブが見つかりません")
    return job


def require_done(job_id: str):
    job = job_or_404(job_id)
    if job.data["status"] != "done":
        raise ServiceError(409, "まだ採譜が終わっていません")
    return job


# --------------------------------------------------------------------------- 一覧に出す固定情報
def static_info() -> dict:
    from .sources import PLATFORM_LABEL

    return {
        "modes": [{"id": m, "label": MODE_LABEL[m]} for m in MODES],
        "leftHand": [{"id": k, "label": v} for k, v in LEFT_HAND_LABEL.items()],
        "separation": [{"id": k, "label": v["label"]} for k, v in QUALITY.items()],
        "tunings": [{"id": k, "label": v[0], "open": v[1]} for k, v in TUNINGS.items()],
        "platforms": PLATFORM_LABEL,
    }


# --------------------------------------------------------------------------- 楽譜
def get_score(job_id: str, mode: str, instrument: str) -> dict:
    require_done(job_id)
    check_mode(mode)
    if check_instrument(instrument) == "guitar":
        return pipeline.ensure_guitar_score(job_id, mode)
    return pipeline.ensure_score(job_id, mode)


def put_score(job_id: str, mode: str, instrument: str, score: dict) -> dict:
    require_done(job_id)
    check_mode(mode)
    score = validate_score(score)
    score["edited"] = True
    prefix = "edited_guitar_" if check_instrument(instrument) == "guitar" else "edited_"
    pipeline.save_json(job_dir(job_id) / f"{prefix}{mode}.json", score)
    return {"ok": True}


def rearrange(job_id: str, mode: str, options: dict, instrument: str, guitar: dict) -> dict:
    require_done(job_id)
    check_mode(mode)
    if check_instrument(instrument) == "guitar":
        return pipeline.ensure_guitar_score(job_id, mode, clean_options(options), clean_guitar(guitar), force=True)
    return pipeline.ensure_score(job_id, mode, clean_options(options), force=True)


def reset(job_id: str, mode: str, instrument: str) -> dict:
    require_done(job_id)
    check_mode(mode)
    if check_instrument(instrument) == "guitar":
        (job_dir(job_id) / f"edited_guitar_{mode}.json").unlink(missing_ok=True)
        return pipeline.ensure_guitar_score(job_id, mode)
    (job_dir(job_id) / f"edited_{mode}.json").unlink(missing_ok=True)
    return pipeline.ensure_score(job_id, mode)


def raw_midi(job_id: str) -> tuple[bytes, str, str]:
    job = require_done(job_id)
    raw = pipeline.load_json(job_dir(job_id) / "raw.json")
    title = job.data["meta"].get("title") or "score"
    return raw_to_midi(raw), safe_filename(title + " (AI生データ)", ".mid"), "audio/midi"


def musicxml(score: dict, options: dict | None) -> str:
    return score_to_musicxml(validate_score(score), options or {})


def fingering(score: dict) -> dict:
    """編集後の運指（ピアノ=指番号、ギター=弦とフレット）と難所を計算し直す。"""
    score = validate_score(score)
    if score.get("instrument") == "guitar":
        return refret(score)
    return rearrange_from_score(score)


def export_tab(score: dict) -> tuple[bytes, str, str]:
    score = validate_score(score)
    if score.get("instrument") != "guitar":
        raise ServiceError(400, "タブ譜はギター譜でだけ書き出せます")
    suffix = f" ({score.get('modeLabel', '')} ギター)"
    return ascii_tab(score).encode("utf-8"), safe_filename(score.get("title", "score") + suffix, ".txt"), "text/plain; charset=utf-8"


def export_midi(score: dict, tempo_scale: float = 1.0) -> tuple[bytes, str, str]:
    score = validate_score(score)
    data = score_to_midi(score, max(0.25, min(2.0, float(tempo_scale or 1.0))))
    suffix = f" ({score.get('modeLabel', '')})" if score.get("modeLabel") else ""
    return data, safe_filename(score.get("title", "score") + suffix, ".mid"), "audio/midi"


def export_musicxml(score: dict, options: dict | None) -> tuple[bytes, str, str]:
    score = validate_score(score)
    xml = score_to_musicxml(score, options or {})
    suffix = f" ({score.get('modeLabel', '')})" if score.get("modeLabel") else ""
    return xml.encode("utf-8"), safe_filename(score.get("title", "score") + suffix, ".musicxml"), "application/vnd.recordare.musicxml+xml"
