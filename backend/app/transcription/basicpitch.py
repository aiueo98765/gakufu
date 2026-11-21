"""Basic Pitch（Spotify, ICASSP 2022）による楽器を問わない多声採譜のアダプタ。

ピアノ専用の ByteDance モデルは、ギター・合唱・シンセなどピアノ以外の音色では
音の取り違えが増える。Basic Pitch は楽器を問わず学習された軽量モデルで、
ギターの独奏・アカペラのハモリ・バンドの伴奏パートに向く。
TensorFlow は Python 3.13 に無いので、同梱の ONNX モデルを ONNX Runtime で動かす。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

_lock = threading.Lock()
_model = None


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        prev = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        try:
            import basic_pitch.inference  # noqa: F401
        finally:
            logging.getLogger().setLevel(prev)
        return True
    except Exception:
        return False


def _get():
    global _model
    if _model is None:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model

        _model = Model(ICASSP_2022_MODEL_PATH)
    return _model


# 楽器ごとの閾値（オンセットの確からしさ・持続の確からしさ・最短の長さ[ms]・音域[Hz]）
PROFILES = {
    "guitar": {"onset": 0.5, "frame": 0.3, "min_ms": 58, "fmin": 70.0, "fmax": 1400.0},
    "voices": {"onset": 0.55, "frame": 0.35, "min_ms": 90, "fmin": 80.0, "fmax": 1200.0},
    "ensemble": {"onset": 0.5, "frame": 0.3, "min_ms": 58, "fmin": 27.5, "fmax": 4200.0},
}


def transcribe(
    wav_path: Path,
    profile: str = "ensemble",
    progress: Callable[[float, str], None] | None = None,
) -> list[dict]:
    """音声ファイル → [{"pitch", "on", "off", "vel"}]（秒）"""
    from basic_pitch.inference import predict

    p = PROFILES.get(profile, PROFILES["ensemble"])
    if progress:
        progress(0.1, "Basic Pitch（楽器を問わない採譜AI）で推論中")
    with _lock:
        _out, _midi, events = predict(
            str(wav_path),
            _get(),
            onset_threshold=p["onset"],
            frame_threshold=p["frame"],
            minimum_note_length=p["min_ms"],
            minimum_frequency=p["fmin"],
            maximum_frequency=p["fmax"],
            multiple_pitch_bends=False,
            melodia_trick=True,
        )
    notes = events_to_notes(events)
    if progress:
        progress(1.0, f"{len(notes)} 音を検出")
    return notes


def events_to_notes(events) -> list[dict]:
    """(開始秒, 終了秒, MIDI, 強さ0-1, …) の列 → アプリの音符形式（ブラウザ版と共通）。"""
    notes = []
    for ev in events:
        on, off, pitch, amp = float(ev[0]), float(ev[1]), int(ev[2]), float(ev[3])
        if off <= on or not 21 <= pitch <= 108:
            continue
        notes.append({"pitch": pitch, "on": on, "off": off, "vel": int(max(1, min(127, 25 + 102 * amp)))})
    notes.sort(key=lambda n: (n["on"], n["pitch"]))
    return notes
