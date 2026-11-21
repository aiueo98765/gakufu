"""深層学習による拍・小節頭（ダウンビート）推定。

Beat This!（Foscarin, Schlüter, Widmer, ISMIR 2024 / CPJKU, JKU Linz）を使う。
メルスペクトログラムを入力にした Transformer で、拍と小節頭を同時に推定する。
ポップス・ロック・クラシックのピアノ演奏を含む大規模データで学習されており、
打楽器の無いピアノ独奏でも拍と小節頭が取れる。

重み（約80MB）は初回に作者の配布元から取得し、torch のキャッシュに置く。
"""
from __future__ import annotations

import threading

import numpy as np

from .config import torch_device

_lock = threading.Lock()
_model = None


def available() -> bool:
    try:
        import beat_this.inference  # noqa: F401

        return True
    except Exception:
        return False


def _get():
    global _model
    if _model is None:
        from beat_this.inference import Audio2Beats

        dev = torch_device()
        _model = Audio2Beats(checkpoint_path="final0", device=dev, float16=False, dbn=False)
    return _model


def track(y: np.ndarray, sr: int) -> dict:
    """モノラル音声 → {"beats": 秒の配列, "downbeats": 秒の配列}"""
    with _lock:
        model = _get()
        beats, downbeats = model(y.astype(np.float32), sr)
    return {"beats": np.asarray(beats, dtype=float), "downbeats": np.asarray(downbeats, dtype=float)}


def meter_from_downbeats(beats: np.ndarray, downbeats: np.ndarray) -> dict:
    """小節頭の間に拍がいくつあるか（最頻値）と、小節頭が拍のどの位置にあるか。"""
    if beats.size < 4 or downbeats.size < 2:
        return {"perBar": None, "phase": 0, "consistency": 0.0}
    idx = np.array([int(np.argmin(np.abs(beats - d))) for d in downbeats])
    gaps = np.diff(idx)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return {"perBar": None, "phase": 0, "consistency": 0.0}
    vals, counts = np.unique(gaps, return_counts=True)
    per = int(vals[np.argmax(counts)])
    consistency = float(counts.max() / gaps.size)
    phase = int(np.bincount(idx % per).argmax()) if per > 0 else 0
    return {"perBar": per, "phase": phase, "consistency": round(consistency, 3), "downbeatIdx": idx.tolist()}
