"""単旋律の音高追跡（歌メロ・ベースライン）。

pYIN で基本周波数を取り、ビブラートやしゃくりを吸収しながら
「同じ音の高さが続く区間」を音符に切る。声の子音や打ち直しは
オンセット検出で区切る。
"""
from __future__ import annotations

from typing import Callable

import numpy as np

SR = 16000
HOP = 320  # 20ms

PROFILES = {
    "vocals": {"fmin": 65.4, "fmax": 1200.0, "frame": 2048, "min_frames": 5, "dev": 0.75},
    "bass": {"fmin": 30.0, "fmax": 400.0, "frame": 4096, "min_frames": 4, "dev": 0.6},
}


def track(
    y: np.ndarray,
    sr: int,
    kind: str = "vocals",
    progress: Callable[[float, str], None] | None = None,
) -> list[dict]:
    import librosa

    try:
        from scipy.ndimage import median_filter
    except ImportError:  # ブラウザ版（scipy なし）
        from ..npcompat import median_filter1d

        def median_filter(x, size):
            return median_filter1d(x, size)

    prof = PROFILES[kind]
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    if y.size < sr // 2:
        return []
    if progress:
        progress(0.05, f"{'歌メロ' if kind == 'vocals' else 'ベース'}の音高追跡 (pYIN)")
    f0, voiced, vprob = librosa.pyin(
        y,
        fmin=prof["fmin"],
        fmax=prof["fmax"],
        sr=sr,
        frame_length=prof["frame"],
        hop_length=HOP,
        fill_na=np.nan,
    )
    rms = librosa.feature.rms(y=y, frame_length=prof["frame"], hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    f0, voiced, vprob, rms = f0[:n], voiced[:n], vprob[:n], rms[:n]
    if progress:
        progress(0.7, "音符への区切り")

    rms_ref = np.percentile(rms[rms > 0], 95) if np.any(rms > 0) else 1.0
    loud = rms > rms_ref * 10 ** (-32 / 20)
    midi = librosa.hz_to_midi(np.where(np.isnan(f0), 1.0, f0))
    ok = voiced & (vprob > 0.25) & loud & ~np.isnan(f0)
    smooth = midi.copy()
    smooth[ok] = median_filter(midi[ok], size=5) if ok.sum() >= 5 else midi[ok]

    onsets = set(
        librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, backtrack=True, units="frames").tolist()
    )

    def rearticulated(t: int, frames: list[int]) -> bool:
        """同じ高さのまま歌い直したか：直前に音量の谷があるときだけ区切る（ビブラートの揺れでは切らない）。"""
        if len(frames) < prof["min_frames"]:
            return False
        ref = float(np.median(rms[frames]))
        dip = float(rms[max(0, t - 3) : t + 1].min())
        return dip < 0.6 * ref

    notes: list[dict] = []
    seg: list[int] = []

    def flush():
        if len(seg) >= prof["min_frames"]:
            vals = smooth[seg]
            pitch = int(np.round(np.median(vals)))
            if 21 <= pitch <= 108:
                level = float(np.median(rms[seg])) / (rms_ref + 1e-9)
                notes.append(
                    {
                        "pitch": pitch,
                        "on": seg[0] * HOP / sr,
                        "off": (seg[-1] + 1) * HOP / sr,
                        "vel": int(np.clip(45 + 70 * np.sqrt(level), 30, 120)),
                    }
                )
        seg.clear()

    dev_run = 0
    for t in range(n):
        if not ok[t]:
            flush()
            dev_run = 0
            continue
        if seg:
            center = np.median(smooth[seg[-12:]])
            if abs(smooth[t] - center) > prof["dev"]:
                dev_run += 1
                if dev_run >= 3:
                    cut = seg[: len(seg) - (dev_run - 1)]
                    rest = seg[len(seg) - (dev_run - 1) :]
                    seg[:] = cut
                    flush()
                    seg.extend(rest)
                    dev_run = 0
            else:
                dev_run = 0
            if t in onsets and rearticulated(t, seg):
                flush()
        seg.append(t)
    flush()

    # 同じ高さで隙間がほぼ無い連続は1音にまとめる（音量の谷がある歌い直しは残す）
    merged: list[dict] = []
    for nt in notes:
        t_on = int(round(nt["on"] * sr / HOP))
        if (
            merged
            and merged[-1]["pitch"] == nt["pitch"]
            and nt["on"] - merged[-1]["off"] < 0.03
            and not (t_on in onsets and float(rms[max(0, t_on - 3) : t_on + 1].min()) < 0.6 * float(np.median(rms[max(0, t_on - 10) : t_on])) )
        ):
            merged[-1]["off"] = nt["off"]
        else:
            merged.append(nt)
    if progress:
        progress(1.0, f"{len(merged)} 音")
    return merged
