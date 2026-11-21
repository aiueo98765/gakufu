"""内蔵の多声採譜エンジン（torch 不要）。

1. 3 bin/半音 の CQT（定Q変換）を取り、チューニングずれを補正する。
2. 88鍵それぞれの「倍音テンプレート」を並べた辞書 W を固定し、
   KL ダイバージェンスの乗法更新 NMF で各鍵の活性度 H を推定する（V ≈ W·H）。
3. 活性度の立ち上がり（オンセット）とヒステリシス閾値で音符を切り出す。
4. 倍音・分数倍音の「幽霊音」を消す。

ピアノ独奏ならそれなりに取れる。本命の bytedance エンジンが使えない環境向け。
"""
from __future__ import annotations

from typing import Callable

import numpy as np

SR = 22050
HOP = 256
BPO = 36  # bins per octave（3 bin / 半音）
N_KEYS = 88
LOW_MIDI = 21


def _templates(n_bins: int) -> np.ndarray:
    W = np.zeros((n_bins, N_KEYS), dtype=np.float32)
    bins = np.arange(n_bins)
    for j in range(N_KEYS):
        base = j * 3
        for h in range(1, 11):
            pos = base + BPO * np.log2(h)
            if pos >= n_bins - 1:
                break
            # 高い倍音ほど弱く、ピアノ特有のわずかな高め（インハーモニシティ）を幅で吸収
            amp = 1.0 / (h ** 0.85)
            W[:, j] += amp * np.exp(-0.5 * ((bins - pos) / 0.7) ** 2)
        s = W[:, j].sum()
        if s > 0:
            W[:, j] /= s
    return W


def _nmf_fixed_w(V: np.ndarray, W: np.ndarray, iters: int = 50, lam: float = 0.08, progress=None) -> np.ndarray:
    eps = 1e-9
    T = V.shape[1]
    H = np.full((W.shape[1], T), V.mean() + eps, dtype=np.float32)
    WT = W.T.copy()
    # 低い鍵ほど倍音が密で他の音を「説明」しやすいので、疎性の罰を強くする
    midi = np.arange(N_KEYS) + LOW_MIDI
    lam_k = lam * (1.0 + 3.0 * np.clip((52 - midi) / 30, 0, 1))
    denom = WT.sum(axis=1, keepdims=True) + lam_k[:, None].astype(np.float32)
    for it in range(iters):
        WH = W @ H + eps
        H *= (WT @ (V / WH)) / denom
        if progress and it % 5 == 0:
            progress(0.35 + 0.4 * it / iters, f"倍音テンプレート分解 {it}/{iters}")
    return H


def transcribe(
    y: np.ndarray,
    sr: int = SR,
    sensitivity: float = 0.5,
    progress: Callable[[float, str], None] | None = None,
) -> list[dict]:
    import librosa
    from scipy.ndimage import median_filter

    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    if progress:
        progress(0.05, "チューニング推定")
    tuning_semitone = float(librosa.estimate_tuning(y=y, sr=sr))
    if progress:
        progress(0.1, "定Q変換 (CQT)")
    n_bins = N_KEYS * 3
    C = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=HOP,
            fmin=librosa.midi_to_hz(LOW_MIDI),
            n_bins=n_bins,
            bins_per_octave=BPO,
            tuning=tuning_semitone * 3,
        )
    ).astype(np.float32)
    C /= C.max() + 1e-9

    W = _templates(n_bins)
    H = _nmf_fixed_w(C, W, progress=progress)
    if progress:
        progress(0.8, "音符の切り出し")

    A = median_filter(H, size=(1, 3))
    A /= np.percentile(A, 99.7) + 1e-9
    L = np.log1p(30 * A)
    D = np.zeros_like(L)
    D[:, 1:] = L[:, 1:] - L[:, :-1]
    D = np.maximum(D, 0)

    # 全体のオンセット強度（どの鍵でも同時に何かが鳴り始めた瞬間）
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    oenv = oenv[: A.shape[1]]
    oenv = oenv / (np.percentile(oenv, 99) + 1e-9)

    s = float(np.clip(sensitivity, 0.0, 1.0))
    frame_thr = 0.16 - 0.10 * s
    onset_thr = 0.45 - 0.25 * s
    min_frames = 3

    T = A.shape[1]
    notes: list[dict] = []
    for j in range(N_KEYS):
        d = D[j]
        a = A[j]
        # 局所最大のオンセット候補
        cand = np.where(
            (d > onset_thr)
            & (d >= np.roll(d, 1))
            & (d >= np.roll(d, -1))
        )[0]
        last_end = -1
        for t in cand:
            if t <= last_end:
                continue
            g = oenv[max(0, t - 2) : t + 3].max() if oenv.size else 1.0
            if g < 0.08 and d[t] < onset_thr * 2:
                continue  # 全体として何も鳴り始めていない場所の小さな山は捨てる
            pk = float(a[t : min(T, t + 6)].max())
            if pk < frame_thr:
                continue
            off_level = max(0.22 * pk, frame_thr * 0.6)
            e = t + 1
            while e < T and a[e] > off_level:
                # 同じ鍵の打ち直し
                if e - t > 4 and d[e] > max(onset_thr, 0.6 * d[t]):
                    break
                e += 1
            if e - t < min_frames:
                continue
            # 打鍵の瞬間の雑音（全帯域に広がる）は持続しないので、直後に残っているか確かめる
            tail = a[t + 2 : min(T, t + 9)]
            if tail.size and float(tail.mean()) < 0.4 * pk:
                continue
            start = max(0, t - 1)
            notes.append(
                {
                    "pitch": LOW_MIDI + j,
                    "on": start * HOP / sr,
                    "off": e * HOP / sr,
                    "vel": int(np.clip(28 + 95 * np.sqrt(min(pk, 1.0)), 1, 127)),
                    "_pk": pk,
                    "_t": start,
                }
            )
            last_end = e - 1

    notes = _remove_ghosts(notes)
    for n in notes:
        n.pop("_pk", None)
        n.pop("_t", None)
    notes.sort(key=lambda n: (n["on"], n["pitch"]))
    if progress:
        progress(1.0, f"{len(notes)} 音を検出")
    return notes


def _remove_ghosts(notes: list[dict]) -> list[dict]:
    """倍音（オクターブ上・12度上・2オクターブ上）と分数倍音の誤検出を消す。"""
    by_pitch: dict[int, list[dict]] = {}
    for n in notes:
        by_pitch.setdefault(n["pitch"], []).append(n)

    def strongest_near(pitch: int, t: int) -> float:
        best = 0.0
        for m in by_pitch.get(pitch, []):
            if abs(m["_t"] - t) <= 2:
                best = max(best, m["_pk"])
        return best

    keep = []
    for n in notes:
        p, t, pk = n["pitch"], n["_t"], n["_pk"]
        ghost = False
        for interval, ratio in ((12, 2.2), (19, 2.8), (24, 3.5), (28, 4.0)):
            if strongest_near(p - interval, t) > ratio * pk:
                ghost = True
                break
        if not ghost:
            # 1オクターブ上がずっと強い同時発音なら、それは分数倍音
            up = strongest_near(p + 12, t)
            if up > 0 and pk < 0.25 * up:
                ghost = True
        if not ghost and p < 60:
            # 低音で、自分の倍音位置（8度・12度・2オクターブ）が軒並み別の強い音で埋まっているなら、
            # それらの音から作られた見かけの低音
            covered = sum(1 for k in (12, 19, 24) if strongest_near(p + k, t) > 1.2 * pk)
            if covered >= 2:
                ghost = True
        if not ghost:
            keep.append(n)
    return keep
