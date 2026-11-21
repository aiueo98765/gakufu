"""採譜結果の雑音（実際には弾かれていない音）を取り除く。

AI 採譜が出しがちな誤検出:
  - 打ち直しの二重検出（同じ鍵が 60ms 以内にもう一度）
  - 倍音の聴き違い（強い音の 12度・2オクターブ・17度…上に、弱い同時発音）
  - 残響・共鳴・録音の雑音から来る、まわりより明らかに弱くて短い音
  - 前後に何もない、ぽつんと短い音
  - 鍵盤の端の、弱い音
強さ（ベロシティ）は録音の音量で全体が上下するので、絶対値ではなく
「前後2秒の音の強さの中央値」との差で判定する。
超プロでも雑音は残さない（弾かれた音は削らない）。モードで厳しさだけを変える。
"""
from __future__ import annotations

import numpy as np

# モードごとの厳しさ: (まわりより弱い音を消す比, 短い音とみなす秒, 孤立判定の秒)
LEVEL = {
    "virtuoso": (0.55, 0.06, 0.25),
    "standard": (0.62, 0.08, 0.3),
    "easy": (0.68, 0.10, 0.35),
}
HARMONIC_INTERVALS = (19, 24, 28, 31, 34, 36)


def _local_ref(on: np.ndarray, vel: np.ndarray, win: float = 2.0) -> np.ndarray:
    order = np.argsort(on)
    s_on, s_vel = on[order], vel[order]
    ref = np.empty_like(vel, dtype=float)
    lo = hi = 0
    out = np.empty(on.size)
    for k, t in enumerate(s_on):
        while s_on[lo] < t - win:
            lo += 1
        while hi < s_on.size and s_on[hi] <= t + win:
            hi += 1
        out[k] = float(np.median(s_vel[lo:hi]))
    ref[order] = out
    return ref


def clean(notes: list[dict], mode: str = "virtuoso") -> tuple[list[dict], dict]:
    """雑音を除いた音符と、理由ごとの除去数を返す。role=melody/bass（単旋律追跡の結果）は対象外。"""
    stats = {"retrigger": 0, "harmonic": 0, "weak": 0, "isolated": 0, "edge": 0}
    target = [dict(n) for n in notes if n.get("role") not in ("melody", "bass")]
    others = [n for n in notes if n.get("role") in ("melody", "bass")]
    if len(target) < 8:
        return notes, stats
    ratio, short, iso = LEVEL.get(mode, LEVEL["virtuoso"])

    # 1. 打ち直しの二重検出をまとめる
    target.sort(key=lambda n: (n["pitch"], n["on"]))
    merged: list[dict] = []
    for n in target:
        p = merged[-1] if merged else None
        if p and p["pitch"] == n["pitch"] and n["on"] - p["on"] < 0.06:
            p["off"] = max(p["off"], n["off"])
            p["vel"] = max(p["vel"], n["vel"])
            stats["retrigger"] += 1
            continue
        merged.append(n)
    target = sorted(merged, key=lambda n: (n["on"], n["pitch"]))

    on = np.array([n["on"] for n in target])
    vel = np.array([float(n.get("vel", 80)) for n in target])
    dur = np.array([n["off"] - n["on"] for n in target])
    pit = np.array([n["pitch"] for n in target])
    ref = _local_ref(on, vel)
    drop = np.zeros(len(target), dtype=bool)

    # 2. 倍音の聴き違い
    j0 = 0
    for i in range(len(target)):
        while on[j0] < on[i] - 0.04:
            j0 += 1
        j = j0
        while j < len(target) and on[j] <= on[i] + 0.04:
            if j != i:
                iv = pit[i] - pit[j]
                if iv in HARMONIC_INTERVALS and vel[i] <= vel[j] - 8:
                    drop[i] = True
                elif iv == 12 and vel[i] <= vel[j] - 25 and dur[i] <= dur[j]:
                    drop[i] = True
            j += 1
    stats["harmonic"] = int(drop.sum())

    # 3. まわりより明らかに弱い音（短いほど厳しく）
    weak = (vel < ref * ratio) | ((dur < short) & (vel < ref * (ratio + 0.15)))
    stats["weak"] = int((weak & ~drop).sum())
    drop |= weak

    # 4. 前後に何もない、ぽつんと短い音
    for i in range(len(target)):
        if drop[i] or dur[i] >= 0.12:
            continue
        near = np.abs(on - on[i]) < iso
        near[i] = False
        if not near[~drop].any() and vel[i] < ref[i] * 1.1:
            drop[i] = True
            stats["isolated"] += 1

    # 5. 鍵盤の両端の弱い音
    edge = ((pit < 28) | (pit > 100)) & (vel < ref - 5) & ~drop
    stats["edge"] = int(edge.sum())
    drop |= edge

    kept = [n for n, d in zip(target, drop) if not d] + others
    kept.sort(key=lambda n: (n["on"], n["pitch"]))
    stats["removed"] = int(sum(stats[k] for k in ("retrigger", "harmonic", "weak", "isolated", "edge")))
    stats["kept"] = len(kept)
    return kept, stats
