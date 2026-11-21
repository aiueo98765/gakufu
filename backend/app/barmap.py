"""小節マップ（小節ごとの長さ）の推定。

J-POP などでは 4/4 の途中に 2/4 や 3/4 の小節が1つ挟まることがよくある。
小節の長さを曲全体で固定すると、その後ろの小節線がすべて2拍ずれる。

拍ごとに「小節の何拍目か」を隠れマルコフモデルで追う:
  状態    : 小節内の位置 0..P-1（P = 拍子の拍数）
  遷移    : 次の拍へ（0）/ 小節の途中で次の小節を始める＝短い小節（罰 COSTS）
  放出    : 小節頭らしさ（低音・長い音・和音の変わり目・Beat This! の小節頭）× 位置ごとの重み
短い小節は「はっきりした根拠が数小節続くときだけ」入るよう罰を重くしてある。
"""
from __future__ import annotations

import numpy as np

# 短い小節を入れる罰（拍数ごと）。4/4 の中の 2/4 はよくあるが、3/4 はまれなので重く
COSTS = {2: 8.0, 3: 14.0}
SHORT_LENGTHS = {4: (2, 3), 3: (2,), 2: ()}


def _template(P: int) -> np.ndarray:
    t = {4: [1.0, -0.45, 0.25, -0.45], 3: [1.0, -0.5, -0.5], 2: [1.0, -0.6]}.get(P, [1.0] + [-1.0 / max(1, P - 1)] * (P - 1))
    v = np.asarray(t, dtype=float)
    return v - v.mean()


def downbeat_evidence(beats: np.ndarray, notes: list[dict], dnn_downbeats: np.ndarray | None) -> np.ndarray:
    """拍ごとの小節頭らしさ（z 値の合成）。"""
    from .rhythm import _accent

    acc = _accent(beats, notes)
    if dnn_downbeats is not None and dnn_downbeats.size:
        ibi = float(np.median(np.diff(beats)))
        ev = np.zeros(beats.size)
        for t in dnn_downbeats:
            i = int(np.argmin(np.abs(beats - t)))
            if abs(beats[i] - t) < 0.3 * ibi:
                ev[i] += 1.0
        if ev.std() > 0:
            acc = acc + 1.2 * (ev - ev.mean()) / ev.std()
    return acc


def bar_starts(evidence: np.ndarray, P: int, phase: int, allow_irregular: bool = True) -> list[int]:
    """小節頭になる拍の番号の列を返す。phase は全体最適の位相（初期状態の手がかり）。"""
    n = evidence.size
    if n == 0:
        return [0]
    tpl = _template(P)
    shorts = SHORT_LENGTHS.get(P, ()) if allow_irregular else ()
    # 状態 s = 小節内の位置。初期状態は phase に合わせて少しだけ優遇
    init = np.full(P, -1.0)
    init[(P - phase) % P] = 0.0
    score = init + tpl * evidence[0]
    back = np.zeros((n, P), dtype=np.int16)
    for i in range(1, n):
        new = np.full(P, -np.inf)
        arg = np.zeros(P, dtype=np.int16)
        for s in range(P):
            # 通常: s-1 → s
            prev = (s - 1) % P
            cand = score[prev]
            src = prev
            if s == 0:
                # 短い小節: 位置 L-1 で終わって次の小節頭へ
                for L in shorts:
                    c = score[L - 1] - COSTS.get(L, 8.0)
                    if c > cand:
                        cand, src = c, L - 1
            new[s] = cand + tpl[s] * evidence[i]
            arg[s] = src
        score = new
        back[i] = arg
    s = int(np.argmax(score))
    path = np.zeros(n, dtype=np.int16)
    for i in range(n - 1, -1, -1):
        path[i] = s
        s = back[i, s]
    starts = [i for i in range(n) if path[i] == 0]
    if not starts or starts[0] != 0:
        # 先頭は弱起（途中から始まる小節）として 0 を入れる
        starts = [0] + starts
    return starts
