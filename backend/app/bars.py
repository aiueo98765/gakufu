"""小節マップ（各小節の開始位置）の共通処理。

楽譜データの "bars" は小節の開始位置（4分音符単位）の昇順の列で、最後の要素は曲の終わり。
小節 i は [bars[i], bars[i+1]) 。途中に 2/4 や 3/4 の短い小節があってもよい。
"bars" が無い古いデータは measureBeats ごとの一定長として扱う。
"""
from __future__ import annotations

import bisect
import math


def regular(total: float, mb: float) -> list[float]:
    n = max(1, math.ceil(total / mb - 1e-9))
    return [round(i * mb, 6) for i in range(n + 1)]


def normalize(starts: list[float], total: float, mb: float) -> list[float]:
    """開始位置の列を整える：0 から始め、重複・近すぎる区切りを除き、曲の終わりまで一定長で延ばす。"""
    out: list[float] = [0.0]
    for s in sorted(starts):
        s = round(float(s), 6)
        if s - out[-1] >= 1 - 1e-6:  # 1拍未満の小節は作らない
            out.append(s)
    while out[-1] < total - 1e-6:
        out.append(round(out[-1] + mb, 6))
    # 最後の要素を「曲の終わり」にする（終わりが小節の途中ならその小節の終わりまで）
    if out[-1] - total > 1e-6 and len(out) >= 2 and out[-2] >= total - 1e-6:
        out.pop()
    if len(out) == 1:
        out.append(round(mb, 6))
    return out


def of_score(score: dict) -> list[float]:
    mb = float(score.get("measureBeats") or 4)
    total = float(score.get("totalBeats") or mb)
    end = max([total] + [n["start"] + n["dur"] for n in score.get("notes", [])])
    b = score.get("bars")
    if not b or len(b) < 2:
        return regular(end, mb)
    b = [float(x) for x in b]
    while b[-1] < end - 1e-6:
        b.append(round(b[-1] + mb, 6))
    return b


def index(pos: float, bars: list[float]) -> int:
    i = bisect.bisect_right(bars, pos + 1e-9) - 1
    return max(0, min(i, len(bars) - 2))


def count(bars: list[float]) -> int:
    return max(1, len(bars) - 1)


def length(i: int, bars: list[float]) -> float:
    return bars[i + 1] - bars[i]


def is_bar_start(pos: float, bars: list[float]) -> bool:
    i = bisect.bisect_left(bars, pos - 1e-6)
    return i < len(bars) and abs(bars[i] - pos) < 1e-6


def strong(pos: float, bars: list[float]) -> bool:
    """小節頭、または4拍以上の小節の真ん中（4/4 の3拍目）。"""
    i = index(pos, bars)
    s, L = bars[i], length(i, bars)
    rel = pos - s
    return abs(rel) < 1e-6 or (L >= 4 - 1e-6 and abs(rel - L / 2) < 1e-6)
