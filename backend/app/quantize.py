"""拍ごとに最適な分割を選んで量子化する。

速いパッセージや連符を正しく譜面にするため、1拍ごと・片手ごとに
「ずれの二乗和 + 細かい分割ほど高くなる複雑さペナルティ + 別の音が同じ点に潰れる罰」
が最小の分割を選ぶ。

- 単純拍子（拍=4分音符）: 2/4/8 分割、3/6 は3連符・6連符
- 複合拍子（拍=付点4分音符）: 3/6/12 分割（8分・16分・32分）、2/4 は2連符・4連符
位置はすべて4分音符=1 の単位で、beat_len（1 または 1.5）が1拍の長さ。
"""
from __future__ import annotations

import math

PENALTY_SIMPLE = {1: 0.0, 2: 0.0015, 4: 0.004, 8: 0.012, 3: 0.007, 6: 0.016}
PENALTY_COMPOUND = {1: 0.0, 3: 0.0015, 6: 0.005, 12: 0.014, 2: 0.010, 4: 0.018}
LOOKAHEAD = 1 / 16  # 拍頭の直前に来た音は次の拍として扱う（拍の割合）


def _beat_of(x: float, beat_len: float) -> int:
    return int(math.floor(x / beat_len + LOOKAHEAD))


MIN_UNIT_SEC = 0.075  # これより短い音価（速い曲の32分など）は、人の演奏と採譜の揺れに埋もれるので選びにくくする


def choose_subdivisions(
    notes: list[dict], allowed: list[int], beat_len: float = 1.0, beat_sec: float | None = None
) -> dict[int, int]:
    pen = PENALTY_COMPOUND if beat_len != 1.0 else PENALTY_SIMPLE
    by_beat: dict[int, list[float]] = {}
    for n in notes:
        b = _beat_of(n["start"], beat_len)
        by_beat.setdefault(b, []).append(n["start"] / beat_len - b)
    chosen: dict[int, int] = {}
    for b, fr in by_beat.items():
        fr = sorted(fr)
        best = None
        for S in allowed:
            err = 0.0
            slots: dict[int, float] = {}
            collide = 0.0
            for f in fr:
                k = round(f * S)
                err += (f - k / S) ** 2
                if k in slots and abs(slots[k] - f) > 0.07:
                    collide += 1
                slots.setdefault(k, f)
            p = pen.get(S, 0.02)
            if beat_sec:
                unit = beat_sec / S
                if unit < MIN_UNIT_SEC:
                    p += 0.012 * ((MIN_UNIT_SEC / unit) ** 2 - 1)
            cost = err + p * (1 + 0.25 * len(fr)) + 0.03 * collide
            if best is None or cost < best[0]:
                best = (cost, S)
        chosen[b] = best[1]
    return chosen


def quantize(
    notes: list[dict],
    allowed: list[int],
    default_sub: int,
    min_units: int = 1,
    beat_len: float = 1.0,
    beat_sec: float | None = None,
) -> list[dict]:
    """notes(start/end は4分音符単位) を量子化した新しいリストを返す。orig_start を残す。"""
    if not notes:
        return []
    subs = choose_subdivisions(notes, allowed, beat_len, beat_sec)

    def q(x: float) -> float:
        b = _beat_of(x, beat_len)
        S = subs.get(b, default_sub)
        f = x / beat_len - b
        return (b + round(f * S) / S) * beat_len

    out = []
    for n in notes:
        b = _beat_of(n["start"], beat_len)
        S = subs.get(b, default_sub)
        s = q(n["start"])
        e = q(n["end"])
        unit = beat_len / S
        if e < s + unit * min_units - 1e-9:
            e = s + unit * min_units
        m = dict(n)
        m["orig_start"] = n.get("orig_start", n["start"])
        m["start"] = round(s, 6)
        m["end"] = round(e, 6)
        out.append(m)
    return out


def cleanup(notes: list[dict]) -> list[dict]:
    """同じ手・同じ高さの重なりを解消し、完全な重複を消す。"""
    out: list[dict] = []
    groups: dict[tuple[str, int], list[dict]] = {}
    for n in notes:
        groups.setdefault((n.get("hand", "R"), n["pitch"]), []).append(n)
    for (_h, _p), g in groups.items():
        g.sort(key=lambda n: (n["start"], -n["end"]))
        kept: list[dict] = []
        for n in g:
            if kept and abs(kept[-1]["start"] - n["start"]) < 1e-6:
                kept[-1]["end"] = max(kept[-1]["end"], n["end"])
                kept[-1]["vel"] = max(kept[-1].get("vel", 80), n.get("vel", 80))
                continue
            if kept and kept[-1]["end"] > n["start"]:
                kept[-1]["end"] = n["start"]
            kept.append(n)
        out.extend(k for k in kept if k["end"] - k["start"] > 1e-6)
    out.sort(key=lambda n: (n["start"], n["pitch"]))
    return out


def thin_collisions(notes: list[dict]) -> list[dict]:
    """速いパッセージを粗い格子に量子化したとき、別々の音が同じ点に潰れて
    和音に化けたものを、格子に近かった方だけ残して間引く。"""
    out = []
    groups: dict[tuple[str, float], list[dict]] = {}
    for n in notes:
        groups.setdefault((n.get("hand", "R"), n["start"]), []).append(n)
    for (_h, s), g in groups.items():
        if len(g) == 1:
            out.extend(g)
            continue
        g.sort(key=lambda n: n.get("orig_start", s))
        blocks: list[list[dict]] = []
        for n in g:
            if blocks and abs(n.get("orig_start", s) - blocks[-1][0].get("orig_start", s)) < 0.07:
                blocks[-1].append(n)
            else:
                blocks.append([n])
        best = min(blocks, key=lambda b: abs(b[0].get("orig_start", s) - s))
        out.extend(best)
    out.sort(key=lambda n: (n["start"], n["pitch"]))
    return out
