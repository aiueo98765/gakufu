"""右手・左手の振り分け。

同時に鳴り始める音のかたまり（クラスタ）ごとに「どこで上下に分けるか」を状態とし、
手の開き（10度を超えると苦しい）・音域の自然さ・手の位置の連続性をコストにして
ビタビ（動的計画法）で全体最適な分け方を求める。
"""
from __future__ import annotations

import math

CLUSTER_SEC = 0.05


def _clusters(notes: list[dict], key: str = "on") -> list[list[dict]]:
    order = sorted(notes, key=lambda n: (n[key], n["pitch"]))
    out: list[list[dict]] = []
    tol = CLUSTER_SEC if key == "on" else 1e-6
    for n in order:
        if out and n[key] - out[-1][0][key] <= tol:
            out[-1].append(n)
        else:
            out.append([n])
    for c in out:
        c.sort(key=lambda n: n["pitch"])
    return out


def _emission(pitches: list[int], s: int, forced: list[str | None]) -> float:
    L, R = pitches[:s], pitches[s:]
    c = 0.0
    for hand_notes, is_left in ((L, True), (R, False)):
        if not hand_notes:
            continue
        span = hand_notes[-1] - hand_notes[0]
        c += max(0, span - 12) * 1.5 + max(0, span - 15) * 8
        c += max(0, len(hand_notes) - 5) * 25
        # 中央ド(60)を境に、上は右手・下は左手が自然という弱い事前分布
        if is_left:
            c += sum(max(0, p - 66) * 0.22 for p in hand_notes)
            c += sum(max(0, p - 60) * 0.1 for p in hand_notes)
        else:
            c += sum(max(0, 53 - p) * 0.22 for p in hand_notes)
            c += sum(max(0, 60 - p) * 0.1 for p in hand_notes)
    for i, f in enumerate(forced):
        if f == "R" and i < s:
            c += 40
        elif f == "L" and i >= s:
            c += 40
    return c


def _center(p: list[int]) -> float | None:
    return sum(p) / len(p) if p else None


def _transition(prev: list[int], ps: int, cur: list[int], cs: int, dt: float) -> float:
    c = 0.0
    w = 0.7 if dt < 0.2 else (0.4 if dt < 0.5 else (0.15 if dt < 1.2 else 0.05))
    for a, b in ((prev[:ps], cur[:cs]), (prev[ps:], cur[cs:])):
        ca, cb = _center(a), _center(b)
        if ca is not None and cb is not None:
            c += max(0.0, abs(cb - ca) - 7) * w
    # 直前の右手より上に左手が来る（素早い交差）は少しだけ嫌う
    if prev[ps:] and cur[:cs] and dt < 0.25 and cur[cs - 1] > prev[ps] + 2:
        c += 1.5
    if prev[:ps] and cur[cs:] and dt < 0.25 and cur[cs] < prev[ps - 1] - 2:
        c += 1.5
    return c


def split_hands(notes: list[dict], key: str = "on") -> None:
    """notes に "hand": "R"|"L" を書き込む。role=melody は右、role=bass は左に固定。"""
    if not notes:
        return
    clusters = _clusters(notes, key)
    states: list[list[int]] = []
    pitches: list[list[int]] = []
    emis: list[list[float]] = []
    for cl in clusters:
        p = [n["pitch"] for n in cl]
        forced = [("R" if n.get("role") == "melody" else "L" if n.get("role") == "bass" else None) for n in cl]
        pitches.append(p)
        st = list(range(len(p) + 1))
        states.append(st)
        emis.append([_emission(p, s, forced) for s in st])

    times = [cl[0][key] for cl in clusters]
    cost = list(emis[0])
    back: list[list[int]] = [[0] * len(states[0])]
    for i in range(1, len(clusters)):
        dt = times[i] - times[i - 1]
        if key != "on":
            dt *= 0.5  # 拍単位のときはおおよそ秒に寄せる
        new_cost = []
        new_back = []
        for cs in states[i]:
            best, arg = math.inf, 0
            for ps in states[i - 1]:
                v = cost[ps] + _transition(pitches[i - 1], ps, pitches[i], cs, dt)
                if v < best:
                    best, arg = v, ps
            new_cost.append(best + emis[i][cs])
            new_back.append(arg)
        cost = new_cost
        back.append(new_back)
    s = min(range(len(cost)), key=lambda k: cost[k])
    for i in range(len(clusters) - 1, -1, -1):
        for j, n in enumerate(clusters[i]):
            n["hand"] = "L" if j < s else "R"
        s = back[i][s]
