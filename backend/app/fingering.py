"""指番号（運指）の自動付与。

Parncutt ら (1997) の運指コストモデルを簡略化して使う。
指の組ごとに「楽に届く幅」「無理なく届く幅」「物理的な限界」を半音で持ち、
はみ出し量・弱い指(4,5)・黒鍵の親指・指くぐり などに罰点をつけ、
片手ごとに音（和音）の並び全体をビタビで最小化する。

利用者が指を手で決めた音（fingerLocked）はその指に固定したまま残りを最適化する。
"""
from __future__ import annotations

import itertools
import math

from .theory import is_black

# 右手。(MinPrac, MinComf, MinRel, MaxRel, MaxComf, MaxPrac)  距離 = 後の指の音 - 前の指の音
SPAN = {
    (1, 2): (-5, -3, 1, 5, 8, 10),
    (1, 3): (-4, -2, 3, 7, 10, 12),
    (1, 4): (-3, -1, 5, 9, 12, 14),
    (1, 5): (-1, 1, 7, 10, 13, 15),
    (2, 3): (1, 1, 1, 2, 3, 5),
    (2, 4): (1, 1, 3, 4, 5, 7),
    (2, 5): (2, 2, 5, 6, 8, 10),
    (3, 4): (1, 1, 1, 2, 2, 4),
    (3, 5): (1, 1, 3, 4, 5, 7),
    (4, 5): (1, 1, 1, 2, 3, 5),
}


def _limits(fa: int, fb: int):
    if fa < fb:
        return SPAN[(fa, fb)]
    a, b, c, d, e, f = SPAN[(fb, fa)]
    return (-f, -e, -d, -c, -b, -a)


def pair_cost(fa: int, pa: int, fb: int, pb: int, hand: str, dt: float, held: bool = False) -> float:
    """音 a（指 fa）から音 b（指 fb）へ移る罰点。"""
    d = (pb - pa) if hand == "R" else (pa - pb)
    if fa == fb:
        if pa == pb:
            return 0.4 if dt < 0.12 else 0.0  # 速い同音連打は指替えが望ましい
        # 同じ指で別の鍵へ：レガートが切れる。かなり間が空くときだけ許す
        if dt > 1.4:
            return 1.0 + 0.25 * abs(d)
        if dt > 0.7:
            return 4.0 + 1.0 * abs(d)
        return 7.0 + 1.5 * abs(d)
    mn_p, mn_c, mn_r, mx_r, mx_c, mx_p = _limits(fa, fb)
    c = 0.0
    if d > mx_c:
        c += 2 * (d - mx_c)
    if d < mn_c:
        c += 2 * (mn_c - d)
    if d > mx_p:
        c += 10 * (d - mx_p)
    if d < mn_p:
        c += 10 * (mn_p - d)
    thumb = 1 in (fa, fb)
    if d < mn_r:
        c += (mn_r - d) * (2 if thumb else 1)
    if d > mx_r:
        c += (d - mx_r) * (1 if thumb else 2)
    if fb in (4, 5):
        c += 0.6
    if fa == 3 and fb == 4:
        c += 1
        if not is_black(pa) and is_black(pb):
            c += 1
    ba, bb = is_black(pa), is_black(pb)
    if fb == 1 and bb:
        c += 1 + (2 if not ba else 0)
    if fb == 5 and bb and not ba:
        c += 2
    # 親指くぐり / 指越え
    passing = (fb == 1 and fa != 1 and d > 0) or (fa == 1 and fb != 1 and d < 0)
    if passing:
        thumb_black = bb if fb == 1 else ba
        other_black = ba if fb == 1 else bb
        if thumb_black and not other_black:
            c += 3
        elif thumb_black == other_black:
            c += 1
    if held:
        c += 0.5
    # 間が空けば手を置き直せる。速いほど厳しく。
    if dt > 1.2:
        c *= 0.25
    elif dt > 0.6:
        c *= 0.55
    elif dt < 0.13:
        c *= 1.35
    return c


def chord_internal_cost(pitches: list[int], fingers: tuple[int, ...], hand: str) -> float:
    """1つの和音を押さえる指の広げ具合。pitches は低→高。"""
    if len(pitches) < 2:
        return 0.0
    pairs = list(zip(pitches, fingers))
    if hand == "L":
        pairs = pairs[::-1]  # 左手は高い音が若い番号
    c = 0.0
    for (pa, fa), (pb, fb) in zip(pairs, pairs[1:]):
        d = abs(pb - pa)
        mn_p, mn_c, mn_r, mx_r, mx_c, mx_p = _limits(min(fa, fb), max(fa, fb))
        if d > mx_c:
            c += 1.5 * (d - mx_c)
        if d > mx_p:
            c += 12 * (d - mx_p)
        if d < max(mn_c, 0):
            c += 1.5 * (max(mn_c, 0) - d)
        # 「楽な幅」から外れる分も少しだけ（C-E-G を 5-3-1 にする等）
        if d < mn_r:
            c += 0.5 * (mn_r - d)
        elif d > mx_r:
            c += 0.5 * (d - mx_r)
    if any(f in (4, 5) for f in fingers) and len(fingers) <= 2 and max(pitches) - min(pitches) < 7:
        c += 0.5
    return c


def _candidates(pitches: list[int], hand: str, locked: list[int | None]) -> list[tuple[int, ...]]:
    k = len(pitches)
    if k == 0:
        return [()]
    if k > 5:
        k = 5
    out = []
    for combo in itertools.combinations(range(1, 6), k):
        fingers = combo if hand == "R" else tuple(reversed(combo))
        ok = True
        for f, lk in zip(fingers, locked[:k]):
            if lk and f != lk:
                ok = False
                break
        if ok:
            out.append(fingers)
    if not out:  # 固定指どうしが矛盾していたら固定を無視
        out = [c if hand == "R" else tuple(reversed(c)) for c in itertools.combinations(range(1, 6), k)]
    return out


def _events(notes: list[dict], hand: str) -> list[dict]:
    hn = sorted([n for n in notes if n.get("hand") == hand], key=lambda n: (n["start"], n["pitch"]))
    events: list[dict] = []
    for n in hn:
        if events and abs(events[-1]["start"] - n["start"]) < 1e-6:
            events[-1]["notes"].append(n)
        else:
            events.append({"start": n["start"], "notes": [n]})
    for e in events:
        e["notes"].sort(key=lambda n: n["pitch"])
        if len(e["notes"]) > 5:
            # 6音以上は片手で押さえられない。外側5音にだけ指をつける
            e["extra"] = e["notes"][:-5] if hand == "R" else e["notes"][5:]
            e["notes"] = e["notes"][-5:] if hand == "R" else e["notes"][:5]
        e["pitches"] = [n["pitch"] for n in e["notes"]]
        e["end"] = max(n["start"] + n["dur"] for n in e["notes"])
    return events


def assign_fingering(score: dict) -> None:
    """score["notes"] に "finger" を書き込む。音ごとの移動コストも "_fcost" として返す。"""
    sec_per_beat = 60.0 / max(20.0, float(score.get("tempo", 100)))
    notes = score["notes"]
    for hand in ("R", "L"):
        events = _events(notes, hand)
        if not events:
            continue
        cands = []
        for e in events:
            locked = [n.get("finger") if n.get("fingerLocked") else None for n in e["notes"]]
            cs = _candidates(e["pitches"], hand, locked)
            scored = [(f, chord_internal_cost(e["pitches"], f, hand)) for f in cs]
            # 物理的に無理な押さえ方は、他に手があるなら候補から外す
            good = [x for x in scored if x[1] < 25]
            cands.append(good or scored)
        cost = [ic for (_f, ic) in cands[0]]
        back: list[list[int]] = [[0] * len(cands[0])]
        for i in range(1, len(events)):
            prev, cur = events[i - 1], events[i]
            dt = (cur["start"] - prev["start"]) * sec_per_beat
            held_until = prev["end"]
            new_cost, new_back = [], []
            for f_cur, ic in cands[i]:
                best, arg = math.inf, 0
                for j, (f_prev, _pic) in enumerate(cands[i - 1]):
                    t = _transition(prev, f_prev, cur, f_cur, hand, dt, held_until > cur["start"] + 1e-6)
                    v = cost[j] + t
                    if v < best:
                        best, arg = v, j
                new_cost.append(best + ic)
                new_back.append(arg)
            cost = new_cost
            back.append(new_back)
        k = min(range(len(cost)), key=lambda x: cost[x])
        path = [0] * len(events)
        for i in range(len(events) - 1, -1, -1):
            path[i] = k
            k = back[i][k]
        prev_choice = None
        for i, e in enumerate(events):
            fingers, ic = cands[i][path[i]]
            step = ic
            if prev_choice is not None:
                pe, pf = prev_choice
                dt = (e["start"] - pe["start"]) * sec_per_beat
                step += _transition(pe, pf, e, fingers, hand, dt, pe["end"] > e["start"] + 1e-6)
            for n, f in zip(e["notes"], fingers):
                if not n.get("fingerLocked"):
                    n["finger"] = int(f)
                n["_fcost"] = round(step, 2)
            for n in e.get("extra", []):
                n.pop("finger", None)
                n["_fcost"] = round(step, 2)
            prev_choice = (e, fingers)


def _transition(pe: dict, pf: tuple[int, ...], ce: dict, cf: tuple[int, ...], hand: str, dt: float, held: bool) -> float:
    # 外側どうし（最低音どうし・最高音どうし）の移動で代表させる
    c = pair_cost(pf[0], pe["pitches"][0], cf[0], ce["pitches"][0], hand, dt)
    if len(pf) > 1 or len(cf) > 1:
        c = (c + pair_cost(pf[-1], pe["pitches"][-1], cf[-1], ce["pitches"][-1], hand, dt)) / 2
    if held:
        # 伸ばしている音の指は使えない
        for n, f in zip(pe["notes"], pf):
            if n["start"] + n["dur"] > ce["start"] + 1e-6 and f in cf:
                idx = cf.index(f)
                if ce["pitches"][idx] != n["pitch"]:
                    c += 8
    return c
