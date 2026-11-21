"""小節ごとの難しさ分析と、曲全体の推定難易度。

速さ（1秒あたりの打鍵数）・和音の大きさと開き・跳躍の速さ・連符/ポリリズム・
運指コスト・両手の独立 を 0〜1 にならして重みづけする。
"""
from __future__ import annotations

import math

import numpy as np

from .theory import is_black

LEVELS = [
    (2, "入門"),
    (4, "初級"),
    (6, "中級"),
    (8, "上級"),
    (10, "超上級 (Virtuoso)"),
]

SEMITONE_TO_DEGREE = [1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7]


def interval_degree(semitones: int) -> int:
    o, r = divmod(abs(semitones), 12)
    return SEMITONE_TO_DEGREE[r] + 7 * o


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _tuplet_beats(notes: list[dict]) -> set[int]:
    out = set()
    for n in notes:
        for x in (n["start"], n["start"] + n["dur"]):
            fr = x - math.floor(x)
            if fr > 1e-6 and abs(fr * 4 - round(fr * 4)) > 1e-4 and abs(fr * 8 - round(fr * 8)) > 1e-4:
                out.add(int(math.floor(x)))
    return out


def measure_beats(score: dict) -> int:
    """1小節の長さ（4分音符単位）。6/8 なら 3、12/8 なら 6。"""
    if score.get("measureBeats"):
        return int(round(float(score["measureBeats"])))
    ts = score["timeSignature"]
    den = int(ts[1]) if len(ts) > 1 else 4
    return int(round(int(ts[0]) * 4 / den))


def analyze(score: dict, threshold: float = 0.45) -> dict:
    from . import bars as BARS

    bpm_m = measure_beats(score)
    tempo = float(score.get("tempo", 100))
    sec_per_beat = 60.0 / max(tempo, 20)
    bars = BARS.of_score(score)
    n_meas = BARS.count(bars)
    notes = score["notes"]
    measures = []
    by_meas: dict[int, dict[str, list[dict]]] = {}
    for n in notes:
        m = BARS.index(n["start"], bars)
        by_meas.setdefault(m, {"R": [], "L": []}).setdefault(n.get("hand", "R"), []).append(n)
    tup = {h: _tuplet_beats([n for n in notes if n.get("hand") == h]) for h in ("R", "L")}

    for m in range(n_meas):
        hands = by_meas.get(m, {"R": [], "L": []})
        per_hand = {}
        meas_len = BARS.length(m, bars)
        meas_sec = meas_len * sec_per_beat
        for h in ("R", "L"):
            hn = sorted(hands.get(h, []), key=lambda n: (n["start"], n["pitch"]))
            if not hn:
                per_hand[h] = None
                continue
            starts = sorted({n["start"] for n in hn})
            nps = len(starts) / meas_sec
            groups: dict[float, list[int]] = {}
            for n in hn:
                groups.setdefault(n["start"], []).append(n["pitch"])
            max_chord = max(len(v) for v in groups.values())
            max_span = max(max(v) - min(v) for v in groups.values())
            leap_speed = 0.0
            seq = [(s, sorted(groups[s])) for s in starts]
            for (s0, p0), (s1, p1) in zip(seq, seq[1:]):
                dt = max((s1 - s0) * sec_per_beat, 0.06)
                jump = max(abs(p1[0] - p0[0]), abs(p1[-1] - p0[-1]))
                if jump >= 7:
                    leap_speed = max(leap_speed, jump / dt)
            fcost = float(np.mean([n.get("_fcost", 0.0) for n in hn]))
            black = sum(1 for n in hn if is_black(n["pitch"])) / len(hn)
            beats_in = range(int(bars[m]), int(math.ceil(bars[m + 1])))
            tuplets = sum(1 for b in beats_in if b in tup[h])
            per_hand[h] = {
                "nps": round(nps, 2),
                "maxChord": max_chord,
                "maxSpan": max_span,
                "leapSpeed": round(leap_speed, 1),
                "fingerCost": round(fcost, 2),
                "black": round(black, 2),
                "tuplets": tuplets,
            }

        reasons: list[str] = []
        score_h = []
        for h, d in per_hand.items():
            if not d:
                continue
            s_speed = _clip01((d["nps"] - 3.5) / 10)
            s_chord = 0.5 * _clip01((d["maxChord"] - 2) / 3) + 0.5 * _clip01((d["maxSpan"] - 9) / 7)
            s_leap = _clip01((d["leapSpeed"] - 18) / 70)
            s_rhythm = _clip01(d["tuplets"] / max(1, meas_len))
            s_finger = _clip01(d["fingerCost"] / 6)
            s = 0.36 * s_speed + 0.2 * s_chord + 0.18 * s_leap + 0.08 * s_rhythm + 0.18 * s_finger
            score_h.append(s)
            hand_ja = "右手" if h == "R" else "左手"
            if s_speed >= 0.45:
                reasons.append(f"{hand_ja}: 高速パッセージ（毎秒{d['nps']:.0f}音）")
            if s_chord >= 0.5:
                reasons.append(f"{hand_ja}: 広い和音（{interval_degree(d['maxSpan'])}度・{d['maxChord']}音）")
            if s_leap >= 0.45:
                reasons.append(f"{hand_ja}: 速い大跳躍")
            if s_finger >= 0.5:
                reasons.append(f"{hand_ja}: 指使いが窮屈")
            if d["tuplets"]:
                reasons.append(f"{hand_ja}: 連符")
        both = per_hand.get("R") and per_hand.get("L")
        indep = 0.0
        if both and per_hand["R"]["nps"] > 5 and per_hand["L"]["nps"] > 5:
            indep = 0.12
            reasons.append("両手とも速い（両手の独立）")
        if both and (per_hand["R"]["tuplets"] > 0) != (per_hand["L"]["tuplets"] > 0):
            indep += 0.08
            reasons.append("左右で違う連符（ポリリズム）")
        total_s = (max(score_h) if score_h else 0.0) + indep
        measures.append(
            {
                "index": m,
                "score": round(_clip01(total_s), 3),
                "reasons": reasons,
                "hands": per_hand,
            }
        )

    scores = np.array([x["score"] for x in measures]) if measures else np.zeros(1)
    nonempty = np.array([x["score"] for x in measures if x["hands"].get("R") or x["hands"].get("L")]) if measures else scores
    if nonempty.size == 0:
        nonempty = scores
    p90 = float(np.percentile(nonempty, 90))
    mean = float(nonempty.mean())
    overall = 1 + 9 * _clip01(0.65 * p90 + 0.35 * mean) ** 0.85
    overall = round(overall, 1)
    label = next(lbl for lim, lbl in LEVELS if overall <= lim)

    spans = []
    for x in measures:
        if x["score"] >= threshold:
            if spans and spans[-1]["endMeasure"] == x["index"] - 1:
                spans[-1]["endMeasure"] = x["index"]
                spans[-1]["peak"] = max(spans[-1]["peak"], x["score"])
                for r in x["reasons"]:
                    if r not in spans[-1]["reasons"]:
                        spans[-1]["reasons"].append(r)
            else:
                spans.append({"startMeasure": x["index"], "endMeasure": x["index"], "peak": x["score"], "reasons": list(x["reasons"])})

    for n in notes:
        n.pop("_fcost", None)
    return {
        "overall": overall,
        "label": label,
        "threshold": threshold,
        "measures": measures,
        "spans": spans,
        "difficultMeasures": [x["index"] for x in measures if x["score"] >= threshold],
    }
